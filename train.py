import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import csv
from pathlib import Path
import numpy as np
import warnings

# Silence sharding warning from DistilBERT
warnings.filterwarnings('ignore', message='.*were not sharded.*')

from data import load_datasets
from model import get_model, get_tokenizer, categories


def focal_loss_fn(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal loss for multi-label classification.
    
    Args:
        logits: predicted logits (before sigmoid)
        targets: ground truth binary labels
        alpha: weighting factor for positive class (default: 0.25)
        gamma: focusing parameter (default: 2.0)
    """
    probs = torch.sigmoid(logits)
    bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_loss = alpha_t * focal_weight * bce_loss
    
    return focal_loss.mean()


def pcl_loss_fn(logits, pcl_scores):
    """
    Binary PCL loss: target 1 if pcl_score > 0, else 0
    """
    targets = (pcl_scores > 0).float()
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return loss


def select_hard_negatives(model, input_ids_all, attention_mask_all,
                          negative_indices, num_positives, device):
    """
    Select hard negatives from non-PCL samples.
    - 70% from top 5% most difficult (highest predicted probability)
    - 30% random
    - Total number equals num_positives
    
    Args:
        model: The model to use for predictions
        input_ids_all: All input_ids tensors
        attention_mask_all: All attention_mask tensors
        negative_indices: Indices of negative samples (pcl_score == 0)
        num_positives: Number of positive samples (to match)
        device: Device to run inference on
    
    Returns:
        Selected indices from negative_indices
    """
    model.eval()
    
    # Get predictions for all negative samples
    neg_input_ids = input_ids_all[negative_indices]
    neg_attention_mask = attention_mask_all[negative_indices]
    
    predictions = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(neg_input_ids), batch_size):
            batch_ids = neg_input_ids[i:i+batch_size].to(device)
            batch_mask = neg_attention_mask[i:i+batch_size].to(device)
            
            logits_dict = model(batch_ids, batch_mask)
            pcl_logits = logits_dict['pcl']
            pcl_probs = torch.sigmoid(pcl_logits)
            predictions.extend(pcl_probs.cpu().numpy())
    
    predictions = np.array(predictions)
    
    # Calculate number of hard and random negatives needed
    num_hard = int(num_positives * 0.7)
    num_random = num_positives - num_hard
    
    # Get top 15% most difficult (highest probability for negatives)
    top_15_percent_count = max(1, int(len(predictions) * 0.15))
    hard_candidate_indices = np.argsort(predictions)[-top_15_percent_count:]
    
    # Sample from hard candidates
    if len(hard_candidate_indices) >= num_hard:
        hard_selected = np.random.choice(hard_candidate_indices, size=num_hard, replace=False)
    else:
        # If not enough hard candidates, take all and fill remainder with random
        hard_selected = hard_candidate_indices
        num_random += (num_hard - len(hard_selected))
    
    # Sample random negatives (excluding already selected hard negatives)
    remaining_indices = np.setdiff1d(np.arange(len(predictions)), hard_selected)
    if len(remaining_indices) >= num_random:
        random_selected = np.random.choice(remaining_indices, size=num_random, replace=False)
    else:
        random_selected = remaining_indices
    
    # Combine and map back to original indices
    selected_local_indices = np.concatenate([hard_selected, random_selected])
    selected_global_indices = negative_indices[selected_local_indices]
    
    print(f"  Selected {len(hard_selected)} hard negatives (from top {top_15_percent_count}), {len(random_selected)} random negatives")
    print(f"  Hard negative probability range: [{predictions[hard_selected].min():.3f}, {predictions[hard_selected].max():.3f}]")
    
    return selected_global_indices


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train PCL category classifier')
    parser.add_argument('--classifier', type=str, choices=['linear', 'attention'], 
                        default='linear', help='Type of classifier to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for optimizer')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', 
                        help='Directory to save the best model')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_df, val_df, _, _ = load_datasets()
    print(f"Train set size: {len(train_df)}")
    print(f"Val set size: {len(val_df)}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer()
    
    # Tokenize texts
    print("Tokenizing training set...")
    train_encodings = tokenizer(
        train_df['text'].tolist(),
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    print("Tokenizing validation set...")
    val_encodings = tokenizer(
        val_df['text'].tolist(),
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Prepare labels: convert scores to binary labels
    # Score 1 or 2 = presence (1), score 0 = absence (0)
    print("Preparing labels...")
    train_labels = (train_df[categories] > 0).astype(int).values.copy()
    train_labels = torch.FloatTensor(train_labels)
    
    val_labels = (val_df[categories] > 0).astype(int).values.copy()
    val_labels = torch.FloatTensor(val_labels)
    
    # Prepare PCL scores
    print("Preparing PCL scores...")
    train_pcl_scores = torch.FloatTensor(train_df['pcl_score'].fillna(0).values)
    val_pcl_scores = torch.FloatTensor(val_df['pcl_score'].fillna(0).values)
    
    # Store all training data for hard negative mining
    train_input_ids = torch.stack([enc.squeeze(0) for enc in train_encodings['input_ids']])
    train_attention_masks = torch.stack([enc.squeeze(0) for enc in train_encodings['attention_mask']])
    
    # Identify positive and negative samples
    positive_mask = train_pcl_scores > 0
    positive_indices = torch.where(positive_mask)[0].numpy()
    negative_indices = torch.where(~positive_mask)[0].numpy()
    
    print(f"Training data: {len(positive_indices)} positives, {len(negative_indices)} negatives")
    
    # Initial training indices: all positives + random negatives (balanced)
    num_positives = len(positive_indices)
    selected_negative_indices = np.random.choice(negative_indices, size=min(num_positives, len(negative_indices)), replace=False)
    train_indices = np.concatenate([positive_indices, selected_negative_indices])
    np.random.shuffle(train_indices)
    
    # Create initial dataset with balanced data
    train_dataset = TensorDataset(
        train_input_ids[train_indices],
        train_attention_masks[train_indices],
        train_labels[train_indices],
        train_pcl_scores[train_indices]
    )
    
    val_dataset = TensorDataset(
        torch.stack([enc.squeeze(0) for enc in val_encodings['input_ids']]),
        torch.stack([enc.squeeze(0) for enc in val_encodings['attention_mask']]),
        val_labels,
        val_pcl_scores
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    print(f"Loading model with classifier type: {args.classifier}")
    model = get_model(classifier_type=args.classifier)
    model = model.to(device)
    print("Training LoRA adapters + classifier (base model frozen)...")
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    loss_history = []
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / f'best_model_{args.classifier}.pt'
    loss_csv_path = output_dir / f'loss_history_{args.classifier}.csv'
    
    print(f"\nStarting training with early stopping (patience={args.patience})...")
    print(f"Best model will be saved to: {best_model_path}\n")
    
    for epoch in range(args.epochs):
        # Hard negative mining before every 5th epoch
        if epoch > 0 and epoch % 5 == 0:
            print(f"\n[Epoch {epoch + 1}] Performing hard negative mining...")
            selected_negative_indices = select_hard_negatives(
                model, 
                train_input_ids, 
                train_attention_masks,
                negative_indices, 
                num_positives, 
                device
            )
            
            # Rebuild dataset with all positives + selected negatives
            train_indices = np.concatenate([positive_indices, selected_negative_indices])
            np.random.shuffle(train_indices)
            
            train_dataset = TensorDataset(
                train_input_ids[train_indices],
                train_attention_masks[train_indices],
                train_labels[train_indices],
                train_pcl_scores[train_indices]
            )
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            print(f"  Rebuilt training set with {len(train_indices)} samples (balanced)")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_cat_loss = 0.0
        train_pcl_loss = 0.0
        
        for batch_idx, (input_ids, attention_mask, labels, pcl_scores) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            pcl_scores = pcl_scores.to(device)
            
            # Forward pass
            logits_dict = model(input_ids, attention_mask)
            
            # Stack category logits in category order
            category_logits = torch.stack([logits_dict[cat] for cat in categories], dim=1)
            pcl_logits = logits_dict['pcl']
            
            # Compute losses
            cat_loss = focal_loss_fn(category_logits, labels)
            pcl_loss = pcl_loss_fn(pcl_logits, pcl_scores)
            loss = cat_loss + pcl_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_cat_loss += cat_loss.item()
            train_pcl_loss += pcl_loss.item()
        
        train_loss /= len(train_loader)
        train_cat_loss /= len(train_loader)
        train_pcl_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_cat_loss = 0.0
        val_pcl_loss = 0.0
        
        with torch.no_grad():
            for input_ids, attention_mask, labels, pcl_scores in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                pcl_scores = pcl_scores.to(device)
                
                # Forward pass
                logits_dict = model(input_ids, attention_mask)
                
                # Stack category logits in category order
                category_logits = torch.stack([logits_dict[cat] for cat in categories], dim=1)
                pcl_logits = logits_dict['pcl']
                
                # Compute losses
                cat_loss = focal_loss_fn(category_logits, labels)
                pcl_loss = pcl_loss_fn(pcl_logits, pcl_scores)
                loss = cat_loss + pcl_loss
                
                val_loss += loss.item()
                val_cat_loss += cat_loss.item()
                val_pcl_loss += pcl_loss.item()
        
        val_loss /= len(val_loader)
        val_cat_loss /= len(val_loader)
        val_pcl_loss /= len(val_loader)
        
        loss_history.append({
            'epoch': epoch + 1, 
            'train_loss': train_loss, 
            'val_loss': val_loss,
            'train_cat_loss': train_cat_loss,
            'val_cat_loss': val_cat_loss,
            'train_pcl_loss': train_pcl_loss,
            'val_pcl_loss': val_pcl_loss
        })
        
        print(f"Epoch {epoch + 1:3d} | Train: {train_loss:.4f} (cat: {train_cat_loss:.4f}, pcl: {train_pcl_loss:.4f}) | Val: {val_loss:.4f} (cat: {val_cat_loss:.4f}, pcl: {val_pcl_loss:.4f})", end="")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(" ✓ (best model saved)")
        else:
            patience_counter += 1
            print(f" (patience: {patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    # Save loss history to CSV
    with open(loss_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'train_cat_loss', 'val_cat_loss', 'train_pcl_loss', 'val_pcl_loss'])
        writer.writeheader()
        writer.writerows(loss_history)
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Loss history saved to: {loss_csv_path}")


if __name__ == '__main__':
    main()

