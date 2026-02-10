import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
import pandas as pd
from pathlib import Path

from data import load_datasets
from model import get_model, get_tokenizer, categories


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate predictions on dev set')
    parser.add_argument('--classifier', type=str, choices=['linear', 'attention'], 
                        required=True, help='Type of classifier used')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--output', type=str, default='dev_predictions.csv', 
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading dev dataset...")
    _, _, dev_df, _ = load_datasets()
    print(f"Dev set size: {len(dev_df)}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer()
    
    # Tokenize dev texts
    print("Tokenizing dev set...")
    dev_encodings = tokenizer(
        dev_df['text'].tolist(),
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Create dataset
    dev_dataset = TensorDataset(
        dev_encodings['input_ids'],
        dev_encodings['attention_mask']
    )
    
    # Create dataloader
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    print(f"Loading model with classifier type: {args.classifier}")
    model = get_model(classifier_type=args.classifier)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Generate predictions
    print("Generating predictions...")
    all_category_predictions = []
    all_pcl_predictions = []
    
    with torch.no_grad():
        for input_ids, attention_mask in dev_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Forward pass (sigmoid is applied automatically in eval mode)
            preds_dict = model(input_ids, attention_mask)
            
            # Stack category predictions in category order
            category_preds = torch.stack([preds_dict[cat] for cat in categories], dim=1)
            all_category_predictions.append(category_preds.cpu())
            
            # Get PCL predictions
            pcl_preds = preds_dict['pcl']
            all_pcl_predictions.append(pcl_preds.cpu())
    
    # Concatenate all predictions
    all_category_predictions = torch.cat(all_category_predictions, dim=0).numpy()
    all_pcl_predictions = torch.cat(all_pcl_predictions, dim=0).numpy()
    
    # Create output dataframe
    results = pd.DataFrame({
        'par_id': dev_df['par_id'].values,
        'pcl': all_pcl_predictions  # PCL prediction first (main task)
    })
    
    # Add predicted scores for each category
    for idx, category in enumerate(categories):
        results[category] = all_category_predictions[:, idx]
    
    # Save to CSV
    results.to_csv(args.output, index=False)
    print(f"\nPredictions saved to: {args.output}")
    print(f"Output shape: {results.shape}")
    print(f"\nPCL prediction statistics:")
    print(f"  Mean: {all_pcl_predictions.mean():.4f}")
    print(f"  Std:  {all_pcl_predictions.std():.4f}")
    print(f"  Min:  {all_pcl_predictions.min():.4f}")
    print(f"  Max:  {all_pcl_predictions.max():.4f}")
    print("\nSample predictions:")
    print(results.head())


if __name__ == '__main__':
    main()
