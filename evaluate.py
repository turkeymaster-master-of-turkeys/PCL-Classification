import pandas as pd
import argparse
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    accuracy_score, hamming_loss
)

from data import load_datasets
from model import categories


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate predictions on dev set')
    parser.add_argument('--predictions', type=str, required=True, 
                        help='Path to predictions CSV file')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Threshold for binary classification (default: 0.5)')
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from: {args.predictions}")
    preds_df = pd.read_csv(args.predictions)
    print(f"Predictions shape: {preds_df.shape}")
    
    # Load dev dataset to get ground truth
    print("\nLoading dev dataset...")
    _, _, dev_df, _ = load_datasets()
    print(f"Dev set size: {len(dev_df)}")
    
    # Merge predictions with ground truth based on par_id
    merged = preds_df.merge(dev_df[['par_id'] + categories], on='par_id', suffixes=('_pred', '_true'))
    print(f"Merged size: {len(merged)}")
    
    # Extract predicted scores and ground truth labels
    y_pred_scores = merged[[f'{cat}_pred' for cat in categories]].values
    y_true = (merged[[f'{cat}_true' for cat in categories]] > 0).astype(int).values
    
    # Convert predicted scores to binary using threshold
    y_pred = (y_pred_scores >= args.threshold).astype(int)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS (threshold={args.threshold})")
    print(f"{'='*60}\n")
    
    # Overall metrics
    print("OVERALL METRICS:")
    print(f"  Micro F1:      {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"  Macro F1:      {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Weighted F1:   {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  Micro Prec:    {precision_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"  Macro Prec:    {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Micro Recall:  {recall_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"  Macro Recall:  {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Hamming Loss:  {hamming_loss(y_true, y_pred):.4f}")
    print(f"  Subset Acc:    {accuracy_score(y_true, y_pred):.4f}")
    
    # Per-category metrics
    print(f"\n{'='*60}")
    print("PER-CATEGORY METRICS:")
    print(f"{'='*60}")
    print(f"{'Category':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print(f"{'-'*60}")
    
    per_class_metrics = []
    for idx, category in enumerate(categories):
        y_true_cat = y_true[:, idx]
        y_pred_cat = y_pred[:, idx]
        
        prec = precision_score(y_true_cat, y_pred_cat, zero_division=0)
        rec = recall_score(y_true_cat, y_pred_cat, zero_division=0)
        f1 = f1_score(y_true_cat, y_pred_cat, zero_division=0)
        support = y_true_cat.sum()
        
        print(f"{category:<10} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {support:>8.0f}")
        
        per_class_metrics.append({
            'category': category,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'support': support
        })
    
    print(f"{'-'*60}")
    
    # Additional analysis
    print(f"\n{'='*60}")
    print("PREDICTION STATISTICS:")
    print(f"{'='*60}")
    print(f"Total samples: {len(y_true)}")
    print(f"Total positive labels (ground truth): {y_true.sum()}")
    print(f"Total positive predictions: {y_pred.sum()}")
    print(f"Avg labels per sample (ground truth): {y_true.sum(axis=1).mean():.2f}")
    print(f"Avg predictions per sample: {y_pred.sum(axis=1).mean():.2f}")
    
    # Category-wise prediction counts
    print(f"\nPer-category prediction counts:")
    print(f"{'Category':<10} {'True Pos':>10} {'Predicted':>10}")
    print(f"{'-'*35}")
    for idx, category in enumerate(categories):
        print(f"{category:<10} {y_true[:, idx].sum():>10.0f} {y_pred[:, idx].sum():>10.0f}")
    
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
