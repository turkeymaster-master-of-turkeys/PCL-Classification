import pandas as pd
from ast import literal_eval
from pathlib import Path
from typing import Tuple, Dict
import re
from io import StringIO
import numpy as np

DATA_DIR = Path(__file__).parent / "data"


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load datasets from PCL and category files, split by provided CSV files.
    
    Returns:
        train_df, val_df, dev_df, test_df
    """
    
    # Category mapping
    category_map = {
        'Unbalanced_power_relations': 'unb',
        'Shallow_solution': 'shal',
        'Presupposition': 'pres',
        'Authority_voice': 'auth',
        'Metaphors': 'met',
        'Compassion': 'comp',
        'The_poorer_the_merrier': 'merr'
    }
    category_cols = list(category_map.values())
    
    # 1. Load PCL data (text and PCL scores)
    print("Loading PCL data from dontpatronizeme_pcl.tsv...")
    pcl_path = DATA_DIR / "dontpatronizeme_pcl.tsv"
    with open(pcl_path, "r", encoding="utf-8") as fh:
        pcl_lines = [ln for ln in fh if re.match(r"^\d+\t@@", ln)]
    
    pcl_df = pd.read_csv(
        StringIO("".join(pcl_lines)),
        sep="\t",
        header=None,
        names=["par_id", "text_id", "keyword", "country", "text", "pcl_score"],
        engine="python",
    )
    pcl_df['par_id'] = pcl_df['par_id'].astype(int)
    pcl_df['pcl_score'] = pd.to_numeric(pcl_df['pcl_score'], errors='coerce').fillna(0).astype(int)
    
    # Keep one row per par_id (text and pcl_score)
    pcl_data = pcl_df[['par_id', 'text', 'pcl_score']].drop_duplicates(subset=['par_id'])
    
    # 2. Load category data
    print("Loading category data from dontpatronizeme_categories.tsv...")
    cat_path = DATA_DIR / "dontpatronizeme_categories.tsv"
    with open(cat_path, "r", encoding="utf-8") as fh:
        cat_lines = [ln for ln in fh if re.match(r"^\d+\t@@", ln)]
    
    cat_df = pd.read_csv(
        StringIO("".join(cat_lines)),
        sep="\t",
        header=None,
        names=["par_id", "text_id", "text", "keyword", "country", "start", "end", "span_text", "category", "score"],
        engine="python",
    )
    cat_df['par_id'] = cat_df['par_id'].astype(int)
    cat_df['score'] = pd.to_numeric(cat_df['score'], errors='coerce').fillna(0).astype(int)
    
    # Aggregate category scores per paragraph (max score per category)
    category_data = {}
    for par_id, group in cat_df.groupby('par_id'):
        category_data[par_id] = {}
        for _, row in group.iterrows():
            category = row['category']
            score = row['score']
            if pd.notna(category):
                abbr = category_map.get(category)
                if abbr:
                    category_data[par_id][abbr] = max(category_data[par_id].get(abbr, 0), score)
    
    # Add category columns to pcl_data
    for abbr in category_cols:
        pcl_data[abbr] = pcl_data['par_id'].map(lambda x: category_data.get(x, {}).get(abbr, 0))
    
    # 3. Split into train/dev/test using CSV files
    print("Loading train/dev split files...")
    train_ids_df = pd.read_csv(DATA_DIR / "train_semeval_parids-labels.csv")
    train_ids_df['par_id'] = train_ids_df['par_id'].astype(int)
    train_ids_df['label'] = train_ids_df['label'].apply(literal_eval)
    
    dev_ids_df = pd.read_csv(DATA_DIR / "dev_semeval_parids-labels.csv")
    dev_ids_df['par_id'] = dev_ids_df['par_id'].astype(int)
    dev_ids_df['label'] = dev_ids_df['label'].apply(literal_eval)
    
    # Merge with PCL data
    train_df = train_ids_df.merge(pcl_data, on='par_id', how='left')
    dev_df = dev_ids_df.merge(pcl_data, on='par_id', how='left')
    
    # Fill missing values
    train_df['pcl_score'] = train_df['pcl_score'].fillna(0).astype(int)
    dev_df['pcl_score'] = dev_df['pcl_score'].fillna(0).astype(int)
    for abbr in category_cols:
        train_df[abbr] = train_df[abbr].fillna(0).astype(int)
        dev_df[abbr] = dev_df[abbr].fillna(0).astype(int)
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(
        DATA_DIR / "task4_test.tsv",
        sep="\t",
        header=None,
        names=['test_id', 'text_id', 'keyword', 'country', 'text']
    )
    test_df['pcl_score'] = 0
    for abbr in category_cols:
        test_df[abbr] = 0
    
    # 4. Split train into train (80%) and validation (20%)
    val_idx = np.random.choice(train_df.index, size=int(0.2 * len(train_df)), replace=False)
    val_df = train_df.loc[val_idx].copy().reset_index(drop=True)
    train_df = train_df.drop(val_idx).copy().reset_index(drop=True)
    
    print(f"Train set: {len(train_df)} samples (80%)")
    print(f"Validation set: {len(val_df)} samples (20%)")
    print(f"Dev set: {len(dev_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, val_df, dev_df, test_df


def print_label_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Get distribution of labels in the dataset."""
    num_positive = [score > 0 for score in df['pcl_score']]
    total_samples = len(num_positive)
    positive_samples = sum(num_positive)
    percentage = (positive_samples / total_samples * 100) if total_samples > 0 else 0
    print(f"Positive samples: {positive_samples} ({percentage:.1f}%)")
    for i in range(1, 5):
        count = sum(df['pcl_score'] == i)
        p = (count / positive_samples * 100) if positive_samples > 0 else 0
        print(f"  Score {i}: {count} ({p:.1f}%)")
    print(f"Negative samples: {total_samples - positive_samples} ({100 - percentage:.1f}%)")
    
    # Analyze category scores
    print("\n--- Category Analysis ---")
    
    # Define category order (abbreviations) to match labels file
    category_cols = [
        'unb',
        'shal',
        'pres',
        'auth',
        'met',
        'comp',
        'merr'
    ]
    
    # Print category sample counts
    print(f"\nSamples per category:")
    for category in category_cols:
        count = (df[category] > 0).sum()
        print(f"  {category}: {count}")
    
    # Print score distribution per category
    print(f"\nScore distribution per category:")
    for category in category_cols:
        scores = df[df[category] > 0][category].value_counts().sort_index()
        if len(scores) > 0:
            print(f"  {category}:")
            total = len(df[df[category] > 0])
            for score in scores.index:
                count = scores[score]
                percentage = (count / total) * 100
                print(f"    Score {score}: {count} ({percentage:.1f}%)")

    # Co-occurrence (counts) and covariance matrix for categories
    print("\n--- Category Co-occurrence & Covariance ---")
    # Binary presence matrix: 1 if category score > 0 else 0
    df_bin = df[category_cols].gt(0).astype(int)

    # Co-occurrence counts
    cooccurrence = df_bin.T.dot(df_bin)
    print("\nCo-occurrence counts (rows=A, cols=B):")
    print(cooccurrence.to_string())

    # Conditional co-occurrence: for each category A, percent of A's occurrences
    # that also contain B (i.e. cooccurrence[A,B] / cooccurrence[A,A] * 100)
    diag = cooccurrence.values.diagonal()
    diag_series = pd.Series(diag, index=cooccurrence.index)
    with pd.option_context('display.float_format', '{:6.1f}'.format):
        cond_pct = cooccurrence.div(diag_series, axis=0).fillna(0) * 100
        print("\nConditional co-occurrence (%) - rows=A (base), cols=B:")
        print(cond_pct.round(1).to_string())

    # Also print correlation matrix (binary presence)
    corr_matrix = df_bin.corr()
    print("\nCorrelation matrix (binary presence):")
    print(corr_matrix.round(4).to_string())


if __name__ == "__main__":
    train, val, dev, test = load_datasets()
    
    print("\n--- Train Set ---")
    print(train.head())
    print(f"\nTrain shape: {train.shape}")
    
    print("\n--- Validation Set ---")
    print(val.head())
    print(f"\nValidation shape: {val.shape}")
    
    print("\n--- Dev Set ---")
    print(dev.head())
    print(f"\nDev shape: {dev.shape}")
    
    print("\n--- Test Set ---")
    print(test.head())
    print(f"\nTest shape: {test.shape}")
    
    print("\nLabel distribution in Train set:")
    print_label_distribution(train)
