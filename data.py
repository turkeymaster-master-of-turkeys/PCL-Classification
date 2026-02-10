import pandas as pd
from ast import literal_eval
from pathlib import Path
from typing import Tuple, Dict
import re
from io import StringIO

DATA_DIR = Path(__file__).parent / "data"


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    # Load text data and category annotations
    print("Loading text data from dontpatronizeme_categories.tsv...")
    cat_path = DATA_DIR / "dontpatronizeme_categories.tsv"
    # Read file and keep only the actual data rows (skip disclaimer/header lines)
    with open(cat_path, "r", encoding="utf-8") as fh:
        raw_lines = fh.readlines()

    data_lines = [ln for ln in raw_lines if re.match(r"^\d+\t@@", ln)]

    if not data_lines:
        # Fallback: try to read entire file as TSV
        cat_df = pd.read_csv(cat_path, sep="\t", header=None, engine="python")
    else:
        names = [
            "par_id",
            "text_id",
            "text",
            "keyword",
            "country",
            "start",
            "end",
            "span_text",
            "category",
            "score",
        ]
        cat_df = pd.read_csv(
            StringIO("".join(data_lines)),
            sep="\t",
            header=None,
            names=names,
            engine="python",
        )

    # Keep only unique paragraphs (first occurrence of par_id)
    cat_df['par_id'] = pd.to_numeric(cat_df['par_id'], errors='coerce').astype('Int64')
    cat_df['score'] = pd.to_numeric(cat_df['score'], errors='coerce').astype('Int64')
    text_lookup = (
        cat_df.loc[cat_df['par_id'].notna(), ['par_id', 'text']]
        .drop_duplicates(subset=['par_id'])
        .set_index('par_id')
    )
    
    # Aggregate category scores per paragraph (max score per category)
    print("Aggregating category scores...")
    category_scores = {}
    for par_id, group in cat_df[cat_df['par_id'].notna()].groupby('par_id'):
        categories_dict = {}
        for _, row in group.iterrows():
            category = row['category']
            score = row['score']
            if pd.notna(category) and pd.notna(score):
                # Keep max score for each category
                if category not in categories_dict:
                    categories_dict[category] = score
                else:
                    categories_dict[category] = max(categories_dict[category], score)
        category_scores[par_id] = categories_dict
    
    # Load PCL scores
    print("Loading PCL scores from dontpatronizeme_pcl.tsv...")
    pcl_path = DATA_DIR / "dontpatronizeme_pcl.tsv"
    with open(pcl_path, "r", encoding="utf-8") as fh:
        raw_lines = fh.readlines()
    
    pcl_data_lines = [ln for ln in raw_lines if re.match(r"^\d+\t@@", ln)]
    
    if pcl_data_lines:
        pcl_names = [
            "par_id",
            "text_id",
            "keyword",
            "country",
            "text",
            "pcl_score",
        ]
        pcl_df = pd.read_csv(
            StringIO("".join(pcl_data_lines)),
            sep="\t",
            header=None,
            names=pcl_names,
            engine="python",
        )
        pcl_df['par_id'] = pd.to_numeric(pcl_df['par_id'], errors='coerce').astype('Int64')
        pcl_df['pcl_score'] = pd.to_numeric(pcl_df['pcl_score'], errors='coerce').astype('Int64')
        pcl_lookup = pcl_df[['par_id', 'pcl_score']].drop_duplicates(subset=['par_id']).set_index('par_id')
    else:
        pcl_lookup = pd.DataFrame()
    
    # Load train labels
    print("Loading train labels...")
    train_labels = pd.read_csv(DATA_DIR / "train_semeval_parids-labels.csv")
    train_labels['par_id'] = train_labels['par_id'].astype(int)
    # Parse label string to list
    train_labels['label'] = train_labels['label'].apply(literal_eval)
    # Merge with text
    train_df = train_labels.merge(
        text_lookup.reset_index(),
        on='par_id',
        how='left'
    )[['par_id', 'text', 'label']]
    
    # Add PCL score
    train_df['pcl_score'] = train_df['par_id'].map(
        lambda x: pcl_lookup.loc[x, 'pcl_score'] if x in pcl_lookup.index else None
    )
    
    # Map original category names to abbreviations and add as separate columns
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

    for long_name, abbr in category_map.items():
        train_df[abbr] = train_df['par_id'].map(
            lambda x, ln=long_name: category_scores.get(x, {}).get(ln, 0)
        )
    
    # Load dev labels
    print("Loading dev labels...")
    dev_labels = pd.read_csv(DATA_DIR / "dev_semeval_parids-labels.csv")
    dev_labels['par_id'] = dev_labels['par_id'].astype(int)
    # Parse label string to list
    dev_labels['label'] = dev_labels['label'].apply(literal_eval)
    # Merge with text
    dev_df = dev_labels.merge(
        text_lookup.reset_index(),
        on='par_id',
        how='left'
    )[['par_id', 'text', 'label']]
    
    # Add PCL score
    dev_df['pcl_score'] = dev_df['par_id'].map(
        lambda x: pcl_lookup.loc[x, 'pcl_score'] if x in pcl_lookup.index else None
    )
    
    # Add category scores as separate columns (using abbreviations)
    for long_name, abbr in category_map.items():
        dev_df[abbr] = dev_df['par_id'].map(
            lambda x, ln=long_name: category_scores.get(x, {}).get(ln, 0)
        )
    
    # Load test data (no labels)
    print("Loading test data...")
    test_df = pd.read_csv(
        DATA_DIR / "task4_test.tsv",
        sep="\t",
        header=None,
        names=['test_id', 'text_id', 'keyword', 'country', 'text']
    )
    
    # Add PCL score as None for test set
    test_df['pcl_score'] = None
    
    # Add category scores as separate columns (all 0 for test set, abbreviations)
    for abbr in category_cols:
        test_df[abbr] = 0
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Dev set: {len(dev_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, dev_df, test_df


def print_label_statistics(df: pd.DataFrame) -> Dict[str, int]:
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
    train, dev, test = load_datasets()
    
    print("\n--- Train Set ---")
    print(train.head())
    print(f"\nTrain shape: {train.shape}")
    
    print("\n--- Dev Set ---")
    print(dev.head())
    print(f"\nDev shape: {dev.shape}")
    
    print("\n--- Test Set ---")
    print(test.head())
    print(f"\nTest shape: {test.shape}")
    
    print("\nLabel distribution in Train set:")
    print_label_statistics(train)
