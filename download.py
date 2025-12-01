"""
Dataset Download Script for AI Lab ML Service
Downloads and prepares datasets for sentiment classification and keyword extraction.

Datasets:
- Sentiment: emotion dataset (mapped to 3 classes: positive, neutral, negative)
- Keywords: midas/inspec dataset for keyword extraction

License Information:
- emotion dataset: Apache 2.0 License
- midas/inspec: Academic use, CC-BY-4.0

Author: AI Lab Project
Date: 2025-12-01
"""

import os
import random
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Dataset configuration
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def download_sentiment_dataset():
    """
    Download and process emotion dataset for 3-way sentiment classification.
    Maps 6 emotions to 3 classes: positive, neutral, negative
    """
    print("=" * 60)
    print("Downloading Sentiment Dataset (emotion)")
    print("=" * 60)

    # Load emotion dataset
    # Dataset has labels: 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
    dataset = load_dataset("emotion")

    # Mapping strategy:
    # Positive (label=2): joy (1), love (2)
    # Negative (label=0): sadness (0), anger (3), fear (4)
    # Neutral (label=1): surprise (5)
    label_mapping = {
        0: 0,  # sadness -> negative
        1: 2,  # joy -> positive
        2: 2,  # love -> positive
        3: 0,  # anger -> negative
        4: 0,  # fear -> negative
        5: 1,  # surprise -> neutral
    }

    label_names = {0: "negative", 1: "neutral", 2: "positive"}

    def map_labels(example):
        example['label'] = label_mapping[example['label']]
        example['label_name'] = label_names[example['label']]
        return example

    # Process train, validation, test splits
    train_data = dataset['train'].map(map_labels)
    val_data = dataset['validation'].map(map_labels)
    test_data = dataset['test'].map(map_labels)

    # Convert to pandas for easy saving
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    # Save to CSV
    train_df.to_csv(DATA_DIR / "sentiment_train.csv", index=False)
    val_df.to_csv(DATA_DIR / "sentiment_val.csv", index=False)
    test_df.to_csv(DATA_DIR / "sentiment_test.csv", index=False)

    print(f"✓ Sentiment dataset downloaded and processed")
    print(f"  - Train: {len(train_df)} samples")
    print(f"  - Val: {len(val_df)} samples")
    print(f"  - Test: {len(test_df)} samples")
    print(f"  - Classes: {label_names}")
    print(f"  - Saved to: {DATA_DIR}")

    # Print class distribution
    print("\nClass Distribution (Train):")
    print(train_df['label_name'].value_counts())

    return train_df, val_df, test_df


def download_keyword_dataset():
    """
    Download and process Inspec dataset for keyword extraction.
    """
    print("\n" + "=" * 60)
    print("Downloading Keyword Extraction Dataset (Inspec)")
    print("=" * 60)

    # Load Inspec dataset
    dataset = load_dataset("midas/inspec", "extraction")

    # Check what columns are available
    print(f"Available columns: {dataset['train'].column_names}")

    # Process the dataset
    def process_keywords(example):
        # Handle different possible column names
        kw_field = None
        if 'extractive_keyphrases' in example:
            kw_field = 'extractive_keyphrases'
        elif 'keyphrases' in example:
            kw_field = 'keyphrases'
        elif 'keywords' in example:
            kw_field = 'keywords'

        # Join keywords into a semicolon-separated string
        if kw_field and example[kw_field]:
            keywords_list = example[kw_field]
            example['keywords_str'] = "; ".join(keywords_list)
            example['keywords_list'] = keywords_list
        else:
            example['keywords_str'] = ""
            example['keywords_list'] = []
        return example

    # Process all splits
    train_data = dataset['train'].map(process_keywords)
    val_data = dataset['validation'].map(process_keywords)
    test_data = dataset['test'].map(process_keywords)

    # Convert to pandas
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    # Select relevant columns (check which exist)
    available_cols = train_df.columns.tolist()
    columns_to_keep = ['document', 'keywords_str']

    # Add keywords_list if it exists
    if 'keywords_list' in available_cols:
        columns_to_keep.append('keywords_list')

    train_df = train_df[columns_to_keep]
    val_df = val_df[columns_to_keep]
    test_df = test_df[columns_to_keep]

    # Save to CSV
    train_df.to_csv(DATA_DIR / "keywords_train.csv", index=False)
    val_df.to_csv(DATA_DIR / "keywords_val.csv", index=False)
    test_df.to_csv(DATA_DIR / "keywords_test.csv", index=False)

    print(f"✓ Keyword dataset downloaded and processed")
    print(f"  - Train: {len(train_df)} samples")
    print(f"  - Val: {len(val_df)} samples")
    print(f"  - Test: {len(test_df)} samples")
    print(f"  - Saved to: {DATA_DIR}")

    # Print statistics
    if 'keywords_list' in train_df.columns:
        avg_keywords = train_df['keywords_list'].apply(len).mean()
        print(f"\nDataset Statistics (Train):")
        print(f"  - Avg keywords per document: {avg_keywords:.2f}")

    avg_doc_length = train_df['document'].apply(lambda x: len(x.split())).mean()
    print(f"  - Avg document length: {avg_doc_length:.2f} words")

    return train_df, val_df, test_df


def save_metadata():
    """
    Save dataset metadata and license information.
    """
    metadata = {
        "project": "AI Lab ML Service - LoRA Fine-tuning",
        "date_downloaded": "2025-12-01",
        "random_seed": RANDOM_SEED,
        "datasets": {
            "sentiment": {
                "name": "emotion",
                "source": "huggingface:emotion",
                "license": "Apache 2.0",
                "url": "https://huggingface.co/datasets/emotion",
                "classes": ["negative", "neutral", "positive"],
                "mapping": "6 emotions mapped to 3 sentiment classes"
            },
            "keywords": {
                "name": "inspec",
                "source": "huggingface:midas/inspec",
                "license": "CC-BY-4.0 (Academic use)",
                "url": "https://huggingface.co/datasets/midas/inspec",
                "task": "keyword extraction",
                "format": "document -> list of keywords"
            }
        }
    }

    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n✓ Metadata saved to data/metadata.json")


def main():
    """
    Main function to download all datasets.
    """
    print("\n" + "=" * 60)
    print("AI LAB ML SERVICE - DATASET DOWNLOAD")
    print("=" * 60)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Output Directory: {DATA_DIR}")
    print("=" * 60 + "\n")

    # Download datasets
    sentiment_train, sentiment_val, sentiment_test = download_sentiment_dataset()
    keyword_train, keyword_val, keyword_test = download_keyword_dataset()

    # Save metadata
    save_metadata()

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Run notebooks/01_explore.ipynb for EDA")
    print("2. Run baseline.py for zero-shot evaluation")
    print("3. Run lora_trainer.py to fine-tune models")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
