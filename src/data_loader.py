"""
Data Loading and Preprocessing Module
Handles loading and preprocessing of sentiment and keyword extraction datasets.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import numpy as np

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class SentimentDataset(Dataset):
    """
    Dataset class for sentiment classification.
    """
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class KeywordDataset(Dataset):
    """
    Dataset class for keyword extraction (seq2seq).
    """
    def __init__(
        self,
        documents: List[str],
        keywords: List[str],
        tokenizer: AutoTokenizer,
        max_source_length: int = 512,
        max_target_length: int = 64
    ):
        self.documents = documents
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        document = str(self.documents[idx])
        keywords = str(self.keywords[idx])

        # Tokenize source (document)
        source_encoding = self.tokenizer(
            document,
            truncation=True,
            padding='max_length',
            max_length=self.max_source_length,
            return_tensors='pt'
        )

        # Tokenize target (keywords)
        target_encoding = self.tokenizer(
            keywords,
            truncation=True,
            padding='max_length',
            max_length=self.max_target_length,
            return_tensors='pt'
        )

        # Get label ids (replace padding token id with -100 for loss calculation)
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }


def load_sentiment_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load sentiment classification data (train, val, test).

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    data_path = Path(data_dir)

    train_df = pd.read_csv(data_path / "sentiment_train.csv")
    val_df = pd.read_csv(data_path / "sentiment_val.csv")
    test_df = pd.read_csv(data_path / "sentiment_test.csv")

    return train_df, val_df, test_df


def load_keyword_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load keyword extraction data (train, val, test).

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    data_path = Path(data_dir)

    train_df = pd.read_csv(data_path / "keywords_train.csv")
    val_df = pd.read_csv(data_path / "keywords_val.csv")
    test_df = pd.read_csv(data_path / "keywords_test.csv")

    return train_df, val_df, test_df


def create_sentiment_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 128
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for sentiment classification.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        tokenizer: Tokenizer for the model
        batch_size: Batch size for DataLoader
        max_length: Maximum sequence length

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = SentimentDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_length
    )

    val_dataset = SentimentDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_length
    )

    test_dataset = SentimentDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        tokenizer,
        max_length
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def create_keyword_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_source_length: int = 512,
    max_target_length: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for keyword extraction.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        tokenizer: Tokenizer for the model
        batch_size: Batch size for DataLoader
        max_source_length: Maximum source sequence length
        max_target_length: Maximum target sequence length

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = KeywordDataset(
        train_df['document'].tolist(),
        train_df['keywords_str'].tolist(),
        tokenizer,
        max_source_length,
        max_target_length
    )

    val_dataset = KeywordDataset(
        val_df['document'].tolist(),
        val_df['keywords_str'].tolist(),
        tokenizer,
        max_source_length,
        max_target_length
    )

    test_dataset = KeywordDataset(
        test_df['document'].tolist(),
        test_df['keywords_str'].tolist(),
        tokenizer,
        max_source_length,
        max_target_length
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_label_names() -> Dict[int, str]:
    """
    Get label names for sentiment classification.

    Returns:
        Dictionary mapping label IDs to names
    """
    return {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
