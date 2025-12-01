"""
Baseline Evaluation Script
Evaluates zero-shot performance of pre-trained models before LoRA fine-tuning.
This establishes baseline metrics for comparison.
"""

import torch
import json
from pathlib import Path
from src.data_loader import (
    load_sentiment_data,
    load_keyword_data,
    create_sentiment_dataloaders,
    create_keyword_dataloaders,
    get_label_names
)
from src.model_utils import load_sentiment_model, load_keyword_model
from src.evaluator import (
    evaluate_sentiment_model,
    evaluate_keyword_model,
    plot_confusion_matrix,
    print_metrics
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM

# Set random seed
SEED = 42
torch.manual_seed(SEED)

# Configuration
SENTIMENT_MODEL = "distilbert-base-uncased"
KEYWORD_MODEL = "t5-small"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate_sentiment_baseline():
    """
    Evaluate zero-shot sentiment classification baseline.
    """
    print("\n" + "="*60)
    print("SENTIMENT CLASSIFICATION - BASELINE EVALUATION")
    print("="*60)

    # Load data
    print("\nLoading data...")
    train_df, val_df, test_df = load_sentiment_data()
    print(f"✓ Loaded {len(test_df)} test samples")

    # Load pre-trained model (without LoRA)
    print(f"\nLoading baseline model: {SENTIMENT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)

    # For zero-shot, we load a pre-trained model with 3 labels
    # Note: This won't have good performance since it's not trained on this task
    # We'll use it as is for baseline
    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL,
        num_labels=3,
        ignore_mismatched_sizes=True  # Initialize new classification head
    )
    print("✓ Model loaded (untrained classification head)")

    # Create test dataloader
    print("\nCreating dataloaders...")
    _, _, test_loader = create_sentiment_dataloaders(
        train_df, val_df, test_df, tokenizer, batch_size=32
    )

    # Evaluate
    print("\nEvaluating on test set...")
    label_names_dict = get_label_names()
    results = evaluate_sentiment_model(
        model, test_loader, label_names=label_names_dict
    )

    # Print metrics
    print_metrics(results, "Sentiment Baseline")

    # Plot confusion matrix
    cm_path = RESULTS_DIR / "confusion_matrices" / "sentiment_baseline.png"
    cm_path.parent.mkdir(exist_ok=True, parents=True)
    plot_confusion_matrix(
        results['confusion_matrix'],
        list(label_names_dict.values()),
        str(cm_path),
        "Sentiment Classification - Baseline (Zero-shot)"
    )

    # Save results
    results_to_save = {
        'model': SENTIMENT_MODEL,
        'type': 'baseline_zero_shot',
        'accuracy': float(results['accuracy']),
        'f1_macro': float(results['f1_macro']),
        'f1_weighted': float(results['f1_weighted']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_positive': float(results['f1_positive']),
        'f1_neutral': float(results['f1_neutral']),
        'f1_negative': float(results['f1_negative']),
        'mean_latency_ms': float(results['mean_latency_ms']),
        'p95_latency_ms': float(results['p95_latency_ms'])
    }

    with open(RESULTS_DIR / "sentiment_baseline.json", "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n✓ Results saved to {RESULTS_DIR / 'sentiment_baseline.json'}")

    return results


def evaluate_keyword_baseline():
    """
    Evaluate zero-shot keyword extraction baseline.
    """
    print("\n" + "="*60)
    print("KEYWORD EXTRACTION - BASELINE EVALUATION")
    print("="*60)

    # Load data
    print("\nLoading data...")
    train_df, val_df, test_df = load_keyword_data()
    print(f"✓ Loaded {len(test_df)} test samples")

    # Load pre-trained model (without LoRA)
    print(f"\nLoading baseline model: {KEYWORD_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(KEYWORD_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(KEYWORD_MODEL)
    print("✓ Model loaded (pre-trained, not fine-tuned)")

    # Create test dataloader
    print("\nCreating dataloaders...")
    _, _, test_loader = create_keyword_dataloaders(
        train_df, val_df, test_df, tokenizer, batch_size=16
    )

    # Evaluate (limit to 100 samples for speed)
    print("\nEvaluating on test set (100 samples)...")
    results = evaluate_keyword_model(
        model, test_loader, tokenizer, num_samples=100
    )

    # Print metrics
    print_metrics(results, "Keyword Baseline")

    # Print some examples
    print("\nSample Predictions:")
    print("-" * 60)
    for i in range(min(3, len(results['predictions']))):
        print(f"\nExample {i+1}:")
        print(f"Reference:  {results['references'][i]}")
        print(f"Prediction: {results['predictions'][i]}")
    print("-" * 60)

    # Save results
    results_to_save = {
        'model': KEYWORD_MODEL,
        'type': 'baseline_zero_shot',
        'rouge1': float(results['rouge1']),
        'rouge2': float(results['rouge2']),
        'rougeL': float(results['rougeL']),
        'mean_latency_ms': float(results['mean_latency_ms']),
        'p95_latency_ms': float(results['p95_latency_ms']),
        'num_samples_evaluated': len(results['predictions'])
    }

    with open(RESULTS_DIR / "keyword_baseline.json", "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n✓ Results saved to {RESULTS_DIR / 'keyword_baseline.json'}")

    return results


def main():
    """
    Run all baseline evaluations.
    """
    print("\n" + "="*60)
    print("BASELINE EVALUATION - ZERO-SHOT PERFORMANCE")
    print("="*60)
    print("\nThis script evaluates pre-trained models without fine-tuning")
    print("to establish baseline metrics for comparison.")
    print("="*60)

    # Check if data exists
    data_dir = Path("data")
    if not (data_dir / "sentiment_train.csv").exists():
        print("\n❌ ERROR: Dataset not found!")
        print("Please run 'python download.py' first to download the datasets.")
        return

    # Evaluate sentiment baseline
    sentiment_results = evaluate_sentiment_baseline()

    # Evaluate keyword baseline
    keyword_results = evaluate_keyword_baseline()

    # Summary
    print("\n" + "="*60)
    print("BASELINE EVALUATION COMPLETE")
    print("="*60)
    print("\nSummary:")
    print(f"  Sentiment - Accuracy: {sentiment_results['accuracy']:.4f}, F1: {sentiment_results['f1_macro']:.4f}")
    print(f"  Keywords  - ROUGE-1: {keyword_results['rouge1']:.4f}, ROUGE-L: {keyword_results['rougeL']:.4f}")
    print("\nNext Steps:")
    print("  1. Review baseline metrics")
    print("  2. Run LoRA fine-tuning: python lora_trainer.py")
    print("  3. Compare pre/post fine-tuning performance")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
