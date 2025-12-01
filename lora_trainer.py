"""
LoRA/QLoRA Training Script
Main script for fine-tuning sentiment and keyword extraction models using LoRA/QLoRA.
"""

import torch
import argparse
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
from src.trainer import train_sentiment_model, train_keyword_model
from src.evaluator import (
    evaluate_sentiment_model,
    evaluate_keyword_model,
    plot_confusion_matrix,
    save_sample_predictions,
    save_keyword_predictions,
    print_metrics
)

# Set random seed
SEED = 42
torch.manual_seed(SEED)

# Configuration
CONFIG = {
    'sentiment': {
        'model_name': 'distilbert-base-uncased',  # ~67M params
        'lora_r': 16,
        'learning_rate': 2e-4,
        'num_epochs': 3,
        'batch_size': 16,
        'use_qlora': True
    },
    'keyword': {
        'model_name': 't5-small',  # ~60M params
        'lora_r': 16,
        'learning_rate': 2e-4,
        'num_epochs': 3,
        'batch_size': 8,
        'use_qlora': True
    }
}


def train_sentiment():
    """
    Train sentiment classification model with LoRA/QLoRA.
    """
    print("\n" + "="*70)
    print("MODEL-A: SENTIMENT CLASSIFICATION - LoRA FINE-TUNING")
    print("="*70)

    config = CONFIG['sentiment']

    # Load data
    print("\n[1/5] Loading data...")
    train_df, val_df, test_df = load_sentiment_data()
    print(f"✓ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Load model
    print(f"\n[2/5] Loading model: {config['model_name']}")
    print(f"  - LoRA rank: {config['lora_r']}")
    print(f"  - QLoRA: {config['use_qlora']}")
    model, tokenizer = load_sentiment_model(
        model_name=config['model_name'],
        num_labels=3,
        use_qlora=config['use_qlora'],
        lora_r=config['lora_r']
    )

    # Create dataloaders
    print("\n[3/5] Creating dataloaders...")
    train_loader, val_loader, test_loader = create_sentiment_dataloaders(
        train_df, val_df, test_df, tokenizer, batch_size=config['batch_size']
    )
    print(f"✓ Batch size: {config['batch_size']}")

    # Train
    print(f"\n[4/5] Training for {config['num_epochs']} epochs...")
    history = train_sentiment_model(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        output_dir="models",
        model_name="sentiment_model"
    )

    # Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    model_path = Path("models") / "sentiment_model_best"
    from transformers import AutoModelForSequenceClassification

    eval_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    eval_tokenizer = tokenizer

    label_names_dict = get_label_names()
    results = evaluate_sentiment_model(
        eval_model, test_loader, label_names=label_names_dict
    )

    # Print metrics
    print_metrics(results, "Sentiment Model (Fine-tuned)")

    # Save confusion matrix
    results_dir = Path("results")
    cm_path = results_dir / "confusion_matrices" / "sentiment_finetuned.png"
    cm_path.parent.mkdir(exist_ok=True, parents=True)
    plot_confusion_matrix(
        results['confusion_matrix'],
        list(label_names_dict.values()),
        str(cm_path),
        "Sentiment Classification - After LoRA Fine-tuning"
    )

    # Save sample predictions
    save_sample_predictions(
        texts=test_df['text'].tolist(),
        true_labels=results['labels'],
        pred_labels=results['predictions'],
        probabilities=results['probabilities'],
        output_path=str(results_dir / "sample_predictions_sentiment.csv"),
        label_names=label_names_dict,
        num_samples=50
    )

    # Save results
    results_to_save = {
        'model': config['model_name'],
        'type': 'lora_finetuned',
        'lora_rank': config['lora_r'],
        'epochs': config['num_epochs'],
        'learning_rate': config['learning_rate'],
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

    with open(results_dir / "sentiment_finetuned.json", "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n✓ Results saved to {results_dir}")
    print("="*70 + "\n")

    return results


def train_keywords():
    """
    Train keyword extraction model with LoRA/QLoRA.
    """
    print("\n" + "="*70)
    print("MODEL-B: KEYWORD EXTRACTION - LoRA FINE-TUNING")
    print("="*70)

    config = CONFIG['keyword']

    # Load data
    print("\n[1/5] Loading data...")
    train_df, val_df, test_df = load_keyword_data()
    print(f"✓ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Load model
    print(f"\n[2/5] Loading model: {config['model_name']}")
    print(f"  - LoRA rank: {config['lora_r']}")
    print(f"  - QLoRA: {config['use_qlora']}")
    model, tokenizer = load_keyword_model(
        model_name=config['model_name'],
        use_qlora=config['use_qlora'],
        lora_r=config['lora_r']
    )

    # Create dataloaders
    print("\n[3/5] Creating dataloaders...")
    train_loader, val_loader, test_loader = create_keyword_dataloaders(
        train_df, val_df, test_df, tokenizer, batch_size=config['batch_size']
    )
    print(f"✓ Batch size: {config['batch_size']}")

    # Train
    print(f"\n[4/5] Training for {config['num_epochs']} epochs...")
    history = train_keyword_model(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        output_dir="models",
        model_name="keyword_model"
    )

    # Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    model_path = Path("models") / "keyword_model_best"
    from transformers import AutoModelForSeq2SeqLM

    eval_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    eval_tokenizer = tokenizer

    results = evaluate_keyword_model(
        eval_model, test_loader, eval_tokenizer, num_samples=100
    )

    # Print metrics
    print_metrics(results, "Keyword Model (Fine-tuned)")

    # Print some examples
    print("\nSample Predictions:")
    print("-" * 70)
    for i in range(min(3, len(results['predictions']))):
        print(f"\nExample {i+1}:")
        print(f"Reference:  {results['references'][i]}")
        print(f"Prediction: {results['predictions'][i]}")
    print("-" * 70)

    # Save sample predictions
    results_dir = Path("results")
    save_keyword_predictions(
        documents=test_df['document'].tolist()[:len(results['predictions'])],
        true_keywords=results['references'],
        pred_keywords=results['predictions'],
        output_path=str(results_dir / "sample_predictions_keywords.csv"),
        num_samples=50
    )

    # Save results
    results_to_save = {
        'model': config['model_name'],
        'type': 'lora_finetuned',
        'lora_rank': config['lora_r'],
        'epochs': config['num_epochs'],
        'learning_rate': config['learning_rate'],
        'rouge1': float(results['rouge1']),
        'rouge2': float(results['rouge2']),
        'rougeL': float(results['rougeL']),
        'mean_latency_ms': float(results['mean_latency_ms']),
        'p95_latency_ms': float(results['p95_latency_ms']),
        'num_samples_evaluated': len(results['predictions'])
    }

    with open(results_dir / "keyword_finetuned.json", "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n✓ Results saved to {results_dir}")
    print("="*70 + "\n")

    return results


def compare_results():
    """
    Compare baseline vs fine-tuned results.
    """
    results_dir = Path("results")

    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON: BASELINE vs FINE-TUNED")
    print("="*70)

    # Load sentiment results
    try:
        with open(results_dir / "sentiment_baseline.json") as f:
            sent_baseline = json.load(f)
        with open(results_dir / "sentiment_finetuned.json") as f:
            sent_finetuned = json.load(f)

        print("\nSENTIMENT CLASSIFICATION:")
        print("-" * 70)
        print(f"{'Metric':<20} {'Baseline':<15} {'Fine-tuned':<15} {'Improvement':<15}")
        print("-" * 70)
        print(f"{'Accuracy':<20} {sent_baseline['accuracy']:<15.4f} {sent_finetuned['accuracy']:<15.4f} {(sent_finetuned['accuracy']-sent_baseline['accuracy']):<15.4f}")
        print(f"{'F1 (macro)':<20} {sent_baseline['f1_macro']:<15.4f} {sent_finetuned['f1_macro']:<15.4f} {(sent_finetuned['f1_macro']-sent_baseline['f1_macro']):<15.4f}")
        print(f"{'Latency (ms)':<20} {sent_baseline['mean_latency_ms']:<15.2f} {sent_finetuned['mean_latency_ms']:<15.2f} {(sent_finetuned['mean_latency_ms']-sent_baseline['mean_latency_ms']):<15.2f}")
        print("-" * 70)
    except FileNotFoundError:
        print("\n⚠ Sentiment baseline results not found. Run baseline.py first.")

    # Load keyword results
    try:
        with open(results_dir / "keyword_baseline.json") as f:
            kw_baseline = json.load(f)
        with open(results_dir / "keyword_finetuned.json") as f:
            kw_finetuned = json.load(f)

        print("\nKEYWORD EXTRACTION:")
        print("-" * 70)
        print(f"{'Metric':<20} {'Baseline':<15} {'Fine-tuned':<15} {'Improvement':<15}")
        print("-" * 70)
        print(f"{'ROUGE-1':<20} {kw_baseline['rouge1']:<15.4f} {kw_finetuned['rouge1']:<15.4f} {(kw_finetuned['rouge1']-kw_baseline['rouge1']):<15.4f}")
        print(f"{'ROUGE-2':<20} {kw_baseline['rouge2']:<15.4f} {kw_finetuned['rouge2']:<15.4f} {(kw_finetuned['rouge2']-kw_baseline['rouge2']):<15.4f}")
        print(f"{'ROUGE-L':<20} {kw_baseline['rougeL']:<15.4f} {kw_finetuned['rougeL']:<15.4f} {(kw_finetuned['rougeL']-kw_baseline['rougeL']):<15.4f}")
        print(f"{'Latency (ms)':<20} {kw_baseline['mean_latency_ms']:<15.2f} {kw_finetuned['mean_latency_ms']:<15.2f} {(kw_finetuned['mean_latency_ms']-kw_baseline['mean_latency_ms']):<15.2f}")
        print("-" * 70)
    except FileNotFoundError:
        print("\n⚠ Keyword baseline results not found. Run baseline.py first.")

    print("\n" + "="*70 + "\n")


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description="LoRA/QLoRA Fine-tuning")
    parser.add_argument(
        '--model',
        type=str,
        choices=['sentiment', 'keyword', 'both'],
        default='both',
        help='Which model to train'
    )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("LORA/QLORA FINE-TUNING - AI LAB ML SERVICE")
    print("="*70)
    print(f"Random Seed: {SEED}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*70)

    # Check if data exists
    data_dir = Path("data")
    if not (data_dir / "sentiment_train.csv").exists():
        print("\n❌ ERROR: Dataset not found!")
        print("Please run 'python download.py' first to download the datasets.")
        return

    # Train models
    if args.model in ['sentiment', 'both']:
        train_sentiment()

    if args.model in ['keyword', 'both']:
        train_keywords()

    # Compare results
    if args.model == 'both':
        compare_results()

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Review results in results/ directory")
    print("  2. Check confusion matrices in results/confusion_matrices/")
    print("  3. Deploy with: python app.py")
    print("  4. Create MCP server: python mcp_server.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
