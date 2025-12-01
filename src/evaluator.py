"""
Evaluation Module
Handles evaluation metrics for sentiment classification and keyword extraction.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import time


def evaluate_sentiment_model(
    model,
    dataloader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    label_names: Dict[int, str] = None
) -> Dict:
    """
    Evaluate sentiment classification model.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        label_names: Mapping of label IDs to names

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Measure inference time
            start_time = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            inference_time = time.time() - start_time
            inference_times.append(inference_time / len(input_ids))  # Per sample

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Inference latency
    mean_latency = np.mean(inference_times) * 1000  # Convert to ms
    p95_latency = np.percentile(inference_times, 95) * 1000

    # Build results
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'mean_latency_ms': mean_latency,
        'p95_latency_ms': p95_latency,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

    # Add per-class F1 scores
    if label_names:
        for idx, label_name in label_names.items():
            results[f'f1_{label_name}'] = f1_per_class[idx]

    return results


def evaluate_keyword_model(
    model,
    dataloader,
    tokenizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_samples: int = None
) -> Dict:
    """
    Evaluate keyword extraction model using ROUGE scores.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        tokenizer: Tokenizer for decoding
        device: Device to run evaluation on
        num_samples: Number of samples to evaluate (None = all)

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    all_rouge1 = []
    all_rouge2 = []
    all_rougeL = []
    inference_times = []
    predictions = []
    references = []

    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            # Measure inference time
            start_time = time.time()
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
            inference_time = time.time() - start_time
            inference_times.append(inference_time / len(input_ids))

            # Decode predictions and references
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Calculate ROUGE scores
            for pred, ref in zip(preds, refs):
                # Skip empty references
                ref = ref.replace("-100", "").strip()
                if not ref:
                    continue

                scores = scorer.score(ref, pred)
                all_rouge1.append(scores['rouge1'].fmeasure)
                all_rouge2.append(scores['rouge2'].fmeasure)
                all_rougeL.append(scores['rougeL'].fmeasure)

                predictions.append(pred)
                references.append(ref)

                sample_count += 1
                if num_samples and sample_count >= num_samples:
                    break

            if num_samples and sample_count >= num_samples:
                break

    # Calculate average scores
    rouge1 = np.mean(all_rouge1)
    rouge2 = np.mean(all_rouge2)
    rougeL = np.mean(all_rougeL)

    # Inference latency
    mean_latency = np.mean(inference_times) * 1000
    p95_latency = np.percentile(inference_times, 95) * 1000

    results = {
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougeL': rougeL,
        'mean_latency_ms': mean_latency,
        'p95_latency_ms': p95_latency,
        'predictions': predictions,
        'references': references
    }

    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix"
):
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix
        label_names: List of label names
        output_path: Path to save plot
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {output_path}")


def save_sample_predictions(
    texts: List[str],
    true_labels: List,
    pred_labels: List,
    probabilities: List,
    output_path: str,
    label_names: Dict[int, str] = None,
    num_samples: int = 50
):
    """
    Save sample predictions to CSV.

    Args:
        texts: Input texts
        true_labels: True labels
        pred_labels: Predicted labels
        probabilities: Prediction probabilities
        output_path: Path to save CSV
        label_names: Mapping of label IDs to names
        num_samples: Number of samples to save
    """
    # Select random samples
    indices = np.random.choice(len(texts), min(num_samples, len(texts)), replace=False)

    data = []
    for idx in indices:
        text = texts[idx]
        true_label = true_labels[idx]
        pred_label = pred_labels[idx]
        prob = probabilities[idx]

        # Get confidence (max probability)
        confidence = np.max(prob)

        # Convert to label names if provided
        if label_names:
            true_label_name = label_names[true_label]
            pred_label_name = label_names[pred_label]
        else:
            true_label_name = str(true_label)
            pred_label_name = str(pred_label)

        data.append({
            'input_text': text,
            'true_label': true_label_name,
            'predicted_label': pred_label_name,
            'confidence': confidence
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✓ Sample predictions saved to {output_path} ({len(df)} samples)")


def save_keyword_predictions(
    documents: List[str],
    true_keywords: List[str],
    pred_keywords: List[str],
    output_path: str,
    num_samples: int = 50
):
    """
    Save keyword extraction predictions to CSV.

    Args:
        documents: Input documents
        true_keywords: True keywords
        pred_keywords: Predicted keywords
        output_path: Path to save CSV
        num_samples: Number of samples to save
    """
    # Select random samples
    indices = np.random.choice(len(documents), min(num_samples, len(documents)), replace=False)

    data = []
    for idx in indices:
        data.append({
            'document': documents[idx][:200] + "...",  # Truncate for readability
            'true_keywords': true_keywords[idx],
            'predicted_keywords': pred_keywords[idx]
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✓ Keyword predictions saved to {output_path} ({len(df)} samples)")


def print_metrics(results: Dict, model_name: str = "Model"):
    """
    Print evaluation metrics in a formatted way.

    Args:
        results: Dictionary with evaluation results
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation Results")
    print(f"{'='*60}")

    # Sentiment metrics
    if 'accuracy' in results:
        print(f"Accuracy:       {results['accuracy']:.4f}")
        print(f"F1 (macro):     {results['f1_macro']:.4f}")
        print(f"F1 (weighted):  {results['f1_weighted']:.4f}")
        print(f"Precision:      {results['precision']:.4f}")
        print(f"Recall:         {results['recall']:.4f}")

        # Per-class F1
        if 'f1_positive' in results:
            print(f"\nPer-class F1:")
            print(f"  Positive:     {results['f1_positive']:.4f}")
            print(f"  Neutral:      {results['f1_neutral']:.4f}")
            print(f"  Negative:     {results['f1_negative']:.4f}")

    # Keyword metrics
    if 'rouge1' in results:
        print(f"ROUGE-1:        {results['rouge1']:.4f}")
        print(f"ROUGE-2:        {results['rouge2']:.4f}")
        print(f"ROUGE-L:        {results['rougeL']:.4f}")

    # Latency metrics
    if 'mean_latency_ms' in results:
        print(f"\nInference Latency:")
        print(f"  Mean:         {results['mean_latency_ms']:.2f} ms")
        print(f"  P95:          {results['p95_latency_ms']:.2f} ms")

    print(f"{'='*60}\n")
