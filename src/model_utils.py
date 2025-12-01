"""
Model Utilities Module
Handles model loading, LoRA/QLoRA configuration, and model utilities.
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from typing import Tuple, Optional
import random
import numpy as np

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_bnb_config() -> BitsAndBytesConfig:
    """
    Get BitsAndBytes configuration for 4-bit quantization (QLoRA).

    Returns:
        BitsAndBytesConfig for QLoRA
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    return bnb_config


def get_lora_config(
    task_type: TaskType,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None
) -> LoraConfig:
    """
    Get LoRA configuration.

    Args:
        task_type: Type of task (SEQ_CLS or SEQ_2_SEQ_LM)
        r: LoRA rank (must be <= 32)
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout probability
        target_modules: Modules to apply LoRA to (None = auto)

    Returns:
        LoraConfig for PEFT
    """
    if r > 32:
        raise ValueError(f"LoRA rank must be <= 32, got {r}")

    # Default target modules if not specified
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        task_type=task_type,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False
    )

    return lora_config


def load_sentiment_model(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 3,
    use_qlora: bool = True,
    lora_r: int = 16
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load a sentiment classification model with LoRA/QLoRA.

    Args:
        model_name: HuggingFace model name (must be <= 110M params)
        num_labels: Number of classification labels
        use_qlora: Whether to use QLoRA (4-bit quantization)
        lora_r: LoRA rank

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with optional quantization
    if use_qlora:
        bnb_config = get_bnb_config()
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            quantization_config=bnb_config,
            device_map="auto"
        )
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    # Configure LoRA
    lora_config = get_lora_config(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.1
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


def load_keyword_model(
    model_name: str = "t5-small",
    use_qlora: bool = True,
    lora_r: int = 16
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Load a keyword extraction model (seq2seq) with LoRA/QLoRA.

    Args:
        model_name: HuggingFace model name (must be <= 110M params)
        use_qlora: Whether to use QLoRA (4-bit quantization)
        lora_r: LoRA rank

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with optional quantization
    if use_qlora:
        bnb_config = get_bnb_config()
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Configure LoRA
    lora_config = get_lora_config(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.1
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


def count_parameters(model) -> int:
    """
    Count total parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> Tuple[int, int, float]:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (trainable_params, total_params, trainable_percentage)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100 * trainable_params / total_params

    return trainable_params, total_params, trainable_percentage


def get_model_info(model) -> dict:
    """
    Get information about a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    trainable, total, percentage = count_trainable_parameters(model)

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": f"{percentage:.2f}%",
        "model_size_mb": total * 4 / (1024 ** 2),  # Approximate size in MB
    }


def save_model(model, tokenizer, output_dir: str):
    """
    Save a model and tokenizer.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
    """
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Model saved to {output_dir}")


def load_trained_model(
    model_path: str,
    model_type: str = "sentiment"
) -> Tuple:
    """
    Load a trained model and tokenizer.

    Args:
        model_path: Path to saved model
        model_type: Type of model ('sentiment' or 'keyword')

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if model_type == "sentiment":
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    elif model_type == "keyword":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, tokenizer
