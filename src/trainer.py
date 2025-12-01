"""
Training Module
Handles LoRA/QLoRA training for sentiment and keyword extraction models.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json


class LoRATrainer:
    """
    Trainer class for LoRA/QLoRA fine-tuning.
    """
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "models",
        model_name: str = "model"
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.model_name = model_name

        self.output_dir.mkdir(exist_ok=True)

        # Move model to device
        self.model.to(device)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc="Training")

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss

    def validate(self) -> float:
        """
        Validate the model.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_dataloader)
        return avg_loss

    def train(self, num_epochs: int, save_best: bool = True) -> Dict:
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_best: Whether to save the best model

        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Output: {self.output_dir / self.model_name}")
        print(f"{'='*60}\n")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            # Train
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")

            # Validate
            val_loss = self.validate()
            print(f"Val Loss:   {val_loss:.4f}")

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"LR:         {current_lr:.6f}")

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)

            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(suffix="best")
                print(f"✓ Best model saved (val_loss: {val_loss:.4f})")

        # Save final model
        self.save_model(suffix="final")
        print(f"\n✓ Training complete!")

        return self.history

    def save_model(self, suffix: str = ""):
        """
        Save the model and tokenizer.

        Args:
            suffix: Suffix to add to the model name
        """
        save_dir = self.output_dir / f"{self.model_name}_{suffix}" if suffix else self.output_dir / self.model_name
        save_dir.mkdir(exist_ok=True, parents=True)

        self.model.save_pretrained(save_dir)

        # Save training history
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)


def create_optimizer_and_scheduler(
    model,
    train_dataloader,
    num_epochs: int,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.1
):
    """
    Create optimizer and learning rate scheduler.

    Args:
        model: Model to optimize
        train_dataloader: Training dataloader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_ratio: Ratio of warmup steps

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    # Calculate total steps
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    return optimizer, scheduler


def train_sentiment_model(
    model,
    tokenizer,
    train_dataloader,
    val_dataloader,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    output_dir: str = "models",
    model_name: str = "sentiment_model"
) -> Dict:
    """
    Train sentiment classification model with LoRA.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        output_dir: Output directory
        model_name: Name for saving the model

    Returns:
        Training history
    """
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, train_dataloader, num_epochs, learning_rate
    )

    # Create trainer
    trainer = LoRATrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        model_name=model_name
    )

    # Train
    history = trainer.train(num_epochs=num_epochs)

    # Save tokenizer
    tokenizer.save_pretrained(Path(output_dir) / f"{model_name}_best")
    tokenizer.save_pretrained(Path(output_dir) / f"{model_name}_final")

    return history


def train_keyword_model(
    model,
    tokenizer,
    train_dataloader,
    val_dataloader,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    output_dir: str = "models",
    model_name: str = "keyword_model"
) -> Dict:
    """
    Train keyword extraction model with LoRA.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        output_dir: Output directory
        model_name: Name for saving the model

    Returns:
        Training history
    """
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, train_dataloader, num_epochs, learning_rate
    )

    # Create trainer
    trainer = LoRATrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        model_name=model_name
    )

    # Train
    history = trainer.train(num_epochs=num_epochs)

    # Save tokenizer
    tokenizer.save_pretrained(Path(output_dir) / f"{model_name}_best")
    tokenizer.save_pretrained(Path(output_dir) / f"{model_name}_final")

    return history
