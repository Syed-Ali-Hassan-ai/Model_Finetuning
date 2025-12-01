"""
MCP (Model Context Protocol) Server
Provides conversational tools for controlling ML training, evaluation, and inference.

Available Tools:
- set_learning_rate(lr: float) - Adjust learning rate
- train_model(model_type: str, epochs: int) - Trigger training
- evaluate_model(model_type: str) - Run evaluation
- get_metrics() - Retrieve current stats
- predict_sentiment(text: str) - Sentiment inference
- extract_keywords(text: str) - Keyword inference
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM


class MCPServer:
    """
    MCP Server for conversational model control.
    """

    def __init__(self):
        self.config_file = Path("mcp_config.json")
        self.load_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load models
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        self.keyword_model = None
        self.keyword_tokenizer = None
        self._load_models()

    def load_config(self):
        """Load or create MCP configuration."""
        if self.config_file.exists():
            with open(self.config_file) as f:
                self.config = json.load(f)
        else:
            self.config = {
                "learning_rate": 2e-4,
                "epochs": 3,
                "batch_size": 16,
                "lora_rank": 16
            }
            self.save_config()

    def save_config(self):
        """Save MCP configuration."""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)

    def _load_models(self):
        """Load trained models."""
        sentiment_path = Path("models/sentiment_model_best")
        keyword_path = Path("models/keyword_model_best")

        if sentiment_path.exists():
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_path)
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_path)
            self.sentiment_model.to(self.device)
            self.sentiment_model.eval()

        if keyword_path.exists():
            self.keyword_model = AutoModelForSeq2SeqLM.from_pretrained(keyword_path)
            self.keyword_tokenizer = AutoTokenizer.from_pretrained(keyword_path)
            self.keyword_model.to(self.device)
            self.keyword_model.eval()

    # ========== MCP TOOLS ==========

    def set_learning_rate(self, lr: float) -> Dict[str, Any]:
        """
        Set the learning rate for training.

        Args:
            lr: New learning rate (e.g., 2e-4, 1e-4, 5e-5)

        Returns:
            Status message
        """
        if lr <= 0 or lr > 0.01:
            return {
                "success": False,
                "message": f"Invalid learning rate: {lr}. Must be between 0 and 0.01"
            }

        old_lr = self.config["learning_rate"]
        self.config["learning_rate"] = lr
        self.save_config()

        return {
            "success": True,
            "message": f"Learning rate updated: {old_lr} → {lr}",
            "old_value": old_lr,
            "new_value": lr
        }

    def train_model(self, model_type: str, epochs: int = None) -> Dict[str, Any]:
        """
        Trigger model training.

        Args:
            model_type: Type of model ('sentiment' or 'keyword')
            epochs: Number of epochs (optional, uses config if not specified)

        Returns:
            Training status
        """
        if model_type not in ["sentiment", "keyword"]:
            return {
                "success": False,
                "message": f"Invalid model type: {model_type}. Use 'sentiment' or 'keyword'"
            }

        epochs = epochs or self.config["epochs"]

        try:
            print(f"\n🚀 Starting {model_type} model training for {epochs} epochs...")
            print(f"   Learning rate: {self.config['learning_rate']}")

            # Run training script
            result = subprocess.run(
                ["python", "lora_trainer.py", "--model", model_type],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"{model_type} model training completed successfully",
                    "model_type": model_type,
                    "epochs": epochs,
                    "learning_rate": self.config["learning_rate"]
                }
            else:
                return {
                    "success": False,
                    "message": f"Training failed: {result.stderr[:200]}"
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Training error: {str(e)}"
            }

    def evaluate_model(self, model_type: str = "both") -> Dict[str, Any]:
        """
        Run model evaluation.

        Args:
            model_type: Type of model ('sentiment', 'keyword', or 'both')

        Returns:
            Evaluation metrics
        """
        results = {}

        try:
            if model_type in ["sentiment", "both"]:
                sentiment_file = Path("results/sentiment_finetuned.json")
                if sentiment_file.exists():
                    with open(sentiment_file) as f:
                        results["sentiment"] = json.load(f)

            if model_type in ["keyword", "both"]:
                keyword_file = Path("results/keyword_finetuned.json")
                if keyword_file.exists():
                    with open(keyword_file) as f:
                        results["keyword"] = json.load(f)

            if results:
                return {
                    "success": True,
                    "message": "Evaluation metrics retrieved",
                    "metrics": results
                }
            else:
                return {
                    "success": False,
                    "message": "No evaluation results found. Train models first."
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Evaluation error: {str(e)}"
            }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current model performance metrics.

        Returns:
            Dictionary with metrics for both models
        """
        return self.evaluate_model("both")

    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment of text.

        Args:
            text: Input text to analyze

        Returns:
            Sentiment prediction with confidence
        """
        if self.sentiment_model is None or self.sentiment_tokenizer is None:
            return {
                "success": False,
                "message": "Sentiment model not loaded. Train model first."
            }

        try:
            # Tokenize
            inputs = self.sentiment_tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[0]
                pred_label = torch.argmax(logits, dim=-1).item()

            label_names = {0: "negative", 1: "neutral", 2: "positive"}

            return {
                "success": True,
                "text": text,
                "label": label_names[pred_label],
                "confidence": float(probs[pred_label]),
                "probabilities": {
                    label_names[i]: float(probs[i])
                    for i in range(3)
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Prediction error: {str(e)}"
            }

    def extract_keywords(self, text: str, max_keywords: int = 10) -> Dict[str, Any]:
        """
        Extract keywords from text.

        Args:
            text: Input text
            max_keywords: Maximum number of keywords

        Returns:
            Extracted keywords
        """
        if self.keyword_model is None or self.keyword_tokenizer is None:
            return {
                "success": False,
                "message": "Keyword model not loaded. Train model first."
            }

        try:
            # Tokenize
            inputs = self.keyword_tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.keyword_model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True
                )

            # Decode
            keywords_str = self.keyword_tokenizer.decode(outputs[0], skip_special_tokens=True)
            keywords_list = [kw.strip() for kw in keywords_str.split(';') if kw.strip()]
            keywords_list = keywords_list[:max_keywords]

            return {
                "success": True,
                "text": text[:200] + "..." if len(text) > 200 else text,
                "keywords": keywords_list,
                "keywords_string": "; ".join(keywords_list)
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Keyword extraction error: {str(e)}"
            }


def interactive_mode():
    """
    Run MCP server in interactive mode.
    """
    print("\n" + "="*70)
    print("MCP SERVER - CONVERSATIONAL MODEL CONTROL")
    print("="*70)
    print("\nAvailable commands:")
    print("  set_lr <value>              - Set learning rate")
    print("  train <sentiment|keyword>   - Train model")
    print("  evaluate <sentiment|keyword|both> - Evaluate model")
    print("  metrics                     - Get current metrics")
    print("  predict <text>              - Predict sentiment")
    print("  keywords <text>             - Extract keywords")
    print("  help                        - Show this help")
    print("  exit                        - Exit")
    print("="*70 + "\n")

    server = MCPServer()

    while True:
        try:
            command = input("MCP> ").strip()

            if not command:
                continue

            parts = command.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "exit":
                print("Goodbye!")
                break

            elif cmd == "help":
                print("\nAvailable commands:")
                print("  set_lr <value>")
                print("  train <sentiment|keyword>")
                print("  evaluate <sentiment|keyword|both>")
                print("  metrics")
                print("  predict <text>")
                print("  keywords <text>")
                print("  help")
                print("  exit\n")

            elif cmd == "set_lr":
                if len(parts) < 2:
                    print("Usage: set_lr <value>")
                else:
                    lr = float(parts[1])
                    result = server.set_learning_rate(lr)
                    print(json.dumps(result, indent=2))

            elif cmd == "train":
                if len(parts) < 2:
                    print("Usage: train <sentiment|keyword>")
                else:
                    model_type = parts[1]
                    result = server.train_model(model_type)
                    print(json.dumps(result, indent=2))

            elif cmd == "evaluate":
                model_type = parts[1] if len(parts) > 1 else "both"
                result = server.evaluate_model(model_type)
                print(json.dumps(result, indent=2))

            elif cmd == "metrics":
                result = server.get_metrics()
                print(json.dumps(result, indent=2))

            elif cmd == "predict":
                if len(parts) < 2:
                    print("Usage: predict <text>")
                else:
                    text = parts[1]
                    result = server.predict_sentiment(text)
                    print(json.dumps(result, indent=2))

            elif cmd == "keywords":
                if len(parts) < 2:
                    print("Usage: keywords <text>")
                else:
                    text = parts[1]
                    result = server.extract_keywords(text)
                    print(json.dumps(result, indent=2))

            else:
                print(f"Unknown command: {cmd}. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    interactive_mode()
