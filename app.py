"""
FastAPI Service for ML Model Deployment
Provides REST API endpoints for sentiment classification and keyword extraction.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import json
import time
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="AI Lab ML Service",
    description="LoRA Fine-tuned Models for Sentiment Classification and Keyword Extraction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
sentiment_model = None
sentiment_tokenizer = None
keyword_model = None
keyword_tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Label mapping
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


# Pydantic models for request/response
class SentimentRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for sentiment", min_length=1, max_length=512)


class SentimentResponse(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float


class KeywordRequest(BaseModel):
    text: str = Field(..., description="Text to extract keywords from", min_length=1, max_length=2048)
    max_keywords: int = Field(10, description="Maximum number of keywords", ge=1, le=20)


class KeywordResponse(BaseModel):
    text: str
    keywords: List[str]
    keywords_string: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    device: str


class MetricsResponse(BaseModel):
    sentiment_metrics: Dict
    keyword_metrics: Dict


@app.on_event("startup")
async def load_models():
    """
    Load models on startup.
    """
    global sentiment_model, sentiment_tokenizer, keyword_model, keyword_tokenizer

    print("Loading models...")

    # Load sentiment model
    sentiment_path = Path("models/sentiment_model_best")
    if sentiment_path.exists():
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_path)
        sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_path)
        sentiment_model.to(device)
        sentiment_model.eval()
        print(f"✓ Sentiment model loaded from {sentiment_path}")
    else:
        print(f"⚠ Sentiment model not found at {sentiment_path}")

    # Load keyword model
    keyword_path = Path("models/keyword_model_best")
    if keyword_path.exists():
        keyword_model = AutoModelForSeq2SeqLM.from_pretrained(keyword_path)
        keyword_tokenizer = AutoTokenizer.from_pretrained(keyword_path)
        keyword_model.to(device)
        keyword_model.eval()
        print(f"✓ Keyword model loaded from {keyword_path}")
    else:
        print(f"⚠ Keyword model not found at {keyword_path}")

    print(f"✓ Models loaded on device: {device}")


@app.get("/", response_model=Dict)
async def root():
    """
    Root endpoint.
    """
    return {
        "message": "AI Lab ML Service API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Sentiment prediction",
            "POST /keywords": "Keyword extraction",
            "GET /health": "Health check",
            "GET /metrics": "Model metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded={
            "sentiment": sentiment_model is not None,
            "keyword": keyword_model is not None
        },
        device=device
    )


@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment of input text.
    """
    if sentiment_model is None or sentiment_tokenizer is None:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded")

    try:
        # Tokenize
        inputs = sentiment_tokenizer(
            request.text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_label = torch.argmax(logits, dim=-1).item()
        inference_time = (time.time() - start_time) * 1000

        # Format response
        probabilities = {
            LABEL_NAMES[i]: float(probs[i])
            for i in range(len(LABEL_NAMES))
        }

        return SentimentResponse(
            text=request.text,
            label=LABEL_NAMES[pred_label],
            confidence=float(probs[pred_label]),
            probabilities=probabilities,
            inference_time_ms=inference_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/keywords", response_model=KeywordResponse)
async def extract_keywords(request: KeywordRequest):
    """
    Extract keywords from input text.
    """
    if keyword_model is None or keyword_tokenizer is None:
        raise HTTPException(status_code=503, detail="Keyword model not loaded")

    try:
        # Tokenize
        inputs = keyword_tokenizer(
            request.text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = keyword_model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
        inference_time = (time.time() - start_time) * 1000

        # Decode
        keywords_str = keyword_tokenizer.decode(outputs[0], skip_special_tokens=True)
        keywords_list = [kw.strip() for kw in keywords_str.split(';') if kw.strip()]

        # Limit keywords
        keywords_list = keywords_list[:request.max_keywords]

        return KeywordResponse(
            text=request.text[:200] + "..." if len(request.text) > 200 else request.text,
            keywords=keywords_list,
            keywords_string="; ".join(keywords_list),
            inference_time_ms=inference_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Keyword extraction error: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get model performance metrics.
    """
    results_dir = Path("results")

    sentiment_metrics = {}
    keyword_metrics = {}

    # Load sentiment metrics
    sentiment_file = results_dir / "sentiment_finetuned.json"
    if sentiment_file.exists():
        with open(sentiment_file) as f:
            sentiment_metrics = json.load(f)

    # Load keyword metrics
    keyword_file = results_dir / "keyword_finetuned.json"
    if keyword_file.exists():
        with open(keyword_file) as f:
            keyword_metrics = json.load(f)

    return MetricsResponse(
        sentiment_metrics=sentiment_metrics,
        keyword_metrics=keyword_metrics
    )


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("AI LAB ML SERVICE - FastAPI Server")
    print("="*60)
    print("\nStarting server...")
    print("Docs available at: http://localhost:8000/docs")
    print("="*60 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
