"""
Streamlit UI for HuggingFace Spaces
Interactive web interface for sentiment classification and keyword extraction.
"""

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="AI Lab ML Service",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_sentiment_model():
    """Load sentiment classification model."""
    model_path = Path("models/sentiment_model_best")
    if model_path.exists():
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return model, tokenizer
    return None, None


@st.cache_resource
def load_keyword_model():
    """Load keyword extraction model."""
    model_path = Path("models/keyword_model_best")
    if model_path.exists():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return model, tokenizer
    return None, None


@st.cache_data
def load_metrics():
    """Load model performance metrics."""
    sentiment_metrics = {}
    keyword_metrics = {}

    sentiment_file = Path("results/sentiment_finetuned.json")
    if sentiment_file.exists():
        with open(sentiment_file) as f:
            sentiment_metrics = json.load(f)

    keyword_file = Path("results/keyword_finetuned.json")
    if keyword_file.exists():
        with open(keyword_file) as f:
            keyword_metrics = json.load(f)

    return sentiment_metrics, keyword_metrics


def predict_sentiment(text, model, tokenizer):
    """Predict sentiment of text."""
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_label = torch.argmax(logits, dim=-1).item()

    label_names = {0: "negative", 1: "neutral", 2: "positive"}

    return {
        "label": label_names[pred_label],
        "confidence": float(probs[pred_label]),
        "probabilities": {
            label_names[i]: float(probs[i])
            for i in range(3)
        }
    }


def extract_keywords_func(text, model, tokenizer, max_keywords=10):
    """Extract keywords from text."""
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )

    keywords_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    keywords_list = [kw.strip() for kw in keywords_str.split(';') if kw.strip()]
    keywords_list = keywords_list[:max_keywords]

    return keywords_list


# Main UI
def main():
    # Header
    st.markdown('<p class="main-header">🤖 AI Lab ML Service</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">LoRA Fine-tuned Models for NLP Tasks</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("📊 About")
    st.sidebar.info("""
    This demo showcases two LoRA fine-tuned models:

    **Model-A:** 3-way Sentiment Classification
    - Positive / Neutral / Negative

    **Model-B:** Keyword Extraction
    - Extractive keyword generation

    **Technology Stack:**
    - LoRA/QLoRA fine-tuning
    - DistilBERT & T5-small
    - Streamlit UI
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**AI Lab Project** | AIL406 Lab-8")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Sentiment Analysis", "🔑 Keyword Extraction", "📈 Model Metrics"])

    # Tab 1: Sentiment Analysis
    with tab1:
        st.header("Sentiment Analysis")
        st.write("Analyze the sentiment of your text (Positive, Neutral, or Negative)")

        # Load model
        sentiment_model, sentiment_tokenizer = load_sentiment_model()

        if sentiment_model is None:
            st.error("⚠️ Sentiment model not found. Please train the model first.")
        else:
            # Input
            col1, col2 = st.columns([2, 1])

            with col1:
                text_input = st.text_area(
                    "Enter text to analyze:",
                    height=150,
                    placeholder="Type or paste your text here..."
                )

            with col2:
                st.markdown("**Examples:**")
                if st.button("Positive Example", use_container_width=True):
                    text_input = "This is an amazing product! I absolutely love it and highly recommend it to everyone."
                    st.rerun()
                if st.button("Neutral Example", use_container_width=True):
                    text_input = "The product arrived on time. It works as described in the manual."
                    st.rerun()
                if st.button("Negative Example", use_container_width=True):
                    text_input = "Very disappointed with this purchase. Poor quality and didn't meet my expectations."
                    st.rerun()

            if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
                if text_input:
                    with st.spinner("Analyzing..."):
                        result = predict_sentiment(text_input, sentiment_model, sentiment_tokenizer)

                        # Display results
                        st.success("✅ Analysis Complete!")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Predicted Sentiment", result["label"].upper())
                        with col2:
                            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                        with col3:
                            emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}
                            st.markdown(f"### {emoji[result['label']]}")

                        # Probability chart
                        st.subheader("Probability Distribution")
                        probs_df = pd.DataFrame({
                            'Sentiment': list(result['probabilities'].keys()),
                            'Probability': list(result['probabilities'].values())
                        })

                        fig = px.bar(
                            probs_df,
                            x='Sentiment',
                            y='Probability',
                            color='Sentiment',
                            color_discrete_map={
                                'positive': '#2ecc71',
                                'neutral': '#95a5a6',
                                'negative': '#e74c3c'
                            }
                        )
                        fig.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.warning("Please enter some text to analyze.")

    # Tab 2: Keyword Extraction
    with tab2:
        st.header("Keyword Extraction")
        st.write("Extract important keywords from your document")

        # Load model
        keyword_model, keyword_tokenizer = load_keyword_model()

        if keyword_model is None:
            st.error("⚠️ Keyword model not found. Please train the model first.")
        else:
            # Input
            col1, col2 = st.columns([2, 1])

            with col1:
                doc_input = st.text_area(
                    "Enter document:",
                    height=200,
                    placeholder="Paste your document or text here..."
                )

            with col2:
                max_kw = st.slider("Max keywords", 5, 20, 10)
                st.markdown("**Example:**")
                if st.button("Load Example Document", use_container_width=True):
                    doc_input = "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models. These models enable computer systems to improve their performance on a specific task through experience. Deep learning, a subfield of machine learning, uses neural networks with multiple layers to process data."
                    st.rerun()

            if st.button("🔑 Extract Keywords", type="primary", use_container_width=True):
                if doc_input:
                    with st.spinner("Extracting keywords..."):
                        keywords = extract_keywords_func(doc_input, keyword_model, keyword_tokenizer, max_kw)

                        # Display results
                        st.success("✅ Extraction Complete!")

                        st.subheader("Extracted Keywords")

                        # Display as pills
                        keyword_html = " ".join([
                            f'<span style="background-color:#1f77b4;color:white;padding:5px 15px;margin:5px;border-radius:20px;display:inline-block;">{kw}</span>'
                            for kw in keywords
                        ])
                        st.markdown(keyword_html, unsafe_allow_html=True)

                        st.markdown("---")

                        # Display as list
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Keywords List:**")
                            for i, kw in enumerate(keywords, 1):
                                st.write(f"{i}. {kw}")

                        with col2:
                            st.markdown("**Copy-friendly format:**")
                            st.code("; ".join(keywords))

                else:
                    st.warning("Please enter some text to extract keywords from.")

    # Tab 3: Metrics
    with tab3:
        st.header("Model Performance Metrics")
        st.write("Evaluation results after LoRA fine-tuning")

        sentiment_metrics, keyword_metrics = load_metrics()

        if sentiment_metrics or keyword_metrics:
            col1, col2 = st.columns(2)

            # Sentiment metrics
            with col1:
                st.subheader("📊 Sentiment Model")
                if sentiment_metrics:
                    st.metric("Accuracy", f"{sentiment_metrics.get('accuracy', 0):.4f}")
                    st.metric("F1 Score (macro)", f"{sentiment_metrics.get('f1_macro', 0):.4f}")
                    st.metric("Precision", f"{sentiment_metrics.get('precision', 0):.4f}")
                    st.metric("Recall", f"{sentiment_metrics.get('recall', 0):.4f}")

                    st.markdown("**Per-class F1:**")
                    st.write(f"- Positive: {sentiment_metrics.get('f1_positive', 0):.4f}")
                    st.write(f"- Neutral: {sentiment_metrics.get('f1_neutral', 0):.4f}")
                    st.write(f"- Negative: {sentiment_metrics.get('f1_negative', 0):.4f}")

                    st.markdown("**Inference Latency:**")
                    st.write(f"- Mean: {sentiment_metrics.get('mean_latency_ms', 0):.2f} ms")
                    st.write(f"- P95: {sentiment_metrics.get('p95_latency_ms', 0):.2f} ms")
                else:
                    st.info("No metrics available yet.")

            # Keyword metrics
            with col2:
                st.subheader("📊 Keyword Model")
                if keyword_metrics:
                    st.metric("ROUGE-1", f"{keyword_metrics.get('rouge1', 0):.4f}")
                    st.metric("ROUGE-2", f"{keyword_metrics.get('rouge2', 0):.4f}")
                    st.metric("ROUGE-L", f"{keyword_metrics.get('rougeL', 0):.4f}")

                    st.markdown("**Inference Latency:**")
                    st.write(f"- Mean: {keyword_metrics.get('mean_latency_ms', 0):.2f} ms")
                    st.write(f"- P95: {keyword_metrics.get('p95_latency_ms', 0):.2f} ms")

                    st.markdown("**Training Configuration:**")
                    st.write(f"- Model: {keyword_metrics.get('model', 'N/A')}")
                    st.write(f"- LoRA Rank: {keyword_metrics.get('lora_rank', 'N/A')}")
                    st.write(f"- Epochs: {keyword_metrics.get('epochs', 'N/A')}")
                else:
                    st.info("No metrics available yet.")

        else:
            st.warning("⚠️ No metrics found. Please train and evaluate the models first.")

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#666;font-size:0.9rem;">Built with LoRA/QLoRA Fine-tuning | AI Lab Project 2025</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
