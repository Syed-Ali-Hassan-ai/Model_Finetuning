# AI Lab ML Service - Project Report Template

**AIL406 Lab-8: LoRA Fine-tuning & Multi-Platform Deployment**

---

## Report Structure (6 Pages Maximum)

Use this template to structure your project report. Fill in each section with your actual results and findings.

---

## Page 1: Title & Introduction

### Title Page

```
AI Lab ML Service: LoRA Fine-tuning & Multi-Platform Deployment

Course: AIL406 Lab-8
Student Name: [Your Name]
Roll Number: [Your Roll Number]
Date: [Submission Date]
Semester: [Current Semester]
```

### Abstract (150-200 words)

Template:
```
This project implements an end-to-end machine learning service featuring
LoRA fine-tuned models for [mention tasks]. We developed two lightweight
models: Model-A for [sentiment classification] achieving [X%] accuracy,
and Model-B for [keyword extraction] achieving [X] ROUGE-L score. The
service is deployed via [FastAPI + Ngrok] and [HuggingFace Spaces], with
conversational control through MCP. Our results demonstrate that LoRA
fine-tuning provides [mention key improvement] while maintaining
[inference latency < 500ms].
```

### 1. Introduction (0.5 page)

- Brief overview of the project goals
- Mention the two tasks: sentiment classification and keyword extraction
- State the key technologies: LoRA/QLoRA, FastAPI, HF Spaces, MCP

---

## Page 2: Dataset Description & Methodology

### 2. Dataset Description (0.5 page)

#### 2.1 Sentiment Dataset

Fill in your actual numbers:
```
- **Source**: Emotion dataset from HuggingFace
- **License**: Apache 2.0 (https://huggingface.co/datasets/emotion)
- **Total Samples**: [X] (Train: [Y], Val: [Z], Test: [W])
- **Classes**: Positive, Neutral, Negative
- **Mapping**: 6 original emotions → 3 sentiment classes
- **Average Text Length**: [X] words
- **Class Distribution**:
  - Positive: [X]%
  - Neutral: [Y]%
  - Negative: [Z]%
```

#### 2.2 Keyword Dataset

Fill in your actual numbers:
```
- **Source**: Inspec dataset from HuggingFace (midas/inspec)
- **License**: CC-BY-4.0 (https://huggingface.co/datasets/midas/inspec)
- **Total Samples**: [X] (Train: [Y], Val: [Z], Test: [W])
- **Task**: Extractive keyword generation
- **Average Keywords per Document**: [X]
- **Average Document Length**: [X] words
```

### 3. Methodology (0.5 page)

#### 3.1 Model Architecture

**Model-A: Sentiment Classification**
```
- Base Model: DistilBERT (67M parameters)
- LoRA Configuration:
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.1
  - Target Modules: q_proj, v_proj
- Quantization: 4-bit (QLoRA)
- Trainable Parameters: [X]% of total
```

**Model-B: Keyword Extraction**
```
- Base Model: T5-small (60M parameters)
- LoRA Configuration:
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.1
  - Target Modules: q_proj, v_proj
- Quantization: 4-bit (QLoRA)
- Trainable Parameters: [X]% of total
```

#### 3.2 Training Configuration

```
- Learning Rate: 2e-4
- Optimizer: AdamW
- Scheduler: Linear with warmup
- Epochs: 3
- Batch Size: 16 (sentiment), 8 (keywords)
- Random Seed: 42 (for reproducibility)
```

---

## Page 3: Baseline Results & Training

### 4. Baseline Metrics (0.5 page)

#### 4.1 Sentiment Classification (Zero-shot)

Copy from `results/sentiment_baseline.json`:

| Metric | Value |
|--------|-------|
| Accuracy | [0.XXXX] |
| F1 (macro) | [0.XXXX] |
| Precision | [0.XXXX] |
| Recall | [0.XXXX] |
| F1 Positive | [0.XXXX] |
| F1 Neutral | [0.XXXX] |
| F1 Negative | [0.XXXX] |
| Mean Latency | [XX.XX] ms |
| P95 Latency | [XX.XX] ms |

#### 4.2 Keyword Extraction (Zero-shot)

Copy from `results/keyword_baseline.json`:

| Metric | Value |
|--------|-------|
| ROUGE-1 | [0.XXXX] |
| ROUGE-2 | [0.XXXX] |
| ROUGE-L | [0.XXXX] |
| Mean Latency | [XXX.XX] ms |
| P95 Latency | [XXX.XX] ms |

### 5. Training Process (0.5 page)

Include:
- Training duration (approximate time taken)
- Hardware used (GPU model or CPU)
- Training curves (if available from notebooks)
- Any observations during training

Example text:
```
Training was conducted on [GPU/CPU model] with the following observations:
- Sentiment model converged after [X] epochs
- Keyword model showed [describe learning pattern]
- Total training time: [X] hours
- Memory usage: [X] GB
```

---

## Page 4: Fine-tuning Results & Analysis

### 6. Final Metrics After LoRA Fine-tuning (0.5 page)

#### 6.1 Sentiment Classification

Copy from `results/sentiment_finetuned.json`:

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| Accuracy | [0.XXXX] | [0.XXXX] | +[0.XXXX] |
| F1 (macro) | [0.XXXX] | [0.XXXX] | +[0.XXXX] |
| Precision | [0.XXXX] | [0.XXXX] | +[0.XXXX] |
| Recall | [0.XXXX] | [0.XXXX] | +[0.XXXX] |
| Mean Latency | [XX.XX] ms | [XX.XX] ms | [+/-X] ms |

**Per-class F1 Scores:**
- Positive: [0.XXXX]
- Neutral: [0.XXXX]
- Negative: [0.XXXX]

#### 6.2 Keyword Extraction

Copy from `results/keyword_finetuned.json`:

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| ROUGE-1 | [0.XXXX] | [0.XXXX] | +[0.XXXX] |
| ROUGE-2 | [0.XXXX] | [0.XXXX] | +[0.XXXX] |
| ROUGE-L | [0.XXXX] | [0.XXXX] | +[0.XXXX] |
| Mean Latency | [XXX.XX] ms | [XXX.XX] ms | [+/-X] ms |

### 7. Confusion Matrix Analysis (0.5 page)

**Insert confusion matrix image**: `results/confusion_matrices/sentiment_finetuned.png`

Analysis:
```
The confusion matrix shows that:
- The model performs best on [class] with [X]% accuracy
- Common misclassifications occur between [class1] and [class2]
- Overall, the model shows [describe pattern]
```

---

## Page 5: Deployment & Architecture

### 8. System Architecture (0.5 page)

Include a diagram showing:
```
┌─────────────────┐
│   User Input    │
└────────┬────────┘
         │
    ┌────▼────┐
    │ FastAPI │────► Ngrok Tunnel ──► Public HTTPS
    └────┬────┘
         │
    ┌────▼─────────┐
    │ LoRA Models  │
    │ - Sentiment  │
    │ - Keywords   │
    └──────────────┘
```

### 9. Deployment Methods (0.5 page)

#### 9.1 FastAPI + Ngrok

```
- **FastAPI Server**: Runs on port 8000
- **Ngrok Tunnel**: Provides public HTTPS URL
- **Endpoints**:
  - POST /predict (sentiment)
  - POST /keywords (keyword extraction)
  - GET /health (health check)
  - GET /metrics (performance metrics)
- **Public URL**: [Insert your Ngrok URL screenshot]
```

#### 9.2 HuggingFace Spaces

```
- **Platform**: HuggingFace Spaces (Streamlit)
- **UI Features**:
  - Tab 1: Sentiment Analysis with confidence scores
  - Tab 2: Keyword Extraction with adjustable count
  - Tab 3: Performance metrics dashboard
- **Space URL**: [Insert your HF Space URL]
- **Screenshot**: [Insert Streamlit UI screenshot]
```

#### 9.3 MCP Conversational Control

```
- **Tools Implemented**:
  - set_learning_rate(lr)
  - train_model(model_type, epochs)
  - evaluate_model(model_type)
  - get_metrics()
  - predict_sentiment(text)
  - extract_keywords(text)
- **Screenshot**: [Insert MCP terminal screenshot]
```

---

## Page 6: Results Summary & Conclusion

### 10. Sample Predictions (0.5 page)

Include 3-5 examples from `results/sample_predictions_sentiment.csv`:

| Input Text | True Label | Predicted | Confidence |
|------------|------------|-----------|------------|
| [text] | [label] | [label] | [0.XX] |
| [text] | [label] | [label] | [0.XX] |
| [text] | [label] | [label] | [0.XX] |

Include 2-3 examples from `results/sample_predictions_keywords.csv`:

| Document (excerpt) | True Keywords | Predicted Keywords |
|-------------------|---------------|-------------------|
| [text...] | [keywords] | [keywords] |
| [text...] | [keywords] | [keywords] |

### 11. Performance Summary (0.25 page)

Key achievements:
```
✓ Sentiment model achieves [X]% accuracy (improvement: +[Y]%)
✓ Keyword model achieves [X] ROUGE-L score (improvement: +[Y])
✓ Inference latency < 500ms for both models ✓
✓ Successfully deployed on 2 platforms (Ngrok, HF Spaces) ✓
✓ MCP integration for conversational control ✓
```

### 12. Conclusion (0.25 page)

Template:
```
This project successfully demonstrates end-to-end ML service development
with LoRA fine-tuning. Our results show that:

1. **Performance Gains**: LoRA fine-tuning improved sentiment accuracy
   by [X]% and keyword ROUGE-L by [Y], demonstrating effectiveness of
   parameter-efficient fine-tuning.

2. **Deployment**: Multi-platform deployment enables both API access
   (FastAPI + Ngrok) and user-friendly interface (HF Spaces), making
   the models accessible to diverse users.

3. **Efficiency**: With only [X]% trainable parameters and inference
   latency < 500ms, the system balances performance with resource
   efficiency.

Future work could explore:
- Expanding to multilingual support
- Implementing active learning for continuous improvement
- Optimizing inference further with model distillation
```

---

## Appendix: Screenshots to Include

### Required Screenshots:

1. **Ngrok URL Working**
   - Terminal showing public URL
   - Browser showing API docs (http://YOUR_URL/docs)
   - Successful API response

2. **HuggingFace Space Running**
   - Streamlit UI showing sentiment analysis
   - Keyword extraction tab
   - Metrics dashboard

3. **MCP Chat Interface**
   - Terminal showing MCP commands
   - Example predictions
   - Metrics retrieval

4. **Confusion Matrix**
   - Image from `results/confusion_matrices/sentiment_finetuned.png`

5. **Training Output**
   - Terminal showing training progress
   - Final metrics

### How to Take Screenshots:

**macOS**: Cmd + Shift + 4
**Windows**: Win + Shift + S
**Linux**: PrtScn or Screenshot app

---

## Deliverables Checklist

Before submission, ensure you have:

- [ ] 6-page PDF report (this template filled out)
- [ ] GitHub repository with all code
- [ ] README.md with one-liner installation
- [ ] Confusion matrices (PNG files)
- [ ] 50 sample predictions CSV
- [ ] Screenshots of working deployments
- [ ] 2-minute demo video (optional but recommended)

---

## Tips for Writing

1. **Be Concise**: Stick to 6 pages maximum
2. **Use Tables**: Present metrics in tables for clarity
3. **Include Visuals**: Confusion matrices, architecture diagrams
4. **Cite Sources**: Dataset licenses, library documentation
5. **Quantify Results**: Always include numbers, not just "improved"
6. **Proofread**: Check for typos and formatting consistency

---

**Good luck with your project submission!**
