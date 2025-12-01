# AI Lab ML Service: LoRA Fine-tuning & Multi-Platform Deployment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Complete end-to-end machine learning service featuring LoRA fine-tuned models for sentiment classification and keyword extraction, with multi-platform deployment capabilities.

## Project Overview

This project implements a production-ready ML service for **AIL406 Lab-8** that demonstrates:

- ✨ **LoRA/QLoRA Fine-tuning** of lightweight models (≤110M parameters)
- 🚀 **Multi-Platform Deployment** (FastAPI + Ngrok, HuggingFace Spaces)
- 🤖 **MCP Integration** for conversational model control
- 📊 **Comprehensive Metrics** and performance analysis
- 🔄 **CI/CD Pipeline** for automated deployment

### Models

| Model | Task | Base Architecture | Parameters | LoRA Rank |
|-------|------|-------------------|------------|-----------|
| **Model-A** | 3-way Sentiment Classification | DistilBERT | ~67M | 16 |
| **Model-B** | Keyword Extraction | T5-small | ~60M | 16 |

### Datasets

- **Sentiment**: Emotion dataset (mapped to positive/neutral/negative)
  - License: Apache 2.0
  - Source: https://huggingface.co/datasets/emotion

- **Keywords**: Inspec dataset for keyword extraction
  - License: CC-BY-4.0
  - Source: https://huggingface.co/datasets/midas/inspec

## Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended, optional)
- 8GB+ RAM
- 10GB+ disk space

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Model_Finetuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### One-Command Setup

```bash
# Download datasets, run baseline, train models
python download.py && python baseline.py && python lora_trainer.py
```

## Usage Guide

### Phase 1: Data Preparation & Baseline

#### Step 1: Download Datasets

```bash
python download.py
```

This will:
- Download emotion dataset for sentiment classification
- Download Inspec dataset for keyword extraction
- Split into train/val/test sets
- Save metadata in `data/metadata.json`

#### Step 2: Exploratory Data Analysis

```bash
jupyter notebook notebooks/01_explore.ipynb
```

Analyze:
- Dataset statistics
- Class distribution
- Text length distribution
- Sample data

#### Step 3: Baseline Evaluation

```bash
python baseline.py
```

Establishes zero-shot performance metrics before fine-tuning.

**Expected Output:**
- Baseline metrics saved to `results/sentiment_baseline.json`
- Baseline metrics saved to `results/keyword_baseline.json`
- Confusion matrix: `results/confusion_matrices/sentiment_baseline.png`

### Phase 2: LoRA Fine-tuning

#### Train Both Models

```bash
python lora_trainer.py --model both
```

Or train individually:

```bash
# Sentiment model only
python lora_trainer.py --model sentiment

# Keyword model only
python lora_trainer.py --model keyword
```

**Training Configuration:**
- Learning Rate: 2e-4
- Epochs: 3
- Batch Size: 16 (sentiment), 8 (keywords)
- LoRA Rank: 16
- Quantization: 4-bit (QLoRA)

**Outputs:**
- Trained models: `models/sentiment_model_best/`, `models/keyword_model_best/`
- Metrics: `results/sentiment_finetuned.json`, `results/keyword_finetuned.json`
- Sample predictions: `results/sample_predictions_*.csv`
- Confusion matrices: `results/confusion_matrices/`

### Phase 3: Deployment

#### Option 1: FastAPI + Ngrok

##### Setup Ngrok (First Time)

1. Sign up at https://ngrok.com/ (FREE account)
2. Get auth token from https://dashboard.ngrok.com/get-started/your-authtoken
3. Install ngrok:
   ```bash
   # macOS
   brew install ngrok

   # Linux
   snap install ngrok

   # Windows
   choco install ngrok
   ```
4. Configure token:
   ```bash
   ngrok config add-authtoken YOUR_AUTH_TOKEN
   ```

##### Run the Service

```bash
python ngrok_setup.py
```

This will:
- Start FastAPI server on port 8000
- Create public HTTPS tunnel via Ngrok
- Test all endpoints
- Display public URL

**API Endpoints:**

```bash
# Health check
curl https://YOUR_NGROK_URL/health

# Sentiment prediction
curl -X POST https://YOUR_NGROK_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'

# Keyword extraction
curl -X POST https://YOUR_NGROK_URL/keywords \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning is a branch of AI...", "max_keywords": 10}'

# Get metrics
curl https://YOUR_NGROK_URL/metrics
```

**Interactive Documentation:** Visit `https://YOUR_NGROK_URL/docs`

#### Option 2: Local FastAPI Server

```bash
python app.py
```

Access at: http://localhost:8000

### Phase 4: MCP Integration

The MCP (Model Context Protocol) server provides conversational control over training and inference.

```bash
python mcp_server.py
```

**Available Commands:**

```bash
MCP> set_lr 1e-4              # Adjust learning rate
MCP> train sentiment          # Trigger training
MCP> evaluate both            # Run evaluation
MCP> metrics                  # Get current metrics
MCP> predict This is great!   # Sentiment prediction
MCP> keywords <text>          # Extract keywords
MCP> help                     # Show help
MCP> exit                     # Exit
```

### Phase 5: HuggingFace Spaces Deployment

#### Setup HuggingFace (First Time)

1. Create account at https://huggingface.co/
2. Create new Space:
   - Go to https://huggingface.co/new-space
   - Select "Streamlit" as SDK
   - Choose a name (e.g., `ai-lab-ml-service`)
3. Get access token:
   - Go to https://huggingface.co/settings/tokens
   - Create new token with "write" permissions

#### Deploy to Spaces

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login --token YOUR_HF_TOKEN

# Push to Space
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git add .
git commit -m "Initial deployment"
git push hf main
```

Or use the Dockerfile:

```bash
# Build locally
docker build -t ml-service .

# Run locally
docker run -p 7860:7860 ml-service

# Access at http://localhost:7860
```

#### Local Streamlit Testing

```bash
streamlit run space.py
```

## Project Structure

```
Model_Finetuning/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── Dockerfile               # Docker configuration
│
├── download.py              # Dataset download script
├── baseline.py              # Zero-shot baseline evaluation
├── lora_trainer.py          # Main training script
├── app.py                   # FastAPI service
├── ngrok_setup.py           # Ngrok tunnel setup
├── mcp_server.py            # MCP conversational server
├── space.py                 # Streamlit UI for HF Spaces
│
├── src/                     # Source modules
│   ├── __init__.py
│   ├── data_loader.py       # Data loading utilities
│   ├── model_utils.py       # Model utilities
│   ├── trainer.py           # Training logic
│   └── evaluator.py         # Evaluation metrics
│
├── data/                    # Datasets (auto-generated)
│   ├── sentiment_train.csv
│   ├── sentiment_val.csv
│   ├── sentiment_test.csv
│   ├── keywords_train.csv
│   ├── keywords_val.csv
│   ├── keywords_test.csv
│   └── metadata.json
│
├── models/                  # Trained models (auto-generated)
│   ├── sentiment_model_best/
│   ├── sentiment_model_final/
│   ├── keyword_model_best/
│   └── keyword_model_final/
│
├── results/                 # Results and metrics
│   ├── confusion_matrices/
│   ├── sentiment_baseline.json
│   ├── sentiment_finetuned.json
│   ├── keyword_baseline.json
│   ├── keyword_finetuned.json
│   ├── sample_predictions_sentiment.csv
│   └── sample_predictions_keywords.csv
│
├── notebooks/               # Jupyter notebooks
│   └── 01_explore.ipynb    # Exploratory data analysis
│
└── .github/
    └── workflows/
        └── deploy.yml       # CI/CD pipeline
```

## Performance Metrics

### Expected Results

After LoRA fine-tuning, you should see:

**Sentiment Classification:**
- Accuracy: > 0.85
- F1 Score (macro): > 0.83
- Inference Latency: < 50ms per sample

**Keyword Extraction:**
- ROUGE-1: > 0.35
- ROUGE-L: > 0.28
- Inference Latency: < 200ms per sample

### Viewing Results

```bash
# View metrics
cat results/sentiment_finetuned.json
cat results/keyword_finetuned.json

# View sample predictions
head -n 20 results/sample_predictions_sentiment.csv
head -n 20 results/sample_predictions_keywords.csv

# View confusion matrices
open results/confusion_matrices/sentiment_finetuned.png
```

## Technical Details

### LoRA Configuration

```python
LoraConfig(
    task_type=TaskType.SEQ_CLS,  # or SEQ_2_SEQ_LM
    r=16,                         # Rank ≤ 32
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)
```

### QLoRA (4-bit Quantization)

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
```

### Reproducibility

All experiments use seed `42`:
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size in lora_trainer.py
CONFIG = {
    'sentiment': {'batch_size': 8},  # Reduce from 16
    'keyword': {'batch_size': 4}     # Reduce from 8
}
```

#### 2. Ngrok Authentication Failed

```bash
# Reconfigure auth token
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

#### 3. Models Not Loading

```bash
# Verify models exist
ls -la models/

# Retrain if needed
python lora_trainer.py --model both
```

#### 4. HuggingFace Upload Fails

```bash
# Check token permissions
huggingface-cli whoami

# Re-login
huggingface-cli login --token YOUR_TOKEN
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Code Formatting

```bash
# Install black
pip install black

# Format code
black src/ *.py
```

## Citation

If you use this project, please cite:

```bibtex
@misc{ailab2025mlservice,
  title={AI Lab ML Service: LoRA Fine-tuning \& Multi-Platform Deployment},
  author={AI Lab Project},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/username/Model_Finetuning}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Datasets**: emotion (Hugging Face), midas/inspec (Hugging Face)
- **Libraries**: Transformers, PEFT, bitsandbytes, FastAPI, Streamlit
- **Course**: AIL406 Lab-8, AI Lab Project 2025

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

**Built with ❤️ for AI Lab Project 2025**
