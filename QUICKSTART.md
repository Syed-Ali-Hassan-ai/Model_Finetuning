# Quick Start Guide - AI Lab ML Service

## 🎉 Project Setup Complete!

All files have been created and committed to your git repository. Here's what you need to do next.

---

## ⚡ Next Steps (In Order)

### Step 1: Set Up Python Environment (5 minutes)

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Datasets (5 minutes)

```bash
python download.py
```

This will download:
- ✅ Emotion dataset for sentiment classification
- ✅ Inspec dataset for keyword extraction
- ✅ Split into train/val/test sets

**Expected output:** Files in `data/` directory

### Step 3: Run Exploratory Data Analysis (10 minutes)

```bash
jupyter notebook notebooks/01_explore.ipynb
```

This will show you:
- Dataset statistics
- Class distributions
- Sample data
- Data quality checks

### Step 4: Establish Baseline (10 minutes)

```bash
python baseline.py
```

This evaluates zero-shot performance before fine-tuning.

**Expected output:**
- `results/sentiment_baseline.json`
- `results/keyword_baseline.json`
- Confusion matrix PNG

### Step 5: Train Models with LoRA (1-2 hours)

```bash
python lora_trainer.py --model both
```

**Note:** This is the longest step. On GPU it takes ~30-60 minutes, on CPU 1-2 hours.

**Expected output:**
- `models/sentiment_model_best/`
- `models/keyword_model_best/`
- `results/sentiment_finetuned.json`
- `results/keyword_finetuned.json`
- Confusion matrices
- Sample predictions CSV

**Performance targets:**
- Sentiment accuracy: > 85%
- Keyword ROUGE-L: > 0.28
- Inference latency: < 500ms

---

## 🚀 Deployment Options

After training, choose your deployment method:

### Option A: Local Testing (Fastest)

```bash
# Test FastAPI locally
python app.py
# Visit: http://localhost:8000/docs

# Test Streamlit UI locally
streamlit run space.py
# Visit: http://localhost:8501

# Test MCP server
python mcp_server.py
```

### Option B: Ngrok Tunnel (Public Access)

**First time setup (5 minutes):**
1. Sign up at https://ngrok.com/ (FREE)
2. Get auth token from https://dashboard.ngrok.com/get-started/your-authtoken
3. Install ngrok:
   ```bash
   brew install ngrok  # macOS
   # OR
   snap install ngrok  # Linux
   ```
4. Configure:
   ```bash
   ngrok config add-authtoken YOUR_TOKEN
   ```

**Then run:**
```bash
python ngrok_setup.py
```

You'll get a public HTTPS URL like: `https://xxxx.ngrok-free.app`

### Option C: HuggingFace Spaces (Free Hosting)

**First time setup (10 minutes):**
1. Create account at https://huggingface.co/
2. Create new Space (Streamlit SDK)
3. Get access token (with Write permission)
4. Install HF CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

**Deploy:**
```bash
# Add remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Push code
git push hf main

# Upload models (large files)
huggingface-cli upload YOUR_USERNAME/YOUR_SPACE_NAME models/sentiment_model_best --repo-type=space
huggingface-cli upload YOUR_USERNAME/YOUR_SPACE_NAME models/keyword_model_best --repo-type=space
```

---

## 📋 Project Structure Overview

```
Model_Finetuning/
├── README.md                    # Main documentation
├── SETUP_GUIDE.md              # Detailed setup for external services
├── PROJECT_REPORT_TEMPLATE.md  # Template for your 6-page report
├── QUICKSTART.md               # This file!
│
├── download.py                 # Step 2: Download datasets
├── baseline.py                 # Step 4: Baseline evaluation
├── lora_trainer.py             # Step 5: Train models
│
├── app.py                      # FastAPI server
├── ngrok_setup.py              # Ngrok tunnel
├── mcp_server.py               # MCP conversational control
├── space.py                    # Streamlit UI
│
├── src/                        # Source modules (don't modify)
├── notebooks/                  # EDA notebook
├── data/                       # Datasets (auto-generated)
├── models/                     # Trained models (auto-generated)
└── results/                    # Metrics and outputs (auto-generated)
```

---

## 🎯 Deliverables for Submission

When you're ready to submit, you'll need:

### 1. Code Repository
- ✅ Already done! All code is committed to git
- Just share the GitHub repository URL

### 2. Project Report (6 pages)
- Use `PROJECT_REPORT_TEMPLATE.md` as your guide
- Include:
  - Dataset description
  - Baseline metrics
  - Fine-tuned metrics
  - Confusion matrices
  - Screenshots of deployments

### 3. Results Package (ZIP)
Create a ZIP with:
```bash
zip -r results_package.zip \
  results/confusion_matrices/ \
  results/sample_predictions_sentiment.csv \
  results/sample_predictions_keywords.csv \
  results/*_finetuned.json \
  results/*_baseline.json
```

### 4. Demo Screenshots
Capture these:
- ✅ Ngrok URL working (terminal + browser)
- ✅ HuggingFace Space UI (all 3 tabs)
- ✅ MCP terminal showing commands
- ✅ Confusion matrices
- ✅ Training output

### 5. Demo Video (Optional, 2 minutes)
Script:
- 0:00-0:20: Project overview
- 0:20-0:40: Show datasets & metrics
- 0:40-1:00: FastAPI + Ngrok demo
- 1:00-1:30: HF Space demo
- 1:30-1:50: MCP conversational control
- 1:50-2:00: Results summary

---

## 🆘 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in `lora_trainer.py`:
```python
CONFIG = {
    'sentiment': {'batch_size': 8},  # was 16
    'keyword': {'batch_size': 4}      # was 8
}
```

### Issue: "Models not found"
**Solution:** Make sure training completed:
```bash
ls -la models/sentiment_model_best/
ls -la models/keyword_model_best/
```

### Issue: "Ngrok authentication failed"
**Solution:** Reconfigure token:
```bash
ngrok config add-authtoken YOUR_TOKEN
ngrok config check
```

### Issue: "HuggingFace upload failed"
**Solution:** Check token permissions:
```bash
huggingface-cli whoami
# Token should have "Write" permission
```

---

## 📊 Expected Performance

After completing all steps, you should see:

**Sentiment Classification:**
- ✅ Accuracy: > 0.85
- ✅ F1 Score: > 0.83
- ✅ Latency: < 50ms

**Keyword Extraction:**
- ✅ ROUGE-1: > 0.35
- ✅ ROUGE-L: > 0.28
- ✅ Latency: < 200ms

**Improvements over baseline:**
- Sentiment: +15-25% accuracy
- Keywords: +10-20% ROUGE scores

---

## 🔗 Important Links

### Documentation
- **Main README**: [README.md](README.md)
- **Setup Guide**: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Report Template**: [PROJECT_REPORT_TEMPLATE.md](PROJECT_REPORT_TEMPLATE.md)

### External Services
- Ngrok Dashboard: https://dashboard.ngrok.com/
- HuggingFace Dashboard: https://huggingface.co/settings/tokens
- Dataset Licenses:
  - Emotion: https://huggingface.co/datasets/emotion
  - Inspec: https://huggingface.co/datasets/midas/inspec

### API Documentation (After deployment)
- Local: http://localhost:8000/docs
- Ngrok: https://YOUR_URL/docs
- HF Space: https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE

---

## ⏱️ Time Estimates

| Task | Time | Can Skip? |
|------|------|-----------|
| Environment Setup | 5 min | ❌ No |
| Download Datasets | 5 min | ❌ No |
| EDA | 10 min | ✅ Yes (but recommended) |
| Baseline Evaluation | 10 min | ✅ Yes (but recommended) |
| LoRA Training | 1-2 hours | ❌ No |
| Local Testing | 5 min | ✅ Yes |
| Ngrok Setup | 10 min | ✅ Yes (if not demoing) |
| HF Spaces Setup | 15 min | ✅ Yes (if not demoing) |
| Write Report | 2-3 hours | ❌ No |

**Minimum time to complete:** ~3-4 hours (setup + training + report)
**Full completion:** ~5-6 hours (including all deployment options)

---

## 💡 Pro Tips

1. **Start training early** - It takes 1-2 hours, so start it before doing other tasks
2. **Monitor GPU usage** - Use `nvidia-smi` to check if GPU is being used
3. **Save screenshots as you go** - Don't wait until the end
4. **Test locally first** - Make sure everything works before deploying
5. **Use the templates** - Don't write documentation from scratch
6. **Version control** - Commit frequently with good messages

---

## ✅ Ready to Start?

Run this one command to verify everything is set up:

```bash
python -c "import torch; import transformers; import peft; print('✅ All libraries installed!')"
```

If that works, you're ready to go! Start with Step 1 above.

---

**Need help?** Check:
1. [README.md](README.md) for usage instructions
2. [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed external service setup
3. [PROJECT_REPORT_TEMPLATE.md](PROJECT_REPORT_TEMPLATE.md) for report structure

**Good luck with your AI Lab project! 🚀**
