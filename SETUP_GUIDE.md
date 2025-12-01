# Complete Setup Guide - AI Lab ML Service

This guide walks you through setting up all external services needed for this project.

## Table of Contents

1. [Python Environment Setup](#1-python-environment-setup)
2. [Ngrok Setup (for Public Tunnel)](#2-ngrok-setup)
3. [HuggingFace Setup (for Spaces Deployment)](#3-huggingface-setup)
4. [GitHub Secrets (for CI/CD)](#4-github-secrets)
5. [Testing the Setup](#5-testing-the-setup)

---

## 1. Python Environment Setup

### Install Python 3.10+

#### On macOS:
```bash
# Using Homebrew
brew install python@3.10
```

#### On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

#### On Windows:
- Download from https://www.python.org/downloads/
- During installation, check "Add Python to PATH"

### Create Virtual Environment

```bash
# Navigate to project directory
cd Model_Finetuning

# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Verify activation (should show venv path)
which python
```

### Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### GPU Support (Optional but Recommended)

If you have an NVIDIA GPU:

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 2. Ngrok Setup

Ngrok creates a secure public tunnel to your local server. This is needed for the FastAPI + Ngrok deployment.

### Step 1: Create Free Account

1. Go to https://ngrok.com/
2. Click "Sign up" (top right)
3. Sign up with:
   - Email
   - GitHub
   - Google
4. Verify your email address

### Step 2: Download Ngrok

#### On macOS:
```bash
brew install ngrok
```

#### On Linux:
```bash
# Download
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | \
  sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
  echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | \
  sudo tee /etc/apt/sources.list.d/ngrok.list && \
  sudo apt update && sudo apt install ngrok

# Or using snap
sudo snap install ngrok
```

#### On Windows:
```bash
# Using Chocolatey
choco install ngrok

# Or download from: https://ngrok.com/download
# Extract ngrok.exe and add to PATH
```

### Step 3: Get Your Auth Token

1. Log in to https://dashboard.ngrok.com/
2. Go to "Getting Started" → "Your Authtoken"
3. Copy your authtoken (looks like: `2abc...xyz123`)

### Step 4: Configure Ngrok

```bash
# Add your authtoken
ngrok config add-authtoken YOUR_AUTH_TOKEN_HERE

# Verify configuration
ngrok config check

# Test ngrok
ngrok http 8000
# Press Ctrl+C to stop
```

### Step 5: Test with Project

```bash
# Make sure models are trained
python lora_trainer.py --model both

# Start ngrok tunnel with FastAPI
python ngrok_setup.py
```

You should see:
```
Public URL: https://xxxx-xx-xx-xxx-xxx.ngrok-free.app
```

### Ngrok Free Tier Limits

- **Connections**: 40 connections/minute
- **Duration**: No time limit
- **Features**: HTTPS, custom domains (paid), IP whitelisting (paid)

### Troubleshooting Ngrok

**Issue**: "ERR_NGROK_108: The authtoken you specified is properly formed, but invalid"
- **Solution**: Re-copy authtoken from dashboard, ensure no extra spaces

**Issue**: "ERR_NGROK_105: No authtoken was specified"
- **Solution**: Run `ngrok config add-authtoken YOUR_TOKEN`

**Issue**: Port 8000 already in use
- **Solution**: Change port in `ngrok_setup.py` and `app.py`

---

## 3. HuggingFace Setup

HuggingFace Spaces hosts your Streamlit UI for free.

### Step 1: Create Account

1. Go to https://huggingface.co/
2. Click "Sign Up"
3. Create account with email or GitHub
4. Verify your email

### Step 2: Create Access Token

1. Log in to HuggingFace
2. Click your profile picture → "Settings"
3. Go to "Access Tokens" (left sidebar)
4. Click "New token"
5. Configure:
   - **Name**: `ml-service-deploy`
   - **Role**: Write
   - **Scopes**: Select all (or at least `repo.write`)
6. Click "Generate"
7. **IMPORTANT**: Copy token immediately (starts with `hf_...`)

### Step 3: Create a Space

1. Go to https://huggingface.co/new-space
2. Fill in details:
   - **Owner**: Your username
   - **Space name**: `ai-lab-ml-service` (or your choice)
   - **License**: MIT
   - **SDK**: Streamlit
   - **Space hardware**: CPU basic (free)
   - **Visibility**: Public
3. Click "Create Space"

### Step 4: Install HuggingFace CLI

```bash
# Install CLI
pip install huggingface_hub

# Login with your token
huggingface-cli login
# When prompted, paste your token (hf_...)

# Verify login
huggingface-cli whoami
```

### Step 5: Deploy to Space

```bash
# Add HF Space as remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Push code (first time)
git add .
git commit -m "Initial deployment"
git push hf main

# For subsequent updates
git push hf main
```

### Step 6: Upload Trained Models

Since trained models are large, use `huggingface-cli`:

```bash
# Upload sentiment model
huggingface-cli upload YOUR_USERNAME/YOUR_SPACE_NAME \
  models/sentiment_model_best \
  --repo-type=space

# Upload keyword model
huggingface-cli upload YOUR_USERNAME/YOUR_SPACE_NAME \
  models/keyword_model_best \
  --repo-type=space
```

### Step 7: Verify Deployment

1. Go to `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
2. Wait for "Building" → "Running" (may take 2-5 minutes)
3. Click "App" tab to view your Streamlit UI
4. Test sentiment analysis and keyword extraction

### HuggingFace Free Tier

- **CPU**: 2 vCPU, 16GB RAM (free)
- **Storage**: 50GB (free)
- **Upgrades**: GPU ($0.60/hour), Persistent storage (varies)

### Troubleshooting HuggingFace

**Issue**: "Build failed"
- **Solution**: Check `requirements.txt` has correct versions
- Check logs in Space settings → "Logs"

**Issue**: Models not loading
- **Solution**: Verify models are in correct paths: `models/sentiment_model_best/`

**Issue**: "Invalid token"
- **Solution**: Generate new token, ensure "Write" permissions

---

## 4. GitHub Secrets (for CI/CD)

To enable automated deployment via GitHub Actions:

### Step 1: Add Secrets to Repository

1. Go to your GitHub repository
2. Click "Settings" → "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Add the following secrets:

**Secret 1: HF_TOKEN**
- **Name**: `HF_TOKEN`
- **Value**: Your HuggingFace token (from Section 3, Step 2)

**Secret 2: HF_SPACE_NAME**
- **Name**: `HF_SPACE_NAME`
- **Value**: `YOUR_USERNAME/YOUR_SPACE_NAME`
  - Example: `johndoe/ai-lab-ml-service`

### Step 2: Test CI/CD Pipeline

```bash
# Create a version tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# Check Actions tab in GitHub
# Workflow should trigger automatically
```

### Step 3: Monitor Workflow

1. Go to repository → "Actions" tab
2. Click on the running workflow
3. View logs for each step
4. Verify deployment succeeds

---

## 5. Testing the Setup

### Test 1: Local Training

```bash
# Download datasets
python download.py

# Run baseline
python baseline.py

# Train models
python lora_trainer.py --model both
```

**Expected Output:**
- Datasets in `data/`
- Trained models in `models/`
- Results in `results/`

### Test 2: Local API Server

```bash
# Start FastAPI server
python app.py

# In another terminal, test endpoints
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

### Test 3: Ngrok Tunnel

```bash
# Start tunnel
python ngrok_setup.py

# Copy public URL from output
# Test from any device:
curl https://YOUR_NGROK_URL/health
```

### Test 4: MCP Server

```bash
# Start MCP server
python mcp_server.py

# Try commands
MCP> predict This is a great day!
MCP> metrics
MCP> exit
```

### Test 5: Streamlit UI (Local)

```bash
# Run Streamlit locally
streamlit run space.py

# Open browser to: http://localhost:8501
# Test sentiment analysis and keyword extraction
```

### Test 6: HuggingFace Space

1. Visit: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
2. Wait for app to load
3. Test both tabs (Sentiment, Keywords)
4. Check metrics tab

---

## Quick Reference Card

### Essential Commands

```bash
# Environment
source venv/bin/activate           # Activate venv (macOS/Linux)
venv\Scripts\activate              # Activate venv (Windows)

# Training
python download.py                 # Download datasets
python baseline.py                 # Run baseline
python lora_trainer.py             # Train models

# Deployment
python app.py                      # Start FastAPI
python ngrok_setup.py              # Start Ngrok tunnel
python mcp_server.py               # Start MCP server
streamlit run space.py             # Start Streamlit

# HuggingFace
huggingface-cli login              # Login to HF
git push hf main                   # Deploy to Space
```

### Important URLs

- Ngrok Dashboard: https://dashboard.ngrok.com/
- HuggingFace Dashboard: https://huggingface.co/settings/tokens
- Your HF Space: https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
- API Docs (local): http://localhost:8000/docs
- Streamlit (local): http://localhost:8501

---

## Getting Help

### Common Issues & Solutions

1. **"ModuleNotFoundError"**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`

2. **"CUDA out of memory"**
   - Reduce batch size in `lora_trainer.py`
   - Or train on CPU (slower): remove CUDA check

3. **"Models not found"**
   - Run training first: `python lora_trainer.py`
   - Check `models/` directory exists

4. **"Ngrok tunnel failed"**
   - Check authtoken: `ngrok config check`
   - Verify port 8000 is free: `lsof -i :8000`

5. **"HuggingFace push failed"**
   - Check token: `huggingface-cli whoami`
   - Verify token has "Write" permission

### Support Resources

- **Project Issues**: GitHub Issues page
- **Ngrok Help**: https://ngrok.com/docs
- **HuggingFace Help**: https://huggingface.co/docs
- **PyTorch Forum**: https://discuss.pytorch.org/

---

**Setup complete! You're ready to build and deploy your ML service.**
