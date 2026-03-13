# SafePrompt – LLM Prompt Safety Guardrail

Full-stack ML system that classifies unsafe prompts before they reach an LLM — using a fine-tuned DistilBERT model, a FastAPI inference server, and a Next.js demo UI.

📊 **GitHub:** [https://github.com/spraka52/safeprompt-guardrail](https://github.com/spraka52/safeprompt-guardrail)

---

## ⚡ Model Performance

- **Base Model:** `distilbert-base-uncased`
- **Dataset:** Jigsaw Toxic Comment Classification (~160K samples)
- **Training:** 3 epochs, lr=2e-5, batch size 16
- **Loss:** Binary Cross-Entropy (multi-label)
- **Hardware:** Mac MPS backend (Apple Silicon)
- **API Inference Latency:** <100ms per request

---

## 🏗️ Architecture

```
[Browser / Next.js UI]
        ↕ HTTP (fetch)
[FastAPI Inference Server :8000]
        ↕
[SafePromptPredictor]
        ↕
[Fine-tuned DistilBERT — models/safeprompt_distilbert]
```

---

## 🚀 Features

- ✅ **Multi-label Classification** — detects 6 harm categories simultaneously
- ✅ **Safety Categories** — toxic, severe toxic, obscene, threat, insult, identity hate
- ✅ **Real-time Inference** — FastAPI `/predict` endpoint with per-label confidence scores
- ✅ **Safe/Unsafe Verdict** — threshold at 0.5 across all labels
- ✅ **Hardware Auto-detection** — runs on MPS (Apple Silicon), CUDA, or CPU
- ✅ **Live Demo UI** — Next.js frontend with per-category score bars
- ✅ **CORS Enabled** — frontend and backend can run independently

---

## 🛠️ Tech Stack

### ML / Backend
- **Python 3.13**
- **PyTorch** — model inference with MPS/CUDA/CPU auto-detection
- **Hugging Face Transformers** — DistilBERT fine-tuning and tokenization
- **Hugging Face Datasets** — Jigsaw dataset loading and preprocessing
- **FastAPI** — REST inference server
- **Uvicorn** — ASGI server
- **Pydantic v2** — request/response validation
- **scikit-learn** — F1 metrics during training

### Frontend
- **Next.js 15** — React framework with App Router
- **React 19**
- **TypeScript 5**
- **Tailwind CSS 3** — utility-first styling

---

## 📊 Safety Categories

| Category | Description |
|---|---|
| `toxic` | General toxic content |
| `severe_toxic` | Severely toxic / extreme hostility |
| `obscene` | Obscene or vulgar language |
| `threat` | Threats of violence or harm |
| `insult` | Personal insults |
| `identity_hate` | Hate speech targeting identity groups |

---

## 🔧 Local Development

### Prerequisites
- Python 3.10+
- Node.js 18+
- Trained model weights in `backend/models/safeprompt_distilbert/`

### 1. Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the inference server
uvicorn main:app --reload --port 8000
```

Backend runs on `http://localhost:8000`

**Environment variables (optional):**
```bash
MODEL_DIR=/absolute/path/to/models/safeprompt_distilbert  # defaults to backend/models/safeprompt_distilbert
```

### 2. Frontend Setup

```bash
cd frontend
npm install

# Create .env.local
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

npm run dev
```

Frontend runs on `http://localhost:3000`

---

## 🧪 API Reference

### Health Check

```http
GET /health
```

```json
{ "status": "ok", "model_loaded": true }
```

### Predict

```http
POST /predict
Content-Type: application/json

{
  "text": "Your prompt here"
}
```

**Response:**
```json
{
  "text": "Your prompt here",
  "predictions": {
    "toxic": 0.03,
    "severe_toxic": 0.01,
    "obscene": 0.02,
    "threat": 0.01,
    "insult": 0.02,
    "identity_hate": 0.01
  },
  "is_safe": true,
  "flagged_categories": []
}
```

**Test with curl:**
```bash
# Health check
curl http://localhost:8000/health

# Run prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}'
```

### Interactive Docs

```
http://localhost:8000/docs
```

---

## 🏋️ Training

```bash
cd backend
python -m training.train
```

**Training config (defaults):**
```python
model_name      = "distilbert-base-uncased"
max_length      = 128
train_batch_size = 16
eval_batch_size  = 32
lr              = 2e-5
num_epochs      = 3
output_dir      = "models/safeprompt_distilbert"
data_path       = "data/train.csv"
```

**Dataset:** Download the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) dataset and place `train.csv` in `backend/data/`.

---

## 🗂️ Project Structure

```
safeprompt-guardrail/
├── backend/
│   ├── main.py              # FastAPI app — /health, /predict
│   ├── predictor.py         # SafePromptPredictor (tokenize + inference)
│   ├── requirements.txt
│   └── training/
│       ├── train.py         # Fine-tuning script
│       └── dataset.py       # Jigsaw dataset loader
├── frontend/
│   └── src/app/
│       ├── page.tsx         # Demo UI with score bars
│       ├── layout.tsx
│       └── globals.css
└── Readme.md
```

---

## 🗺️ Roadmap

- [ ] **Confidence calibration** — Platt scaling or temperature scaling
- [ ] **Streaming inference** — Server-Sent Events for real-time feedback
- [ ] **Batch endpoint** — classify multiple prompts in one request
- [ ] **Model versioning** — A/B test checkpoints via env var
- [ ] **Docker Compose** — one-command local setup
- [ ] **Benchmark suite** — latency and F1 regression tests
- [ ] **Deployment** — Vercel (frontend) + Railway/Fly.io (backend)

---

## 📄 License

MIT License

---

## 👤 Author

**Shreya Prakash**
- LinkedIn: [linkedin.com/in/shreya-prakash2199](https://linkedin.com/in/shreya-prakash2199)
- GitHub: [github.com/spraka52](https://github.com/spraka52)
- Email: spraka52@asu.edu
