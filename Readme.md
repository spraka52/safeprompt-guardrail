# SafePrompt – LLM Safety Guardrail

SafePrompt is a full-stack ML project that detects unsafe or harmful prompts before they are sent to an LLM.

It uses a fine-tuned DistilBERT model trained on a multi-label toxicity dataset and exposes predictions via a FastAPI service, with a Next.js frontend demo.

## Features
- Multi-label prompt safety classification
- Categories: toxicity, threats, insults, identity hate, obscene content
- REST API for inference
- Web UI for live demo

## Tech Stack
**ML / Backend**
- Python, PyTorch
- Hugging Face Transformers
- FastAPI

**Frontend**
- Next.js
- TypeScript
- Tailwind CSS

## Training
- Base model: `distilbert-base-uncased`
- Dataset: Jigsaw Toxic Comment Classification
- Loss: Binary Cross Entropy (multi-label)
- Trained on Mac MPS backend

## Running Locally

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
