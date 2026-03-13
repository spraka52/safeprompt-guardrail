import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predictor import SafePromptPredictor

app = FastAPI(title="SafePrompt API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_DEFAULT_MODEL_DIR = Path(__file__).parent / "models" / "safeprompt_distilbert"
MODEL_DIR = os.getenv("MODEL_DIR", str(_DEFAULT_MODEL_DIR))
predictor: SafePromptPredictor | None = None


@app.on_event("startup")
def load_model():
    global predictor
    predictor = SafePromptPredictor(MODEL_DIR)


class PromptRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    text: str
    predictions: dict[str, float]
    is_safe: bool
    flagged_categories: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PromptRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    predictions = predictor.predict(req.text)
    flagged = [label for label, score in predictions.items() if score >= 0.5]

    return PredictionResponse(
        text=req.text,
        predictions=predictions,
        is_safe=len(flagged) == 0,
        flagged_categories=flagged,
    )
