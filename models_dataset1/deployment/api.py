from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


APP_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = APP_DIR.parent / "csv_models" / "artifacts" / "best_model.joblib"
ARTIFACT_PATH = Path(os.getenv("MODEL_ARTIFACT_PATH", str(DEFAULT_ARTIFACT)))
API_KEY = os.getenv("API_KEY", "").strip()


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(default_factory=dict)
    top_k: int = Field(default=3, ge=1, le=10)


class PredictResponse(BaseModel):
    predicted_label: str
    predicted_probability: float
    top_k: list[dict[str, float]]
    model_name: str
    feature_count: int


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    model_name: str
    feature_cols: list[str]
    classes: list[str]


def _load_bundle(path: Path) -> ModelBundle:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError("Model artifact must be a dictionary payload")

    required = {"model", "model_name", "feature_cols", "classes"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Model artifact is missing keys: {sorted(missing)}")

    return ModelBundle(
        model=payload["model"],
        model_name=str(payload["model_name"]),
        feature_cols=[str(col) for col in payload["feature_cols"]],
        classes=[str(cls) for cls in payload["classes"]],
    )


def _normalize_probs(probs: np.ndarray, size: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=float).reshape(-1)
    if probs.size != size:
        return np.full(size, 1.0 / float(size), dtype=float)
    probs = np.clip(probs, 0.0, np.inf)
    total = float(probs.sum())
    if total <= 0:
        return np.full(size, 1.0 / float(size), dtype=float)
    return probs / total


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_vals = np.exp(shifted)
    denom = float(exp_vals.sum())
    if denom <= 0:
        return np.full_like(exp_vals, 1.0 / len(exp_vals), dtype=float)
    return exp_vals / denom


def _predict(bundle: ModelBundle, features: dict[str, float], top_k: int) -> PredictResponse:
    vector = np.array([float(features.get(name, 0.0)) for name in bundle.feature_cols], dtype=float).reshape(1, -1)

    if hasattr(bundle.model, "predict_proba"):
        probs = np.asarray(bundle.model.predict_proba(vector)[0], dtype=float)
    elif hasattr(bundle.model, "decision_function"):
        decision = np.asarray(bundle.model.decision_function(vector)).reshape(-1)
        if decision.size == 1 and len(bundle.classes) == 2:
            positive = 1.0 / (1.0 + np.exp(-decision[0]))
            probs = np.array([1.0 - positive, positive], dtype=float)
        else:
            probs = _softmax(decision)
    else:
        probs = np.zeros(len(bundle.classes), dtype=float)
        probs[int(bundle.model.predict(vector)[0])] = 1.0

    probs = _normalize_probs(probs, len(bundle.classes))
    pred_idx = int(np.argmax(probs))
    ranked_idx = np.argsort(probs)[::-1][: min(top_k, len(bundle.classes))]

    return PredictResponse(
        predicted_label=bundle.classes[pred_idx],
        predicted_probability=float(probs[pred_idx]),
        top_k=[{bundle.classes[int(i)]: float(probs[int(i)])} for i in ranked_idx],
        model_name=bundle.model_name,
        feature_count=len(bundle.feature_cols),
    )


bundle = _load_bundle(ARTIFACT_PATH)
MODEL_NAME = bundle.model_name
FEATURE_COLS = bundle.feature_cols
CLASSES = bundle.classes

app = FastAPI(title="Uganda Sign Language Disease API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "artifact_path": str(ARTIFACT_PATH),
        "feature_count": len(FEATURE_COLS),
    }


@app.get("/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    return {
        "status": "ready",
        "model_name": MODEL_NAME,
        "feature_count": len(FEATURE_COLS),
        "class_count": len(CLASSES),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, _: None = Depends(_require_api_key)) -> PredictResponse:
    return _predict(bundle, request.features, request.top_k)