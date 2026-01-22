#!/usr/bin/env python3
"""FastAPI service wrapping the logistic baseline for deployment validation."""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist


MODEL_PATH = Path(__file__).parent / "models" / "logistic_l1.pkl"
# The feature order mirrors data_preprocessing.py; keeping a single list here
# avoids “tool magic” and shows reviewers exactly what the service expects.
FEATURES = [
    "url_length",
    "domain_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "has_ip",
    "is_https",
    "has_port",
    "num_params",
    "is_ai_endpoint",
    "prompt_tokens",
    "completion_tokens",
    "latency_ms",
]


class RequestContext(BaseModel):
    record_id: Optional[str]
    owasp_category: Optional[str]
    atlas_tactic: Optional[str]


class RequestPayload(BaseModel):
    features: conlist(float, min_length=len(FEATURES), max_length=len(FEATURES))
    context: Optional[RequestContext] = None


class BatchRequest(BaseModel):
    requests: List[RequestPayload]


app = FastAPI(title="Inference Phishing Detector")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phishing-detector")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle.get("scaler")
MODEL_VERSION = bundle.get("model_version", MODEL_PATH.name)
THRESHOLD = 0.03


PROMPT_INDEX = FEATURES.index("prompt_tokens")
COMPLETION_INDEX = FEATURES.index("completion_tokens")


def predict(features: np.ndarray) -> float:
    sample = features.reshape(1, -1)
    if scaler is not None:
        sample = scaler.transform(sample)
    prob = model.predict_proba(sample)[0, 1]
    return float(prob)


def emit_monitor_event(features: np.ndarray, context: Optional[RequestContext], prob: float, decision: int, latency_ms: float):
    prompt_tokens = float(features[PROMPT_INDEX])
    completion_tokens = float(features[COMPLETION_INDEX])
    record = {
        "request_id": uuid.uuid4().hex,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": decision,
        "probability": prob,
        "latency_ms": latency_ms,
        "owasp_category": context.owasp_category if context else "N/A",
        "atlas_tactic": context.atlas_tactic if context else "N/A",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "model_version": MODEL_VERSION,
        "feature_version": "v2025.08.10",
        "record_id": context.record_id if context else None,
    }
    logger.info("MONITOR %s", json.dumps(record))


@app.post("/predict")
def predict_single(payload: RequestPayload):
    features = np.array(payload.features, dtype=float)
    if len(features) != len(FEATURES):
        raise HTTPException(status_code=400, detail="Feature length mismatch")
    start = time.perf_counter()
    prob = predict(features)
    latency_ms = (time.perf_counter() - start) * 1000
    decision = int(prob >= THRESHOLD)
    emit_monitor_event(features, payload.context, prob, decision, latency_ms)
    return {"probability": prob, "decision": decision}


@app.post("/predict_batch")
def predict_batch(payload: BatchRequest):
    results = []
    for req in payload.requests:
        features = np.array(req.features, dtype=float)
        start = time.perf_counter()
        prob = predict(features)
        latency_ms = (time.perf_counter() - start) * 1000
        decision = int(prob >= THRESHOLD)
        emit_monitor_event(features, req.context, prob, decision, latency_ms)
        results.append({"probability": prob, "decision": decision})
    return {"results": results}
