from pathlib import Path
import sys

import pandas as pd
from fastapi.testclient import TestClient


BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BASE_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

import server  # noqa: E402  (loaded after sys.path tweak)


TEST_FILE = BASE_DIR / "data" / "processed" / "v2025.08.10" / "test_data.csv"


def _load_sample_features() -> list[float]:
    df = pd.read_csv(TEST_FILE)
    df = df[server.FEATURES].fillna(0)
    sample = df.iloc[0].astype(float).tolist()
    assert len(sample) == len(server.FEATURES)
    return sample


def test_predict_endpoint_returns_probability_and_decision():
    client = TestClient(server.app)
    payload = {"features": _load_sample_features()}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["decision"] in (0, 1)


def test_predict_endpoint_validates_feature_length():
    client = TestClient(server.app)
    payload = {"features": [0.0] * (len(server.FEATURES) - 1)}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_batch_matches_single_predictions():
    client = TestClient(server.app)
    sample = _load_sample_features()
    payload = {"requests": [{"features": sample}, {"features": sample}]}
    response = client.post("/predict_batch", json=payload)

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    probs = [result["probability"] for result in results]
    for prob in probs:
        assert 0.0 <= prob <= 1.0
    assert probs[0] == probs[1]


def test_predict_accepts_context_metadata():
    client = TestClient(server.app)
    sample = _load_sample_features()
    payload = {
        "features": sample,
        "context": {
            "record_id": "test-record",
            "owasp_category": "LLM01",
            "atlas_tactic": "TA0031",
        },
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "probability" in body and "decision" in body
