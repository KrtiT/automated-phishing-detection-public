# AI Phishing Detection – Code Implementation

**Author:** Krti Tallam  
**Date:** November 23, 2025  
**Project:** Automated Phishing Detection for Frontier AI Inference  

## Overview

This directory houses the reproducible pipeline used in Chapter 3–4: data preprocessing for the Sept–Nov 2025 snapshot, baseline training/evaluation, FastAPI deployment, and latency replay harnesses. All paths are relative to the repository root unless otherwise noted.

## Key Scripts

| File | Description |
| --- | --- |
| `data_preprocessing.py` | Materializes train/val/test CSVs under `data/processed/<snapshot>/` from the raw feeds described in `data/README.md`. |
| `train_baselines.py` | Applies SMOTE to the training split, trains Logistic-L1 + class-weighted Random Forest, and writes model artifacts to `code/models/` (generated locally, not committed). |
| `evaluate_baselines.py` | Computes calibrated threshold metrics, ROC points, and confusion matrices (outputs `evaluation_summary.json`, `roc_curve.png`, etc.). |
| `run_feature_ablations.py` | Drops feature families to quantify robustness (produces `feature_ablations.json`). |
| `generate_feature_correlation.py` | Builds Pearson correlation matrix/heat map from processed splits; outputs JSON to `data/processed/<snapshot>/eval/` and `reports/Fig3-FeatureCorrelation.png`. |
| `server.py` | FastAPI microservice exposing `/predict` + `/predict_batch`, loading `code/models/logistic_l1.pkl` + optional scaler (generated after training). |
| `burst_test.py` | Replay harness with ASGI and HTTP modes. Use `--mode asgi` (default) inside this sandbox or `--mode http --base-url http://<host>:<port>` when a uvicorn service is bound. Supports Unix domain sockets via `--uds` and configurable totals/output paths. |
| `mirrored_latency.py` | Legacy single-thread timing helper retained for reference (superseded by `burst_test.py`). |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Regenerate processed splits
#    Use week1_snapshot_v2.yml to include honeypot Batch 2 (v2025.12.10)
python data_preprocessing.py --config ../data/configs/week1_snapshot_v2.yml
#    Or stream the full URLhaus feed (add --max-rows to downsample)
# python data_preprocessing.py --config ../data/configs/week1_snapshot_urlhaus_full.yml --max-rows 2000000
#    Public multi-million scale (URLhaus + Umbrella Top 1M + Majestic Million)
# python data_preprocessing.py --config ../data/configs/public_millions_urlhaus_umbrella_majestic.yml

# 3. Train and evaluate baselines (generates code/models/* locally)
python train_baselines.py --config ../data/configs/week1_snapshot_v2.yml
python evaluate_baselines.py --config ../data/configs/week1_snapshot_v2.yml
# Public multi-million scale baselines (uses downsampling knobs in the config)
# python train_baselines.py --config ../data/configs/public_millions_urlhaus_umbrella_majestic.yml
# python evaluate_baselines.py --config ../data/configs/public_millions_urlhaus_umbrella_majestic.yml

# 4. Launch the FastAPI microservice (non-sandbox)
uvicorn server:app --host 0.0.0.0 --port 8000

# 5. Run burst replay
# ASGI fallback (sandbox)
python burst_test.py --mode asgi --concurrency 32

# HTTP/uvicorn mode (outside sandbox)
uvicorn server:app --host 0.0.0.0 --port 8000
python burst_test.py --mode http --base-url http://127.0.0.1:8000 --concurrency 64 --total 1023
```

## Performance Snapshot (Sept–Nov 2025 Freeze, v2025.12.10)

- **Logistic-L1** – Accuracy/Precision/Recall/F1 = 1.000 on the 473-request test split at calibrated threshold 0.03; latency p95 ≈ 0.060 ms.
- **Random Forest** – Accuracy/Precision/Recall/F1 = 1.000 on the same split at calibrated threshold 0.15; latency p95 ≈ 14.5 ms (CPU-only) so it serves as an audit-only challenger.
- **FastAPI Burst Replay** – Prior run on v2025.08.10: 341 held-out requests at avg 7.40 ms (p95 11.72 ms, p99 12.03 ms). Replay refresh for v2025.12.10 is pending.

## Feature Vector (15 Inline Signals)

1. URL length, domain length, path length
2. Counts of dots, hyphens, underscores, slashes
3. `has_ip`, `is_https`, `has_port`, `num_params`
4. `is_ai_endpoint`, `prompt_tokens`, `completion_tokens`, `latency_ms`

`code/models/feature_metadata.json` (generated after training) and the `FEATURES` list in `server.py` keep this ordering explicit for reviewers.

### Request Schema

`POST /predict` accepts:

```
{
  "features": [...15 floats...],
  "context": {
    "record_id": "optional-id",
    "owasp_category": "LLM01",
    "atlas_tactic": "TA0031"
  }
}
```

`context` is optional but recommended for governance/monitoring; when supplied, the service emits structured logs per `reports/DeploymentMonitoringFields.md`. `burst_test.py` automatically attaches context pulled from the dataset, so replayed traffic exercises the same logging path.

## Next Steps

1. Expand automated regression tests (currently `tests/test_server.py` + `tests/test_data_integrity.py`) to cover preprocessing edge cases and burst harness logging.
2. Capture production-style REST traces + monitoring field definitions to finish `reports/DeploymentValidationPlan.md` Step 3–4.
3. Re-run training/evaluation after Batch 2+ labeling to document how the models behave on noisier freezes.

## Testing

```
cd ..  # repo root
pytest
```

The suite exercises the FastAPI endpoints (feature-length validation, batch parity) and verifies that the processed datasets still expose all `server.FEATURES`. Extend this as more modules stabilize to stay aligned with `documents/Testing_Protocol.md`.

## Dependencies

Key packages live in `requirements.txt` (FastAPI, httpx, pandas, numpy, scikit-learn, joblib, yaml). Python 3.10+ recommended.

## Notes

- Sandbox constraint: `burst_test.py` uses FastAPI’s ASGI transport because localhost binding is blocked. In real deployments run `uvicorn server:app --host 0.0.0.0 --port 8000` and point the script at that endpoint.
- `baseline_phishing_detector.py` remains for historical context but is not part of the current evaluation stack.
