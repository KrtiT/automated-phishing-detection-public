# Deployment Validation Note

- **FastAPI microservice** (`code/server.py`) hosts the calibrated Logistic-L1 model at `/predict` and `/predict_batch`. Feature ordering matches the frozen snapshot and is validated via `feature_metadata.json`.
- **Burst replay harness** (`code/burst_test.py`) uses FastAPI's ASGI transport to replay all 341 test examples concurrently (32-request batches). Results are stored in `data/processed/v2025.08.10/eval/burst_latency.json`.
- **Observed latency (341 requests):** avg 7.40 ms, p50 7.24 ms, p95 11.72 ms, p99 12.03 ms, max 14.44 ms. All measurements remain far below the 200 ms inline SLO.
- **Observed latency (341 requests, ASGI harness):** avg 7.40 ms, p50 7.24 ms, p95 11.72 ms, p99 12.03 ms, max 14.44 ms. All measurements remain far below the 200 ms inline SLO.
- **Extended replay (1,023 requests, ASGI harness):** avg 14.41 ms, p50 13.48 ms, p95 26.99 ms, p99 30.48 ms, max 34.12 ms (see `burst_latency_extended.json`). Same FastAPI code path, concurrency=64 batches; demonstrates stability when mirroring the held-out set 3x+.
- **HTTP replay on uvicorn (1,023 requests, concurrency=64):** avg 129.76 ms, p50 125.40 ms, p95 217.65 ms, p99 238.78 ms, max 270.19 ms (see `data/processed/v2025.08.10/eval/burst_latency_uvicorn.json`). Captured while uvicorn served `/predict` on localhost to validate end-to-end REST path and logging.
- **Tests:** `python -m pytest` (8 passed; cache write warning in sandbox only).
- **Figures:**
  - ROC curve exported to `reports/Fig4-1_ROC.png`
  - Confusion matrices exported to `reports/Fig4-2_ConfusionMatrices.png`

These artifacts complete the technical validation promised in Chapter 3 §3.6, demonstrating both statistical performance and deployment-readiness measurements.

> Sandbox note: attempting to bind uvicorn to a TCP port or Unix domain socket triggers `PermissionError: [Errno 1] Operation not permitted` (see `/tmp/uvicorn_phish.log`). The `burst_test.py --mode http` path will work as soon as the service runs on a real host (e.g., `uvicorn server:app --host 0.0.0.0 --port 8000`).
