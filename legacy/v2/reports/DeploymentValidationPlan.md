## Deployment Validation TODOs

1. **FastAPI microservice prototype** – ✅ `code/server.py` + `feature_metadata.json` wrap the calibrated logistic classifier. Latency recorded via `mirrored_latency.json` (single-shot) and `burst_latency.json` (ASGI replay).
2. **Burst replay** – ✅ `code/burst_test.py` replays the 341 held-out requests in 32-request batches (ASGI transport due to sandbox). Avg 7.40 ms, p95 11.72 ms, p99 12.03 ms, max 14.44 ms logged in `data/processed/v2025.08.10/eval/burst_latency.json`. TODO: rerun against deployed uvicorn instance for 1K+ burst when ports are accessible.
3. **Integration artifacts** – ✅ `reports/deployment_trace_sample.json` captures request, feature order, response, and latency for Appendix B. Extend with production trace once the service runs behind a gateway.
4. **Monitoring hooks** – ✅ `reports/DeploymentMonitoringFields.md` enumerates decision/latency/governance fields + alert thresholds for Prometheus/ELK ingestion. `code/server.py` now emits JSON log lines (prefixed `MONITOR`) with those fields, and `reports/DeploymentRunbook.md` documents how to launch uvicorn + replay traffic outside the sandbox.
