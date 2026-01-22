# Deployment Runbook (FastAPI Logistic Service)

## 1. Launch the Service

```bash
cd Automated_Phishing_Detection_Praxis/code
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 2
```

- Logs include `MONITOR {...}` JSON lines that match `reports/DeploymentMonitoringFields.md`.
- Environment variables: `PYTHONPATH=.` so the `server` module resolves correctly.

## 2. Run Burst Replay Over HTTP

```bash
python burst_test.py \
  --mode http \
  --base-url http://127.0.0.1:8000 \
  --concurrency 64 \
  --total 1023 \
  --output ../data/processed/v2025.08.10/eval/burst_latency_uvicorn.json
```

- `--uds /tmp/uvicorn.sock` can be used if the service binds a Unix domain socket instead of TCP.
- The script automatically attaches context (record_id, OWASP category, MITRE tactic) so the server logs governance-facing fields for every request.

## 3. Capture Trace Sample

While the replay runs, tail the uvicorn log and copy one `MONITOR` line plus representative request/response pairs for Appendix B. (In this sandbox the socket bind is blocked; see `reports/SandboxConstraints.md` for the error message.)

## 4. Tear Down

```bash
pkill -f "uvicorn server:app"
```

Re-run the steps whenever a new snapshot/model is deployed to refresh the latency JSON and deployment notes.
