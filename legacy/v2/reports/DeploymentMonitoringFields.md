# Deployment Monitoring Fields

To close `DeploymentValidationPlan.md` items 3–4, the FastAPI service will emit the following structured fields to Prometheus/ELK (or any metrics/logging stack). Each prediction log line contains both governance context and performance signals.

| Field | Type | Description |
| --- | --- | --- |
| `request_id` | string | UUID generated per request for traceability + correlation with upstream gateway logs. |
| `timestamp` | RFC3339 string | UTC time the request was processed. |
| `decision` | int | 0 = benign, 1 = phishing (threshold 0.03 for logistic path). |
| `probability` | float | Model probability of phishing, stored for calibration drift analysis. |
| `latency_ms` | float | End-to-end inference latency as recorded in FastAPI middleware. Alert if >200 ms p95 in 5‑minute windows. |
| `owasp_category` | string | OWASP LLM Top 10 label surfaced for governance dashboards (mirrors dataset column). |
| `atlas_tactic` | string | MITRE ATLAS tactic code to align with Section 3.2 taxonomy. |
| `prompt_tokens` | int | Observed token count (sanitized) to monitor for denial-of-wallet bursts. |
| `completion_tokens` | int | Completion length for paired billing analysis. |
| `model_version` | string | Hash or semantic version of the logistic model bundle (`logistic_l1.pkl`). |
| `feature_version` | string | `v2025.08.10` or future snapshot identifier. |
| `trace_sample` | json | (Optional) Sample of normalized feature vector for Appendix logging; see `deployment_trace_sample.json`. |

Alerts:
- **Latency**: Warn at 150 ms p95, Critical at 200 ms p95 sustained over 3 consecutive 5‑minute windows.
- **Volume**: Alert if phishing decision rate deviates ±3σ from trailing 7‑day average (potential distribution shift).
- **Data Drift**: Alert when median probability drops below 0.4 for >1 hour, prompting recalibration.

These fields tie directly back to the labeling/feature metadata captured in `data/README.md` and support Chapter 3 §3.6’s “policies and evidence” requirement.
