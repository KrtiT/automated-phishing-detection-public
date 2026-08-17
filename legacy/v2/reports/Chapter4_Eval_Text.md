### Chapter 4 – Evaluation Narrative Snippet

Calibration on the Sept–Nov validation window lowered the operating thresholds to 0.03 (Logistic-L1) and 0.06 (Random Forest), which eliminated the slow-burn misses observed at the default 0.5 cutoff. On the 341-request test window (291 benign / 50 phishing), both detectors achieved 100% accuracy, precision, recall, and F1, producing identical confusion matrices [[291, 0], [0, 50]]. The ROC curve figure (`roc_curve.png`) shows overlapping AUC=1.0 traces. A McNemar test (b=0, c=0; χ²=0, p=1.0) indicates no statistically significant gap between the classifiers once thresholds are tuned, so deployment decisions hinge on latency rather than accuracy.

Mirrored-traffic replay of the same test set confirmed a p95 inference cost of 0.059 ms for the logistic model versus 14.6 ms for the Random Forest ensemble (`mirrored_latency.json`), leaving ample headroom beneath the 200 ms inline budget. Consequently, the logistic path remains the preferred production configuration, with Random Forest retained as an audit-only challenger.

Ablations (Table 4-2) are tied to the AI snapshot (v2025.12.10) and show that dropping URI punctuation features is the only scenario that materially degrades recall/F1. Burst/latency rows in Table 4-3 likewise correspond to the AI snapshot; the “Public Millions” row reports model inference microbenchmarks only.

### Public Scale Check (Multi-Million Rows)

To align with a standard large-scale baseline, we also evaluated on a public multi-million row dataset (`v2025.public_millions`) constructed from URLhaus (malicious URLs) plus Umbrella Top 1M and Majestic Million domain lists (benign). This yields 2,111,133 total rows with stratified random splits (80/10/10). Using URL-only features and calibrated thresholds from the validation split, both models achieve near-perfect metrics (see `reports/Table4-1_ModelMetrics.csv`). For tractability, training uses a fixed downsample of 200k rows while evaluation runs on the full validation/test splits.

Dataset sources and terms are listed in `reports/Public_Datasets_Citations.md` for straightforward citation (URLhaus, Umbrella Top 1M, Majestic Million).

Public-scale robustness checks: we ran feature ablations on a 200k training downsample (Table 4-2, Public block) and a 5k-request ASGI burst replay on the public test split (Table 4-3, Public row), showing stable accuracy and acceptable latency at scale.
