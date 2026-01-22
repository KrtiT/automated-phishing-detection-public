## Sept–Nov 2025 Evaluation Highlights (Dec 10 snapshot)

- Logistic-L1 (threshold 0.03) and Random Forest (0.15) each achieved 100% accuracy/precision/recall/F1 on the 473-request test window (352 benign / 121 phishing) after adding honeypot Batch 2.
- ROC-AUC remains 1.0 for both models; ROC points saved in `data/processed/v2025.12.10/eval/roc_*.csv`.
- McNemar’s test (b=0, c=0) yields χ²=0, p=1.0, indicating no statistically significant difference once thresholds are calibrated.
- Confusion matrices and threshold-search results are captured in `evaluation_summary.json` under `data/processed/v2025.12.10/eval/`.
- Mirrored latency replay pending refresh for the expanded snapshot; prior figures remain valid for baseline latency narratives.
- Feature correlation heat map (`reports/Fig3-FeatureCorrelation.png`) and matrix JSON (`data/processed/v2025.12.10/eval/correlation_matrix.json`) back the feature-selection rationale in Chapter 3.
