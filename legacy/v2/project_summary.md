# Automated Phishing Detection for Frontier AI Inference
## Project Summary

Based on HW#4 presentation by Krti Tallam

### Project Title
Automated Phishing Detection for Frontier AI Inference

### Project Overview
This praxis/thesis project focuses on developing an automated system for detecting phishing attempts targeting frontier AI inference systems.

### Current Accomplishments (Nov 2025)

1. **Snapshot + Labeling** – Twelve-week Sept–Nov 2025 window frozen with 1,669/318/341 chronological splits. Sixty manual annotations (30 phishing / 30 benign) logged with OWASP LLM Top 10, MITRE ATLAS, and NIST AI RMF tags plus κ = 1.0 inter-rater stats.
2. **Baseline Modeling** – Logistic-L1 + class-weighted Random Forest retrained on the snapshot, calibrated (0.03 / 0.06) via validation F1 search, and evaluated with ROC/CM, feature ablations, and McNemar tests. Artifacts live under `data/processed/v2025.08.10/eval/` and `reports/`.
3. **Deployment Validation** – FastAPI microservice (`code/server.py`) plus ASGI burst replay harness (`code/burst_test.py`) show avg 7.4 ms / p95 11.7 ms latency across 341 held-out requests. Notes + figures saved in `reports/DeploymentValidationNotes.md` and `reports/Fig4-*.png`.
4. **Governance Documentation** – LabelingProtocol, Week1_Data_Freeze_Log, Sandbox/Calibration notes, and AnnotationScalingPlan capture policies, assumptions, and plans to scale to ≥150 labeled samples.

### Immediate Next Steps
1. Execute Batches 2–3 of labeling (per AnnotationScalingPlan) or explicitly document deferral in Chapter 3 if scope remains at 60 samples.
2. Capture REST traces + monitoring field definitions for the FastAPI service to close `reports/DeploymentValidationPlan.md`.
3. Port the refreshed Chapter 3–4 text, tables (Table 4‑1/2/3), and figures (Fig 4‑1/2) into `main-thesis/Praxis-MAIN-TALLAM.docx`.
4. Stand up the minimal automated test suite and record results in `reports/Testing_Log.md` (to be created) before final submission.
