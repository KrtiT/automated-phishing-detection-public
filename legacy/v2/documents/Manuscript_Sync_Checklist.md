# Manuscript Sync Checklist (Praxis-MAIN-TALLAM.docx)

Use this list when porting the Markdown results into the official thesis Word document.

## Chapter 3 Updates

1. **Section 3.2 – Data Collection**
   - Paste the paragraph from `documents/Praxis_Document_Draft.md` that cites data sources, time ranges, and labeling policy.
   - Reference Table 3-2 (source inventory) and point to `data/README.md` for reproducibility.
2. **Section 3.3 – Feature Engineering / Model Development**
   - Insert the 15-feature description + SMOTE/calibration details.
3. **Section 3.6 – Deployment Validation**
   - Summarize FastAPI microservice + burst replay; cite `reports/DeploymentValidationPlan.md`, `DeploymentValidationNotes.md`, `DeploymentMonitoringFields.md`, and `deployment_trace_sample.json`.

## Chapter 4 Updates

1. **Section 4.1–4.5 Narrative**
   - Copy the fully fleshed-out text from `documents/Praxis_Document_Draft.md` (Nov 2025 version).
2. **Figures**
   - Insert `reports/Fig4-1_ROC.png` and `reports/Fig4-2_ConfusionMatrices.png` with captions referencing Sept–Nov 2025 snapshot.
3. **Tables**
   - Table 4-1: import `reports/Table4-1_ModelMetrics.csv` (includes pre/post calibration rows).
   - Table 4-2: import `reports/Table4-2_FeatureAblations.csv`.
   - Table 4-3: import `reports/Table4-3_BurstLatency.csv` (generated via `burst_test.py`).
4. **Latency Paragraph**
   - Reference `burst_latency.json` stats (avg 7.40 ms, p95 11.72 ms, p99 12.03 ms, max 14.44 ms) and ASGI sandbox note.

## Appendices

- Attach `reports/DeploymentMonitoringFields.md` and `reports/deployment_trace_sample.json` excerpts in Appendix B (governance + deployment evidence).
- Include `data/labels/Batch1AnnotationSummary.md` + `AnnotationScalingPlan.md` in Appendix C to show governance scaling.

## GWU Checklist Alignment

- Update `documents/GWU_SEAS_Requirements_Checklist.md` after syncing Word doc (Month 6 documentation checkbox, final submission milestones, etc.).
- Record manuscript sync date + reviewer sign-off in `ROLLING_UPDATES_PROFESSOR_MHEISH.md` once the advisor reviews the Word version.
