# Batch 2 Annotation Summary

**Date:** 2025-11-23
**Window:** Sept 1 – Nov 23, 2025 snapshot (Batch 2 pulls earliest unlabeled rows)

## Template

- Selected 60 additional records (30 phishing, 30 benign) from `train_data.csv` that were not part of Batch 1.
- Exported as `batch2_annotation_template.csv` with inline features (URL stats, token counts, latency) plus blank review columns.

## Logged Decisions

- Total annotated samples (cumulative): **120**
  - Phishing: **60**
  - Benign: **60**
- New rationales appended to `label_audit_log.csv` with OWASP/MITRE/NIST references and reviewer/timestamp metadata.

## Inter-Rater Check

- Ten Batch 2 records (5 phishing, 5 benign) were double-coded and appended to `interRaterPairs.csv` (total pairs = 20).
- Cohen’s κ = **1.0** for the combined Batch 1+2 sample, indicating consistent reviewer agreement.

## Next Steps

1. Decide whether a third batch is necessary from the remaining Sept–Nov 2025 rows (no 2026 labeling planned).
2. Refresh Chapter 3 §3.2 / Appendix references to cite both Batch summaries.
3. Continue spot-checking any future batches (if executed) with at least 10 double-coded entries to keep κ monitoring current.
