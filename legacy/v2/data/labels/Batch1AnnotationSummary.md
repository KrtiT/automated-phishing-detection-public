# Batch 1 Annotation Summary

**Date:** 2025-11-17  
**Window:** Sept 1 – Nov 23, 2025 snapshot (1,669 / 318 / 341 splits)

## Template

- Extracted 40 mixed records (20 phishing, 20 benign) from the training split
  into `batch1_annotation_template.csv`. Each record includes the original
  `record_id`, timestamp, source feed, feature metrics, and blank columns for
  reviewer label/OWASP/MITRE entries.
- Additional samples were pulled directly from `train_data.csv` to reach the
  60-label target once the template was exhausted.

## Logged Decisions

- Total annotated samples: **60**
  - Phishing: **30**
  - Benign: **30**
- Audit log lives at `data/labels/label_audit_log.csv` and mirrors the template
  columns so we can trace each example back to the raw feed.

## Inter-Rater Check

- Ten records (5 per class) were double-coded and stored in
  `data/labels/interRaterPairs.csv`; Cohen’s kappa = **1.0** (perfect agreement).
- As batches grow, we will repeat the calculation and append summaries to
  `interRaterStats.json`.

## Next Steps

1. Continue annotating future batches (goal: 150+ total samples) following the
   same protocol.
2. Incorporate label distribution + kappa metrics into Chapter 3 §3.6 text.
3. Automate annotation entry (e.g., lightweight CLI form) if reviewer load
   increases.
