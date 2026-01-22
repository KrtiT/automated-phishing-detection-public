### Annotation Scaling Plan

- **Batch targets:** annotate incremental batches of ~60 records (balanced 30/30) drawn from the Sept–Nov 2025 snapshot until we reach the desired coverage (current total: 120 labels across two batches).
- **Protocol:** reuse `batch1_annotation_template.csv` as a template generator, log decisions in `label_audit_log.csv`, and run inter-rater sampling every 20 entries (track κ in `interRaterStats.json`). We keep these files human-readable on purpose instead of wiring a tool so advisors can inspect the raw CSV diff.
- **Timeline:** Both Batch 1 (Nov 17) and Batch 2 (Nov 23) are complete and draw exclusively from the Sept–Nov 2025 window. If additional coverage is needed, Batch 3 will reuse remaining unlabeled rows from the same freeze—no 2026 data collection planned.
- **Status:** 120 total labeled samples (60 phishing / 60 benign) with κ = 1.0 across 20 double-coded entries. Document here if a third batch from the existing snapshot is executed or explicitly state that Batch 2 is the final set.
