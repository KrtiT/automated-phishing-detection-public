# Week 1 Data Freeze Execution Log

**Date:** 2025-11-16 22:20 UTC  
**Operator:** Krti Tallam (CLI workspace)  
**Artifacts:** `data/configs/week1_snapshot.yml`, `data/processed/v2025.08.10/*`

## Actions Completed

1. **Snapshot Definition** – Authored `data/README.md` describing the 12-week
   window (2025-09-01 → 2025-11-23), source inventory, split policy, and OWASP /
   MITRE / NIST-aligned labeling rules that satisfy Section 3.2 requirements.
2. **Config Materialization** – Added `data/configs/week1_snapshot.yml` capturing
   freeze date, split boundaries, source URIs, reviewer roster, and output file
   names. This config is referenced by all preprocessing steps to guarantee
   reproducibility.
3. **Label Traceability** – Created `data/labels/label_audit_log.csv` header so
   reviewer rationales can be appended during annotation sessions.
4. **Processing Pipeline** – Refactored `code/data_preprocessing.py` to honor
   the YAML config, enforce chronological splits, attach OWASP/MITRE labels, and
   emit datasets plus stats files under the `v2025.08.10` folder.
5. **Dataset Generation (Initial)** – Executed the pipeline via:
   ```bash
   python code/data_preprocessing.py \
     --config data/configs/week1_snapshot.yml \
     --data-dir data
   ```
   Each rerun updates `data/processed/v2025.08.10/dataset_stats.json` with the
   active sample counts and feature list.
6. **Snapshot Refresh (Nov 16 Update)** – Intermediate run using sanitized feeds
   (small batch) produced 164/70/114 splits with 71 phishing exemplars.
7. **Snapshot Expansion (Nov 17 Update)** – Final sanitized exports (320 GPT
   telemetry rows, 200 OpenPhish, 240 PhishTank, 260 URLhaus) established the
   authoritative splits: **1,669 train / 318 validation / 341 test** with 888
   phishing events in the training window. All raw CSV/JSON sources live under
   `data/raw/` and are referenced by `week1_snapshot.yml` for auditability.

## Resulting Files

| File | Purpose |
| --- | --- |
| `data/processed/v2025.08.10/train_data.csv` | Chronological training split covering weeks 1-6 |
| `.../val_data.csv` | Weeks 7-9 validation window for threshold tuning |
| `.../test_data.csv` | Weeks 10-12 held-out evaluation data |
| `.../dataset_stats.json` | Snapshot metadata (counts, feature list, config ref) |

## Outstanding Items

- Continue manual labeling per the published protocol so at least 30 phishing
  and 30 benign decisions are documented with inter-rater statistics.
- Integrate the latest evaluation figures and latency metrics into Chapters 3–4
  of the Praxis manuscript.
- Execute the deployment validation plan (microservice wrapper, burst replay,
  monitoring hooks) and archive the resulting logs.

This log should be cited in Chapter 3 §3.2 and Appendix data provenance notes to
demonstrate that the Week 1 SLO (frozen datasets + documentation) has been met.
