# Week 1 Data Snapshot Plan

This note freezes the telemetry, public feeds, and adversarial references that
drive Chapter 3 (§3.2 data collection) and unlock Chapter 4 experiments. It
documents the exact time ranges, retrieval steps, and file names so another
researcher can recreate the same train/validation/test splits.

> **Repository note**: the public GitHub repository does not include
> `data/raw/` or `data/processed/` contents. Use the source inventory below
> to populate `data/raw/`, then regenerate processed artifacts with the
> preprocessing pipeline.

## Snapshot Overview

- **Window**: 12 continuous weeks (2025-09-01 through 2025-11-23 UTC)
- **Train/val/test**: weeks 1-6 (train), 7-9 (validation), 10-12 (test)
- **Freeze date**: 2025-11-23 window (updated Dec 10) — all feature tables and
  models reference raw artifacts pulled on/before this date.
- **Current snapshot volume (Dec 10 update)**: 320 GPT telemetry rows, 200
  OpenPhish entries, 240 PhishTank entries, 260 URLhaus entries, 220 honeypot
  detections → 1,901/442/473 chronological splits with 998 phishing exemplars
  in training.

## Source Inventory

| Source | Artifact | Coverage | Storage Path |
| --- | --- | --- | --- |
| GPT-4 / GPT-4o pilot telemetry | `telemetry/gpt4o_snapshot.csv` | Requests spanning 2025-09-01 → 2025-11-23 from two pilot clusters. Includes prompt/completion token counts, auth context, latency, truncated payload hashes. | `data/raw/telemetry/` |
| OpenPhish daily feed | `openphish_2025-11-20.csv` | Latest CSV before freeze; rerun weekly but keep the 2025-11-20 export for reproducibility. | `data/raw/openphish/` |
| PhishTank verified JSON | `phishtank_verified_2025-11-25.json` | Snapshot token-authenticated download performed 2025-11-25 02:15 UTC. | `data/raw/phishtank/` |
| URLhaus recent CSV | `urlhaus_recent_2025-11-25.csv` | Pulled 2025-11-25 03:00 UTC with abuse.ch TLP:C feed. | `data/raw/urlhaus/` |
| URLhaus full historical CSV (millions) | `urlhaus_full.csv.gz` | Complete URLhaus feed (public, redistributable) for large-scale experiments; download from https://urlhaus.abuse.ch/downloads/csv/ (extract to `csv.txt`). | `data/raw/urlhaus/` |
| Honeypot Batch 2 (internal) | `honeypot_batch2.csv` | Honeypot-triggered phishing captures within the window (Sept–Nov 2025), preserved with feed labels. | `data/raw/honeypot/` |
| JailbreakBench v0.4 prompts | `jailbreakbench_v0.4_prompts.jsonl` | Reference bank for similarity features and qualitative exemplars. | `data/raw/jailbreak/` |
| Advisor-approved adversarial prompts | `advisor_redteam_batch3.jsonl` | Closed-set catalog (25 prompts) reviewed 2025-07-28. | `data/raw/redteam/` |
| Umbrella Top 1M domains (public) | `top-1m.csv` | Benign domain list used to scale public experiments to multi-million rows. | `data/raw/benign/` |
| Majestic Million domains (public) | `majestic_million.csv` | Benign domain list used to scale public experiments to multi-million rows. | `data/raw/benign/` |

> **Access**: public feeds (OpenPhish, PhishTank, URLhaus, JailbreakBench) are
> reproducible with curl/wget scripts in `data/scripts/`; GPT telemetry and the
> advisor catalog live in encrypted storage. The CLI snapshot dated 2025-11-17
> uses sanitized CSV/JSON extracts stored under `data/raw/` so the preprocessing
> pipeline runs without falling back to mock generators.

## Chronological Splits

| Split | Weeks | Calendar Range | Artifacts |
| --- | --- | --- | --- |
| Train | 1-6 | 2025-09-01 → 2025-10-12 | `features/train.parquet`, `labels/train_labels.parquet` |
| Validation | 7-9 | 2025-10-13 → 2025-11-02 | `features/val.parquet`, `labels/val_labels.parquet` |
| Test | 10-12 | 2025-11-03 → 2025-11-23 | `features/test.parquet`, `labels/test_labels.parquet` |

Splits are generated chronologically (no shuffling) to mimic deployment drift.

## Labeling Policy (Phishing vs. Benign)

1. **Prompt Injection / Instruction Hijack** → OWASP LLM Top 10: `LLM01` or
   `LLM04`, MITRE ATLAS tactics `TA0042`, `TA0031`. Classified as `phishing`.
2. **Insecure Output Handling Attempts** (e.g., injecting markup that leads to
   downstream privilege escalation) → OWASP `LLM05`, `LLM06`. `phishing`.
3. **Resource Exhaustion / Denial of Wallet** where token consumption exceeds
   97.5th percentile for the respective workload family **and** burst rate
   breaches baseline +3σ → `phishing` (`LLM09`).
4. **Benign** traffic must satisfy all: adheres to published policy templates,
   token + latency profile within 5th–95th percentile of historical workload,
   no similarity (<0.75 cosine) to public jailbreak corpora, and no anomalous
   auth-context churn within 10 minutes.

All labels reference governance vocabulary from the NIST AI RMF (Mapping, Measure,
Manage, Govern functions). Reviewer decisions are logged in
`data/labels/label_audit_log.csv` with timestamps and rationale text. Public
feed labels (OpenPhish, PhishTank, URLhaus) are preserved as published; GPT
telemetry uses the policy rules above. No relabeling is performed in the frozen
snapshot.

## Derived Feature Tables

`data/processed/` stores versioned feature bundles (latest first):

- `v2025.12.10/train_data.csv`
- `v2025.12.10/val_data.csv`
- `v2025.12.10/test_data.csv`
- `v2025.urlhaus_full/train_data.csv` (generated when URLhaus full feed is provided)
- `v2025.urlhaus_full/val_data.csv`
- `v2025.urlhaus_full/test_data.csv`
- `v2025.public_millions/train_data.csv` (URLhaus + Umbrella + Majestic)
- `v2025.public_millions/val_data.csv`
- `v2025.public_millions/test_data.csv`
- `v2025.08.10/train_data.csv`
- `v2025.08.10/val_data.csv`
- `v2025.08.10/test_data.csv`

Each file contains prompt features (token entropy, imperative ratio), session
bursts, and infrastructure-level latency/cost metrics plus the `owasp_category`
column for traceability.

### Feature Selection + Correlation

- Correlation matrix JSON: `data/processed/v2025.12.10/eval/correlation_matrix.json`
- Heat map figure: `reports/Fig3-FeatureCorrelation.png`
- Regenerate with `python code/generate_feature_correlation.py --config data/configs/week1_snapshot_v2.yml`.
  (Skip or downsample for URLhaus-full if memory-bound.)

## Reproduction Steps

1. Download/copy raw sources into `data/raw/` according to the table above.
   - For multi-million–row experiments, download `urlhaus_full.csv.gz` and place
     it at `data/raw/urlhaus/urlhaus_full.csv.gz`.
2. Run `python code/data_preprocessing.py --config data/configs/week1_snapshot_v2.yml`
   to materialize the latest processed splits (v2025.12.10). The previous
   version (v2025.08.10) is reproducible with `week1_snapshot.yml`.
   - For URLhaus full, run `python code/data_preprocessing.py --config data/configs/week1_snapshot_urlhaus_full.yml --max-rows 2000000`
     after extracting the download to `data/raw/urlhaus/csv.txt`; adjust `--max-rows`
     to fit memory.
   - For a public multi-million scale dataset, run `python code/data_preprocessing.py --config data/configs/public_millions_urlhaus_umbrella_majestic.yml`.
3. Commit the resulting statistics JSON (`data/processed/v2025.12.10/dataset_stats.json`
   or `data/processed/v2025.urlhaus_full/dataset_stats.json`) and attach SHA256
   fingerprints for encrypted telemetry blobs in the private registry.

## Next Actions (Week 1 SLO – Status)

- [x] Populate `data/raw/` with sanitized exports of GPT telemetry + public feeds (hashes listed in `documents/Week1_Data_Freeze_Log.md`).
- [x] Finalize `week1_snapshot.yml` with storage URIs + access tokens (stored in the private password vault; template committed here).
- [x] Capture label audit samples (30 phishing / 30 benign) with OWASP/MITRE/NIST rationale in `data/labels/label_audit_log.csv` and `Batch1AnnotationSummary.md`.
- [x] Hand off frozen feature tables (v2025.12.10) to modeling; referenced by `train_baselines.py` and Chapter 4 experiments.

## Data Availability & Privacy

- Public feeds (OpenPhish, PhishTank, URLhaus, JailbreakBench) can be re-downloaded via the scripts referenced above.
- GPT-4/GPT-4o telemetry and advisor red-team prompts contain sensitive metadata; only sanitized aggregates (token counts, latency, hashed contexts) are stored under `data/raw/telemetry/` in this repo. Full payloads remain in encrypted storage with SHA256 fingerprints recorded in `documents/Week1_Data_Freeze_Log.md`.
- External readers replicating this work should replace the telemetry files with their own sanitized traces and rerun `data_preprocessing.py` to regenerate the processed splits.

This README doubles as Section 3.2 evidentiary support so reviewers can audit
the data snapshot used in Chapter 4 results.
