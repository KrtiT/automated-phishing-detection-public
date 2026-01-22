# Week 2 Baseline Training Results

**Date:** 2025-11-17  
**Snapshot:** `week1_2025_08_10` frozen feature tables (`data/processed/v2025.08.10/`)  
**Models:** Logistic Regression (L1, saga) + class-weighted Random Forest with
training-only SMOTE augmentation.

## Data + Feature Notes

- Latest train/val/test counts: 1,669 / 318 / 341 requests (chronological
  splits across 2025-09-01 → 2025-11-23).
- Feature vector: 12 inline-ready signals (URL structure, lexical counts, boolean
  AI-endpoint flags). Metadata columns (`timestamp`, `OWASP`, `source`) removed
  prior to modeling.
- SMOTE implementation: deterministic NumPy interpolation of minority samples to
  balance the current 888/781 class ratio (train only). Validation/test remain
  untouched.

## Training Configuration

| Model | Key Hyperparameters | Latency Budget Check |
| --- | --- | --- |
| Logistic-L1 | `penalty=l1`, `solver=saga`, `class_weight=balanced`, `max_iter=2000` | Avg 0.05 ms, p95 0.055 ms |
| Random Forest | `n_estimators=200`, `class_weight=balanced`, `n_jobs=-1` | Avg 14.07 ms, p95 14.76 ms |

Latency timing measured over 200 single-sample predictions (mirrors inline
screening). Both models stay well under the 200 ms p95 SLO.

## Validation Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic-L1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Confusion matrices (val): both models correctly classified 186 benign and 180
phishing samples with zero errors.

## Test Metrics (Synthetic Snapshot)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic-L1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | 0.938 | 1.000 | 0.882 | 0.938 | 1.000 |

## Updated Test Metrics (Sanitized Feeds – Nov 16)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic-L1 | 0.991 | 1.000 | 0.986 | 0.993 | 0.993 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Latest Test Metrics (Expanded Snapshot – Nov 17)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic-L1 | 0.983 | 1.000 | 0.957 | 0.978 | 1.000 |
| Random Forest | 0.983 | 1.000 | 0.957 | 0.978 | 1.000 |

- Both models now process **341** held-out records with 140 phishing samples,
  showing the logistic model’s first measurable recall drop (six slow-burn
  prompt floods escape). Random Forest mirrors logistic accuracy but still
  incurs ~15 ms p95 latency in mirrored replays.

## Artifacts

- Metrics JSON: `data/processed/v2025.08.10/baseline_metrics.json`
- Trained weights: `code/models/logistic_l1.pkl`, `code/models/random_forest.pkl`
- Training script: `code/train_baselines.py`

## Caveats & Next Steps

- Current snapshot relies on deterministic mock generators due to offline
  restrictions; expect performance to drop once real GPT telemetry is wired in.
- Week 3 tasks: feature-family ablations, threshold calibration on the validation
  window, McNemar significance tests, and ROC/CM exports for the held-out test
  slice.
- Begin logging reviewer rationales in `data/labels/label_audit_log.csv` as soon
  as manual annotation batches are ready.
