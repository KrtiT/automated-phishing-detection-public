# Week 3 Evaluation Package

**Date:** 2025-11-17  
**Snapshot:** `week1_2025_08_10` (same 12-week freeze)  
**Artifacts:** `code/evaluate_baselines.py`, `data/processed/v2025.08.10/eval/*`

## Threshold Calibration

- Performed grid search (0.01–0.99, step 0.01) on the validation window for
  each baseline to maximize F1 before locking the operating point.
- Optimal thresholds now land at **0.03** for Logistic-L1 and **0.06** for Random
  Forest based on the Sept–Nov validation window. These settings eliminate the
  slow-burn misses seen at the default 0.5 cutoff.

## Validation Metrics @ Calibrated Threshold (Expanded Snapshot)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion Matrix |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic-L1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | [[291, 0], [0, 27]] |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | [[291, 0], [0, 27]] |

## Test Metrics @ Calibrated Threshold (Expanded Snapshot)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion Matrix |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic-L1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | [[291, 0], [0, 50]] |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | [[291, 0], [0, 50]] |

Calibrating thresholds removes the six slow-burn misses observed at the default
0.5 cutoff and keeps both models in lockstep on the 341-record test window.

## ROC Curves

- Exported CSVs (`roc_logistic.csv`, `roc_random_forest.csv`) contain FPR/TPR
  traces over the test window so Chapter 4 can produce publication-quality plots
  without rerunning model inference.

## Feature-Family Ablations

- Scenarios evaluated: baseline (all features) plus four removals (length,
  punctuation counts, security flags, AI-endpoint indicator). Each scenario
  retrains both models with SMOTE applied to the training split only.
- Logistic-L1 is highly sensitive to URI punctuation signals: dropping that
  family reduces test recall from 95.7% to **58%** (F1 = 0.667), while removing
  only length cues still leaves recall at 88%.
- Random Forest stays at 100% accuracy across all ablations thanks to the deep
  ensemble capacity, underscoring why the logistic path is preferable for
  interpretability and low latency but must retain the high-informational URL
  tokens.
- JSON artifact: `data/processed/v2025.08.10/eval/feature_ablations.json` lists
  per-scenario validation/test metrics for direct citation in Chapter 4.

## McNemar Test (Test Window)

- Contingency (logistic correct / RF incorrect vs. RF correct / logistic
  incorrect): **b = 0**, **c = 0** after applying the calibrated thresholds.
- Test statistic `χ² = 0`, p-value `1.0`, indicating no measurable difference
  between the models on the Sept–Nov snapshot.

## Files Produced

| File | Description |
| --- | --- |
| `data/processed/v2025.08.10/eval/evaluation_summary.json` | Aggregated metrics, thresholds, and McNemar output |
| `data/processed/v2025.08.10/eval/roc_logistic.csv` | ROC points for Logistic-L1 on the test set |
| `data/processed/v2025.08.10/eval/roc_random_forest.csv` | ROC points for Random Forest on the test set |
| `data/processed/v2025.08.10/eval/feature_ablations.json` | Feature-family ablation results |
| `data/processed/v2025.08.10/eval/mirrored_latency.json` | Request-level mirrored latency metrics |

## Mirrored Latency Measurements

- Replayed all **341** test requests through each model. Logistic-L1 stays below
  **0.059 ms p95** (avg 0.054 ms), while Random Forest clocks **14.63 ms p95** and
  exhibits a worst-case tail of 32.4 ms. Both remain inside the <200 ms SLO, but
  the logistic path leaves more headroom for inline deployment.

## Label Audit Log

- Added three exemplar decisions to `data/labels/label_audit_log.csv` covering
  a denial-of-wallet burst, an OpenPhish credential lure, and a benign GPT-4
  completion. Each entry maps to OWASP/MITRE categories and records reviewer
  rationale/timestamps to satisfy Section 3.2/3.6 traceability requirements.

## Next Steps

- Once real telemetry replaces the mock feeds, rerun both `train_baselines.py`
  and `evaluate_baselines.py` to capture realistic confusion matrices and ROC
  curves.
- Extend the evaluation notebook with feature-family ablations and mirrored
  latency measurements (Week 3 milestone).
