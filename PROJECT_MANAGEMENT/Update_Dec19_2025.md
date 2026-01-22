# Dec 19, 2025 Advisor Update (Draft)

## Response to previous feedback
- Explicitly document label provenance (using source dataset labels; no relabeling) in Chapter 3/data sheets.
- Restate sampling window (Sept 1–Nov 23, 2025) plus filenames/ID structure for reproducibility.
- Add correlation/heat map with feature-selection rationale tied to baseline features.
- Expand dataset beyond ~1.6k rows before the next training/eval cycle.

## Progress since last meeting
- Logged Meeting #10 outcomes and action items in `ROLLING_UPDATES_PROFESSOR_MHEISH.md` and added the required 4-bullet template.
- Expanded dataset with Honeypot Batch 2 (220 new phishing rows) via `week1_snapshot_v2.yml`; new splits: 1,901 / 442 / 473 with 998 phishing exemplars in training.
- Regenerated processed data, stats, and correlation artifacts for `v2025.12.10` (`reports/Fig3-FeatureCorrelation.png`, JSON in `data/processed/v2025.12.10/eval/`).
- Retrained/evaluated baselines on the expanded snapshot: Logistic-L1 @0.03 and Random Forest @0.15 both score 100% accuracy/recall on the 473-request test set.
- Prepared the task list for any remaining documentation updates and latency replay refresh.

## Issues encountered and setbacks
- Need to refresh burst/latency replay to reflect the expanded snapshot; current latency figures reference the earlier freeze.
- Tight timing to propagate new stats/figures into Chapter 3–4 text before the Dec 19 check-in.

## Tasks for the next 2 weeks
- Update methodology/data documentation with label provenance, sampling window, filenames, and ID structure (include honeypot feed info).
- Refresh burst/latency replay on `v2025.12.10` and fold results into Chapter 4 once available.
- Build the concise meeting deck/notes using the four bullet areas above.
