# Manuscript, presentation and evidence map

This index connects every manuscript table and figure, and every advisor
slide, to the same retained research record. It is an artifact index, not
an additional experiment or a substitute for the literature bibliography.

**Current text:** [repository-linked manuscript](manuscript/Tallam_Krti_Praxis_Integrated_2026-10-04.pdf).
**Data and frozen code:** [research-record-2026-10-04](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/research-record-2026-10-04).
The earlier manuscript assets in that sealed release predate this editorial
integration. Use this directory's current Word/PDF/Markdown links for reading
and the research release for its unchanged data archives.

## Citation and source conventions

The manuscript uses author–date citations for outside scholarship and formally
cites the versioned research record as Tallam (2026). Original dataset
creators remain credited in the bibliography and [license record](../../research-archive/2026-10-04/LICENSES.md).
This repository citation does not replace those original attributions.

Paths below are relative to this package. The [archive guide](../../research-archive/2026-10-04/README.md#locating-the-evidence)
continues the trace from the aggregate files to licensed inputs, prepared
partitions, fitted artifacts, row-level predictions, request records and freezes.
The [recomputation receipt](../../research-archive/2026-10-04/recomputation.json)
records the saved-observation arithmetic and its authenticated inputs.

## Tables

| Item | Source artifacts | Interpretation |
|---|---|---|
| 2.1 — Closest literature and contribution boundaries | [Tallam_Krti_Praxis_Integrated_2026-10-04.md](manuscript/Tallam_Krti_Praxis_Integrated_2026-10-04.md); [README.md](literature/README.md); [source-claims.json](literature/source-claims.json) | Literature synthesis; author–date citations in each row and the manuscript bibliography remain authoritative. |
| 3.1 — Component-to-experiment map | [CODE_AND_ENVIRONMENT.json](provenance/CODE_AND_ENVIRONMENT.json); [rq1-transformer-cascade-contract-v2.json](provenance/original/measured-source/data/rq1-transformer-cascade-contract-v2.json); [rq2-gmm-development-contract-v1.json](provenance/original/measured-source/data/rq2-gmm-development-contract-v1.json) | Design map, not an empirical result. |
| 4.1 — External composition | [source-contingency.csv](aggregate-data/source-contingency.csv) | Source/tier strata and reference-label roles. |
| 4.2a — Primary confusion counts | [primary-results.json](aggregate-data/primary-results.json); [primary-gates.csv](aggregate-data/primary-gates.csv) | Internal and external strata use distinct denominators. |
| 4.2b — Primary rates and bounds | [primary-results.json](aggregate-data/primary-results.json); [primary-gates.csv](aggregate-data/primary-gates.csv) | Observed FPR and one-sided exact bound are distinct. |
| 4.3 — Six paired recall contrasts | [paired-contrasts.csv](aggregate-data/paired-contrasts.csv) | Domain-clustered intervals; contrasts are percentage points. |
| 4.4 — GMM training selection | [complete-secondary-results.json](aggregate-data/complete-secondary-results.json); [rq2-gmm-development-contract-v1.json](provenance/original/measured-source/data/rq2-gmm-development-contract-v1.json) | historical.rq2-gmm-development-v1-summary.json.data.candidates contains the six retained training fits. |
| 4.5 — Monitor audit and external alerts | [complete-secondary-results.json](aggregate-data/complete-secondary-results.json); [external-monitor-windows.csv](aggregate-data/external-monitor-windows.csv) | Overlapping window rates, not URL-level false-positive rates. |
| 4.6 — Twenty-five operational groups | [operational-groups.csv](aggregate-data/operational-groups.csv); [operational-runs.csv](aggregate-data/operational-runs.csv) | Five-repeat pooled quantiles. |
| 4.7 — Throughput and drain ranges | [operational-runs.csv](aggregate-data/operational-runs.csv) | Ranges across repeats, not uncertainty intervals. |
| 4.8 — Ranking and calibration | [secondary-metrics.csv](aggregate-data/secondary-metrics.csv); [calibration-bins.csv](aggregate-data/calibration-bins.csv) | Selected rows; complete metrics remain in CSV. |
| 4.9 — Additional tiers and controls | [secondary-metrics.csv](aggregate-data/secondary-metrics.csv) | Tranco alerts are label-free. |
| 4.10 — All permutation comparators | [secondary-metrics.csv](aggregate-data/secondary-metrics.csv); [complete-secondary-results.json](aggregate-data/complete-secondary-results.json) | Consumed-label provenance limitation remains; no seed selection. |
| 4.11 — Transformer seed sensitivity | [secondary-metrics.csv](aggregate-data/secondary-metrics.csv); [seed-logical-invocations.csv](aggregate-data/seed-logical-invocations.csv) | Accepted secondary operating points, not primary replacements. |
| 4.12 — McNemar and Holm contrasts | [primary-results.json](aggregate-data/primary-results.json) | Positive-only contrasts; numerical underflow is not exact zero. |
| 4.13 — Development probes | [complete-secondary-results.json](aggregate-data/complete-secondary-results.json); [probe-decisions-and-scores.csv](aggregate-data/probe-decisions-and-scores.csv); [probe-monitors-and-scores.csv](aggregate-data/probe-monitors-and-scores.csv) | Label-free development accounting. |
| 4.14 — All twenty-two primary gates | [primary-gates.csv](aggregate-data/primary-gates.csv) | Nine pass, thirteen fail; unrounded operands govern. |
| 4.15 — D population admission | [verification.json](aggregate-data/detection-D/verification.json); [domain-size-distribution.csv](aggregate-data/detection-D/domain-size-distribution.csv) | Exclusion-reason incidences overlap. |
| 4.16 — D frozen-threshold comparison | [detection-metrics.csv](aggregate-data/detection-D/detection-metrics.csv); [verification.json](aggregate-data/detection-D/verification.json) | 8,622 eligible records; recall/specificity tradeoff. |
| 4.17 — D requirements | [verification.json](aggregate-data/detection-D/verification.json); [scheme-invariance.csv](aggregate-data/detection-D/scheme-invariance.csv) | Exact invariance does not change the unsupported D conjunction. |
| 4.18 — Ten primary S pairs | [primary-pairs.csv](aggregate-data/service-S/primary-pairs.csv) | Success-only p95; paired ratio, not pooled-quantile ratio. |
| 4.19 — Eight S groups | [group-metrics.csv](aggregate-data/service-S/group-metrics.csv) | Synthetic service workload; unchanged structural scorer. |
| 4.20 — Five S requirements | [requirements.csv](aggregate-data/service-S/requirements.csv); [verification.json](aggregate-data/service-S/verification.json) | Four pass; strict response agreement fails. |
| 5.1 — Technical contributions | [primary-gates.csv](aggregate-data/primary-gates.csv); [verification.json](aggregate-data/detection-D/verification.json); [verification.json](aggregate-data/service-S/verification.json) | Interpretive synthesis of the cited measurements. |
| 5.2 — Decisions at each claim level | [primary-gates.csv](aggregate-data/primary-gates.csv); [verification.json](aggregate-data/detection-D/verification.json); [requirements.csv](aggregate-data/service-S/requirements.csv) | Original H1–H3 remain distinct from D and S. |
| A.1 — Evidence reading map | [advisor-deck-scope-crosswalk-20261001.md](provenance/advisor-deck-scope-crosswalk-20261001.md); [primary-gates.csv](aggregate-data/primary-gates.csv) | Navigation index, not additional data. |
| B.1 — Full eighty-arm S schedule | [arm-metrics.csv](aggregate-data/service-S/arm-metrics.csv) | The interrupted 45-arm schedule is never pooled. |

## Figures

| Item | Source artifacts | Interpretation |
|---|---|---|
| 3.1 — Implemented dataflow | [gwu-system-dataflow-20261001.pdf](manuscript/gwu-system-dataflow-20261001.pdf); [draw_system_dataflow_20261001.py](analysis-scripts/dissertation/draw_system_dataflow_20261001.py) | Schematic of frozen stages and accounting boundaries, not measured performance. |
| 3.2 — Source and partition roles | [followup-source-partitions.pdf](manuscript/followup-source-partitions.pdf); [comparison-specification-v1.md](provenance/followup/comparison-specification-v1.md); [verification.json](aggregate-data/detection-D/verification.json) | Permitted information flow; overlap screening is not temporal validation. |
| 4.1 — Paired D recall difference | [followup-paired-recall.pdf](manuscript/followup-paired-recall.pdf); [verification.json](aggregate-data/detection-D/verification.json) | paired_uncertainty.recall_difference; effect and 97.5% interval in percentage points. |
| 4.2 — D calibration and bin populations | [followup-calibration.pdf](manuscript/followup-calibration.pdf); [calibration-bins.csv](aggregate-data/detection-D/calibration-bins.csv) | Fixed bins and all counts; empty bins have no reliability point. |

## Secondary analyses beyond displayed tables

All declared products remain available, including material not selected for a displayed table:

- [Complete secondary results](aggregate-data/complete-secondary-results.json) and [verification scope](aggregate-data/secondary-verification.json): 153 population/model rows and retained control qualifications.
- [Calibration bins](aggregate-data/calibration-bins.csv): all 430 original bins; [low-FPR score curves](aggregate-data/low-fpr-score-curves.csv): descriptive thresholds, not test-set retuning.
- [Prevalence projections](aggregate-data/prevalence-projections.csv): 129 scenario rows, not estimated deployment prevalence.
- [Monitor windows](aggregate-data/external-monitor-windows.csv) and [PSI features](aggregate-data/external-psi-features.csv): 396 windows and 3,432 feature scores.
- [Seed invocation records](aggregate-data/seed-logical-invocations.csv), [probe decisions](aggregate-data/probe-decisions-and-scores.csv) and [probe monitors](aggregate-data/probe-monitors-and-scores.csv): all retained streams, including adverse results.

## Presentation

There are **25 slides**, ordered by the PowerPoint presentation manifest,
not by slide-part filenames. The [speaker notes](advisor/Tallam_Praxis_Advisor_Integrated_2026-10-04_Speaker_Notes.txt)
match the embedded notes in that order and include tagged public evidence links.
The visible slides and presentation PDF are unchanged from the measured-results edition.

| Slide | Topic | Public evidence |
|---|---|---|
| 1 | Automated Phishing Detection | [RESEARCH_STORY.md](RESEARCH_STORY.md) |
| 2 | The questions promised in August and September | [advisor-deck-scope-crosswalk-20261001.md](provenance/advisor-deck-scope-crosswalk-20261001.md); [primary-gates.csv](aggregate-data/primary-gates.csv) |
| 3 | Evidence coverage and population boundaries | [source-contingency.csv](aggregate-data/source-contingency.csv); [verification.json](aggregate-data/verification.json) |
| 4 | RQ1: internal recall does not ensure external specificity | [primary-results.json](aggregate-data/primary-results.json) |
| 5 | RQ1: structural gain; no incremental fixed-cascade gain | [paired-contrasts.csv](aggregate-data/paired-contrasts.csv) |
| 6 | RQ2: detecting departure did not make routing useful | [primary-gates.csv](aggregate-data/primary-gates.csv); [external-monitor-windows.csv](aggregate-data/external-monitor-windows.csv) |
| 7 | RQ3: low transformer use did not guarantee low latency | [primary-gates.csv](aggregate-data/primary-gates.csv); [operational-runs.csv](aggregate-data/operational-runs.csv) |
| 8 | The complete service matrix includes adverse load results | [operational-groups.csv](aggregate-data/operational-groups.csv) |
| 9 | Secondary evidence qualifies—not repairs—the primary result | [secondary-verification.json](aggregate-data/secondary-verification.json); [complete-secondary-results.json](aggregate-data/complete-secondary-results.json) |
| 10 | One engineering process, with distinct evidence stages | [comparison-specification-v1.md](provenance/followup/comparison-specification-v1.md) |
| 11 | Diagnosis selects two bounded interventions | [diagnosis-and-design.md](provenance/followup/diagnosis-and-design.md) |
| 12 | Detection comparison: freeze first, evaluate every admitted row | [verification.json](aggregate-data/detection-D/verification.json) |
| 13 | D: full operating-point tradeoff at the frozen thresholds | [detection-metrics.csv](aggregate-data/detection-D/detection-metrics.csv); [verification.json](aggregate-data/detection-D/verification.json) |
| 14 | Achieved property: exact scheme invariance | [scheme-invariance.csv](aggregate-data/detection-D/scheme-invariance.csv) |
| 15 | Measured service improvement; strict contract distinguished | [verification.json](aggregate-data/service-S/verification.json); [requirements.csv](aggregate-data/service-S/requirements.csv) |
| 16 | All ten primary pairs: unchanged structural scorer, c64 | [primary-pairs.csv](aggregate-data/service-S/primary-pairs.csv) |
| 17 | Controls locate the measured benefit in the client path | [group-metrics.csv](aggregate-data/service-S/group-metrics.csv) |
| 18 | Technical contribution: measured changes, explicit contracts | [RESEARCH_STORY.md](RESEARCH_STORY.md) |
| 19 | Completed research and dissertation evidence | [VERIFICATION.md](VERIFICATION.md); [service-recovery-amendment-v2.md](provenance/followup/service-recovery-amendment-v2.md) |
| 20 | Appendix: every H1 gate | [primary-gates.csv](aggregate-data/primary-gates.csv); [paired-contrasts.csv](aggregate-data/paired-contrasts.csv) |
| 21 | Appendix: every H2 gate | [primary-gates.csv](aggregate-data/primary-gates.csv) |
| 22 | Appendix: every H3 gate | [primary-gates.csv](aggregate-data/primary-gates.csv) |
| 23 | Appendix: full operational matrix (1/3) | [operational-groups.csv](aggregate-data/operational-groups.csv); [operational-runs.csv](aggregate-data/operational-runs.csv) |
| 24 | Appendix: full operational matrix (2/3) | [operational-groups.csv](aggregate-data/operational-groups.csv); [operational-runs.csv](aggregate-data/operational-runs.csv) |
| 25 | Appendix: full operational matrix (3/3) | [operational-groups.csv](aggregate-data/operational-groups.csv); [operational-runs.csv](aggregate-data/operational-runs.csv) |

## Historical versus current checks

[Integration verification](document-checks/integration/verification.json) and its
[exact text-change ledger](document-checks/integration/editorial-ledger.json)
identify this edition. Earlier citation, source-copy and publication records
remain historical events tied to their original hashes; they are not silently
reissued for a different file. GitHub CI tests source code and package checks;
it does not certify research efficacy, source attribution or university acceptance.
