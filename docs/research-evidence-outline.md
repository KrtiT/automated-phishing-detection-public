# Research Evidence Outline

This file maps the research questions to code-backed evidence. It is not
manuscript prose and does not contain an interpretation of results.

| Item | Value |
|---|---|
| Protocol version | `1.4` |
| Protocol SHA-256 | `2c2956e7cf958f9d2d948a2b1b665e214e12b175d84e4b73c766cc0a6e3be4de` |
| RQ1 baseline contract | `data/rq1-baseline-contract.json` (`rq1-baselines-v1`) |
| RQ1 baseline contract SHA-256 | `594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4` |
| Development source | PhiUSIIL, UCI dataset 967 |
| Development source schema | `2` |
| Source-freeze release tag | `phiusiil-development-v1` |
| Source-freeze release | [`phiusiil-development-v1`](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/phiusiil-development-v1) |
| Development preparation | `complete` |
| Preparation record | `reports/phiusiil-preparation-summary.json` |
| Current technical milestone | RQ1 v1.4 baseline attempt `stopped_nonconverged`; no baseline result accepted; a prospective numerical-convergence amendment must be frozen before any retry. |
| External source | PhishVN v4, reserved for the frozen external evaluation |
| Current hypothesis status | H1 `undecided`; H2 `undecided`; H3 `undecided` |

An absent final artifact or denominator leaves the related item `not_run` or
`undecided` unless an active execution is explicitly recorded as `in_progress`.
An in-progress run is not completed evidence, and no result is inferred from
another experiment.

The [`phiusiil-development-v1`](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/phiusiil-development-v1)
GitHub Release is the source-freeze record for the completed preparation
milestone. It holds the exact licensed UCI archive outside Git history and
connects it to the archive and CSV SHA-256 checksums in source schema version 2.

## Common Audit Record

The evidence index will identify the exact source, license, source hash,
Public Suffix List hash, software commit, environment lock, split manifest,
quarantine counts, model artifacts, thresholds, prediction files, and analysis
outputs. Every reported rate will retain its numerator and denominator; every
interval will identify its method, grouping unit, seed, and source artifact.

The completed development-data run read 235,795 publisher-labeled rows. It
retained 233,536 rows across 197,105 registrable domains and quarantined 2,259
rows: 1,380 invalid or unsupported URLs, 877 same-label canonical duplicates,
and two rows in a conflicting canonical group. Native labels contained 100,945
`0` values and 134,850 `1` values, with no invalid label cell. After the frozen
mapping and quarantine rules, the retained local classes contain 98,687
phishing and 134,849 legitimate rows.

| Split | Domains | Rows | Legitimate (`0`) | Phishing (`1`) |
|---|---:|---:|---:|---:|
| Train | 137,973 | 166,248 | 94,373 | 71,875 |
| Validation | 29,566 | 32,695 | 20,209 | 12,486 |
| Group test | 29,566 | 34,593 | 20,267 | 14,326 |

These counts document preparation only. They do not test H1, H2, or H3.

### Internal Holdout Access Note

On 2026-09-03, after the RQ1 baseline contract and implementation had been
frozen and committed, a broad local repository text search displayed row
content from the ignored PhiUSIIL `group_test.jsonl` file. This was analyst
access and is recorded as such. The `fit-baselines` command does not accept that
file and did not read it: model fitting used the pinned training partition, and
threshold selection used the pinned validation partition. No group-test
prediction or metric was produced, and no contract, implementation, threshold
rule, or hypothesis decision changed in response.

When generated, `access.group_test_accessed=false` in a baseline artifact
describes only the `fit-baselines` process input boundary. It does not negate
the analyst access recorded here.

The group-test file has not been used for fitting, selection, prediction,
metric calculation, or method revision, but it will not be described as unseen
by the analyst. Remaining transformer and cascade procedures will be frozen
without reference to the displayed rows, and any later internal group-test
result will disclose this exposure as a validity limitation. No PhishVN record
was accessed.

### RQ1 Baseline Execution Note

The prescribed RQ1 baseline run started on 2026-09-03 and stopped on 2026-09-04
after approximately eight hours and forty minutes. The full-feature
`Logistic-L1` fit reached the frozen `max_iter=5000` at `tol=1e-8`; scikit-learn
therefore raised a convergence warning, which the contract treats as an error.
The command reported `error: Logistic-L1 did not converge` and exited with code
2. Atomic publication left no model directory or aggregate summary. No model,
threshold, or metric from this attempt is used as research evidence.

The run used implementation commit
`e535586c6162a306a8dac7a5a6546f55dc09136f`. The command accepted only the
pinned PhiUSIIL training and validation files, the preparation summary, and the
feature contract. It accepted no group-test or PhishVN input. The runner
completed the length-only stage in memory before attempting `Logistic-L1`, but
atomic publication emitted neither model and no model, threshold, or metric
from the attempt was reviewed. This is a numerical fitting failure, not
evidence for or against H1. H1, H2, and H3 remain undecided.

## RQ1 and H1

**Question:** What incremental value do structural URL features and
character-level representations provide under registrable-domain-disjoint and
external evaluation?

Required evidence:

- the frozen `rq1-baselines-v1` feature order, predictor exclusions, shared
  logistic configuration, partition use, and threshold-selection rule;
- validation-locked thresholds for length-only, Logistic-L1, transformer-only,
  and cascade models;
- paired predictions on the unscored PhiUSIIL group-test partition, which
  remains excluded from fitting and selection subject to the analyst-access
  limitation recorded above;
- one frozen pass over the primary PhishVN external strata;
- observed FPR counts and the four prespecified recall differences with
  registrable-domain-clustered confidence intervals; and
- a gate table that evaluates every H1 condition without substituting a
  secondary metric.

Current status: baseline contract and feature extraction are `complete`; the
v1.4 fit is `stopped_nonconverged`; no baseline artifact is accepted; H1 is
`undecided`.

## RQ2 and H2

**Question:** Can GMM-based monitoring detect an external source/domain shift
and guide escalation without exceeding the low-FPR operating constraint?

Required evidence:

- GMM component selection, fit artifact, calibration-window scores, and the
  independent false-alert audit;
- label-blind external window scores and the resulting alert trace;
- the future-only routing record showing each alert affects only the next 256
  requests, with overlapping activations unioned; and
- fixed-cascade versus alert-policy errors with the prespecified clustered
  interval and all H2 denominators.

Current status: routing mechanics are `implemented`; monitoring and outcome
evidence are `not_run`; H2 is `undecided`.

## RQ3 and H3

**Question:** What detection, escalation, throughput, and latency tradeoffs
determine whether the fixed cascade is viable inline?

Required evidence:

- certified-registry FPR and exact one-sided upper bounds for both systems;
- the separate Tranco control alert rate and upper bound for both systems;
- cascade-minus-transformer recall and the noninferiority interval;
- the transformer-invocation trace on the frozen 1% reference manifest; and
- five concurrency-64 HTTP runs, pooled p95 latency, and the exact request-error
  numerator over 50,000 measured requests.

Current status: `not_run`; H3 is `undecided`.

## Secondary Evidence

Seed sensitivity, Random Forest, MMD, PSI, controlled perturbations, shortcut
checks, McNemar tests, and Holm-adjusted ablations are reported separately.
They describe robustness but do not replace a primary decision rule.

## Manuscript Map

| Chapter | Evidence role |
|---|---|
| 1 | State the raw-URL problem, operational objective, RQs, hypotheses, contribution boundary, scope, and limitations. |
| 2 | Review structural URL models, character models, domain leakage, low-FPR calibration, source/domain shift, GMM monitoring, and selective cascades. |
| 3 | Describe the frozen label, quarantine, split, model, calibration, monitoring, routing, statistical, and HTTP replay procedures. |
| 4 | Present the audit record first, followed by RQ1, RQ2, RQ3, secondary checks, and one generated gate table. |
| 5 | Interpret the recorded findings, answer each RQ directly, compare with prior work, and delimit mixed or negative results. |
| Appendix | Provide one reproduction and evidence index, with supplemental diagnostics only when needed. |

The working manuscript remains private. Public repository evidence consists of
the protocol, implementation, tests, source and environment locks, aggregate
data-preparation record, and later generated result tables and figures.
