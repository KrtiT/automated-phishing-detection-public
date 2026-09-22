# Research Evidence Outline

This file maps the research questions to code-backed evidence. It is not
manuscript prose and does not contain an interpretation of results.

| Item | Value |
|---|---|
| Protocol version | `1.10` |
| Protocol SHA-256 | `aad6b7cf8d8416bfb37ec19503dbed031c15767ea96d9b76486ca3f3fedebe1b` |
| Historical v1.9 protocol SHA-256 | `f24eac919cb79d24d2248a94b3a74208f7b4d809ad778b963ad2e62315d78a38` |
| RQ2 GMM contract | `data/rq2-gmm-development-contract-v1.json` (`rq2-gmm-development-v1`) |
| RQ2 GMM contract SHA-256 | `22d32088b05e74432704f9671ab76ba28b4f573ead418846b23bc366315cb393` |
| RQ1 baseline contract | `data/rq1-baseline-contract-v2.json` (`rq1-baselines-v2`) |
| RQ1 baseline contract SHA-256 | `05d6d0831def7d26448c8dbdc8117800ea2448cdfc2aca2ad95489f22d2d11ba` |
| RQ1 transformer/cascade contract | `data/rq1-transformer-cascade-contract-v2.json` (`rq1-transformer-cascade-v2`) |
| RQ1 transformer/cascade contract SHA-256 | `686c0d86b33b8a6c2e09cd6e174003db0bd2f7c30b087faf5470e6a270524213` |
| Superseded unrun transformer/cascade contract | `data/rq1-transformer-cascade-contract-v1.json` (`rq1-transformer-cascade-v1`) |
| Superseded unrun transformer/cascade contract SHA-256 | `aeaa84534c4cadf0459cf6d2f010dc802684d4801cce563ce18242f36359fb54` |
| Review-hardening repository commit | `0793ca3dbc36e49b561cd0ac74968a4644060426` |
| Transformer/cascade retry producer commit | `e866441f2ff858472d031b8d358fd469897c6a65` |
| Transformer bundle verifier commit | `ef4e4567df979fef3afc91f8ad8097691f94d1ad` |
| Accepted transformer/cascade summary | [`reports/rq1-transformer-cascade-v2-summary.json`](../reports/rq1-transformer-cascade-v2-summary.json) |
| Accepted transformer/cascade summary SHA-256 | `41499aa388babe60442de7231b4087f67a53f96f340568a7cc58a3268606a2fd` |
| Accepted transformer/cascade retry receipt | [`reports/rq1-transformer-cascade-v2-retry-execution.json`](../reports/rq1-transformer-cascade-v2-retry-execution.json) |
| Accepted transformer/cascade retry receipt SHA-256 | `9be2519db4782fe71840235f81e58c4382e504d98a9d888512271107b16e4957` |
| Historical RQ1 baseline contract | `data/rq1-baseline-contract.json` (`rq1-baselines-v1`) |
| Historical RQ1 baseline contract SHA-256 | `594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4` |
| Development source | PhiUSIIL, UCI dataset 967 |
| Development source schema | `2` |
| Source-freeze release tag | `phiusiil-development-v1` |
| Source-freeze release | [`phiusiil-development-v1`](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/phiusiil-development-v1) |
| Development preparation | `complete` |
| Preparation record | `reports/phiusiil-preparation-summary.json` |
| Current technical milestone | See the [current execution record](advisor-approval/approval-status.md). Frozen contracts and historical rows are not live run status. |
| External source | Mendeley Data repository Version 4 (PhishVN v3.1.0), reserved for the frozen external evaluation |
| Current hypothesis status | H1 `undecided`; H2 `not_supported`; H3 `undecided` |

An absent final artifact or denominator leaves the related item `not_run` or
`undecided`. Frozen contracts and dated artifacts retain their own historical
status fields; the current conclusion above incorporates the later accepted
development records. No result is inferred from another experiment.

The [`phiusiil-development-v1`](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/phiusiil-development-v1)
GitHub Release is the source-freeze record for the completed preparation
milestone. It holds the exact licensed UCI archive outside Git history and
connects it to the archive and CSV SHA-256 checksums in source schema version 2.

The September 3 advisor report, SHA-256
`b72da89a4cc8a5b06f6ca88d79fe78dd54e3199a96b7450209ea53b4a4c04215`,
directed the study to complete and freeze the source-provenance release, then
conduct systematic hypothesis testing with all gates, thresholds, features,
and train/validation procedures locked before test results, with particular
attention to H1 and the GMM. The public `phiusiil-development-v1` release
completed that requested source freeze after the meeting.

## Common Audit Record

The evidence index will identify the exact source, license, source hash,
Public Suffix List hash, software commit, environment lock, split manifest,
quarantine counts, model artifacts, thresholds, prediction files, and analysis
outputs. Every reported rate will retain its numerator and denominator; every
interval will identify its method, grouping unit, seed, and source artifact.

Manual review is permitted only as separately reported post hoc descriptive
error analysis and cannot assign or override labels, change quarantine or
inclusion, thresholds, features, model or procedure choices, gates, or
hypothesis decisions.

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

On 2026-09-09, a second broad local wording search displayed row content from
the same ignored group-test file. The displayed rows informed no model,
threshold, gate, routing, or scientific-procedure change; the separate v2
transformer publication correction arose from code review and changed no
scientific field. No fit, score, metric, or PhishVN access occurred. The
partition remains analyst-exposed but model-unscored.

When generated, `access.group_test_accessed=false` in a baseline artifact
describes only the `fit-baselines` process input boundary. It does not negate
the analyst access recorded here.

The group test is analyst-exposed but model-unscored. Its file has not been used
for fitting, selection, prediction, or metric calculation, and the displayed
rows informed no model, threshold, gate, routing, or scientific-procedure
change. It will not be described as unseen by the analyst. The transformer and
cascade scientific procedure was frozen without reference to the displayed
rows, and any later internal group-test result will disclose this exposure as a
validity limitation. The raw group-test partition receives exactly one later
noninteractive frozen processing pass only after all four RQ1 models,
thresholds, evaluator, manifest specifications and selection rules, software
environment, and hashes are frozen. That pass produces the paired predictions
and realized replay manifests. Later HTTP runs use only those frozen manifests
and do not reopen or rescan the raw partition. No PhishVN record was accessed.

### RQ1 Baseline Execution Note

The immutable v1.4 failure record is protocol v1.4 SHA-256
`2c2956e7cf958f9d2d948a2b1b665e214e12b175d84e4b73c766cc0a6e3be4de`,
contract `rq1-baselines-v1` SHA-256
`594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4`,
contract commit `c79e8aefb47560c6ae982dbd5848cf2b707c99a4`, implementation
commit `e535586c6162a306a8dac7a5a6546f55dc09136f`, and failure-record
commit `67107874b9e46457ed710db42f050e35c1ca5ea2`.

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
evidence for or against H1. At that point, H1, H2, and H3 were undecided.

A later exploratory local tolerance check was intended to use training data
only and to change only the candidate tolerance to `tol=1e-4`. The console
observation was `elapsed_seconds=22119.348848833004` (`22,119.35` seconds),
`n_iter=5000`, and `ConvergenceWarning=true`. The exact command, executed code,
environment, and raw console record were not preserved. The v1.4 implementation
constructed the estimator from a hard-coded `tol=1e-8` rather than the
serialized configuration, so the surviving record cannot verify that
`tol=1e-4` reached the fitted estimator and cannot independently verify its
input boundary. No model or summary artifact from the check was retained. The
observation is provenance-incomplete, does not establish that tolerance alone
failed, and is not research evidence.

### SAGA Convergence Diagnostic

Protocol v1.5 froze a two-model SAGA diagnostic and required two fresh-process
executions. Both v1.5 fresh runs completed the length-only fit at `n_iter=69`
with the same fitted-state SHA-256, then stopped while scoring `Logistic-L1`
with `RuntimeWarning: divide by zero encountered in matmul`. The
`Logistic-L1` aggregate was not produced. The exact aggregate receipt is
[`reports/rq1-saga-convergence-v1-execution.json`](../reports/rq1-saga-convergence-v1-execution.json),
SHA-256 `7309f52f704150f85e6c17d44d96adcde917d2539c7b75264bf775ccec3aa6f4`.
The v1.5 SAGA diagnostic has status `stopped_platform_warning` in the execution
record; it is a diagnostic stop, not a baseline result or RQ/H evidence.

Protocol v1.6 froze the next training-only, two-model SAGA diagnostic before
use. It retained the same two fresh processes, models, feature sets, scaler,
classifier configuration, state digest, and repeatability gates. Scaling and
fitting warnings remained fatal. A narrow scoring-only audit applied on macOS
arm64 when NumPy reported the Accelerate BLAS: one of three exact
`RuntimeWarning` messages from `sklearn.utils.extmath` could be captured only
during `decision_function` or `predict_proba`. The warning had to be recorded.
The scikit-learn scores and probabilities had to be finite and agree with
independent float64 `einsum` and full two-column `expit` calculations at
`rtol=1e-12` and `atol=1e-12`; only the maximum absolute differences could be
reported. The repeat had to match the first run's iteration counts and
fitted-state SHA-256 values.

SAGA leaves the intercept unpenalized. The frozen `liblinear` baseline instead
uses a penalized synthetic intercept at `intercept_scaling=1.0`, so this
diagnostic is not a `rq1-baselines-v1` model. The feature definitions contain
the exact dependency `raw_url_codepoint_length = raw_url_ascii_letter_count +
raw_url_ascii_digit_count + raw_url_other_codepoint_count`, so individual
coefficients and the selected sparsity pattern will not be interpreted as
feature importance.
The exact aggregate receipt is
[`reports/rq1-saga-convergence-v2-execution.json`](../reports/rq1-saga-convergence-v2-execution.json),
SHA-256 `487714ea17a095e369d381da5a27452f3b263f1be1a05db2ebe061eecdefebdf`,
from clean commit `69a67d4e5cb81d49009d6a90e87f4c0c5f2cea87`. Both v1.6 fresh
runs passed with matching iteration counts and fitted-state SHA-256 values.
`length-only` stopped at `n_iter=69`; `Logistic-L1` stopped at `n_iter=4783`.
Each full-model run recorded six allowlisted scoring warnings. Maximum absolute
decision and full-matrix probability differences were
`4.263256414560601e-14` and `5.551115123125783e-16`, respectively. The v1.6
SAGA diagnostic has status `passed_training_only`. It establishes reproducible
training feasibility, not generalization or an RQ/H result.

### RQ1 Baseline v2 Validation

Protocol v1.7 freezes `rq1-baselines-v2` before validation execution. It keeps
the v1 feature vector, predictor exclusions, partition boundaries, score
meaning, threshold-selection rule, and no-search policy. It prospectively
replaces `liblinear` at `tol=1e-8` with the diagnostic's SAGA configuration at
`tol=1e-4`; this also changes the intercept from a penalized synthetic feature
to SAGA's unpenalized intercept. Validation scores must pass the same exact
platform-warning audit and independent full-matrix numerical reference before
the scikit-learn class-1 probability is used for threshold selection.

The command ran once on 2026-09-04 from clean commit
`7ae6c9af85e935c551468f590a7ba43441f58def`, after CI passed. The exact
aggregate [summary](../reports/rq1-baseline-v2-summary.json), SHA-256
`bf5b3a6f0fc705d26852da4dd0053c6111ffc3e500d7a2e95dfba5ad859b279c`,
records `analysis_stage=development_validation_only`. The rq1-baselines-v2
execution has status `completed_development_validation`. This is development
validation only.

| Model | Artifact SHA-256 | `n_iter` | Threshold | Validation recall | Observed FPR | One-sided 95% FPR upper bound |
|---|---|---:|---:|---:|---:|---:|
| `length-only` | `b8b92cfbe29160e769e5e7d80712becc8fc0680cdfd45a44b839ef9bada87799` | 69 | 0.7612031186147 | 0.3214800576645843 | 0.006680191993666189 | 0.007702192373035135 |
| `Logistic-L1` | `71a3e24a0283a31ba188bc7dd60b18c1b708370b9ca275d5ab1a1004680c968a` | 4783 | 0.2670846328466124 | 0.9842223290084895 | 0.008758473947251225 | 0.009915480854183582 |

The length-only scoring audit recorded no warning and zero reference
difference. The full model recorded six allowlisted scoring warnings; its
finite decision values and full probability matrix matched the independent
reference with maximum absolute differences of `2.1316282072803006e-14` and
`3.3306690738754696e-16`. Both thresholds have status `selected`. These
validation results set operating points; they are not group-test or external
evidence and do not decide H1. Coefficients and sparsity are not interpreted as
feature importance.

The group test remains analyst-exposed but model-unscored. The summary's
`access.group_test_accessed=false` describes the process input boundary and
does not negate the analyst exposure recorded above. No PhishVN record has
been accessed. At that September 4 baseline milestone, H1, H2, and H3 were
undecided. Protocol v1.7 changed the RQ1 baseline method but no RQ/H question,
evidence designation, or decision rule.

Protocol v1.8's byte-preserved `rq1-transformer-cascade-v1`, SHA-256
`aeaa84534c4cadf0459cf6d2f010dc802684d4801cce563ce18242f36359fb54`, is
`superseded_unrun`. Protocol v1.9 freezes the publication-only amendment
`rq1-transformer-cascade-v2`, SHA-256
`686c0d86b33b8a6c2e09cd6e174003db0bd2f7c30b087faf5470e6a270524213`, at
`frozen_not_run`. Its scientific and artifact-content rules are unchanged. The
contract accepts only the pinned training, validation, preparation-summary,
baseline-contract, and `Logistic-L1` artifact roles; it accepts no group-test,
external, PhishVN, runtime-tuning, or test-path input. The implementation is
unit-tested before execution at status `frozen_implemented_not_run`; its first
MPS execution stopped at a stage-one scoring integrity check. Protocol v1.9 separately
corrected the then-unrun GMM allocation status to
a prospective staged freeze; no GMM allocation choice was made. The
implementation and its tests did not open the PhiUSIIL group-test partition or
PhishVN.

The [stopped execution](../reports/rq1-transformer-cascade-v2-execution.json)
preserves its exact code, timing, failure, and no-fit diagnosis. The controlled
retry from producer commit `e866441f2ff858472d031b8d358fd469897c6a65`
subsequently completed development validation. Its accepted
[summary](../reports/rq1-transformer-cascade-v2-summary.json), SHA-256
`41499aa388babe60442de7231b4087f67a53f96f340568a7cc58a3268606a2fd`,
and [retry receipt](../reports/rq1-transformer-cascade-v2-retry-execution.json)
are linked in the table above. The current execution details and independent
bundle audit are in the [execution record](advisor-approval/approval-status.md).

## RQ1 and H1

**Question:** What incremental value do structural URL features and
selective character-model escalation provide under
registrable-domain-disjoint and external evaluation?

The two H1 primary contrasts remain `recall(Logistic-L1) -
recall(length-only)` and `recall(cascade) - recall(Logistic-L1)`. The latter is
the selective system contribution, not a pure causal isolation of
representation. Transformer-only remains a comparator and operational
reference, not a third primary H1 gate. It calibrates the cascade and supports
the H3 comparison.

The immutable `rq1-transformer-cascade-v2` contract retains its freeze-time
status `frozen_not_run`. The implementation's historical pre-execution status
was `frozen_implemented_not_run`, and its first execution stopped with
`stopped_stage_one_integrity_check`. The accepted retry is
`completed_development_validation`. Reviewed verifier commit
`ef4e4567df979fef3afc91f8ad8097691f94d1ad` loaded the named bundle on MPS,
verified the four private hashes and public/private projections, and performed
no fit; its invocation read and scored no research rows.

The [development comparison](advisor-approval/approval-status.md#controlled-retry)
records all four models' validation operating points.

The cascade's validation recall was identical to `Logistic-L1`; its logical
routing mask selected escalation for 3 of 32,695 rows. Calibration computed
transformer scores for every validation row. This development
result does not decide H1 and does not establish measured HTTP savings.

Required evidence:

- the frozen `rq1-baselines-v2` feature order, predictor exclusions, shared
  logistic configuration, partition use, and threshold-selection rule;
- validation-locked thresholds for length-only, Logistic-L1, transformer-only,
  and cascade models;
- paired predictions on the unscored PhiUSIIL group-test partition, which
  remains excluded from fitting and selection subject to the analyst-access
  limitation recorded above;
- one frozen pass over the primary PhishVN external strata;
- observed FPR counts and the two prespecified H1 recall differences with
  registrable-domain-clustered confidence intervals; and
- a gate table that evaluates every H1 condition without substituting a
  secondary metric.

Current status: baseline contract and feature extraction are `complete`; the
v1.4 fit is `stopped_nonconverged`; the exploratory tolerance observation is
`not_accepted_provenance_incomplete`; the v1.5 SAGA diagnostic is
`stopped_platform_warning`; the v1.6 SAGA diagnostic is
`passed_training_only`; the rq1-baselines-v2 execution is
`completed_development_validation`; the validation-set operating points are
recorded above; the transformer/cascade contract is `frozen_not_run` and its
first execution is `stopped_stage_one_integrity_check`; the controlled retry is
`completed_development_validation`; H1 remains `undecided` pending the paired
domain-clustered internal and external comparisons.

## RQ2 and H2

Protocol v1.10 freezes [`rq2-gmm-development-v1`](../data/rq2-gmm-development-contract-v1.json),
SHA-256 `22d32088b05e74432704f9671ab76ba28b4f573ead418846b23bc366315cb393`,
before GMM execution,
and incorporates unchanged transformer v2. The ordered 26-column input combines
the 25 structural features with the pinned portable Logistic-L1 probability.
Scaling and all six diagonal-GMM fits use training only; minimum training BIC
selects the component count, with exact ties favoring the smaller count.

Unique normalized ASCII validation domains are SHA-256 ordered under namespace
`rq2-gmm-validation-v1`, seed `20260816`, digest-byte then domain-byte order.
The first `floor(D/2)` domains form calibration; the rest form audit, preserving
input row order without labels, class counts, or rerolling. The boundary is
`np.quantile(calibration_scores, 0.95, method="linear")`. Audit alerts require
`score > boundary`; the gate is exactly `20 * alert_windows <= complete_windows`.
The boundary is never retuned on audit, and overlapping-window fractions get
no binomial confidence interval. A failed gate still records
`completed_development_validation` and `false_alert_gate_met=false`, not an H2
decision. Private scaler/mixture parameters and membership/window traces remain
outside Git; the public record may contain only aggregate counts, all six
BIC/iteration records, configuration, boundary, audit fractions, and hashes.
Synthetic-only preflight exposed an Accelerate warning; all six candidates
passed with the same NumPy 2.2.6 and `scipy-openblas` 0.3.29. That backend is
required before input reads and its observed name/version are recorded in the
public summary. No research-data fit or warning-policy relaxation was used.

**Question:** Can GMM-based monitoring detect an external source/domain shift
and guide escalation without exceeding the low-FPR operating constraint?

Every complete 256-request window of the retained external stream is a
prespecified external-shift window. The detection-rate numerator is windows
with score strictly greater than the boundary; the denominator is all such
complete windows. Overlapping windows count separately. An incomplete terminal
window is excluded from this rate, but its requests remain routable from a
prior alert. The independent validation-audit false-alert fraction uses the
same complete-window numerator and denominator rule.

Required evidence:

- GMM component selection, fit artifact, calibration-window scores, and the
  independent false-alert audit;
- label-blind external window scores and the resulting alert trace;
- the future-only routing record showing each alert affects only the next 256
  requests, with overlapping activations unioned; and
- fixed-cascade versus alert-policy errors with the prespecified clustered
  interval and all H2 denominators.

### Development Result

The single prescribed GMM run completed on September 17 from clean, published
commit `9983bfddc6ea31e370629cb1b2efd3da383f7ff0`, after
[CI passed](https://github.com/KrtiT/automated-phishing-detection-public/actions/runs/35260883392).
The [aggregate summary](../reports/rq2-gmm-development-v1-summary.json) has
SHA-256 `6f695138a302e854e1e5af590152e289486affe8ccdf75510ca9a5dcaad3b523`.
All six candidates converged without warnings; training BIC selected six
components. Calibration used 16,325 rows and audit used 16,370, with 14,783
domains and 252 complete windows in each stream. The frozen calibration rule
yielded a boundary of `-67.45792380813624`.

The independent audit alerted on **28/252 windows (11.11%)**, exceeding the
prespecified **5%** limit. Thus `false_alert_gate_met=false`; the boundary was
not retuned and the run was not repeated. No independent-binomial interval is
assigned to these overlapping windows. The aggregate does not identify the
cause of this failure. Private fitted parameters and window traces were
retained, with their hashes in the public record.

Current status: routing mechanics are `implemented`; GMM execution is
`completed_development_validation`, with its required false-alert component
failed. H2 is not supported. External monitoring and routing-outcome evidence
remain `not_run` and are needed to characterize RQ2, not to rescue the failed
conjunctive support rule. H1 and H3 remain undecided.

### Post Hoc Audit Description

The [saved-window descriptor](../scripts/describe_gmm_audit.py), committed at
`dc3def7dddfeb2c161d567d89b9cd5fcf0457016` before its recorded execution, reads
only the hash-pinned summary and saved audit trace. It performs no fitting,
new URL scoring, or threshold selection. The
[aggregate description](../reports/rq2-gmm-development-v1-description.json) has
SHA-256 `fe0e8c9fdae32fc48118102b71e0cda7f771b9ab77e49c4146d2f3479c052712`.

| Saved window score | Calibration | Audit |
|---|---:|---:|
| Median | -70.975987 | -70.619527 |
| 90th percentile | -68.156625 | -67.236718 |
| 95th percentile | -67.457924 | -65.330474 |
| Maximum | 482882.365753 | -47.305379 |
| Above the original boundary | 13/252 | 28/252 |

The audit's upper quantiles are higher, but calibration has a much larger
maximum. These observations do not show a uniform shift or identify its cause.
The audit's 28 alerts form nine consecutive-window runs and cover 3,456 unique
row positions, rather than 7,168 independent memberships. Window overlap does
not explain away the failed gate: the frozen rule counts windows, permits at
most 12 of these 252 windows to alert, and still fails at 28.

The audit percentiles above are descriptions, not replacement thresholds.
Calibration's 13 exceedances are compatible with its linearly interpolated
95th percentile and do not imply a coding error. These saved scores cannot
distinguish feature leverage, composition, ordering, or model misspecification
as the cause. Any later feature-level investigation is exploratory and must
leave the fitted monitor, allocation, threshold, and failed result unchanged.

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

## Remaining Executable Work

The corrected transformer/cascade development execution and reviewed no-fit
bundle loader are complete. Artifact-only scoring now includes length-only,
Logistic-L1, transformer, and cascade paths. `selective_inference.py` shares one
singleton scoring path between full paired scoring and selective requests.
Stage one receives the original raw URL; character normalization remains
specific to the transformer. A selective request skips the transformer outside
the band unless a prior alert overrides routing. Forward attempts and validated
successful scores are counted separately. These behaviors have been tested on
synthetic fixtures, not measured as HTTP performance.

The paired evaluator checks record identity and order before calculating
URL-weighted differences. It resamples whole positive-stratum domains with
paired outcomes, retaining the row counts of unequal domains. The exact RNG,
draw sequence, percentile interpolation, and missing-stratum rules are in
[`evaluation-contract-v1.json`](../data/evaluation-contract-v1.json). That
contract is explicitly `prospective_incomplete`: it does not complete the
pre-access freeze or provide a protected-data execution command. Routing must
precede stratum selection; bootstrap resampling never reruns the policy. The
H2 interval is conditional on the realized stream and does not capture
cross-domain temporal dependence from shared routing activations.

Synthetic runtime checks also found that backend and batch shape can change
last-bit scores, including a threshold or band decision at exact equality.
OpenBLAS and MPS can run together, but numerical closeness does not establish
decision equivalence. The evaluation contract now specifies OpenBLAS 0.3.29,
MPS, one numerical thread and one URL per forward as the candidate shared
runtime. Its separately recorded, no-fit development comparison must reproduce
the original reference results and then check exact decisions, band membership,
and calibration/audit window alerts. The first attempt stopped in candidate
preflight before model or validation access: PyTorch's first thread-count query
reset its OpenMP pool after the limiter had run. A fresh-interpreter regression
now checks initialization before limiting and restoration afterward. The
contract permitted one explicitly requested corrective execution with a separate
receipt and authenticated the original preflight-only failure. That comparison
has completed; its
[aggregate receipt](../reports/inference-compatibility-v1-preflight-correction.json)
records `not_equivalent` on 32,695 validation rows. The original reference
reproduced all accepted counts, the original band-selection count of three,
and the saved GMM calibration/audit window scores. The candidate produced zero decision
mismatches for length-only, Logistic-L1 and cascade, zero band mismatches, and
zero GMM alert mismatches in either stream. Transformer-only had one mismatch:
12,373 rather than 12,374 true positives, with false positives unchanged at 178.
Its maximum absolute probability difference was `7.152557373046875e-7`.

The zero-mismatch acceptance rule therefore rejects the candidate runtime;
numerical closeness does not waive that rule. This is a finite development
comparison of the two full scoring paths, not isolation of a particular kernel
or a new hypothesis test. Original thresholds, band, GMM boundary, and the
28/252 failed GMM audit remain unchanged. The comparison evaluated the
transformer for every validation row and is not a selective-work or HTTP
measurement. No further run or alternate-runtime search follows automatically.
The September 21 [inference-method amendment](../data/singleton-inference-amendment-v1.json)
adopts this same singleton convention for future paired evaluation and serving.
It explicitly replaces the future equivalence prerequisite, not the failed
historical comparison. The choice is development-informed. Historically
selected weights, thresholds and band are carried forward without recalibration;
their optimizer/tie-break optimality under singleton scoring is not claimed.
No further compatibility execution, alternate-runtime search or protected-data
access is authorized. The historical contract and receipts remain byte-identical.

`saved_metrics.py` now calculates aligned binary counts and exact one-sided
95% upper bounds. Tranco controls use a separate label-free alert-rate API.
`hypothesis_evaluation.evaluate_primary` assembles the six frozen contrasts and
22 component gates. It rejects overlapping gold/certified/Tranco record IDs,
uses integer comparisons for observed-rate gates, and keeps missing evidence
distinct from zero-denominator or insufficient-domain non-estimability. A
measured failed conjunct yields `not_supported` with `complete=false` when
other components remain unmeasured. The four H3 confidence bounds are reported,
not substituted for the observed-rate gates. Overlapping monitor windows do
not receive binomial intervals.

The evaluator accepts typed physical-reference and HTTP summaries but does not
establish their provenance or perform a benchmark. Its producer must verify
the complete manifest, actual forward counters, warm-up exclusion, total
2,000 ms deadline, response validity and pooling of individual latencies.
The pure evaluator requires exactly five measured concurrency-64 runs of
10,000 requests for H3. Summary construction is not measurement evidence.

For example, the existing public audit can be evaluated without opening data
or loading a model:

```python
import json
from pathlib import Path
from automated_phishing_detection.hypothesis_evaluation import WindowCounts, evaluate_primary

audit = json.loads(Path("reports/rq2-gmm-development-v1-summary.json").read_text())
result = evaluate_primary(audit_windows=WindowCounts(
    audit["audit_alert_count"], audit["audit_window_count"]
))
assert result.hypotheses["H2"].decision == "not_supported"
assert result.hypotheses["H2"].complete is False
```

`policy_replay.py` joins identity-aligned model and monitor scores without labels
or source filters. Each complete 256-request window is evaluated at stride 64;
an alert affects only the next 256 requests, with overlapping overrides unioned.
Its cached-score routing masks are logical selections, not skipped computation.
A synthetic integration test matches replay decisions to selective execution
and checks that the first alert cannot change earlier requests.

`evaluation_stream.build_external_evidence` now validates complete aligned
metadata and saved scores, calls that replay once, then selects the outcome
strata. Gold and certified populations retain all five detector/policy outputs;
Tranco retains label-free cascade and transformer controls. Secondary records
remain in the window and routing population. This establishes the ordering
boundary in code, not the authenticity or completeness of future source inputs.

`evaluation_manifest.build_manifest` validates prepared candidates and selects
the three 10,000-row prevalence references without replacement. Purpose-specific
PCG64 streams separate negative selection, positive selection and final order.
All repeats use the same order; warmup reuses the first 1,000 rows under separate
request IDs. Insufficient capacity is explicit, with no fallback sample.
The [manifest supplement](../data/evaluation-manifest-contract-v1.json) defines
the exact draw sequence and private-payload hash; no real manifest exists from
this increment.

`selective_service.create_app` constructs and runs the scorer on one dedicated
owner thread. A bounded FIFO preserves admission order. Once admitted, a request
finishes even if the client disconnects. A drain closes the previous phase's
IDs before queuing a counter snapshot, preventing a late warmup body from
entering measured work. `http_replay.replay_run` uses real HTTP, bounded
concurrency, no retries and a total 2,000 ms deadline including response
validation. It keeps terminal failures, verifies physical counters and excludes
warmup by phase rather than resetting the model. Its primary summary requires
five complete matching concurrency-64 runs and pools all 50,000 individual
latencies. The [HTTP supplement](../data/http-replay-contract-v1.json) fixes
these choices before measurements. Real-loopback tests use temporary synthetic
CPU scorers, including forwards that succeed or fail after client timeout;
they are integration checks, not latency or detection results.

`execution_preflight.bind_execution` checks an externally supplied reviewed
commit and binding-contract hash against the clean nested checkout, committed
and working source bytes, loaded package origins, public evidence pins and
declared hardware/runtime. `bound_models.load_bound_models` accepts only the
previously recorded artifact identities. `bound_runtime` composes these checks
with the service owner lifecycle, including a post-restoration identity check.
`execution_receipt` reserves an attempt and publishes private evidence before
the public completion marker without replacing existing records. Installed
records survive failure; acceptance requires successful producer exit and
verification of all installed evidence, not merely the presence of a marker.
The internal file/process composition is now implemented; complete secondary
output coverage and external source execution remain integration tasks.

`evaluation_producer.parse_internal_partition` checks the supplied bytes against
an expected hash before parsing, then validates canonical rows, source IDs,
domains, split membership and counts. `produce_internal_evidence` requires a
fresh owner-thread session before any scoring and makes one in-memory pass for
all four detectors, portable monitor features and NLLs. It returns paired
evidence, private payloads, primary evaluation and the three replay manifests,
or an explicit capacity shortfall. Caller-supplied hashes do not authenticate a
file by themselves. `source_runner` now derives those pins from the authenticated
public source and preparation records, checks their exact cross-binding, and
reserves an attempt before reading PSL, models or partition bytes. It hashes the
same PSL and partition buffers that it parses, rejecting aliases and changes to
the open file or its pathname. It composes primary evidence and secondary score
metrics, closes the owner, and rechecks the execution binding before publishing.
Outputs stay outside the clean authenticated checkout. Failure after publication
starts does not replace the installed claim or imply successful completion.

`scripts/run_internal_evaluation.py` uses a fresh worker process. Its parent
requires a successful worker exit and independent completion verification,
including the reservation, claim, outcome and saved-file hashes. Captured child
diagnostics are not forwarded into public records. Both parent and worker stop
before supplied-input path access under the current profile, whose readiness
remains false. The private composition is tested on temporary invented files;
there is no flag to enable protected access. Completing the remaining secondary
models and one-pass outputs requires a reviewed future profile, not a command-line
bypass. Receipt verification establishes execution consistency, not scientific
truth or a replacement for recomputing analyses from saved evidence. It checks
fixed gate and role semantics but does not recompute prediction or manifest
contents; model identities and carried-forward cutoffs remain authenticated by
the worker's loader, not by a second artifact read in the parent.

`phishvn.prepare_external_rows` implements the frozen mapping and preparation
rules on normalized records. Declared all-split coverage and file positions are
checked before whole-group quarantine, domain exclusions and stable-ID duplicate
selection. Retained test records keep source-file order; no duplicate is chosen
because it has a more favorable confidence tier. Published labels, routing
roles and Tranco's label-free role remain distinct. These are synthetic tests
of the normalized interface, not verification of actual PhishVN columns, a
complete source inventory or the PhiUSIIL overlap set. Those inputs must be
authenticated in the staged external preparation and producer integration.

`secondary_metrics.py` implements AP, ROC AUC, observed recall at FPR <= 1%,
confusion metrics, Brier score, ten-bin ECE and prevalence projections. Probability
scores and binary operating-point decisions remain separate, which matters for
a cascade with different stage thresholds. Exact two-sided McNemar tests and
the fixed four-contrast Holm family are secondary, dependence-limited summaries;
they do not replace the primary domain-clustered intervals. Missing strata stay
explicit and retain their place in the multiplicity family. The
[secondary supplement](../data/secondary-analysis-contract-v1.json) also specifies
MMD, PSI, perturbations and shortcut checks. `secondary_drift.py` implements
training-reference selection, biased RBF MMD, featurewise PSI, complete 256/64
windows and independent strict-boundary calibration/audit functions. It receives
already standardized arrays; callers must bind training provenance and the
original calibration/audit allocation. No reference or boundary has been fitted
on research records in this increment.

`secondary_probes.py` preserves original URL spelling and returns three separate
operator outputs with eligibility and changed status. Inputs retain the primary
canonical-url acceptance rules. Uppercase indicators mean presence of at least
one ASCII uppercase letter; the host excludes userinfo and port. Explicit default
ports compare numerically, including leading zeros. Path encoding skips existing
percent escapes; percent-case inspection covers every raw component. Eligible
no-ops remain distinct from ineligible unchanged inputs. Probes inherit no label.
The five formatting indicators are implemented, but their classifier is not fit.
Secondary fit/runtime/calibration contracts and development execution must be
completed before protected evaluation. The supplement is development-informed,
not a retroactive claim that these details preceded all development observations.

The metadata-only preflight accepts no model or dataset paths:

```sh
python scripts/verify_execution_binding.py --repo-root . \
  --expected-revision REVIEWED_COMMIT \
  --expected-contract-sha256 REVIEWED_BINDING_CONTRACT_SHA256
```

Use independently reviewed literal identities, not values discovered and
accepted automatically at execution time. The command checks readiness of
public code and runtime only; its output explicitly reports no research
measurement and no protected-evaluation readiness.
Current code accepts only `data/execution-binding-contract-v2.json`, SHA-256
`887f771381927dfe1b9268a45f4e605baf3e9a7caee2b7005cdfe68b1be516e1`.
Its 26 public pins retain all 23 v1 pins and add the original v1 profile and the
shift and secondary supplements. Runtime requirements are unchanged. The v1
command remains reproducible at commit `d119443b928f9c23840ad7fd9ee1f67d882a8108`;
current code does not fall back to it.

The [workload specification](../data/operational-workloads-v1.json) separates
three measurements. Fixed cascade retains its original primary H3 rules.
The implemented transformer-only path runs character encoding and one physical
forward without structural, logistic or monitor inference. The live shift
workload is now implemented through `live_monitor`, `shift_service` and
`shift_replay`, composed with bound loading by `create_bound_shift_app`.
One unresolved request preserves planned order, including after timeouts.
Warmup drains before a one-time monitor reset; physical counters are not reset.
The live monitor uses the original portable feature and float64 window mean,
256-row windows at stride 64, strict threshold exceedance and future-only
256-request overrides. A computation failure permanently invalidates that run.

The shift client retains request outcomes if controls fail, checks admission
order and complete traces, and keeps inline timeout-drain time within measured
phase wall time. Final drain and trace export are separate. Live traces can be
checked against offline replay, including NLLs, windows, alerts and decisions.
A temporary CPU-model TCP test exercises the real monitor, reset and first
future-only override. These tests are not operational research measurements.
Shift summaries are separate from primary H3 summaries; caller plan hashes and
reference parameters still need authentication by the official producer. The
[shift supplement](../data/shift-execution-contract-v1.json) fixes the executable
conventions without changing the frozen workload or H3 definition.

| Order | Work product | What completion must demonstrate |
|---|---|---|
| 1 | Complete paired evaluator integration | Internal file/process execution, primary/score-metric composition and receipt verification are implemented on fixtures behind the closed readiness gate. Complete remaining secondary-model outputs and external execution before opening that gate. The singleton convention is adopted by explicit amendment, not equivalence acceptance. |
| 2 | Composed H2 policy replay | Full-stream routing followed by outcome-stratum selection and normalized external preparation are implemented on fixtures. Bind the verified publisher schema, complete source inventory and saved-score inputs. Preserve the failed development false-alert component; this characterizes RQ2 and cannot rescue H2. |
| 3 | Selective inference service and real-HTTP harness | Fixed cascade, transformer-only and serialized live-monitor modes are tested, including real TCP, phase reset and offline trace agreement. Bind service, client, process lifecycle and saved outputs in the official producer before measurements. |
| 4 | Frozen evaluation and replay-manifest contracts | Sampling, stream integration, runtime identity and all three workloads have supplements. Secondary score, MMD/PSI and probe calculations are implemented. Complete development provenance/calibration, secondary fits and their contracts before the pre-access freeze. |
| 5 | Independent execution and one gate table | The single internal raw-partition pass produces paired predictions and replay manifests. Later HTTP runs use only those manifests. External schema verification, preparation, and evaluation follow the separate frozen access sequence. |

The existing offline cascade scorer accepts transformer probabilities for every
row and computes a logical invocation mask. That mask is useful for paired
evaluation, but it is not measured transformer work saved or HTTP performance.
The service and client now reconcile responses with physical forward counts on
fixtures; research execution must still measure the complete HTTP path independently.

The GMM's failed 5% audit gate remains a failed mandatory H2 component. Further
external or routing measurements can explain system behavior; they cannot make
the original conjunctive support rule pass. Descriptive failure analysis is
separate from model selection and does not reopen the audit for tuning.

## Manuscript Map

| Chapter | Evidence role |
|---|---|
| 1 | State the raw-URL problem, operational objective, RQs, hypotheses, contribution boundary, scope, and limitations. |
| 2 | Review structural URL models, character models, domain leakage, low-FPR calibration, source/domain shift, GMM monitoring, and selective cascades. |
| 3 | Describe the frozen label, quarantine, split, model, calibration, monitoring, routing, statistical, and HTTP replay procedures. |
| 4 | Present the audit record first, followed by RQ1, RQ2, RQ3, secondary checks, and one generated gate table. |
| 5 | Interpret the recorded findings, answer each RQ directly, compare with prior work, and delimit mixed or negative results. |
| Appendix | Provide one reproduction and evidence index, with supplemental diagnostics only when needed. |

The working manuscript remains private. Public repository evidence currently
consists of the protocol, implementation, tests, source and environment locks,
and aggregate data-preparation, baseline-validation, transformer/cascade, and
GMM-audit records, including the failed false-alert gate. The stopped
transformer execution and its diagnosed scoring mismatch remain recorded beside
the accepted retry. The accepted transformer/cascade record is development
validation only: it does not decide H1 and does not establish measured HTTP
savings. H2 is not supported; H1 and H3 remain undecided. Later result tables
and figures enter this record only after their stated runs.
