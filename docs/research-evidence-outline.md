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
| Historical RQ1 baseline contract | `data/rq1-baseline-contract.json` (`rq1-baselines-v1`) |
| Historical RQ1 baseline contract SHA-256 | `594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4` |
| Development source | PhiUSIIL, UCI dataset 967 |
| Development source schema | `2` |
| Source-freeze release tag | `phiusiil-development-v1` |
| Source-freeze release | [`phiusiil-development-v1`](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/phiusiil-development-v1) |
| Development preparation | `complete` |
| Preparation record | `reports/phiusiil-preparation-summary.json` |
| Current technical milestone | See the [current execution record](advisor-approval/approval-status.md). Frozen contracts and historical rows are not live run status. |
| External source | PhishVN v4, reserved for the frozen external evaluation |
| Current hypothesis status | H1 `undecided`; H2 `undecided`; H3 `undecided` |

An absent final artifact or denominator leaves the related item `not_run` or
`undecided` unless an active execution is explicitly recorded as
`running_development_validation`.
An in-progress run is not completed evidence, and no result is inferred from
another experiment.

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
evidence for or against H1. H1, H2, and H3 remain undecided.

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
been accessed. H1, H2, and H3 remain undecided. Protocol v1.7 changed the RQ1
baseline method but no RQ/H question, evidence designation, or decision rule.

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
preserves its exact code, timing, failure, and no-fit diagnosis. The
[current execution record](advisor-approval/approval-status.md) tracks recovery.
No completed transformer, threshold, or cascade result exists.

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

The immutable `rq1-transformer-cascade-v2` contract is `frozen_not_run`. The
pre-execution implementation was `frozen_implemented_not_run`; its first
execution is `stopped_stage_one_integrity_check`. Reviewed repository commit
`0793ca3dbc36e49b561cd0ac74968a4644060426` records the last implementation
hardening before this publication amendment. No completed transformer,
threshold, or cascade result exists.

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
first execution is `stopped_stage_one_integrity_check`; H1 is `undecided`.

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
failed. This result does not support H2. External monitoring and routing-outcome
evidence remain `not_run`; H1, H2, and H3 remain `undecided`.

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

The next deliverables are analyses and working inference code, not additional
versions of this outline. No raw group-test or external partition is needed to
implement and test the following pieces on synthetic fixtures.

| Order | Work product | What completion must demonstrate |
|---|---|---|
| 1 | Corrected transformer/cascade development execution | Reconstruct the accepted baseline scorer exactly, check its binding before training, preserve the stopped attempt, and verify all private artifacts against the aggregate completion marker. |
| 2 | Portable inference loaders and paired-statistics code | Load each frozen artifact without fitting; compute paired counts and domain-clustered recall/FNR differences. Freeze RNG, domain ordering, cluster weighting, percentile interpolation, and empty-stratum handling before research-data use. |
| 3 | Composed H2 policy replay | Join saved scores, complete-window alerts, and next-256-request routing; apply outcome-stratum filters after label-blind routing; preserve the failed development false-alert component. |
| 4 | Selective inference service and real-HTTP harness | Actually skip transformer inference outside the band and independently count calls. Test concurrency, timeouts, errors, warm-up exclusion, and pooled latency accounting. |
| 5 | Frozen evaluation and replay-manifest contracts | Bind models, thresholds, evaluator, population/order/denominator rules, manifest selection, environment, and hashes before any internal or external pass. |
| 6 | Independent execution and one gate table | The single internal raw-partition pass produces paired predictions and replay manifests. Later HTTP runs use only those manifests. External schema verification, preparation, and evaluation follow the separate frozen access sequence. |

The existing offline cascade scorer accepts transformer probabilities for every
row and computes a logical invocation mask. That mask is useful for paired
evaluation, but it is not measured transformer work saved or HTTP performance.
The future service must prove actual selective execution independently.

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
and aggregate data-preparation, baseline-validation, and GMM-audit records,
including the failed false-alert gate. The stopped transformer execution and
its diagnosed scoring mismatch are also recorded. The implemented
transformer/cascade procedure can produce evidence when run; it is not itself a
model result. Later result tables and figures enter this record only after their
stated runs.
