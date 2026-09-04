# Research Status

**Status:** The August 20 advisor report directed continuation of the
three-question design with objective, reproducible outcome criteria. The
PhiUSIIL development split and aggregate preparation record are complete. The
historical `rq1-baselines-v1` contract remains preserved. Its prescribed fit
stopped with a convergence warning and published no model or summary. A later
local tolerance observation is provenance-incomplete and is not research
evidence. The v1.5 SAGA diagnostic has status `stopped_platform_warning`; the
v1.6 SAGA diagnostic has status `passed_training_only`. Protocol v1.7 froze
`rq1-baselines-v2` before validation. Its single planned execution completed
from clean commit `7ae6c9af85e935c551468f590a7ba43441f58def` and has status
`completed_development_validation`. This is development validation only. H1,
H2, and H3 remain undecided, and no PhishVN record has been accessed.

| Field | Value |
|---|---|
| Protocol | `docs/advisor-approval/2026-08-16-realignment-matrix.md` |
| Protocol version | `1.7` |
| Protocol date | `2026-09-04` |
| Protocol SHA-256 | `ebd5b9f90157d6d21e8f22b7e1937c16dbaf60a577ec4f55cefc4709b604caef` |
| RQ1 baseline contract | `data/rq1-baseline-contract-v2.json` (`rq1-baselines-v2`) |
| RQ1 baseline contract SHA-256 | `05d6d0831def7d26448c8dbdc8117800ea2448cdfc2aca2ad95489f22d2d11ba` |
| Historical RQ1 baseline contract | `rq1-baselines-v1` |
| Historical RQ1 baseline contract SHA-256 | `594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4` |
| Development source schema | `2` |
| Source-freeze release tag | `phiusiil-development-v1` |
| Source-freeze release | [`phiusiil-development-v1`](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/phiusiil-development-v1) |
| Development preparation | `complete` |
| Current technical milestone | Protocol v1.7 baseline amendment `frozen`; v1.4 baseline attempt `stopped_nonconverged`; exploratory tolerance observation `not_accepted_provenance_incomplete`; v1.5 SAGA diagnostic `stopped_platform_warning`; v1.6 SAGA diagnostic `passed_training_only`; rq1-baselines-v2 execution `completed_development_validation`; H1, H2, and H3 remain undecided. |
| Current hypothesis status | H1 `undecided`; H2 `undecided`; H3 `undecided` |
| Legacy base tag | `legacy-v2-clean-2026-08-16` |
| Legacy base SHA | `a5eceecf21ad5ce29c4ab8f8d4de0edc8b73b240` |
| Advisor report date | `2026-08-20` |
| Advisor report SHA-256 | `c4be5b5d5c4eaf89a2494a521522d124aa45351ee8d08dda2f41b01b961a0b7a` |
| Direction | Proceed with the three-question design and make the outcome criteria objective and reproducible. |

## Current Controls

- PhiUSIIL is the only development source. Its native labels are mapped by an
  exact rule, and no researcher assigns or changes a row's outcome.
- Source bytes, the Public Suffix List, the preparation algorithm, and every
  generated split are identified by SHA-256.
- Registrable domains are assigned to one development split by the frozen,
  label-blind rule in the protocol.
- `rq1-baselines-v2` preserves the 25 raw-URL features, their order and
  denominators, partition use, score meaning, and threshold rule from v1. It
  froze SAGA and the validation-scoring audit before the recorded execution.
  Both the active and historical contract hashes are recorded above.
- The PhiUSIIL group test is analyst-exposed but model-unscored. It remains
  excluded from fitting and selection. Its raw partition receives exactly one
  later frozen noninteractive processing pass only after all four RQ1 models,
  thresholds, evaluator, manifest specifications and selection rules, software
  environment, and hashes are frozen. That pass creates the realized frozen
  replay manifests; later HTTP runs consume only those manifests.
- The aggregate [preparation record](../../reports/phiusiil-preparation-summary.json)
  under source schema version 2
  reports source and output hashes, split counts, class counts, and every
  quarantine reason without publishing row-level data.
- The [`phiusiil-development-v1`](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/phiusiil-development-v1)
  GitHub Release is the source-freeze record for the completed preparation
  milestone. It keeps the exact licensed UCI archive outside Git history and
  binds it to the archive and CSV checksums in `data/sources.json`.
- PhishVN is reserved for one external evaluation after the protocol, data
  pipeline, models, thresholds, and analysis code are frozen.
- H1, H2, and H3 remain undecided until their stated evidence and gates have
  been evaluated. Exploratory v2 results do not decide them.

## Execution Audit

On 2026-09-03, a broad local repository text search displayed row content from
the ignored PhiUSIIL `group_test.jsonl` file. This is recorded as analyst
access. The file was not accepted or read by the baseline command, no
group-test prediction or metric was produced, and the exposure did not change
the frozen contract, implementation, threshold rule, or hypothesis status. The
detailed record and mitigation are in the
[research evidence outline](../research-evidence-outline.md). No PhishVN record
was accessed.

When generated, `access.group_test_accessed=false` in a baseline artifact
describes only the `fit-baselines` process input boundary. It does not negate
the analyst access recorded here.

The immutable v1.4 failure record is protocol v1.4 SHA-256
`2c2956e7cf958f9d2d948a2b1b665e214e12b175d84e4b73c766cc0a6e3be4de`,
contract `rq1-baselines-v1` SHA-256
`594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4`,
contract commit `c79e8aefb47560c6ae982dbd5848cf2b707c99a4`, implementation
commit `e535586c6162a306a8dac7a5a6546f55dc09136f`, and failure-record
commit `67107874b9e46457ed710db42f050e35c1ca5ea2`. The prescribed run
started on 2026-09-03 and stopped on 2026-09-04 after approximately eight hours
and forty minutes. The full-feature `Logistic-L1` fit reached `max_iter=5000`
at `tol=1e-8`, and the command exited with
`error: Logistic-L1 did not converge`. Atomic publication left no model
directory or summary file. The command accepted the pinned training,
validation, preparation-summary, and contract inputs only; it accepted no
group-test or PhishVN input. No model, threshold, or metric from the attempt was
reviewed or used as research evidence.

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

Protocol v1.5 froze a prospective two-model SAGA diagnostic, executed twice in
fresh processes with training data only. Both v1.5 fresh runs completed the
length-only fit at `n_iter=69` with the same fitted-state SHA-256, then stopped
while scoring `Logistic-L1` with `RuntimeWarning: divide by zero encountered in
matmul`. The `Logistic-L1` aggregate was not produced. The exact aggregate
receipt is
[`reports/rq1-saga-convergence-v1-execution.json`](../../reports/rq1-saga-convergence-v1-execution.json),
SHA-256 `7309f52f704150f85e6c17d44d96adcde917d2539c7b75264bf775ccec3aa6f4`.
The v1.5 SAGA diagnostic has status `stopped_platform_warning` in the execution
record; it is not a baseline result or RQ/H evidence.

Protocol v1.6 froze the same training-only, two-model SAGA diagnostic under a
narrow scoring audit. Scaling and fitting warnings remained fatal. On macOS
arm64 with the NumPy Accelerate BLAS only, one of three exact `RuntimeWarning`
messages from `sklearn.utils.extmath` could be captured during
`decision_function` or `predict_proba`. The warning had to be recorded, and the
scikit-learn outputs had to be finite and agree with independent float64
`einsum` and full two-column `expit` probability calculations at `rtol=1e-12`
and `atol=1e-12`.

The exact v1.6 receipt is
[`reports/rq1-saga-convergence-v2-execution.json`](../../reports/rq1-saga-convergence-v2-execution.json),
SHA-256 `487714ea17a095e369d381da5a27452f3b263f1be1a05db2ebe061eecdefebdf`,
from clean commit `69a67d4e5cb81d49009d6a90e87f4c0c5f2cea87`. Both v1.6 fresh
runs passed with identical iteration counts and fitted-state SHA-256 values:
`length-only` stopped at `n_iter=69`, and `Logistic-L1` stopped at
`n_iter=4783`. Each full-model run recorded the same six allowlisted scoring
warnings. The maximum absolute decision and full-matrix probability differences
were `4.263256414560601e-14` and `5.551115123125783e-16`, respectively. The
v1.6 SAGA diagnostic has status `passed_training_only`; this establishes
training feasibility only.

Protocol v1.7 froze `rq1-baselines-v2` before validation. The command then ran
once from clean commit `7ae6c9af85e935c551468f590a7ba43441f58def`, after CI passed.
The exact aggregate [summary](../../reports/rq1-baseline-v2-summary.json) has
SHA-256 `bf5b3a6f0fc705d26852da4dd0053c6111ffc3e500d7a2e95dfba5ad859b279c`.
The rq1-baselines-v2 execution has status `completed_development_validation`.
The private `length-only` artifact has SHA-256
`b8b92cfbe29160e769e5e7d80712becc8fc0680cdfd45a44b839ef9bada87799`;
the private `Logistic-L1` artifact has SHA-256
`71a3e24a0283a31ba188bc7dd60b18c1b708370b9ca275d5ab1a1004680c968a`.

| Model | `n_iter` | Threshold | Validation recall | Observed FPR | One-sided 95% FPR upper bound |
|---|---:|---:|---:|---:|---:|
| `length-only` | 69 | 0.7612031186147 | 0.3214800576645843 | 0.006680191993666189 | 0.007702192373035135 |
| `Logistic-L1` | 4783 | 0.2670846328466124 | 0.9842223290084895 | 0.008758473947251225 | 0.009915480854183582 |

The length-only scoring audit recorded no warning and zero reference
difference. The full model recorded six allowlisted scoring warnings; its
finite decision values and full probability matrix matched the independent
reference with maximum absolute differences of `2.1316282072803006e-14` and
`3.3306690738754696e-16`. Both thresholds have status `selected`. These are
validation-set operating points, not group-test, external, or hypothesis
results. The group test remains analyst-exposed but model-unscored. The
summary's `access.group_test_accessed=false` records the process input boundary
and does not negate the earlier analyst exposure. No PhishVN record was
accessed, and H1, H2, and H3 remain undecided.

## Change Record

| Date | Version | Change |
|---|---|---|
| 2026-09-01 | 1.1 | Defined dataset-specific label mappings, quarantine rules, PhishVN v4 evidence strata, the H3 request-error construct, future-only routing, and the bounded contribution. |
| 2026-09-03 | 1.2 | Recorded active implementation under the August 20 direction and froze the PhiUSIIL canonicalization and label-blind domain-allocation algorithms before the development-data run. |
| 2026-09-03 | 1.3 | Clarified the bounded contribution claim, source-derived label provenance, and study-defined target basis; recorded completed preparation and the next baseline milestone. No experiment was run, and no RQ/H method or decision rule changed. |
| 2026-09-03 | 1.4 | Froze the exact RQ1 feature vector, shared logistic configuration, partition use, convergence handling, score meaning, and validation threshold algorithm before fitting. The feature extractor was implemented and tested; no model was fitted, no threshold was selected, and no RQ/H decision rule changed. |
| 2026-09-04 | 1.5 | Recorded the group test as analyst-exposed but model-unscored with one later frozen noninteractive raw-partition pass; preserved the failed `tol=1e-8` baseline, qualified the `tol=1e-4` work as a provenance-incomplete tolerance observation, and froze the SAGA diagnostic before use. No baseline result was accepted, and no research question, hypothesis, evidence designation, or decision gate changed. |
| 2026-09-04 | 1.6 | Recorded the v1.5 diagnostic failure on macOS arm64 with Accelerate and froze a scoring-only warning audit with an independent numerical reference. No baseline result was accepted, and no RQ/H decision rule changed. |
| 2026-09-04 | 1.7 | Recorded that the v1.6 diagnostic passed its prespecified checks as training feasibility only; `rq1-baselines-v2`, SAGA, and the validation scoring audit are frozen before validation. No baseline result was accepted, and no RQ/H decision rule changed. |

Any change to a research question, hypothesis, evidence designation, method,
or decision rule increments the protocol version and records a new hash before
the affected analysis runs.
