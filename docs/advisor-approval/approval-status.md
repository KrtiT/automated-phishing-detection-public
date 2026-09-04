# Research Status

**Status:** The August 20 advisor report directed continuation of the
three-question design with objective, reproducible outcome criteria. The
PhiUSIIL development split and aggregate preparation record are complete. The
`rq1-baselines-v1` contract remains frozen. Its prescribed fit stopped with a
convergence warning and published no model or summary. A later local tolerance
observation is provenance-incomplete and is not research evidence. Protocol
v1.5 freezes a prospective SAGA
convergence diagnostic, which is `not_run`. No baseline model, threshold, or
validation result has been accepted. H1, H2, and H3 remain undecided. No
PhishVN record has been accessed.

| Field | Value |
|---|---|
| Protocol | `docs/advisor-approval/2026-08-16-realignment-matrix.md` |
| Protocol version | `1.5` |
| Protocol date | `2026-09-04` |
| Protocol SHA-256 | `f8579b2e5d85e6de19b2abc653aa9b128715c49c8a3a2fca9f384c2f52d118b6` |
| RQ1 baseline contract | `rq1-baselines-v1` |
| RQ1 baseline contract SHA-256 | `594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4` |
| Development source schema | `2` |
| Source-freeze release tag | `phiusiil-development-v1` |
| Source-freeze release | [`phiusiil-development-v1`](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/phiusiil-development-v1) |
| Development preparation | `complete` |
| Current technical milestone | Protocol v1.5 diagnostic amendment `frozen`; v1.4 baseline attempt `stopped_nonconverged`; exploratory tolerance observation `not_accepted_provenance_incomplete`; prospective SAGA convergence diagnostic `not_run`; no baseline model, threshold, or validation result accepted. |
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
- `rq1-baselines-v1` freezes the 25 raw-URL features, their order and
  denominators, the shared logistic pipeline, partition use, and threshold
  rule before fitting. Its SHA-256 is recorded above.
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

Protocol v1.5 freezes a prospective two-model SAGA diagnostic, executed twice
in fresh processes with training data only. `length-only` uses its one declared
feature, and `Logistic-L1` uses all 25. Both share L1 SAGA with `C=1.0`, balanced
class weights, an intercept, `max_iter=5000`, `tol=1e-4`, and seed `42`. The
diagnostic accepts no validation, group-test, or PhishVN input and publishes no
model or summary. Both models in both fresh-process runs must finish without a
warning, satisfy `0 < n_iter < 5000`, have the declared class and parameter
shapes, and produce only finite scaler values, parameters, decision scores, and
probabilities. The second run must match the first run's iteration counts and
fitted-state SHA-256 values. The prospective SAGA convergence diagnostic is
`not_run`. No baseline model, threshold, or validation result has been
accepted, and H1, H2, and H3 remain undecided.

## Change Record

| Date | Version | Change |
|---|---|---|
| 2026-09-01 | 1.1 | Defined dataset-specific label mappings, quarantine rules, PhishVN v4 evidence strata, the H3 request-error construct, future-only routing, and the bounded contribution. |
| 2026-09-03 | 1.2 | Recorded active implementation under the August 20 direction and froze the PhiUSIIL canonicalization and label-blind domain-allocation algorithms before the development-data run. |
| 2026-09-03 | 1.3 | Clarified the bounded contribution claim, source-derived label provenance, and study-defined target basis; recorded completed preparation and the next baseline milestone. No experiment was run, and no RQ/H method or decision rule changed. |
| 2026-09-03 | 1.4 | Froze the exact RQ1 feature vector, shared logistic configuration, partition use, convergence handling, score meaning, and validation threshold algorithm before fitting. The feature extractor was implemented and tested; no model was fitted, no threshold was selected, and no RQ/H decision rule changed. |
| 2026-09-04 | 1.5 | Recorded the group test as analyst-exposed but model-unscored with one later frozen noninteractive raw-partition pass; preserved the failed `tol=1e-8` baseline, qualified the `tol=1e-4` work as a provenance-incomplete tolerance observation, and froze the SAGA diagnostic before use. No baseline result was accepted, and no research question, hypothesis, evidence designation, or decision gate changed. |

Any change to a research question, hypothesis, evidence designation, method,
or decision rule increments the protocol version and records a new hash before
the affected analysis runs.
