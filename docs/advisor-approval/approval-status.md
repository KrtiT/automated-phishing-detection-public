# Research Status

**Status:** The September 3 advisor report, SHA-256
`b72da89a4cc8a5b06f6ca88d79fe78dd54e3199a96b7450209ea53b4a4c04215`,
directed the study to complete and freeze the source-provenance release, then
conduct systematic hypothesis testing with all gates, thresholds, features,
and train/validation procedures locked before test results, with particular
attention to H1 and the GMM. The public `phiusiil-development-v1` release
completed that requested source freeze after the meeting. Protocol v1.7 froze
`rq1-baselines-v2` before validation; its single execution from clean commit
`7ae6c9af85e935c551468f590a7ba43441f58def` is
`completed_development_validation`. This is development validation only.
Protocol v1.8 froze `rq1-transformer-cascade-v1`; it was never fit and is now
`superseded_unrun`. Protocol v1.9 freezes the publication-only amendment
`rq1-transformer-cascade-v2` at contract status `frozen_not_run`. The procedure
code, tests, CLI, and private/public artifact publication code are complete.
Its historical pre-execution implementation status was
`frozen_implemented_not_run`, and its first execution stopped with status
`stopped_stage_one_integrity_check`. The controlled retry from producer commit
`e866441f2ff858472d031b8d358fd469897c6a65` subsequently completed development
validation. Its accepted public summary has SHA-256
`41499aa388babe60442de7231b4087f67a53f96f340568a7cc58a3268606a2fd`.
Repository commits
`a8ee067bda8fd45d19f5c4b794ba21f58d1947fc` and
`0793ca3dbc36e49b561cd0ac74968a4644060426` record the initial implementation
and review hardening; they are not performance or result runs. The implementation
and its tests did not open the PhiUSIIL group-test partition or PhishVN. GMM
execution is `completed_development_validation`; its independent false-alert
audit failed the 5% gate (28/252 windows, 11.11%), without retuning.
H2 is not supported because that mandatory conjunctive gate failed. H1 and H3
remain undecided. The group test remains analyst-exposed but model-unscored.

The September 18 implementation adds no-fit scoring for all four detectors,
paired domain-clustered recall intervals, and label-blind future-only policy
replay, checked with synthetic fixtures. The shared singleton scoring core
skips transformer execution outside the band unless a prior alert overrides
routing, and counts forward attempts separately from successful scores. The
[evaluation contract](../../data/evaluation-contract-v1.json) specifies a
candidate offline/service runtime and a no-fit development compatibility
comparison. The [first attempt](../../reports/inference-compatibility-v1.json)
stopped during candidate environment preflight, before model or validation
reads. A fresh-process check identified PyTorch thread-initialization ordering;
the contract records one explicit corrective execution with a separate receipt,
unchanged numerical rules, and no automatic retry. That
[comparison](../../reports/inference-compatibility-v1-preflight-correction.json)
completed with status `not_equivalent`: one transformer decision differed among
32,695 validation records. Length-only, Logistic-L1, cascade decisions, inclusive
band membership, and all calibration/audit GMM alerts matched. The candidate
runtime does not satisfy the frozen zero-mismatch requirement and is not
accepted as equivalent. No threshold or original result changed; no further execution is
authorized by this correction. Later supplements specify manifest, operational,
and secondary-analysis conventions. The original contract remains `prospective_incomplete`;
this development comparison supplies no group-test, external, HTTP, or new
hypothesis result. The [implementation outline](../research-evidence-outline.md#remaining-executable-work)
records the next steps without changing the frozen development results.

The September 21 [inference-method amendment](../../data/singleton-inference-amendment-v1.json)
adopts that same singleton convention for future evaluation and serving, rather
than seeking a different runtime that passes the comparison. This is a
development-informed method change, not a passed bridge. Historically selected
weights, thresholds and band are carried forward without recalibration; their
optimality under singleton scoring is not claimed. The executed contract,
both comparison receipts and original development summaries remain unchanged.
The full pre-access freeze remains incomplete, and no new scoring run follows
from this amendment.

Saved-evidence evaluation now computes the six primary paired contrasts and
all 22 H1/H2/H3 component gates. It reports class-specific counts, exact FPR
upper bounds and separate label-free Tranco control-alert rates. Final rate
gates use observed rates, not the upper bounds. It rejects misaligned records,
invalid labels and overlapping external outcome strata. Missing or
non-estimable evidence cannot pass; the original 28/252 audit still establishes
H2 non-support while its remaining measurements are incomplete. These are
tested implementation rules, not newly observed hypothesis results. The
standalone scorer also has a synthetic fresh-owner-thread startup regression.

The September 21 HTTP increment adds the owner-thread FastAPI service and a
real-TCP replay client. Synthetic tests cover queue capacity, late arrivals,
timeouts, failed forwards, cleanup and separate warmup/measured counters.
The client retains failed attempts and pools individual measured latencies;
small fixture runs cannot produce a primary H3 summary. The
[HTTP supplement](../../data/http-replay-contract-v1.json) fixes the queue,
deadline, error, phase-fence and run-order conventions. The first primary
concurrency-1 run is the designated physical-invocation reference, so a later
repeat cannot be chosen for a more favorable count.

The [manifest supplement](../../data/evaluation-manifest-contract-v1.json)
specifies deterministic classwise selection, independent sampling streams,
fixed replay order and private-payload hashing. The saved-score integration now
routes every retained external row before selecting gold, certified and
label-free Tranco outcomes; secondary rows remain in the routing denominator.
These paths have been tested with synthetic inputs only. No protected record
was opened, no compatibility run was repeated and no operational result is
claimed. The original hypothesis status is unchanged.

The execution-binding increment authenticates the reviewed clean commit,
working-tree source bytes, imported package paths, accepted public summaries,
dependency versions and declared Mac/MPS hardware before loading artifacts.
The fixed artifact loader checks the accepted length-only, Logistic-L1,
transformer/cascade and GMM identities; it exposes no refit or alternate-device
option. The HTTP composition loads and checks them on its single owner thread,
then checks the binding again after numerical settings are restored. A separate
metadata-only command performs no model or data read. Durable attempt records
reserve identities before execution and preserve failures instead of overwriting
them. These controls do not themselves constitute a research execution.

The [workload supplement](../../data/operational-workloads-v1.json) specifies
the separate transformer-only comparator and serialized live-monitor workload.
Transformer-only HTTP scoring is implemented and tested without stage-one
inference. The September 21 live-monitor increment implements the separate
serialized shift workload, including actual singleton GMM computation, warmup
state reset, timeout drains, physical counters and private traces. An alert
ending at position t affects only t+1 through t+256. A real-socket synthetic
test matches live decisions and windows to offline replay. Neither workload
replaces the fixed-cascade concurrency-64 H3 benchmark. The
[shift supplement](../../data/shift-execution-contract-v1.json) records these
controls and the remaining process-level checks.

The internal producer now validates supplied partition bytes and constructs
paired scores, monitor features, primary evaluation and all three replay
manifests in memory. It rejects unopened, closed or wrong-thread scoring
sessions before inference. The external adapter applies publisher-label,
duplicate, domain-overlap and quarantine rules to normalized synthetic rows.
The September 22 internal runner adds public source/preparation authentication,
durable reservation before supplied PSL, model-artifact or partition access,
safe single-read handling of PSL and partition bytes, and publication after owner
teardown and final binding checks. The parent process
accepts outputs only after the worker exits successfully and its receipt and
saved files verify. Synthetic tests exercise these boundaries. Both process
entry points remain closed under the current incomplete execution profile;
there is no override flag. External publisher-schema verification and complete
secondary-output integration remain separate work.

The [secondary-analysis supplement](../../data/secondary-analysis-contract-v1.json)
fixes score metrics, calibration bins, low-FPR recall, exact McNemar contrasts
and the four-comparison Holm family. Those calculations are implemented and
tested. The [secondary-development contract](../../data/secondary-development-contract-v1.json)
now specifies the original-scaler MMD/PSI composition and the fixed formatting,
label-permutation and Random Forest fits, including numerical scoring and
separate validation cutoffs. Its SHA-256 is
`f592352593ae64b178a468d5800e780267275a62046a05b31b952f69e424f44f`.
Synthetic tests exercise these paths, portable
model round trips, training-reference separation and the original calibration/
audit allocation. The first research attempt is recorded below. The three
independent URL transformations are implemented;
their full detector/monitor comparison remains to be connected.

The separate [development execution profile](../../data/development-execution-contract-v1.json),
SHA-256 `67146228d636c16f02484998741c7a1545da68b209e693620efab22b2676cd43`,
now binds the accepted training/validation sources and the secondary-development
contract to the unchanged runtime. Its runner reserves an attempt before input
access, freezes training references before reading validation, and saves the
drift comparison and each of the seven tabular fits individually. A failure
preserves earlier outputs and the failed or incomplete attempt; later members
remain unattempted. There is no automatic retry or partial-success marker.
The parent checks the actual worker exit, receipts, output identities and hashes,
and recomputes CP thresholds, AP/AUC and drift calibration/alerts from retained
evidence. It does not refit models or reopen source records.

The September 22 [first attempt](../../reports/secondary-development-v1-attempt-1.json)
ran once from CI-passed commit `61ec4f74716ecfb76fed0638a0f7b3defef6bfb6`.
Drift, formatting and all five permutation controls wrote their child evidence.
Random Forest then stopped with `SecondaryTabularError`; the parent exited 2,
and no aggregate success marker was published. No retry or resume occurred.
The specific failed check and fitted RF state were not retained, so the receipt
alone does not establish the cause. The public record preserves every completed
child summary as preliminary at that stop, not independently accepted output
from the failed original family.

Those summaries report MMD 8/252 and PSI 11/252 audit-window alerts, not URL FPR.
Formatting and some permutation controls show appreciable validation ranking;
the full five-seed results remain visible. This is an unresolved control
observation, not proof of leakage or successful negative-control validation.
It changes no primary model, cutoff, hypothesis gate or original GMM result.

The separate [correction profile](../../data/development-correction-contract-v1.json),
SHA-256 `61739fa0638ae822bf54639cf3a485c0b7b8a3a236a80221d824df7f79a090a9`,
records direct stored-leaf RF arithmetic with the same exact-parity requirement.
The correction runner first audits the seven retained members without refitting,
then allows one RF fit in a fresh attempt. Fitted state precedes later checks;
safe check identifiers and actual worker exits remain in failed-attempt records.
Historical fit-label digests remain unavailable. The [first correction attempt](../../reports/secondary-development-correction-v1-attempt-1.json)
exited 2 in the audit worker and parent before scientific checks or fitting.
Three original formatting-summary floats `0.0` had become integers `0` in the
pinned accounting copy. A receipt-only diagnosis reproduced the exact original
marker hash by restoring just those representations. The failed receipt itself
retained only a generic audit identifier; its diagnosis is reported separately.
The [v2 execution profile](../../data/development-correction-contract-v2.json),
SHA-256 `1d806d536dc77b5a085264950e64b2bde3db4ab02afa33c6adb827d8b7a07f73`,
binds that failure and the unchanged v1 method. It authenticates original marker
bytes, permits only the three declared real-zero normalizations in accounting,
and uses original summaries for all scientific checks. Specific audit identifiers
are now retained. A fresh audit and conditional RF fit follow verification;
neither failed attempt is resumed or promoted to success.

The September 23 [v2 execution](../../reports/secondary-development-correction-v2-summary.json)
completed from CI-passed `f1bebca93aba38b19a208026242c14c8285c27eb`.
Both workers and the parent exited 0, and the final receipt/stage linkage passed
verification. The seven retained members passed their specified no-fit audit.
The single corrected RF fit passed exact parity and retained its fitted-state
checkpoint. At its selected validation cutoff 0.2, RF detected 12,349/12,486
positives with 157/20,209 false positives: recall 98.9028%, observed FPR 0.7769%,
and one-sided 95% CP upper bound 0.8864%. AP was 0.995444 and ROC AUC 0.995279.
These are selected development operating-point results, not held-out or
deployment guarantees. Control provenance limits and the unresolved ranking
variation remain; the failed original aggregate is not retrospectively accepted.
The public execution report SHA-256 is
`663f1117cd33f949b70c35c56810764193f69cae3a37505004d0db27641d829d`.

Additional transformer seeds 43-46 still
need their training runtime, checkpoint-scoring and comparable secondary
calibration contract; seed 42 primary weights and cutoffs remain unchanged.
No additional fitting follows from the no-fit singleton amendment.

Execution binding now uses the fixed
[v2 profile](../../data/execution-binding-contract-v2.json), SHA-256
`887f771381927dfe1b9268a45f4e605baf3e9a7caee2b7005cdfe68b1be516e1`,
which adds the shift/secondary-analysis supplements and the unchanged v1 profile
to its public pins. It remains unchanged; the separate development profile adds
the secondary-development binding without enabling protected evaluation.
The runtime and historical pins are unchanged. External file/schema integration,
remaining secondary development procedures and complete output coverage still
precede the final pre-access review. The development attempt read only the
accepted training/validation, PSL, Logistic-L1 and GMM inputs. It accessed no
protected records. Its original family result remains unaccepted; the separate
September 23 execution accepts the audited members and corrected RF as
development evidence, with the limitations stated above.

The preflight-stop receipt has SHA-256
`6f27b23a88e40455a186ce21cddebec0f9ab827919c663b345a6a14ca14bb670`.
Its `input_sha256` values are declared expected bindings; the stop preceded
verification or decoding of those model and validation inputs.

The completed comparison receipt has SHA-256
`66272dceee640f6b1a7f42f90a5776db4f30673df09660003a4b4250cd286eb5`.
It ran from clean commit `e7483f05ef245da4eeed713342bb41f8b4a4ab13` after
CI run `35424248455` passed. Both scoring processes exited successfully; the
parent returned code 2 because exact compatibility was not established.

| Field | Value |
|---|---|
| Protocol | `docs/advisor-approval/2026-08-16-realignment-matrix.md` |
| Protocol version | `1.10` |
| Protocol date | `2026-09-17` |
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
| Initial transformer/cascade implementation repository commit | `a8ee067bda8fd45d19f5c4b794ba21f58d1947fc` |
| Review-hardening repository commit | `0793ca3dbc36e49b561cd0ac74968a4644060426` |
| Transformer/cascade retry producer commit | `e866441f2ff858472d031b8d358fd469897c6a65` |
| Transformer bundle verifier commit | `ef4e4567df979fef3afc91f8ad8097691f94d1ad` |
| Accepted transformer/cascade summary | [`reports/rq1-transformer-cascade-v2-summary.json`](../../reports/rq1-transformer-cascade-v2-summary.json) |
| Accepted transformer/cascade summary SHA-256 | `41499aa388babe60442de7231b4087f67a53f96f340568a7cc58a3268606a2fd` |
| Accepted transformer/cascade retry receipt | [`reports/rq1-transformer-cascade-v2-retry-execution.json`](../../reports/rq1-transformer-cascade-v2-retry-execution.json) |
| Accepted transformer/cascade retry receipt SHA-256 | `9be2519db4782fe71840235f81e58c4382e504d98a9d888512271107b16e4957` |
| Historical RQ1 baseline contract | `rq1-baselines-v1` |
| Historical RQ1 baseline contract SHA-256 | `594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4` |
| Development source schema | `2` |
| Source-freeze release tag | `phiusiil-development-v1` |
| Source-freeze release | [`phiusiil-development-v1`](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/phiusiil-development-v1) |
| Development preparation | `complete` |
| Current technical milestone | Baseline v2, transformer/cascade, and GMM development validation complete. The transformer bundle is independently verified; the GMM false-alert gate failed at 28/252 windows. Historical attempts remain recorded below. |
| Current hypothesis status | H1 `undecided`; H2 `not_supported`; H3 `undecided` |
| Legacy base tag | `legacy-v2-clean-2026-08-16` |
| Legacy base SHA | `a5eceecf21ad5ce29c4ab8f8d4de0edc8b73b240` |
| Advisor report date | `2026-09-03` |
| Advisor report SHA-256 | `b72da89a4cc8a5b06f6ca88d79fe78dd54e3199a96b7450209ea53b4a4c04215` |
| Direction | Complete and freeze the source-provenance release, then conduct systematic hypothesis testing with all gates, thresholds, features, and train/validation procedures locked before test results, with particular attention to H1 and the GMM. |

Protocol v1.10 incorporates unchanged transformer v2 and freezes
`rq2-gmm-development-v1` before GMM execution. The contract and matrix retain
their freeze-time status; the dated execution record below reports the later
fit and failed false-alert gate. Synthetic-only preflight
exposed an Accelerate warning; the unchanged NumPy 2.2.6 version with
`scipy-openblas` 0.3.29 passed all six synthetic candidate fits. The GMM runtime
therefore pins that backend and rejects Accelerate before input reads, rather
than extending the warning allowlist. No research-data fit informed this choice.

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
- `rq1-transformer-cascade-v2`, SHA-256
  `686c0d86b33b8a6c2e09cd6e174003db0bd2f7c30b087faf5470e6a270524213`,
  preserves v1's scientific and artifact-content rules and replaces only its
  contract identity, schema version, protocol version, and publication semantics.
  Its date is unchanged. Each destination uses a
  temporary path in its own parent and an atomic no-replace install. The private
  directory is installed first; the public summary is installed last and is the
  completion marker. A completed result requires both. A caught in-process
  `BaseException` removes only destinations created by the run. Publication is
  not cross-destination atomic, so abrupt process or host failure can leave the
  private directory without the public summary. Either one-sided state is
  `incomplete_not_result`; the pipeline leaves it unchanged until the operator
  verifies it is stale and removes it before rerun. V1 remains byte-preserved at
  SHA-256 `aeaa84534c4cadf0459cf6d2f010dc802684d4801cce563ce18242f36359fb54`
  with status `superseded_unrun`. V2 retains its freeze-time contract status
  `frozen_not_run`; the implementation's historical pre-execution status was
  `frozen_implemented_not_run`, and its first execution stopped with
  `stopped_stage_one_integrity_check`. The later retry is
  `completed_development_validation`; its accepted summary and receipt are
  linked above.
- `rq2-gmm-development-v1` freezes training-only 26-column scaling, six
  diagonal-GMM fits selected by training BIC, the label-blind validation-domain
  allocation, the linear calibration quantile, and the independent audit gate.
  Its SHA-256 is `22d32088b05e74432704f9671ab76ba28b4f573ead418846b23bc366315cb393`.
  The audit cannot retune the boundary; a failed audit gate is still a completed
  development-validation run, not H2 evidence.
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
- H1 and H3 remain undecided. H2 is not supported because the frozen monitor's
  mandatory independent false-alert gate failed; later characterization cannot
  convert that failed conjunctive gate into support.

Manual review is permitted only as separately reported post hoc descriptive
error analysis and cannot assign or override labels, change quarantine or
inclusion, thresholds, features, model or procedure choices, gates, or
hypothesis decisions.

## Execution Audit

On 2026-09-03, a broad local repository text search displayed row content from
the ignored PhiUSIIL `group_test.jsonl` file. This is recorded as analyst
access. The file was not accepted or read by the baseline command, no
group-test prediction or metric was produced, and the exposure did not change
the frozen contract, implementation, threshold rule, or hypothesis status. The
detailed record and mitigation are in the
[research evidence outline](../research-evidence-outline.md). On 2026-09-09, a
second broad local wording search displayed row content from that ignored file.
The displayed rows informed no model, threshold, gate, routing, or
scientific-procedure change; the separate v2 transformer publication correction
arose from code review and changed no scientific field. No fit, score, metric,
or PhishVN access occurred. The partition remains analyst-exposed but
model-unscored.

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
directory or summary file; its status is `stopped_nonconverged`.
The command accepted the pinned training,
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
accessed. At that September 4 baseline milestone, H1, H2, and H3 were
undecided.

Protocol v1.9 freezes the transformer and selective cascade procedure under
`rq1-transformer-cascade-v2`. The contract accepts only the pinned training,
validation, preparation-summary, baseline-contract, and `Logistic-L1` artifact
roles. It accepts no group-test, external, PhishVN, runtime-tuning, or test-path
input. Repository commits `a8ee067bda8fd45d19f5c4b794ba21f58d1947fc`
and `0793ca3dbc36e49b561cd0ac74968a4644060426` record the initial implementation
and review hardening. The v2 amendment changes publication semantics, not the
model or scientific procedure. The pre-execution implementation was
`frozen_implemented_not_run`; its first execution later stopped with
`stopped_stage_one_integrity_check`.
The controlled retry completed development validation and is recorded below.
The GMM execution is recorded separately after it.

The official transformer run started on 2026-09-17 at 18:21:12 UTC from clean
detached commit `c3a5c815b20121f1ddd06a2f316f904077c00c4f`, after its
[CI run](https://github.com/KrtiT/automated-phishing-detection-public/actions/runs/35257955477)
passed. It stopped at 19:24:27 UTC with exit code 2 after 3,795.28 seconds:
`stage-one artifact threshold does not match the supplied scores`.
The [execution record](../../reports/rq1-transformer-cascade-v2-execution.json)
has status `stopped_stage_one_integrity_check`. No output directory or public
summary was produced by that attempt; it was not an accepted result.

A separate no-fit diagnostic read only the hash-pinned development-validation
partition and accepted Logistic-L1 artifact. The portable scorer used `einsum`
and a C-contiguous feature matrix instead of the original scikit-learn scorer
and Fortran-contiguous matrix. This changed exact score ties: 11,210 threshold
candidates rather than the accepted 11,279. Its threshold difference also
exceeded the recorded numerical audit, although decisions at the unchanged
accepted threshold were identical. Reconstructing the original scorer and
layout reproduced the full threshold record and warning audit exactly.
Commit `cc7edd66409473e7aba5c40c846135699aa1e192` restores that path without
loosening candidate/count checks or tolerances. It moves stage-one binding
before vocabulary construction, tensors, and transformer training. GMM retains
its frozen portable scorer. The stopped attempt remains part of the record.

#### Controlled Retry

The retry started at `2026-09-17T20:42:45Z` from clean, published commit
`e866441f2ff858472d031b8d358fd469897c6a65`, after
[CI passed](https://github.com/KrtiT/automated-phishing-detection-public/actions/runs/35272134401).
The full local suite passed 601 tests; a separate 124-test check passed on
the original Accelerate execution environment. These checks verify software,
not scientific outcomes.

Before the retry, a hash-checked validation-only preflight returned
`passed_no_fit_validation_only`. The complete threshold record and numerical
warning audit matched the accepted baseline exactly, including all 11,279
threshold candidates and the unchanged threshold `0.2670846328466124`.
The preflight script SHA-256 is
`d0949ddbf40fb85b4253f7010eb3b85b3245a04d88eeadb5ada4755b09cf5d37`;
its aggregate receipt SHA-256 is
`ba92ef7432b52e8222d30b9df71a13b04d7629ec0204a281c2f44f7eb26df3df`.
No baseline was refit. The transformer contract, seed, training and selection
rules, and output destinations are unchanged. The environment lock SHA-256 is
`7912cb1be00e009b0fcbaa506fa6eb838c7d4a4743be6e87dc687469b5ccb1af`.
Its only change from the first attempt promotes the already installed
`threadpoolctl==3.6.0` from a transitive to a direct dependency; all installed
dependency versions match the original environment.

At `2026-09-17T20:43:08Z`, the retry was recorded as
`running_development_validation`; that is a dated historical status, not the
current result. The run finished at `2026-09-17T22:19:28Z` after 5,802.94
seconds with exit code 0. The
[retry receipt](../../reports/rq1-transformer-cascade-v2-retry-execution.json),
SHA-256
`9be2519db4782fe71840235f81e58c4382e504d98a9d888512271107b16e4957`,
records `status=completed_development_validation`, `result_accepted=true`, and
`hypotheses_decided_by_this_run=[]`. The original stopped-run record is
preserved byte-for-byte at SHA-256
`2440e4fef8c9035eae702fecad1afb300fe9c3c487c625c8c6e22dff6e4c7786`.
No group-test or PhishVN input was accepted.

The accepted public
[summary](../../reports/rq1-transformer-cascade-v2-summary.json), SHA-256
`41499aa388babe60442de7231b4087f67a53f96f340568a7cc58a3268606a2fd`,
contains development-validation aggregates only.

| Model | Validation recall | Observed FPR | One-sided 95% FPR upper bound | Logical transformer selections |
|---|---:|---:|---:|---:|
| `length-only` | 32.1480% | 0.6680% | 0.7702% | n/a |
| `Logistic-L1` | 98.4222% | 0.8758% | 0.9915% | n/a |
| transformer | 99.1030% | 0.8808% | 0.9968% | 32,695 of 32,695 |
| fixed cascade | 98.4222% | 0.8758% | 0.9915% | 3 of 32,695 |

The cascade's validation recall was identical to `Logistic-L1`; the two rows
have the same aggregate confusion counts. Its logical routing mask selected
escalation for 3 of 32,695 validation rows. Calibration computed transformer
scores for every validation row. This does not decide H1 and does
not establish measured HTTP savings. The required paired domain-clustered
internal and external comparisons and physical selective-execution measurements
have not run.

The public summary pins the four private bundle members:

| Private member | SHA-256 |
|---|---|
| `cascade.json` | `7ac88c784dbc299d436a029904f0c6bde5a8cf741e62028a7635e8672e55119c` |
| `transformer-weights.npz` | `1d4cdef31cb23cb84f093ca61c0afe0318142fa45ae559acd78cf10b49ee5de7` |
| `transformer.json` | `a13b6b7d554db6a9ee5b1689ecef44e2ad8069fcd25967f931101bdd3b256727` |
| `vocabulary.json` | `68bda780006d07b3b849abc366fbe3ccffe8a29e8984b092396d04f4eef43579` |

Reviewed verifier commit `ef4e4567df979fef3afc91f8ad8097691f94d1ad`
passed its [CI run](https://github.com/KrtiT/automated-phishing-detection-public/actions/runs/35310546331),
the 652-test full suite, and the 50 focused loader/CLI tests. The named-bundle
audit returned `verified_artifact_bundle` on MPS. It verified directory mode
`0700`, file modes `0600`, all four hashes, public/private projections,
producer-receipt hashes, and execution-stdout equality. It also checked that
best epoch 5 of 10 follows the frozen minimum-delta and patience rule, the
positive class weight matches the training counts, and all six stage-one
warnings are preserved. The verification invocation performed no fit, and no
research rows were read or scored.

### GMM Development Execution

The single prescribed run started on September 17 at 18:49:07 UTC and finished
at 18:50:39 UTC with exit code 0 (92.78 seconds). It used clean, published
commit `9983bfddc6ea31e370629cb1b2efd3da383f7ff0`, after its
[CI run](https://github.com/KrtiT/automated-phishing-detection-public/actions/runs/35260883392)
passed. Commit `df0e37f` records the preceding method freeze. The exact
[aggregate summary](../../reports/rq2-gmm-development-v1-summary.json) has SHA-256
`6f695138a302e854e1e5af590152e289486affe8ccdf75510ca9a5dcaad3b523`.

All six candidates converged without warnings; training BIC selected six
components. Each validation stream contained 14,783 domains and 252 complete
windows. The calibration stream contained 16,325 rows and audit contained
16,370. The calibration boundary was `-67.45792380813624`; the independent
audit alerted on 28/252 windows (11.11%). This exceeds the prespecified 5%
limit, so `false_alert_gate_met=false`. The boundary was not retuned and the
run was not repeated. Fitted parameters and window traces remain private.
Verification matched both private artifact hashes and permissions, checked
disjoint stream membership and preserved row order, and recomputed the linear
quantile, strict alert count, and gate from the stored window traces.

The status is `completed_development_validation`, not a passed hypothesis.
This monitor failed the required false-alert component and does not support H2.
Later endpoints cannot reverse this failed mandatory gate for the frozen monitor.
External detection and future-routing outcomes remain unmeasured and are still
needed to characterize RQ2, not to rescue its failed support rule. H1 and H3
remain undecided; no group-test or PhishVN input was read by this execution.

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
| 2026-09-09 | 1.8 | Recorded the September 3 advisor direction and completed source-provenance release; narrowed RQ1 to selective character-model escalation; froze `rq1-transformer-cascade-v1`, the manual-review boundary, and the complete-window H2 metric at contract status `frozen_not_run`. The procedure was later implemented and review-hardened at `frozen_implemented_not_run`; no transformer, cascade, or GMM fit was run. |
| 2026-09-09 | 1.9 | Preserved `rq1-transformer-cascade-v1` byte-for-byte as `superseded_unrun` and froze `rq1-transformer-cascade-v2` at `frozen_not_run`. Version 2 changes only contract identity, schema version, protocol version, and publication semantics; its date is unchanged, and it does not change a research question, hypothesis, input, model, training, threshold, cascade, manual-review, or artifact-content rule. The protocol record also corrected the still-unrun GMM allocation status to a prospective staged freeze without selecting an allocation rule. The implementation is `frozen_implemented_not_run`; no transformer, cascade, or GMM fit was run. |
| 2026-09-17 | 1.10 | Froze `rq2-gmm-development-v1` before GMM execution: exact 26-column inputs and training scaler, six diagonal-GMM candidates and training-BIC selection, label-blind domain-hash calibration/audit allocation, complete-window linear quantile, strict audit gate, numerical policy, and private/public publication controls. Incorporates unchanged transformer v2; both transformer contracts remain byte-identical. Transformer execution is `running_development_validation`; GMM execution is `not_run`. No completed transformer or GMM result is claimed and H1, H2, and H3 remain undecided. |

Any change to a research question, hypothesis, evidence designation, method,
or decision rule increments the protocol version and records a new hash before
the affected analysis runs.
