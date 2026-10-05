# Bounded follow-up: evidence, diagnosis and design

Recorded October 1, 2026 (America/Los_Angeles), after the initial study results.
This is an operator-requested extension, not an original preregistered experiment
or an advisor-approved amendment. The original 22 gate decisions, 25 operational
groups, source code and dissertation package remain unchanged.

## What is established

The original study completed all 125 operational cells and all 22 primary checks.
Nine checks passed. The mandatory conjunctions for H1, H2 and H3 did not. A useful
engineering account can report the passing components, identify why they were
insufficient, and evaluate a disclosed redesign. It cannot relabel a conjunction
as supported because some components passed.

The three research questions concern representation value, shift monitoring and
safe escalation, and inline detection/service tradeoffs. They organize both the
initial findings and the follow-up. The extension does not replace them.

### Detection and source representation

The frozen label mapping agrees with the publisher definition: PhiUSIIL native
0 means phishing and maps to 1; native 1 maps to legitimate 0. No inversion defect
has been established. Tranco remains unlabeled, not a certified-negative set.

The earlier preparation amendment consistently uses the publisher's external
`url_norm`. Its documented HTTP prefixes can be publisher-derived, not observed
transport properties. The unchanged extractor uses `is_https` and raw-string
length/composition features. Existing external PSI exports show a median
`is_https` PSI of 5.8347912565135776 across the 132 overlapping windows. Several
other dimensions also shift. PSI is descriptive here, not a causal attribution
or an independent-window significance test.

This justifies testing transport-format invariance. It does not prove that scheme
formatting caused the external false positives. Dropping `is_https` alone would
leave the scheme encoded in raw lengths, character counts, ratios and entropy.
Any representation candidate must remove that information consistently before
extracting all affected features, preserve raw strings separately, and be trained
and calibrated on the same representation used at evaluation.

External gold-plus-certified results also rule out an easy calibration claim:
Logistic-L1 AUC is 0.8512388779578973, but its already-reported, post-test low-FPR
point detects only 20/69 positives with 24/2,497 false positives. The transformer's
corresponding point detects 7/69 with 13/2,497 false positives. These points are
not new calibrated thresholds and must not be deployed or retested as if chosen
without seeing outcomes. Better score calibration alone cannot be assumed to
repair ranking or produce high recall at a stringent risk ceiling.

### Routing

The frozen cascade's feasible-band selection minimizes transformer calls and then
band width, subject to its recall/FPR constraints. It does not require incremental
recall over Logistic-L1. A nearly empty band is therefore compatible with its
actual objective; this is not evidence of a software defect. Drift-triggered
escalation added 218 certified-negative alerts without a gold-positive recall
gain. Detecting a distribution change is not evidence that an unvalidated expert
will correct it. No automatic escalation redesign is selected yet.

### HTTP measurement

The frozen primary fixed-cascade group used no transformer calls and produced
HTTP p95 364.3004101 ms at concurrency 64; its concurrency-1 p95 was 1.488833 ms.
Model reconstruction is real per-call work, but a synthetic cProfile run of
2,000 singleton Logistic-L1 calls took 0.776133457897231 seconds in total.
That instrumented synthetic result does not identify the original service's
bottleneck, and it does not justify attributing hundreds of milliseconds to
estimator construction.

The next diagnostic retained the original HTTP client and a synthetic no-model
service. At concurrency 64, its instrumented p95 was 652.5387833499997 ms. The
client-thread profile assigned 4.962 of 8.598 profiled seconds cumulatively to
httpcore `_assign_requests_to_connections`; socket expiration/readability checks
and repeated optional-runtime import checks were prominent nested costs.
At concurrency 1, instrumented p95 was 1.7289399 ms. No request errors occurred.
See `synthetic-http-client-c*.json` and their profile text files.

These are development diagnostics, not dissertation operational measurements:
the fixture is synthetic, the service runs in another thread, and cProfile adds
overhead. Nevertheless, substantial tail growth without model work makes the
HTTP client's shared connection-pool path a concrete candidate for investigation.
The study's original HTTP endpoint result remains correct for its declared stack.

## Selected implementation experiment

First test worker-owned HTTP/1.1 connection pools. Each closed-loop worker keeps
one persistent client/connection; total simultaneous requests and connections
remain bounded by the same declared concurrency. Keep the same service, singleton
scorer, request schema, input order, warmup, inclusive latency timing, response
validation, two-second deadline, no-retry rule, phase barriers and error accounting.
Use worker zero's client for drains outside measured request phases.

This changes the measurement client, not detection or server inference. Give it
an explicit extension identity. Do not substitute its output into original H3
evidence or claim detector improvement from faster client behavior. Compare the
unchanged and changed clients on a no-model control first; any later research
comparison must evaluate both clients under the separately frozen extension.

Prepared-scorer caching is deferred: the current diagnostic does not establish
it as the dominant problem. Do not build an optimization framework or change
numerical audits merely to obtain favorable timing.

The second bounded candidate, if data eligibility permits, is a transport-format-
invariant structural model. Its precise representation, fitting procedure,
calibration rule and fresh evaluation allocation must be specified before fitting
or accessing new evaluation rows. No character-model retraining, four-way redesign,
post-test threshold adoption or unbounded candidate search is authorized by this
design record.

## Evaluation data and stopping boundary

Hannousse and Yahiouche's 11,430-URL Mendeley benchmark is a metadata-only candidate,
not an admitted test set. Its May 2020 collection predates PhiUSIIL's reported
2022–2023 phishing collection. It could support a carefully delimited retrospective
cross-source evaluation, not a prospective temporal-transfer claim. Publisher
labels are not the original PhishVN gold/certified classes. License, provenance,
prior exposure, complete corpus/domain overlap, row validity and sample size must
be established before admitting it. A new dataset name does not prove independence.

No new dataset rows, protected model artifacts or research measurements have been
accessed in this diagnostic phase. A frozen extension specification and its data
eligibility checks precede that phase. The whole-study hold remains: an absent
required population does not silently disappear from the promised evaluation.

## Implementation acceptance

Synthetic tests must establish persistent worker ownership, unchanged response
and failure semantics, no retry, bounded concurrency, complete cleanup including
partial-start failures, and preserved interruption/checkpoint evidence. Existing
shared-client tests must remain unchanged and pass. A new result wrapper must
not be accepted accidentally by the original primary-summary or record codec.

All synthetic diagnostic attempts and rejected candidates remain recorded. A
speedup, detection gain or hypothesis outcome is not assumed in advance.
