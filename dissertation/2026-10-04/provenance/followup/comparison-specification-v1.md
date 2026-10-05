# Bounded comparison specification v1

Written October 1, 2026, after the original study and synthetic diagnostic runs,
but before retrieving the new benchmark CSV, fitting the transport-neutral model,
or running a final extension comparison. Operator approval is the user's request
to implement the bounded redesign. This is not advisor/institutional approval or
a preregistration made before the initial results. Preserve this version and its
hash; any correction receives a separate dated record before further execution.

## Questions and scope

Two changes only: (D) remove URL scheme spelling from structural model inputs;
(S) replace the shared HTTP connection pool with one persistent pool per worker.
No transformer retraining, GMM retuning, new drift-routing policy, threshold
selection on external results, or change to original H1–H3 decisions. The extension
tests mechanisms relevant to RQ1/RQ3; it does not rescue original H2. The original
125 cells, 22 decisions and deliverable package remain authoritative and intact.

## Data admission before predictions

Candidate: Hannousse and Yahiouche, Mendeley DOI 10.17632/c2gw7fy2j4.3, CC BY 4.0,
file `dataset_B_05_2020.csv`, publisher file ID
`575316f4-ee1d-453e-a04f-7b950915b61b`, SHA-256
`21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c`.
Publisher metadata states 11,430 rows, balanced labels, May 2020. The associated
arXiv preprint, section 4.3, says March 2020; retain this discrepancy and refer to
the 2020 benchmark, not a verified exact collection date. Section 4.1 describes
Alexa-seeded crawling/Yandex legitimate sources and PhishTank/OpenPhish phishing
sources. Shared upstream phishing feeds with PhiUSIIL preclude a claim of fully
independent source mechanisms. This is a retrospective, cross-dataset/domain
comparison, not forward temporal generalization, independently adjudicated ground
truth, or operational prevalence. No local use was found in the research records
searched; this is not proof of absence of every historical exposure.

Download only this publisher CSV after saving this specification's hash. Do not
visit any listed URL, load DOM pickles, or use any of the publisher's 87 features.
Use only the URL index and explicit publisher `phishing`/`legitimate` class values.
An unrecognized header or label stops preparation rather than guessing a mapping.
Require the exact publisher hash, 11,430 input rows and 5,715 of each class.

Reuse canonical-url-v1 and the original pinned Public Suffix List including its
private suffix policy. Quarantine invalid URLs, mixed-label canonical duplicates,
and every URL whose registrable domain occurs anywhere in the original PhiUSIIL
valid-source manifest (including quarantined rows) or the complete retained
PhishVN publisher-source manifest (all three splits). Keep the first source-order
row of each same-label canonical duplicate. Preserve all raw rows, exclusions,
reason incidences, label/domain counts and hashes. Do not repair URLs or change
publisher classes. Count overlap before other filtering as well as final retention.

Admission requires at least 1,000 retained rows and 250 unique registrable domains
in each class. Use every eligible row, with no balancing, sampling or rerolls.
These are minimum-information rules, not a guarantee of power: with 1,000 truly
independent Bernoulli observations, the worst-case 95% margin is about 3.1 points;
within-domain dependence can substantially widen uncertainty. Report observed
cluster counts/sizes and domain-clustered intervals rather than claim that row
count is an effective sample size. If either class fails admission, hold the
whole final extension comparison; do not substitute another favorable dataset or
drop detection while describing the whole extension as complete.

## Detection comparison D

Unchanged comparator: the accepted 25-feature Logistic-L1 model and its existing
PhiUSIIL-validation-selected threshold. Do not refit that comparator. Candidate:
`transport-neutral-structural-v1`, which validates the original URL, substitutes
the fixed modeling prefix `http` while preserving the remainder byte-for-byte
in its Python string representation, extracts structural features, and omits
`is_https`. This removes scheme information consistently from lengths, character
ratios and entropy, not only from a single indicator. Stored raw URLs are intact.

Fit the 24-feature candidate once on the unchanged original training partition
(SHA-256 `575f2fb13a0766020e29d78bf8e633a185b381abde7060bdd1ed04cc4a5e38a0`).
Train-only StandardScaler and LogisticRegression: L1, saga, C=1, balanced class
weights, intercept enabled, max_iter=5000, tol=1e-4, seed=42. Select its threshold
only on the unchanged original validation partition (SHA-256
`970c6568a6400a1fc265b7809ef7bd9d1c297632799cbb88801313bc34ac415a`) using the
existing maximum-recall rule with one-sided 95% Clopper–Pearson FPR upper bound
<=1%, and its existing exact tie rules. Those partitions are development data;
neither is fresh validation for the extension. Any convergence, nonfinite-state
or numerical-integrity failure is retained and stops fitting without a solver,
seed or tolerance search. Freeze both model hashes and thresholds before any
new-benchmark scoring. Score no original group-test/PhishVN rows for new efficacy
claims. Those exposed observations motivated development only.

Evaluate original URL strings and one prespecified scheme-swap companion per
eligible row (HTTP→HTTPS; HTTPS→HTTP; remainder unchanged). Companions are a
metamorphic representation test, not independently labeled historical websites
or evidence that changing a live website's transport preserves phishing status.
The candidate must have exactly equal feature vectors, scores and decisions
across companions; report comparator score differences and decision flips without
inferring that all comparator flips are errors. Do not choose a favorable scheme.

Primary extension detection estimand: paired recall difference, candidate minus
unchanged Logistic-L1, at their frozen thresholds on original eligible strings.
Report both confusion matrices, observed FPR, recall, precision, ROC AUC, average
precision, Brier score and calibration plots. External thresholds are never fitted.
Resample whole registrable-domain groups with replacement, paired across models,
10,000 replicates, PCG64 seed 20261001. Compute row-weighted metrics within each
resample. Report undefined-replicate counts; never silently substitute zeros.
For the primary difference use a two-sided 97.5% percentile interval. The
study-defined D requirement is point gain >=5 percentage points, lower interval
bound >0, and candidate observed FPR <=1%. Report clustered FPR uncertainty too;
the observed budget is not a population-risk guarantee. The effect target is a
follow-up engineering requirement, not an original promise or literature standard.
No seed sensitivity or subgroup significance fishing is planned. Other metrics
and invariance diagnostics are descriptive and all are reported.

## Service comparison S

Isolate client topology from model work with the unchanged synthetic no-model
response control and a singleton structural detector using the accepted Logistic-L1
scorer. Use the same separate-process HTTP service, one serialized scoring owner,
same ordered synthetic absolute-URL manifest, response schema, float64 scoring,
connection ceiling, 1,000 warmups, 10,000 measured requests, two-second deadline,
no retries, phase drains and inclusive client latency. No URL is fetched.
Synthetic strings test service behavior, not detection accuracy. They do not
represent the original 1% prevalence workload; do not pass these results off as
replacement H3/cascade measurements or transformer speedups.

Run concurrency 1 and 64, each workload, both clients, ten paired repetitions.
Use run order shared→worker on odd pairs, worker→shared on even pairs; complete
one pair at a time, fresh service per arm. Retain every scheduled attempt and
all request errors, startup/cleanup failures and interruptions. After an
environmental interruption, stop the remaining schedule and report an incomplete
comparison rather than silently pooling a favorable subset or repeating it.

Primary operational estimand is the paired per-run p95 ratio worker/shared for
the structural-detector workload at concurrency 64. Report all ten ratios and
their median with a two-sided 97.5% paired run-bootstrap percentile interval,
10,000 replicates, PCG64 seed 20261002. Requirement S: median ratio <=0.8, upper
interval bound <1, pooled worker p95 <=200 ms, errors <0.1% of all 100,000 measured
worker requests, and exact paired response agreement apart from request IDs.
Report failure latencies too; keep the original success-only p95 definition and
its explicit error gate rather than hiding errors in a percentile denominator.
Concurrency 1 and no-model control are required descriptive checks, not additional
success opportunities. The 97.5% intervals for D and S apply a Bonferroni allocation
across the two primary comparisons; no independence is assumed between them.
Bootstrap inference is approximate, particularly with ten operational pairs.

Final timing requires actual AC-power observations, thermal/performance checks,
sleep inhibition, no concurrent tests/builds/training/benchmarks, and the user's
reserved Mac. Record the Python/library/hardware versions and every conditions
sample. No development probe or result obtained while these conditions fail is
eligible. Throughput includes all measured requests and their elapsed phase;
report p50/p95/p99, errors, raw counts and resource observations. Do not claim
operational novelty for the standard technique of connection reuse.

## Freeze, review and reporting

Implementation tests use synthetic data only. Commit verified implementation
locally, record exact commit/runtime/specification/input hashes and commands in
a small execution manifest, and retain model-fit evidence before scoring. This
does not open original standalone runners or alter their authorization bindings.
Record code review limitations; no independent model reviewer completed a review.

Integrate the measured extension as development-informed engineering iteration,
with actual chronology once in Methods and explicit stage labels in Results.
State separately what worked, what did not, what changed, and what the comparison
establishes. Preserve all adverse outcomes and initial decisions. Update the new
GWU manuscript/deck only from verified final evidence, not diagnostic timings.
No claim of guaranteed acceptance, invented algorithmic priority, or successful
detector repair is authorized by this specification alone.
