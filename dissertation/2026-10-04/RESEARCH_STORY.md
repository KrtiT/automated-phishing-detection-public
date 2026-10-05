# Research argument and evidence map

## The Argument

An inline phishing gateway must do more than rank URLs accurately. Its input
representation must transfer to different sources, its threshold must control
false alarms, escalation must add useful detections, and the actual request path
must meet a latency and reliability budget. This investigation implements that
path and measures these requirements together. The resulting contribution is
the connection between algorithmic behavior and operational evidence, rather
than a claim that any familiar model family is itself new.

The original design combines a fast structural detector, a selectively invoked
character transformer and a GMM distribution monitor. Evaluation shows where
these components help and where the joint operating conditions are not met.
Diagnosis then motivates two bounded changes: remove an input-representation
dependency on the HTTP/HTTPS scheme, and change HTTP client connection ownership
without changing the structural scorer. Separate comparisons measure the effects.
The original gates stay intact throughout this sequence.

That is an end-to-end engineering investigation: implemented system, completed
evaluation, diagnosed constraints, explicit modifications and measured tradeoffs.
The later comparisons were developed after original results were available; they
are not retrospectively presented as original, prediction-blind hypotheses.

## RQ1 — Representation Value and Generalization

**What incremental value do structural URL features and character-level
representations provide under registrable-domain-disjoint and external evaluation?**

Structural Logistic-L1 increases recall over length-only by 64.0723 percentage
points internally and 76.8116 points on the external gold-positive domains. Both
domain-clustered 95% intervals exclude zero. Internal structural FPR is 0.7747%,
but certified external FPR is 90.5487%. The fixed cascade selects no evaluation
rows for character inference and therefore adds no recall to the structural
model. Transformer-only gains some internal recall descriptively, while its
certified external FPR is 99.2791%.

**Answer:** structural features provide measured representation value in the
paired comparisons; the fixed selective character stage adds none in this
evaluation, and the required low-FPR external transfer is not established.

H1 requires six model/population FPR checks at or below 1% and four paired recall
contrast lower bounds above zero. **Five of ten checks pass; H1 is not supported.**
The exact comparison identities, denominators, bounds and operators are in
[primary-gates.csv](aggregate-data/primary-gates.csv) and
[paired-contrasts.csv](aggregate-data/paired-contrasts.csv). See Sections 3.6–3.8,
4.2 and 5.1 of the manuscript.

## RQ2 — Shift Detection and Useful Escalation

**Can GMM-based monitoring detect an external source/domain shift and
guide escalation without exceeding the low-FPR operating constraint?**

The GMM alerts on 115/132 external windows (87.12%), exceeding the 80% detection
requirement. Its separate reference audit alerts on 28/252 windows (11.11%),
exceeding the 5% limit. Future-only escalation meets neither the required
certified FPR nor a positive lower bound for the external false-negative-rate
reduction. Detection of distribution departure is therefore distinct from a
useful low-FPR routing intervention.

**Answer:** the monitor detects the prescribed external departure, but its
reference specificity and escalation utility do not satisfy the joint constraints.
**One of four checks passes; H2 is not supported.** MMD/PSI comparisons are
reported as secondary evidence and do not replace the frozen primary monitor.

See [external monitor windows](aggregate-data/external-monitor-windows.csv),
[PSI feature scores](aggregate-data/external-psi-features.csv),
[secondary results](aggregate-data/complete-secondary-results.json), and
manuscript Sections 3.9–3.11, 4.3 and 5.1. Window alert fractions are not
per-URL false-positive rates.

## RQ3 — Real Service Cost and Inline Viability

**What detection, escalation, throughput, and latency tradeoffs determine
whether the fixed cascade is viable inline?**

All 125 cells and 25 five-repeat groups are retained, including 1,901 errors
among 1,243,505 requests. The primary invocation, gold-recall noninferiority and
request-error components pass. Both the cascade and transformer fail the
certified external FPR and Tranco alert safeguards; each alerts on all 1,163
Tranco controls. Tranco is a label-free control, not a certified-negative set.
The designated real-HTTP latency component also fails.

**Answer:** avoiding transformer calls reduces inference demand but is not
sufficient to establish inline viability; external false alarms and the measured
request path remain binding constraints. **Three of eight checks pass; H3 is
not supported.** The criteria include certified FPR and Tranco alert rate at or
below 1% for each system, recall lower bound at least −0.02, invocation at most
30%, designated concurrency-64 p95 at most 200 ms, and errors below 0.1%.

See [all cells](aggregate-data/operational-runs.csv),
[all groups](aggregate-data/operational-groups.csv),
[all gates](aggregate-data/primary-gates.csv), and Sections 3.12, 4.4 and 5.1.
The [data dictionary](RESEARCH_DATA_DICTIONARY.txt) distinguishes client outcomes,
server counters, successful latencies, deadlines and pooled reductions.

## From Diagnosis to Measured Modification

### D — Transport-Neutral Representation

The additional benchmark contains 8,622 eligible records, 6,273 domains, 4,651
publisher positives and 3,971 publisher negatives. The candidate removes the
transport-scheme dependency while keeping the comparison's frozen thresholds.
All 8,622 prescribed opposite-scheme pairs have exactly equal candidate features
and scores and zero decision flips; the baseline flips 481 decisions.

This is a verified representation property, not merely a small average score
difference. It does not imply detection efficacy. At the frozen cutoffs the
candidate has 251 fewer false positives and 184 fewer true positives; its recall
difference is −3.9561 percentage points, with a domain-bootstrap 97.5% interval
of [−5.0547, −3.0014] points. Candidate FPR is 92.8985%. D's required recall gain
and low-FPR conditions are not met, so **D is not supported**.

See [D verification](aggregate-data/detection-D/verification.json),
[metrics](aggregate-data/detection-D/detection-metrics.csv),
[invariance](aggregate-data/detection-D/scheme-invariance.csv), and manuscript
Sections 3.15.2–3.15.4, 4.8 and 5.2. Opposite-scheme companions are metamorphic
pairs, not a second independent labeled sample.

### S — Client Connection Ownership

The complete comparison comprises 80 arms: two workloads, two concurrencies,
ten pairs and two clients. Each arm has 1,000 warmups and 10,000 measured attempts.
The primary comparison uses the unchanged structural scorer at concurrency 64.
The worker/shared median paired p95 ratio is 0.20324899 (79.68% lower), with a
97.5% paired-bootstrap ratio interval of [0.18790305, 0.21350824]. Pooled primary
worker successful latency is 72.00 ms p95, with zero errors in 100,000 attempts.

Predictions agree in 99,999/99,999 comparable primary pairs. However, the frozen
strict response rule also includes admission sequence: only 799/100,000
requested pairs match exactly apart from request IDs. Sequence differs for
99,200 pairs, and one shared-client timeout is noncomparable. Four of five
requirements pass, but **S is not supported**. Reporting descriptive prediction
agreement does not replace that strict criterion.

These measurements isolate a large client-path effect under the stated host and
synthetic workload. They do not establish an equivalent improvement for a
transformer, fixed cascade, production traffic or a different server. No paired
statistical comparison is claimed between original H3 and the later S schedule.

See [all arms](aggregate-data/service-S/arm-metrics.csv),
[primary pairs](aggregate-data/service-S/primary-pairs.csv),
[requirements](aggregate-data/service-S/requirements.csv),
[verification](aggregate-data/service-S/verification.json), and Sections 3.15.5,
4.9 and 5.2.

## Complete Secondary Coverage

| Evidence | Public file | Records / scope |
|---|---|---|
| Population/model outputs | [secondary-metrics.csv](aggregate-data/secondary-metrics.csv) | 153 population/model records; 43 mixed-class metric sets |
| Calibration | [calibration-bins.csv](aggregate-data/calibration-bins.csv) | 430 bins |
| Descriptive low-FPR behavior | [low-fpr-score-curves.csv](aggregate-data/low-fpr-score-curves.csv) | 43 curves; not test-set threshold selection |
| Prevalence sensitivity | [prevalence-projections.csv](aggregate-data/prevalence-projections.csv) | 129 fixed-rate projections; not deployment prevalence estimates |
| Drift comparisons | [external-monitor-windows.csv](aggregate-data/external-monitor-windows.csv) | 396 window rows, plus 3,432 PSI feature rows |
| Seed robustness | [seed-logical-invocations.csv](aggregate-data/seed-logical-invocations.csv) | 35 seed/population rows; five seed fits/states, no best-seed selection |
| Formatting probes | [detector probes](aggregate-data/probe-decisions-and-scores.csv), [monitor probes](aggregate-data/probe-monitors-and-scores.csv) | 20 detector and 12 monitor aggregate rows; no labels assigned to transformed URLs |
| Paired significance family | [primary-results.json](aggregate-data/primary-results.json) | Four McNemar/Holm slots, including their eligibility/status |
| Formatting, permutation and RF controls | [complete-secondary-results.json](aggregate-data/complete-secondary-results.json) | All five permutation controls; retained RF correction-v2; unresolved consumed-label provenance remains disclosed |

The secondary results deepen interpretation; they do not rescue a failed primary
gate by substituting a favorable model, threshold or denominator.

## Chronology and Contribution Boundary

The original raw-field preparation stopped before predictions. The disclosed
publisher `url_norm` amendment followed that hold; four later executions were
interrupted. A separately reviewed recovery admitted all 72 eligible cells from
the fourth attempt and measured the remaining 53 complete cells. The completed
matrix therefore spans sessions, with corresponding session/confounding limits.
No fragments from earlier attempts replace accepted adverse outcomes.

The first S schedule stopped after 45 complete arms. A separate full 80-arm
schedule followed the disclosed power-recovery amendment; the interrupted
schedule was not pooled into its primary result. The method and interruption
records remain in [provenance](provenance/), and earlier exact source states
remain in Git history.

The active questions are documented in the
[advisor-deck crosswalk](provenance/advisor-deck-scope-crosswalk-20261001.md).
Earlier feature-fusion and distillation proposals were superseded, not completed.
The work's defensible value is the implemented and traceable joint evaluation,
its diagnostic account of failure mechanisms, the invariant representation, and
the bounded service improvement. Establishing that value does not require an
unsupported novelty claim or a retrospective change to the hypotheses.
