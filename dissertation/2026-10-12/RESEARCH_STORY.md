# The research argument

The dissertation asks when a more expressive representation or a more expensive
expert earns its place in a constrained URL detector. A useful system must do
more than obtain high recall: it must distinguish legitimate inputs, send useful
cases to the expert and serve requests within an explicit cost budget. The work
therefore evaluates the complete implemented path, diagnoses the observed
behavior, changes the implicated component and measures both benefit and cost.

## RQ1 — What do structural and character representations add?

Structural URL features provide a measured recall gain over length alone: 64.07
percentage points internally and 76.81 points on the original external positive
sample. The fixed cascade does not add recall because its validation-selected
zero-width band has no evaluation ties. External specificity also deteriorates:
the structural model flags 2,261 of 2,497 certified negatives. The transformer
shares those errors, repairs none and introduces 218 more. Routing among these
two unchanged decisions cannot resolve their shared false positives.

The diagnostic comparisons distinguish two remedies that might otherwise be
confused. D changes the representation and refits/recalibrates: it achieves exact
invariance across 8,622 pairs but trades 251 fewer false positives for 184 fewer
detections. F holds the model, scaler and threshold fixed and changes only the
scheme prefix. Forced HTTPS reduces the original certified false positives to
50 but reduces gold detections from 69 to 22. These are real measured changes,
not a sufficient low-FPR solution. The technical answer is to evaluate low-FPR
class separation separately from representation stability and overall ranking.

**H1:** five of ten component checks pass; the joint requirement is unsupported.
See Sections 4.2, 4.8 and 4.10.1; Tables 4.2–4.3 and 4.21, and
[F](aggregate-data/F-formatting.csv) with the [corrected ledger](aggregate-data/corrected-primary-gates.csv).

## RQ2 — Can shift monitoring guide safe escalation?

The GMM monitor detects 115 of 132 external windows (87.12%), exceeding its 80%
sensitivity target. Its independent reference audit alerts on 28 of 252 windows
(11.11%), exceeding the 5% false-alert allowance. The future-only routing policy
adds 218 certified false positives without adding a gold detection. Recognizing
a distribution change and choosing a beneficial response are different system
capabilities: the first does not supply missing error complementarity for the second.

**H2:** one of four checks passes; the joint requirement is unsupported. This is
a completed answer with identified mechanisms, not an unfinished experiment.
See Sections 4.3 and 5.1, Tables 4.4–4.5 and the retained
[original aggregates](../2026-10-04/aggregate-data/primary-results.json).
Overlapping windows and policy activations retain their dependence caveats.

## RQ3 — When does selective computation justify its cost?

The original service assessment covers all 125 cells in 25 groups, with
1,243,505 requests and 1,901 errors. Its designated concurrency-64 pooled
all-request p95 is 364.30 ms, above the 200 ms requirement. Sparse calls alone
do not establish acceptable request latency or external false-positive behavior.

R tests a frozen wider gate on a separately selected, retrospective
domain-balanced frame: 1,392 positive and 1,262 negative domains, one URL per
domain. The symmetric gate makes 21 calls, repairs no selected errors and adds
18 false positives. All systems detect the positives. The original 69-pair
95% recall envelope is corrected from a zero-width interpretation to ±6.1533
points; the new R envelope is ±0.3143 points. Two separately specified
recall-only noninferiority statements now meet the unchanged −2-point margin,
conditional on independent labeled pairs. They do not establish campaign
independence, external specificity or the original joint hypothesis.

C measures the computation rather than inferring it from call counts. The
symmetric gate uses 17 expert calls per 1,000 requests on its distinct fixed
workload and costs 1.646 times structural-only summed scoring time. Median run
p95 is 0.223 ms and p99 is 6.769 ms: rare expert work appears in the higher tail.
Always-transformer costs 30.781 times structural scoring. All 30 arms and every
recorded outcome are retained. These are descriptive concurrency-one computation
measurements, not HTTP latencies; five cyclic rotations are not fully balanced.

S addresses a different mechanism: connection ownership in a synthetic,
structural-only closed-loop HTTP workload. Worker-specific clients reduce
median paired p95 by 79.68%, with 99,999/99,999 comparable predictions agreeing.
Strict full-response equality is only 799/100,000 because admission sequence
differs and one shared-client request times out. Four of five requirements pass;
the full S conjunction does not. No transformer-service acceleration is claimed.

**H3:** two of eight original checks pass after correcting paired inference;
the joint requirement is unsupported. See Sections 4.4, 4.9 and 4.10.2–4.10.3;
[R](aggregate-data/R-routing.csv), [paired contrasts](aggregate-data/R-paired-contrasts.csv),
[all C arms](aggregate-data/C-all-arms.csv) and the inherited
[S evidence](../2026-10-04/aggregate-data/service-S/verification.json).

## Reliability checks and the contribution

P links the actual shuffled training labels to each of five new fits, seeds
42–46. All converge; all outcomes remain, including AUC 0.3940–0.6965 and AP
0.4027–0.7683. True validation labels select the descriptive operating points.
This resolves the new fits' input provenance, not historical authentication or
a formal five-permutation significance test. See [all P seeds](aggregate-data/P-all-seeds.csv).

The contribution is a tested engineering account of conditional repair, risk
and computation in one implemented gateway. The useful decision is where to
intervene: routing cannot fix shared fixed-decision errors; invariance does not
guarantee low-FPR separation; and an HTTP improvement must not be mistaken for
better detection or cheaper transformer work. The
[closest-work comparison](CONTRIBUTION_MATRIX.txt) acknowledges existing URL
normalization, shift evaluation and expert fusion rather than claiming those
components as inventions. Every claim follows its measured population and method.

The [evidence map](EVIDENCE_MAP.txt) covers the full manuscript and deck. The
[reproduction guide](REPRODUCTION.md) distinguishes the public frozen source,
public aggregates and retained private record-level verification.
