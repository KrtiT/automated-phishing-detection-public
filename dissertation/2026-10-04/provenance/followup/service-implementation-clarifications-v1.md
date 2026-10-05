# Service implementation details before research execution

Recorded October 1, 2026, before follow-up fitting, new-benchmark prediction or
final service measurements. The scientific requirements in the retained comparison
specification are unchanged.

- The schedule is no-model then unchanged structural-detector, concurrency 1 then
  64 within each workload, and pairs 1 through 10 within each concurrency. Odd
  pairs use shared then worker clients; even pairs use the reverse order.
- The fixed synthetic manifest has 10,000 unique occurrence IDs and reserved
  `.example` URLs with alternating schemes, 257 hosts, distinct paths and 97
  repeating query tokens. It has no phishing labels or class-prevalence claim.
  Both arms use exactly the same ordered records and 1,000-row warmup prefix.
- Existing HTTP record validation accepts wire indices 1 through 5. Pairs 1–5
  and 6–10 are two explicitly identified blocks using these wire indices; the
  outer extension record retains pair 1–10. Every arm has a new service process,
  so this does not reuse server state or confuse occurrences within an arm.
- The nested legacy transport codec requires a prevalence field and workload
  identifier. Its `100`/`fixed_cascade` values are compatibility metadata only;
  the outer record explicitly identifies the actual extension workload and sets
  class prevalence to null. These records cannot be submitted as original-study
  measurements. No transformer is loaded or invoked.
- Readiness uses an unmeasured drain and requires zero counters. The parent owns
  the loopback listening socket and the child service; the child owns one scoring
  thread. Closing the parent's input pipe requests shutdown, and the child's
  actual exit status is retained. Failed startup or cleanup stops the schedule.
- The primary p95 uses successful measured responses. All-request and failure-only
  percentiles, error categories, throughput denominators and physical counters
  are separately retained. Undefined successful-response p95 values cannot become
  favorable ratios or disappear from a subset bootstrap.
- The specification's exact response requirement is applied literally: compare
  all 100,000 paired primary measured responses after excluding only request IDs.
  Admission-sequence differences and noncomparable errors do not satisfy it.
  Prediction-field agreement is also reported separately, so an order difference
  is not misrepresented as a changed model decision. This strict service requirement
  may be unmet even when client speed and prediction agreement improve.
- Conditions are sampled before and after each arm and approximately every five
  seconds during the schedule. Samples retain power, energy mode, available
  thermal/performance reports, process observations and owned sleep assertions.
  This is sampled observation plus operator reservation, not proof of continuously
  idle hardware. Unknown or warning thermal/performance reports stop execution.
  A detected violation cancels the remaining schedule and preserves partial work.

Implementation tests use small synthetic runs and invented records. They are not
final performance results. No additional candidate, threshold or favorable-result
retry has been introduced.
