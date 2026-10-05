# Service S: one bounded environmental recovery

Recorded October 2, 2026, after service-comparison-v1 stopped. The operator
explicitly approved: “Approve one new full schedule after stable AC.” This is
operator authorization, not advisor or institutional approval. The approval
followed disclosure of the interruption and the proposed full-schedule recovery.

## Reason and retained chronology

The original attempt stopped at 2026-10-02T07:38:54.894711Z with ConditionsError
`ac_power_absent`. It completed 45 arms (450,000 measured requests): all 40
no-model arms and five structural-detector concurrency-1 arms. No primary
structural-detector concurrency-64 arm began. All files, including the interrupted
46th arm, condition samples and failure/exit receipts, remain in their original
directory. `service-v1-preservation.json` binds their complete hash inventory.
Restored AC does not retroactively make that attempt complete or eligible.

## Sole methodological change

Supersede only the original specification's prohibition on another schedule
after environmental interruption, for exactly one new full schedule in
`service-comparison-v2`. Do not resume, overwrite, remove, replace or pool the
45 interrupted-schedule arms. Do not choose between schedules using outcomes.
The complete new schedule, if valid, is the sole primary S comparison; the first
attempt is disclosed separately as interrupted evidence. This amendment follows
partial nonprimary observations, not a pristine pre-experiment registration.

The original source commit ef8ba5f0b357cf3dd60c4d663e6297d13334460c remains
unchanged. A separate small launcher imports its frozen schedule, synthetic
requests, guard, arm execution and reducer. The authorization binds that launcher,
its tests, this amendment, the original execution manifest/specification and the
preservation receipt. No model fitting, new efficacy predictions, threshold
selection, candidate search, population change or original-source reopening is
authorized. Original H1–H3 and completed detection D remain unchanged.

## Execution and stopping rule

Finish synthetic launcher tests before measurements. Require at least 180 elapsed
seconds of sampled clean AC, automatic energy mode, normal thermal/performance
reports, owned sleep inhibition and no detected competing workload, sampled at
intervals no shorter than five seconds. This is sampled stability, not proof of
conditions between observations. Retain every sample. Use the same inhibitor
through preflight and measurements, with a fresh condition check at transition.
Only then exclusively create service-comparison-v2. A preflight violation stops
without starting a schedule. An execution violation or error stops the new
schedule, retains all progress and produces no automatic additional attempt.

All 80 arms, their order, ten pairs, both workloads, both concurrencies, both
clients, 1,000 warmups, 10,000 measured requests, timeouts, drains, no-request-retry
rule, strict response agreement including admission sequence, latency definitions,
bootstrap seed/resamples/interval and every S threshold remain unchanged.

## Implementation and verification checklist

- Retain and verify the complete v1 hash inventory and frozen decoder checks.
- Test tamper rejection, no overwrite, sampled preflight, unchanged schedule
  delegation, successful reduction and failure/cleanup preservation using fakes.
- Freeze the launcher/test/amendment/receipt binding before clean-host preflight.
- Execute exactly one additional full schedule; no concurrent tests or rendering.
- Independently recompute saved measurements, then integrate actual chronology
  and all favorable/adverse results into the new GWU manuscript and deck.

Independent review tooling is unavailable in this session; self-review and
synthetic tests are not represented as independent scientific approval.
