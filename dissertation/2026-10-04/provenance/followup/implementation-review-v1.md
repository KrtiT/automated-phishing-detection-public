# Implementation review and execution scope

October 1, 2026. Primary-agent review, not independent scientific approval.

The bounded implementation now connects population admission, a single fixed
candidate fit, frozen-artifact detection comparison and the 80-arm service
comparison. Original measurement code and evidence remain in their separate
frozen checkout. The only changed existing module in the follow-up checkout is
the HTTP client, whose default shared-pool entry point remains covered by its
existing tests. New results carry distinct follow-up identities.

## Review actions and corrections

- Checked the unchanged baseline loader, feature representation, train-only
  scaling and validation-only threshold selection. Added explicit equality checks
  between the frozen operating thresholds and their model artifacts before new
  benchmark evaluation.
- Checked domain-cluster resampling, within-domain pairing, row-weighted ratios,
  exact denominators, fixed seeds and exclusion of undefined bootstrap replicates
  with explicit counts. No test-time threshold or candidate selection is present.
- Checked all 80 service identities and balanced within-pair ordering. Every arm
  starts a separate process and retains launch, checkpoints, raw response/timing
  records, cleanup status and its recomputable summary.
- Found a launch-receipt storage-failure cleanup gap. A synthetic regression test
  reproduced it; moving the receipt write inside the owned-process cleanup scope
  fixed it. The red and green lifecycle test outputs are retained.
- Checked the environmental cancellation path, startup readiness and terminal
  errors. Small real loopback tests exercise both clients with a separate no-model
  process. These are tests, not final timing measurements.
- Checked the literal exact-response gate separately from prediction-only
  agreement. Admission sequence remains included in the former, as specified.
  This conservative gate can fail despite a useful latency reduction; its
  semantics have not been relaxed in pursuit of a passing result.
- Expanded execution identity to pin the interpreter/platform and HTTP/service
  packages as well as numerical libraries. Dated implementation clarifications
  are also bound to the manifest.

## Verification and review limitations

The focused verification receipt reports 213 passing tests before the final two
threshold-binding regressions; the subsequent pre-freeze receipt is the authority
for the final focused count. Ruff check/format and Git whitespace verification
are retained separately. A broad original regression suite was launched earlier
and remains in progress at this review checkpoint. Its result is not represented
as passing until the actual exit and complete output are available.

A read-only independent Codex CLI review was attempted using the saved prompt.
It returned a model/client-version compatibility error before producing any
substantive review. The console and exit are retained. Earlier Claude/Gemini
attempts likewise supplied no completed review. There is therefore no independent
reviewer approval or model consensus to claim. Synthetic tests and this source
review establish implementation evidence, not independent replication.

The limited implementation is ready for a local source freeze after the focused
verification completes. The broader suite can continue before final packaging;
it is not silently discarded. No final service timing is eligible while any
tests are running or actual AC power is absent. Model fitting and detection
scoring are not service timing measurements and do not use those timings as
results. All scientific outcomes, including nonconvergence, are retained.
