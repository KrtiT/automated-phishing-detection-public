# Advisor-deck scope crosswalk — October 1, 2026

Purpose: retain the exact advisor-facing commitments for final synthesis, not
create a new research design. The operator reiterated that the dissertation must
answer the questions in the last two months of advisor decks, with a coherent
argument, data and analysis, without shortcuts. This record changes no gate,
model, measurement, authorization or active execution.

## Exact active questions

These appear verbatim on slide 6 of both the August 20 and September 3 decks.

1. **RQ1:** What incremental value do structural URL features and character-level
   representations provide under registrable-domain-disjoint and external evaluation?
2. **RQ2:** Can GMM-based monitoring detect an external source/domain shift and
   guide escalation without exceeding the low-FPR operating constraint?
3. **RQ3:** What detection, escalation, throughput, and latency tradeoffs determine
   whether the fixed cascade is viable inline?

Use these questions verbatim in the final manuscript and deck. Plain-language
answers may explain them but must not silently substitute new questions.

## Decision rules recorded on September 3, slide 7

- **H1:** At <= 1% observed FPR on both primary test sets, both domain-clustered
  95% bootstrap lower bounds must be > 0: Logistic-L1 over length-only recall,
  and cascade over Logistic-L1 recall.
- **H2:** >= 80% shift-window detection; <= 5% independent reference false alerts;
  routed-policy FPR <= 1%; and a 95% lower bound for external false-negative-rate
  reduction > 0.
- **H3:** Each system: certified-registry FPR <= 1% and Tranco control alert rate
  <= 1%; recall lower bound >= -0.02; invocation <= 30%; real-HTTP p95 <= 200 ms
  at concurrency 64; errors < 0.1%.

The deck requires every gate. The complete frozen contracts resolve all
population, interval, reference-cell and denominator details. Final reporting
must cover 10 H1, 4 H2 and 8 H3 checks, all 25 operational groups and all promised
secondaries in `../plans/2026-09-29-final-rqh-synthesis-checklist.md`.

September 17 slides 5, 7, 8 and 10 carry forward paired internal/external
comparisons, external detection and future-only routing, and actual selective
HTTP measurement. H2's failed 28/252 reference-audit component remains visible;
its non-support does not excuse omitting the other RQ2 analyses.

## Historical distinction

The June manuscript and August 6 deck used earlier feature-fusion, new-technique
GMM detection and 300M-to-30M distillation questions. The August 20 deck explicitly
presented revised questions; the September 3 deck named them active questions.
The realignment plan explicitly substituted selective execution for distillation.
The current experiment cannot be presented as a completed distillation or
text/temporal/network feature-fusion study. Local deck content does not itself
prove advisor approval; operator authority and actual amendment timing remain
separately disclosed. Do not retroactively relabel the old hypotheses as tested.

## Source identities

Paths below are relative to `.context/deliverables/`; hashes cover the actual
PPTX files read, not a regenerated deck or its Python builder.

| File | SHA-256 |
|---|---|
| Tallam_Praxis_Advisor_Update_2026-08-06.pptx | ec81f38ac3aa97dabb57b4fca4f0ced6e90ad583b92a8ccaa4f9e92ec0ee9c79 |
| Tallam_Praxis_Advisor_Update_2026-08-20.pptx | 515da6500c7f09c5d6b187a2b16d7a1e9d534a00ce41d6d2d848286123a6b150 |
| Tallam_Praxis_Advisor_Update_2026-09-03_With_Speaker_Notes.pptx | 4f2dab6cf12fab80c61b9f4e475f7e14b0dc21ef6d81c187a792c1ed31dc7684 |
| Tallam_Praxis_Advisor_Update_2026-09-17_Meeting_Final.pptx | 6b6fc2a36b9cbf50061ca1865c449e87c3492f7733394929f8e32b4934cdf5de |

## Final argument

Lead with one evidence chain: what the frozen data represent; what structural
features and selective character inference add; what distribution alerts detect
and whether future-only routing helps; and what the real service costs. Connect
each conclusion to the original question, complete gate results and supporting
analyses. Preserve negative findings, source limitations, prior exposure and
multi-session confounding. Software checks establish implementation integrity,
not scientific support.
