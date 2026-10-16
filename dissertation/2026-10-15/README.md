# Cross-dataset reliability: manuscript and research evidence

CyberSentinel is a working URL phishing gateway combining a structural classifier,
a character transformer, a distribution monitor and an HTTP service. The Praxis
follows RQ1–RQ3 from design choices through controlled tests, measured achievements
and engineering findings.

- [Dissertation PDF](manuscript/Tallam_Krti_Praxis_Strengthened_2026-10-15.pdf)
  · [Word copy](manuscript/Tallam_Krti_Praxis_Strengthened_2026-10-15.docx).
- [Shuffled-label mechanism and evidence](diagnostics/REPORT.md).
- [Reproduce and verify](REPRODUCTION.md) · [data dictionary](DATA_DICTIONARY.md).
- [Verification](VERIFICATION.md) · [file hashes](SHA256SUMS.txt).

## Research questions and contributions

**RQ1 — Representation and transfer.** The measured internal recall gain is
64.07 percentage points, reaching 98.6807% recall at 0.7747% FPR. Validation controls reveal strong
format–class association, while the fixed-weight scheme intervention changes
external false positives from 2,261 to 50 and gold detections from 69 to 22.
A separate refit verifies scheme invariance across 8,622 pairs and quantifies its
detection trade-off. The experiments identify format dependence; their different
partitions and interventions do not quantify the fraction of internal gain due
to HTTPS or isolate normalization as the cause of the refit's detection changes.

**RQ2 — Monitoring and action.** The monitor flags 87.12% of external windows
and 11.11% of audit windows. Replay selects the expert for 94.8512% of requests,
adds 218 certified false positives and retains 69/69 gold detections. Evaluating
both alerts and subsequent decisions supplies a concrete test of monitoring utility.

**RQ3 — Selective execution and service cost.** Both models flag the same 2,261
of 2,497 certified benign URLs. Selection between their unchanged decisions has
an exact 90.5487% FPR floor on that cohort. On separate data, wider gates select
no repairs and add 18 false positives. Worker-owned clients reduce median paired
HTTP p95 by 79.68% in the single-host synthetic structural-only experiment; a
no-model control supports the client-path explanation. Expert computation is
measured separately, so these results do not establish a net transformer-service
speedup. Comparable predictions agree despite changed admission order.

The shuffled-label analysis supports interpretation of RQ1. Across all 99
additional permutations, AUC spans 0.1046–0.9496; validation-label randomization
returns mean AUC 0.5000. Fixed-score component removal and five stricter
convergence checks support the explanation of small random coefficients ranking
structured validation labels. Collection-source causality is not identified.

The original joint hypotheses H1–H3 remain unsupported, with eight passing
component checks of 22. These decisions retain the stated operating criteria
while the experiments establish specific design properties and consequences.

## Manuscript and evidence structure

The defense reading copy contains 149 total pages, 104 main-body pages,
31 tables, four original figures and 127 navigation entries. The 199-word
abstract fits one page. The existing RQ/hypothesis statements, numerical result
values, equations, references and figure bytes are preserved. Table 4.14 uses
“not met” for unmet conditions, and Table 5.2 adds the engineering insight from
each comparison. Both retain their formal decision meanings.

Sections 3.15–3.16 explain the diagnostic design rationale and the sequence of
specification and execution. Appendices retain the detailed provenance. Earlier
[October 4](../2026-10-04/README.md) and [October 12](../2026-10-12/README.md)
editions and their companion responses retain their original evidence.
The [claim-to-evidence map](document-checks/three-contribution-evidence-map.json)
connects the synthesis to source aggregates and interpretation boundaries.

The [abstract/export check](document-checks/abstract-update-verification.json)
connects the earlier manuscript verification to the author-saved title date and
approved abstract. Earlier receipts retain their own document hashes.

The [preceding prose verification](document-checks/posture-verification.json) extends
that chain through 27 interpretive edits and four updated navigation caches.
All numerical results and formal decisions are preserved. The full local software
suite records 13,313 passed tests, two skipped and zero failures in the
[software receipt](document-checks/software-checks.json).

The [broader revision receipt](document-checks/voice-verification.json) and
[every sentence change](document-checks/SENTENCE_CHANGES.md) document the 72-paragraph
research-voice revision and the two authorized table changes.

The [final Chapter 5 and thesis pass](document-checks/CHAPTER5_SENTENCE_CHANGES.md)
polishes 14 paragraphs, with [preservation checks](document-checks/chapter5-verification.json)
against the preceding edition. The abstract, conclusion and all tables remain exact.

The subsequent [conclusion pass](document-checks/CONCLUSION_SENTENCE_CHANGES.md)
revises only Section 5.6. Its [preservation receipt](document-checks/conclusion-verification.json)
verifies unchanged numbers, formal outcome, interpretation boundaries, all other
manuscript content and formatting. The conclusion fits its existing page; the
Word/PDF pair remains 149 pages.
