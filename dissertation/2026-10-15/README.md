# Cross-dataset reliability: manuscript and control diagnosis

The manuscript explains why strong internal URL detection fails to transfer,
why shared errors limit expert routing, and why representation invariance and
service speed must be evaluated alongside detection quality.

- [Read the dissertation](manuscript/Tallam_Krti_Praxis_Strengthened_2026-10-15.pdf)
  · [Word copy](manuscript/Tallam_Krti_Praxis_Strengthened_2026-10-15.docx).
- [Shuffled-label mechanism and evidence](diagnostics/REPORT.md).
- [Reproduce and verify](REPRODUCTION.md) · [data dictionary](DATA_DICTIONARY.md).
- [Document and numerical verification](VERIFICATION.md) · [file hashes](SHA256SUMS.txt).

## What the evidence adds

The original external cohort contains 2,261 false positives shared by both
models among 2,497 certified negatives. Selecting between those unchanged
binary decisions has a 90.5487% false-positive floor. On the separate routing
cohort, wider gates add 18 false positives without selecting a repair. Universal
positive detection makes recall-only noninferiority a limited practical result.

Scheme interventions identify sensitivity to URL spelling. A separate refit
achieves exact scheme invariance across 8,622 pairs but loses detections and
retains poor specificity. The controls explain another consequence of the
format–class association: small coefficients learned from randomized labels can
still order the unchanged validation labels. All 99 additional permutations
complete without warnings; their AUCs span 0.1046–0.9496.
Fixed-score term removal and five stricter optimization checks support the
mechanism. These diagnostics explain the audited pipeline; they do not identify
collection-source causality or certify that every possible leakage path is absent.

The service experiments distinguish expert computation from request handling.
Worker-owned clients reduce median paired p95 by 79.68% in the tested synthetic,
structural-only HTTP setup. A similar improvement without a model supports a
client/transport explanation. Sparse expert calls increase scoring time by
1.646 times on a different workload; the two measurements cannot be pooled.

The original H1–H3 joint requirements remain unsupported, with eight passing
component checks of 22. The scientific contribution is the evidence explaining
these failures and the design decisions it constrains.

## Reading and evidence structure

The Word/PDF pair is the current reading edition: 149 pages,
31 tables, four original figures and 127 navigation targets.
The abstract and Chapters 1–5 now develop the connected argument. Sections 3.16
and 4.10.4 contain the control methods and mechanism results. Appendices retain
provenance and the distinctions between hypothesis tests and diagnostic analyses.

Earlier [October 4](../2026-10-04/README.md) and
[October 12](../2026-10-12/README.md) editions retain their original evidence.
The [October 12 response](../2026-10-12/advisor/Advisor_Response_2026-10-12.pdf)
and presentation remain historical companion materials, not revised copies of
this edition. All original numerical result tables and hypothesis decisions are
preserved. The new figure in `diagnostics/` supports the control report.
