# Cross-dataset reliability: manuscript and research evidence

CyberSentinel provides an empirical design basis for cascaded URL phishing
detection. The manuscript connects representation, expert selection and execution
cost through controlled comparisons, giving each design decision a measurable
benefit, trade-off and scope.

- [Read the dissertation](manuscript/Tallam_Krti_Praxis_Strengthened_2026-10-15.pdf)
  · [Word copy](manuscript/Tallam_Krti_Praxis_Strengthened_2026-10-15.docx).
- [Shuffled-label mechanism and evidence](diagnostics/REPORT.md).
- [Reproduce and verify](REPRODUCTION.md) · [data dictionary](DATA_DICTIONARY.md).
- [Document and numerical verification](VERIFICATION.md) · [file hashes](SHA256SUMS.txt).

## Research contributions

Structural features increase internal recall by 64.07 percentage points at
0.7747% FPR. Scheme interventions isolate sensitivity to supplied spelling, and
a refit achieves exact invariance across 8,622 paired inputs. Labeled comparisons
quantify the associated detection losses and remaining specificity shortfall,
connecting a verified representation property to its practical consequences.

Paired errors establish an exact routing bound for the evaluated decisions.
The models share 2,261 false positives among 2,497 certified external negatives,
imposing a 90.5487% false-positive floor on selection between their unchanged
binary decisions. Wider gates on a separate cohort add 18 false positives without
selected repairs. This gives redesign a concrete priority: improve the available
decisions before increasing calls. Recall-only noninferiority addresses preservation
at complete observed recall; specificity and corrections determine useful expert value.

Audited controls explain how small random coefficients can rank unchanged
validation labels when format and class are associated. All 99 additional
permutations complete without warnings, spanning AUC 0.1046–0.9496. Fixed-score
term removal and five stricter optimization checks support the mechanism.
These analyses characterize the audited pipeline; they do not identify
collection-source causality or certify every possible leakage path.

The service experiments demonstrate a client-path optimization and quantify
expert work separately. Worker-owned clients reduce median paired p95 by 79.68%
in the tested synthetic structural-only HTTP setup. A similar no-model improvement
supports a client/transport explanation. Sparse expert calls increase scoring
time by 1.646 times on a different workload; combining the results would not
establish a net transformer-service speedup.

The original H1–H3 joint requirements remain unsupported, with eight passing
component checks of 22. Their decisions distinguish gateway readiness from the
measured representation, routing, control and service contributions.

## Reading and evidence structure

The current Word/PDF edition contains 149 pages, 31 tables, four original
figures and 127 navigation targets. The abstract and Chapters 1–5 lead with what
the experiments establish and the engineering choices they inform. Sections 3.16
and 4.10.4 explain the control methods and mechanism. Appendices retain provenance
and the distinctions between hypothesis tests and diagnostic analyses.

Earlier [October 4](../2026-10-04/README.md) and
[October 12](../2026-10-12/README.md) editions retain their original evidence.
The [October 12 response](../2026-10-12/advisor/Advisor_Response_2026-10-12.pdf)
and presentation remain historical companion materials for those editions.
All original numerical result tables and hypothesis decisions are preserved.
The supporting diagnostic figure is in `diagnostics/`.
