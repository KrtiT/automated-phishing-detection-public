# Guide for a research reviewer

**15 October 2026 · authoritative branch: `praxis-realignment-v3`**

The central result is an explanation of why internally successful URL detection
fails external operating requirements, and why plausible representation, routing
and service changes address different parts of that failure.

## Reading route

1. Read the abstract and Chapters 1 and 5 in the
   [manuscript](dissertation/2026-10-15/manuscript/Tallam_Krti_Praxis_Strengthened_2026-10-15.pdf).
   They connect the research questions to the contribution and its design implications.
2. Read Chapters 3–4 for populations, paired estimands, controls and results.
   Sections 3.16 and 4.10.4 explain the shuffled-label mechanism. The
   [diagnostic report](dissertation/2026-10-15/diagnostics/REPORT.md) supplies
   complete aggregate evidence and a standalone figure.
3. Use the [current reproduction guide](dissertation/2026-10-15/REPRODUCTION.md)
   to check all 99 fits and five optimization continuations. Its adapter verifies
   the numerical inputs before a new permutation run. Original private fit receipts
   are required to re-authenticate the five earlier audited fits.
4. Follow the [original archive](research-archive/2026-10-04/README.md) and
   [component evidence](dissertation/2026-10-12/REPRODUCTION.md) for the remaining
   detection, routing and service results. Earlier review responses and slides
   remain companion records for their dated editions.

## Follow a claim to its evidence

| Claim | Manuscript | Supporting record |
|---|---|---|
| Internal feature gains fail to transfer | Sections 4.2 and 5.1 | Tables 4.2–4.3; original archive |
| Shared errors limit unchanged decision selection | Sections 4.8.1 and 4.10.2 | Table 4.22; original paired decisions; October 12 R aggregates |
| Scheme invariance is insufficient for useful detection | Sections 4.8.3 and 4.10.1 | Tables 4.16–4.17 and 4.21; D and F evidence |
| Shuffled training labels can retain substantial ranking | Sections 3.16 and 4.10.4 | [Mechanism report](dissertation/2026-10-15/diagnostics/REPORT.md), all 99 fits, five continuations |
| Expert computation and HTTP cost are distinct | Sections 4.9.2 and 4.10.3 | Tables 4.18–4.19 and 4.23; S and C evidence |
| Original requirements remain unmet | Sections 4.7 and 5.3 | [Corrected decisions](dissertation/2026-10-12/aggregate-data/corrected-primary-gates.csv) |
| Figures, tables and links survive export | Complete Word/PDF | [Verification](dissertation/2026-10-15/VERIFICATION.md) |
| Authorship and assistance are disclosed | Original provenance | [Authorship record](research-archive/2026-10-04/PROVENANCE.md#authorship-and-assistance) |

## Numerical requirements

The primary targets are **study-defined operating requirements**. Their rationale
is operational: a detector needs tolerable false alarms, a useful recall increment,
bounded escalation, responsive service and reliable requests. Prior work motivates
those concerns; it does not uniquely derive the chosen numerical cutoffs.

| Requirement | Interpretation |
|---|---|
| FPR and Tranco alert rate ≤1% | Specificity and label-free control safeguards; these are different denominators |
| Recall-contrast lower bound >0 | Evidence of incremental recall, rather than a favorable point estimate alone |
| Reference-window alerts ≤5%; external-window alerts ≥80% | Separate specificity and sensitivity requirements for the monitor, not URL-level FPR |
| Gold-recall lower bound ≥−0.02 | A two-percentage-point noninferiority margin |
| Physical transformer invocation ≤30% | A compute-use constraint, not a promised latency saving |
| Designated concurrency-64 p95 ≤200 ms; request errors <0.1% | Explicit service requirements under the specified host, workload and accounting rules |

The [corrected gate table](dissertation/2026-10-12/aggregate-data/corrected-primary-gates.csv) is
authoritative for each comparison, operator and denominator. The thresholds were
not relaxed after observing outcomes. Eight component checks pass after correcting
the original 69-pair noninferiority interval; one historical pass is not established.
All three original joint hypotheses are adjudicated as not supported. Useful
positive findings—including structural recall gains, exact scheme invariance and
the measured client-path improvement—are presented with their actual boundaries.

## Manuscript conventions and formal review

The current copy uses the GWU Praxis structure, double spacing, numbered sections,
portrait Letter pages, author–date citations and an unnumbered bibliography.
It contains **149 pages**, 31 tables and four figures. The supplied
D.Eng. guidance sets a 150-page total ceiling and recommends approximately 80
main-body pages; the manuscript meets the total ceiling but remains longer than
that body recommendation. The verification record gives the rendered boundaries.

Both Amir Etemadi and Mazen Mheish are thanked. Academic-integrity clearance,
formal committee roles, scope approval and final submission acceptance remain
university decisions. Publication and software checks do not establish them.
Private correspondence and marked-up historical manuscripts are excluded.
