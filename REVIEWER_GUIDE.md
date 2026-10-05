# Guide for a research reviewer

**Author-review edition · 4 October 2026 · authoritative branch: `praxis-realignment-v3`**

The central question is whether a selective phishing detector can satisfy
representation, false-alarm and real-service constraints together. This repository
connects the implemented system to the data, measurements and written answers.
It distinguishes completed experiments from successful operating requirements.

## A short reading route

1. Read the [research argument](dissertation/2026-10-04/RESEARCH_STORY.md).
   Each research question has a direct answer, its numerical requirements and
   links to evidence. The original H1–H3 decisions and the later D/S comparisons
   are kept separate.
2. Read Chapters 3–5 in the [manuscript](dissertation/2026-10-04/README.md).
   Chapter 3 defines the design and decision rules; Chapter 4 presents the
   measurements; Chapter 5 explains the answers, contributions and limits.
   The manuscript explains the method without requiring the reader to navigate Git.
3. Download the [retained research archive](research-archive/2026-10-04/README.md).
   This includes raw licensed inputs, prepared partitions, fitted artifacts,
   row-level predictions, individual request records, freezes and interruptions.
4. Verify the archive and run the read-only recomputation. This checks saved
   observations; it does not retrain, revisit malicious URLs or manufacture new
   timing evidence. See the archive guide for commands and exact scope.

## Follow a claim to its evidence

| Review question | Where to look | What can be checked |
|---|---|---|
| Do the questions, methods and conclusions form one argument? | [Research story](dissertation/2026-10-04/RESEARCH_STORY.md), manuscript Sections 1.4–1.5, 3.6–3.12, 4.2–4.7 and 5.1 | Exact questions, experiments, denominators, direct answers and all 22 decisions |
| Where did the numerical requirements come from? | [Target rationale](#numerical-requirements), [frozen research basis](docs/research-basis.md) | Study-defined requirements, not invented literature constants |
| Which code produced the measurements? | [Provenance](research-archive/2026-10-04/PROVENANCE.md) | Original and follow-up full commit IDs, distinct publication revision |
| Which data and models were actually used? | [Archive locators](research-archive/2026-10-04/README.md#locating-the-evidence), [licenses](research-archive/2026-10-04/LICENSES.md) | Source versions, hashes, corrected partitions and accepted fitted artifacts |
| Can the reported numbers be recalculated? | `scripts/recompute_research.py`, [archive guide](research-archive/2026-10-04/README.md) | Confusion counts, intervals, gates, request quantiles, paired service comparisons and declared secondary metric arithmetic |
| Were interrupted or adverse results removed? | Archive history families and [chronology](dissertation/2026-10-04/RESEARCH_STORY.md#chronology-and-contribution-boundary) | Original hold, interrupted attempts, 72 retained plus 53 completed cells; separate 45-arm interruption and complete 80-arm schedule |
| Are tables, figures and references usable? | [Document checks](dissertation/2026-10-04/VERIFICATION.md#document-checks) | 27 tables, 302 data rows, four figures, 113 navigation targets and 58 reference entries |
| Who contributed, and what assistance was used? | [Authorship and assistance](research-archive/2026-10-04/PROVENANCE.md#authorship-and-assistance) | Preserved Git history and explicit assistance disclosure; no invented contributor roles |

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

The [gate table](dissertation/2026-10-04/aggregate-data/primary-gates.csv) is
authoritative for each comparison, operator and denominator. The thresholds were
not relaxed after observing outcomes. Nine component checks pass; thirteen do not.
All three original joint hypotheses are adjudicated as not supported. Useful
positive findings—including structural recall gains, exact scheme invariance and
the measured client-path improvement—are presented with their actual boundaries.

## Manuscript conventions and remaining formal review

The supplied copy uses the GWU Praxis structure, numbered sections, portrait
Letter pages, author–date citations and an unnumbered bibliography. The current
layout record counts **150 PDF pages, including 117 main-body pages**. The supplied
December 2023 D.Eng. guidelines set a 150-page total ceiling and suggest an
approximately 80-page body; the additional advising tips recommend a 70–90-page
body and approximately 95–110 pages overall. The copy is at the former ceiling
and above the recommended body length. That distinction is not a length waiver.

Both Amir Etemadi and Mazen Mheish are thanked. Historical advising credit is
not a substitute for confirmation of the final examination roles. Current
academic-integrity clearance, the applicability of program AI-use restrictions,
formal scope approval and final submission approval remain university decisions.
Publication, software tests and a similarity score cannot establish those decisions.
Private correspondence and marked-up historical manuscripts are not published here.
