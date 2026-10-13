# Cross-Dataset Reliability and Computational Trade-offs in Cascaded URL Phishing Detection

**Krti Tallam · author review revision · 12 October 2026**

This edition answers the seven review comments through corrected inference,
completed diagnostic comparisons and a clearer engineering argument. It connects
each question to a method, an observed result and a design decision. It preserves
the original numerical requirements and distinguishes successful properties from
the joint operating requirements that the detector does not meet.

## Read the revision

- [Main manuscript — Word](manuscript/Tallam_Krti_Praxis_Revision_2026-10-12.docx)
  · [PDF](manuscript/Tallam_Krti_Praxis_Revision_2026-10-12.pdf).
- [Response to all seven comments — PDF](advisor/Advisor_Response_2026-10-12.pdf)
  · [Word](advisor/Advisor_Response_2026-10-12.docx)
  · [plain text](advisor/Advisor_Response_2026-10-12.txt).
- [Advisor deck — PDF](advisor/Tallam_Advisor_Revision_2026-10-12.pdf)
  · [PowerPoint](advisor/Tallam_Advisor_Revision_2026-10-12.pptx)
  · [speaker notes](advisor/Speaker_Notes_2026-10-12.txt).
- [Extended methods and chronology — PDF](manuscript/Extended_Methods_and_Chronology_2026-10-12.pdf)
  · [Word](manuscript/Extended_Methods_and_Chronology_2026-10-12.docx).

Start with the main manuscript and response. The supplement preserves the fuller
methods/discussion record removed during condensation; it is supporting history,
not a competing version of the conclusions. The latest Word is authoritative;
the PDF is its checked rendering. There is no regenerated Markdown manuscript.

## Follow the argument to evidence

- [Research story and direct RQ answers](RESEARCH_STORY.md).
- [Seven-comment matrix](RESPONSE_MATRIX.csv) and [manuscript change map](MANUSCRIPT_CHANGE_MAP.txt).
- [Every table, figure and slide mapped](EVIDENCE_MAP.txt).
- [Aggregate data dictionary](DATA_DICTIONARY.txt) and [corrected 22-check ledger](aggregate-data/corrected-primary-gates.csv).
- [Closest-work contribution comparison](CONTRIBUTION_MATRIX.txt).
- [Reproduction guide](REPRODUCTION.md), [verification record](VERIFICATION.txt)
  and [complete file manifest](SHA256SUMS.txt).

The public package includes 35 byte-preserved specification/code/test snapshots,
the corrected ledger, all F/R/C/P aggregates and a [hash-only freeze projection](provenance/freeze-projection.json).
It does **not** contain the new record-level URLs, domains, predictions or private
execution profile. That boundary is deliberate; aggregate checks are not a claim
of complete public reproduction of private observations. The separately released
[original data and freezes](../../research-archive/2026-10-04/README.md) remain available
under their recorded licensing and privacy terms.

The original study has 22 checks, 25 operational groups and 125 cells. Corrected
inference leaves eight passing checks (H1: 5/10; H2: 1/4; H3: 2/8), rather than
the historical nine. H1–H3 remain unsupported as conjunctions. The two new R
recall-only statements are supported under their stated assumptions; they do not
retroactively change those decisions. Every adverse outcome remains reported.

The manuscript has 150 PDF pages, including 106 main-body pages, 31 tables and
four figures; the deck has 16 slides. The total meets the supplied 150-page ceiling,
but the body exceeds the approximately 80-page recommendation. This is a revised
author review package, not institutional acceptance or a length waiver. Both
Amir Etemadi and Mazen Mheish are thanked; formal committee roles remain for the
author and university to confirm.
