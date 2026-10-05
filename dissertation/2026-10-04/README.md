# Dissertation review package — 4 October 2026

**Automated Phishing Detection for Frontier AI Inference — Krti Tallam**

This is the completed research package for author and advisor review. It brings
the questions, implementation, measurements, engineering follow-ups and written
argument together. It is not a claim of university acceptance or advisor approval.

## Read in This Order

[Download this integrated review edition](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/dissertation-integrated-2026-10-04)
for the Word/PDF manuscript, presentation and complete editorial-package ZIP.
Its release notes link the unchanged research-data archives; the two releases
identify different layers of the same evidence record, not different results.

1. [Research story and answers](RESEARCH_STORY.md): what was built, what the
   original evaluation established, why two mechanisms were changed, and what
   the subsequent measurements support.
2. [Manuscript PDF](manuscript/Tallam_Krti_Praxis_Integrated_2026-10-04.pdf)
   or [Word review copy](manuscript/Tallam_Krti_Praxis_Integrated_2026-10-04.docx).
   The [Markdown edition](manuscript/Tallam_Krti_Praxis_Integrated_2026-10-04.md)
   supports searching and line-by-line comparison.
3. [Advisor presentation](advisor/Tallam_Praxis_Advisor_Integrated_2026-10-04.pdf),
   [editable slides](advisor/Tallam_Praxis_Advisor_Integrated_2026-10-04.pptx), and
   [speaker notes](advisor/Tallam_Praxis_Advisor_Integrated_2026-10-04_Speaker_Notes.txt).
4. [Original primary checks](aggregate-data/primary-gates.csv),
   [operational groups](aggregate-data/operational-groups.csv),
   [detection follow-up](aggregate-data/detection-D/verification.json), and
   [service follow-up](aggregate-data/service-S/verification.json).
5. [Every table, figure and slide → source evidence](EVIDENCE_MAP.md),
   [retained data, models and freezes](../../research-archive/2026-10-04/README.md),
   [verification](VERIFICATION.md) and [reproduction guide](REPRODUCTION.md).

## Inventory

| Component | Contents |
|---|---|
| Original primary evaluation | 22 adjudicated checks; nine pass, thirteen fail; all H1–H3 conjunctions not supported |
| Original operations | 125 cells, 25 five-repeat groups; 1,243,505 requests and 1,901 errors |
| Original secondary analyses | Population/model metrics; calibration; descriptive low-FPR curves; prevalence projections; monitor comparisons; seeds; formatting/permutation/RF controls; probes; McNemar/Holm contrasts |
| Detection D | 8,622 benchmark rows; domain-size aggregates; paired recall uncertainty; calibration; scheme invariance |
| Service S | Complete 80-arm schedule, 40 paired comparisons, eight groups, ten primary pairs and all five requirements |
| Manuscript | 150 PDF pages: 17 front pages, 117 main-body pages, then references/appendices; 59 references, 27 tables, four figures, 113 navigation targets |
| Advisor deck | 25 slides; unchanged October 2 visible results; embedded and exported notes now identify the public evidence in actual presentation order |
| Provenance | Frozen revisions, method/repair chronology, preserved historical records, source-copy identities and publication derivative record |
| Analysis source | 38 retained original/follow-up build and verification scripts plus later consistency/citation scripts; controlled-workspace dependencies remain explicit |

The three data namespaces are distinct. Top-level `aggregate-data/` CSV/JSON
files are byte-identical aliases of `aggregate-data/original/`, retained for the
manuscript's Appendix A locators; they are not extra observations. Use
`detection-D/` and `service-S/` only for their respective follow-ups.

[RESEARCH_DATA_DICTIONARY.txt](RESEARCH_DATA_DICTIONARY.txt) defines units,
denominators and follow-up fields.
[ORIGINAL_DATA_DICTIONARY.txt](ORIGINAL_DATA_DICTIONARY.txt) gives the complete
original schema. [DATA_DICTIONARY.txt](DATA_DICTIONARY.txt) is the preserved
October 2 delivery note: its references to earlier layouts and `prior-edition/`
describe that sealed local package, not additional files in this public edition.

## Publication Scope

This folder contains the manuscript, aggregate evidence and its provenance.
The separate [research release](../../research-archive/2026-10-04/README.md)
supplies retained licensed raw datasets, prepared splits, fitted artifacts,
row-level predictions, request records and freeze metadata. The separation keeps
large data out of ordinary Git history without hiding it from reviewers.
Private execution capabilities, correspondence and full host/process logs remain
excluded or explicitly hash-only; full-text third-party literature is not redistributed. The
[literature record](literature/README.md) distinguishes metadata/abstract checks
from full-text checks and states their limits.

The current **repository-linked edition** adds a formal citation to the versioned
research record, corrects data-availability wording and Appendix A locators, and
links the manuscript, deck and archive through the [evidence map](EVIDENCE_MAP.md).
Its [change ledger](document-checks/integration/editorial-ledger.json) and
[verification](document-checks/integration/verification.json) identify the exact
Word/PDF/Markdown files. Scientific tables and figure bytes are unchanged; the
only table edit corrects a section reference in Table A.1. Your original review
file, credential wording and both advisors' acknowledgments are preserved.

The [earlier Word copy record](provenance/publication-copy.json) records only the
preceding privacy cleanup, not this editorial revision. That earlier edition and
its assets remain at the immutable
[research-record-2026-10-04 tag](https://github.com/KrtiT/automated-phishing-detection-public/tree/research-record-2026-10-04/dissertation/2026-10-04).
Use the current links above for reading and that research release for the data.

Formal committee titles, final author review, program length requirements and
current academic-integrity clearance remain unresolved administrative items.
This package does not promise a similarity score or AI-detector outcome.
Assisted editorial preparation must be handled according to the program's rules.

## Integrity

[SHA256SUMS.txt](SHA256SUMS.txt) covers every file in this folder except itself.
From this directory, run `shasum -a 256 -c SHA256SUMS.txt`. The
[source inventory](provenance/public-source-inventory.json) records the earlier
assembly of byte-identical copies, identified by the release tag and original
hashes; current editorial derivatives have the separate integration record above.
Hashes establish file identity;
the archive's separate recomputation checks arithmetic against retained observations.
Neither is an independent repeat of model fitting or measurement.
