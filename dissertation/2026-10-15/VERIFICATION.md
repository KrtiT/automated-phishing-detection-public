# Verification of the defense reading edition

The current focused revision contains **149 total pages and 104 main-body pages**.
The 199-word abstract, conclusion, all 31 tables and four figures remain unchanged.
The [focused revision receipt](document-checks/chapter5-verification.json) compares
the current files with published commit f3e42b5. It records 14 revised paragraphs
in the thesis statement and Sections 5.1–5.2, preserving existing numerical results,
citations, qualifications and formatting. The thesis now includes the already
reported 79.68% median paired p95 reduction with its experimental scope.

[Every sentence changed in this final pass](document-checks/CHAPTER5_SENTENCE_CHANGES.md)
is paired with its earlier wording. Table 5.2 already contained “What it revealed”
and is preserved exactly. The updated Word/PDF pair passes complete text, table,
figure, navigation, citation, hyperlink and package checks. Earlier receipts retain
their original hashes; replaced summaries are preserved as `pre-chapter5-*`.

Current Word SHA-256: `393fabb4894200c968f67c00fed0d533bd1bdf219d1e7713377b2d1a2150221f`

Current PDF SHA-256: `8859901f2f57d4068072cb93276449a1af490051be2006f9b82191eb0ab93879`

## Preceding broader revision and retained verification

The following describes the preceding research-voice edition. Its checks remain
part of the verification chain; the current files are identified above.

The preceding Word/PDF pair contains **149 total pages and 104 main-body pages**,
31 tables, four original figures and 127 checked navigation entries. The
**199-word abstract fits one page**. Times New Roman, double spacing, margins,
paragraph styles and existing run formatting are preserved. The document is within
the supplied 150-page total ceiling; its body exceeds the approximately 80-page
recommendation.

## Scientific and document checks

The [revision receipt](document-checks/voice-verification.json) compares this edition
directly with the preceding published Word file. The
[sentence change log](document-checks/SENTENCE_CHANGES.md) records every changed
sentence, grouped where sentences split or join; the
[structured log](document-checks/voice-changes.json) retains complete paragraph pairs.

- Seventy-two prose paragraphs are revised across the abstract and Chapters 1–5.
  The argument leads with the working gateway, measured achievements and design
  lessons while retaining RQ1–RQ3. Substantive limitations remain in the relevant
  interpretations and Section 5.4. Diagnostic chronology is unchanged.
- Every existing numerical result in the edited paragraphs is preserved. The
  abstract adds internal structural recall of 98.6807% and FPR of 0.7747%, both
  already reported in Table 4.2b. No research measurements were changed or added.
- Table 4.14 changes 14 labels from “fail” to “not met,” without changing decisions.
  Table 5.2 adds “What it revealed” and eight concise insight cells. All existing
  cells retain their text; column widths accommodate the addition. The other
  29 tables retain their exact XML.
- Research questions, hypotheses, formal decision meanings, equations, references,
  figure bytes, acknowledgements, committee details and the author-saved title
  date remain intact. All document nodes outside the declared edits and refreshed
  navigation caches are unchanged; other DOCX package parts are byte-identical.
- Complete paragraph and table-cell coverage passes the PDF audit. All table rows
  and four figures render, no caption is detached, and no figure intersects prose.
  Navigation entries, internal destinations, exported external links and citation
  checks pass. Visual review covers the abstract, thesis, contributions, formal
  adjudication, expanded table, RQ synthesis and conclusion.
- Package inspection finds no tracked revisions, substantive hidden review content,
  local paths or credentials. The copy is for the upcoming defense and does not
  certify a completed examination. The inaccurate practitioner acknowledgement
  remains removed.

Word SHA-256: `056421a8941f5868a488a72634488a1dcf49977a3ff6fc13c3cc98191e4e25fa`

PDF SHA-256: `80c0308579148e70b2e9bb238ec827b03ac08ee344cf3227b1a515f1d35ffb5b`

The [preceding summary](document-checks/pre-chapter5-manuscript-verification.json) identifies those
artifacts. The [preceding prose receipt](document-checks/posture-verification.json),
[abstract/export receipt](document-checks/abstract-update-verification.json) and
[earlier RQ synthesis](document-checks/three-contribution-verification.json) retain
their original hashes and stage-specific meaning. Replaced summaries and render
checks are preserved as `pre-voice-*` records. The
[claim-to-evidence map](document-checks/three-contribution-evidence-map.json) retains
its source hashes; the research evidence is unchanged.

## Research evidence and repository checks

All 99 additional permutation fits and five stricter continuations are retained.
Independent score reconstruction and tie-aware AUC/AP calculations reproduce the
saved metrics. The public checker validates completeness, source hashes and
aggregate arithmetic; historical-fit authentication still requires private receipts.

The complete local software suite previously passed **13,313 tests**, with
**two skipped and zero failures**, on the source and runtime recorded in the
[software receipt](document-checks/software-checks.json). This revision changes
manuscript and documentation files. The
[current repository check](document-checks/repository-checks.json) verifies that the
tested software source fingerprint is unchanged, reruns the 18 historical-publication
regressions and public diagnostic checker, and checks this edition's package,
links, preserved evidence and hashes. These are distinct validation scopes.

On macOS arm64, the recorded numerical runtime uses NumPy 2.2.6/OpenBLAS 0.3.29.
GitHub CI reports separately for each pushed commit; the local receipt does not
certify a new remote run. Document and repository verification do not constitute
institutional approval.
