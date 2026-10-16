# Verification of the defense reading edition

The current Word/PDF pair contains **149 total pages and 104 main-body pages**,
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

The [current summary](document-checks/manuscript-verification.json) identifies these
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
