# Verification of the defense reading edition

The current Word/PDF pair contains **150 total pages and 104 main-body pages**,
31 tables, four original figures and 127 checked navigation entries. Times New
Roman and double spacing are retained. The approved **219-word abstract spans two
pages**. The document is at the supplied 150-page total ceiling; its body exceeds
the approximately 80-page recommendation.

## Scientific and document checks

- All 99 additional permutation fits and five stricter continuations completed
  without warnings. Independent score reconstruction and separate tie-aware
  AUC/AP implementations reproduce the retained metrics. The public checker
  verifies aggregates and rejects missing seeds, changed metrics and changed
  producer source. Historical-fit authentication still requires private receipts.
- The preceding RQ synthesis changed 32 paragraphs and five interpretive cells
  in Table 5.1. Its exact-XML verification retains its original document hash in
  [three-contribution-verification.json](document-checks/three-contribution-verification.json).
  Earlier sentence and framing checks retain their stage-specific meaning.
- The author then saved the document in Word, changing the title date from
  October 2 to November 5th, 2026. Comparison of all 2,254 XML paragraphs finds
  no other text change. All table-cell text, math XML, image dimensions and
  decoded pixels, page sizes, margins and active hyperlink labels/destinations
  are preserved. Word re-encoded images and changed field/style serialization;
  these are not claimed to be package-byte identical to the preceding stage.
- The approved abstract replaces only its existing paragraph. Five front-matter
  page caches were updated. All other XML nodes and package parts are identical
  to the author-saved document. The
  [abstract/export receipt](document-checks/abstract-update-verification.json)
  verifies that chain explicitly.
- Complete text coverage, all 31 tables and four figures pass the export audit.
  No table caption is detached and no figure intersects prose. All 127 navigation
  entries, internal click destinations, exported external links and citation
  checks pass. Effective double spacing is checked through document defaults,
  inherited paragraph styles and direct formatting. Both abstract pages and the
  updated contents received visual review. Package inspection flags no hidden
  review content, local paths or credentials.

The research questions, original result values, formal hypothesis decisions,
equations, bibliography and research evidence remain intact. The author-confirmed
committee roles are recorded for the upcoming defense; the copy does not assert
a completed final examination. The inaccurate practitioner-discussion
acknowledgement remains removed.

Word SHA-256: `500f5ca98cce81f92ac76a9d4f23db01377e1c05a6d55ac73e2c44f7a5f8f384`

PDF SHA-256: `7087f0ab1ab4b97a3284bf35719da6fc6749748a3ef9ec4d65a878bdff7ef4d6`

The current summary is in
[manuscript-verification.json](document-checks/manuscript-verification.json).
Earlier receipts are evidence for their named stages; the current summary and
[final prose receipt](document-checks/posture-verification.json) identify the final artifacts. The
[claim-to-evidence map](document-checks/three-contribution-evidence-map.json)
was established against the preceding verified manuscript; the final update
preserves its underlying scientific statements and sources.

## Final interpretive prose revision

The final pass revises 27 paragraphs to lead with research answers, quantitative
findings and engineering implications. The
[change log](document-checks/posture-changes.json) records each edit. The
[verification receipt](document-checks/posture-verification.json) compares the
result directly with the approved abstract edition: all 31 tables, numerical
result tokens, formal hypothesis decisions, methods disclosures, references,
appendices, image bytes and paragraph formatting are preserved. Four navigation
page caches were refreshed. Every other document node and package part is unchanged.
The approved abstract and author-saved title date remain exact. The discussion,
hypothesis-relationship section and conclusion received visual review after export.

## Repository checks

The complete local software suite passed **13,313 tests**, with **two skipped**
and **zero failures**. Ruff, formatting, dependency-lock, CLI and package-build
checks also passed. These results are recorded in
[software-checks.json](document-checks/software-checks.json), separately from the
focused publication checks in
[repository-checks.json](document-checks/repository-checks.json).
Synthetic execution fixtures authenticate two approved public compatibility-report
projections before using their public hashes in mock bindings. Regression tests
reject their use as original historical evidence and reject tampered reports.
Production binders and frozen research contracts are unchanged.

On macOS arm64, numerical tests use the contract-required NumPy 2.2.6/OpenBLAS
0.3.29 build described in the reproduction instructions. The default Accelerate
build is rejected by the scientific runtime checks. GitHub CI is a separate run
on the pushed commit; the local receipt does not certify its outcome.
