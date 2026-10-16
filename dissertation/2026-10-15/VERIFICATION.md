# Verification of the reading edition

The Word/PDF pair contains 149 total pages and 104 main-body pages,
31 tables, four original figures and 127 checked navigation targets. Double
spacing and Times New Roman are retained. The 161-word abstract fits one page.
The manuscript meets the supplied 150-page total ceiling; the body still exceeds
the approximately 80-page recommendation.

## Scientific and document checks

- All 99 additional permutation fits and five stricter continuations completed
  without warnings. Independent score reconstruction and separate tie-aware
  AUC/AP implementations reproduce the retained metrics.
- The public checker recalculates aggregates and rejects three deliberate
  defects: a missing seed, a changed metric and changed producer source.
- The public replay adapter reconstructs numerical input arrays matching the
  original SHA-256 digests, without requiring the private fit receipts.
- The document audit verifies all original numerical result tables, hypothesis
  statements, references and original figure bytes. Changes are logged for
  95 original paragraphs, ten additions and 13 interpretive table cells.
- All 2,119 nonempty paragraphs and table cells render. No table caption is
  detached and no figure intersects prose. Internal and exported external links,
  references and document privacy checks pass. The changed prose and selected
  final pages received a visual/content review.

A final sentence-structure pass improves 40 paragraphs without removing research
content. Its separate verification confirms unchanged tables, image bytes,
paragraph and run formatting, hyperlinks and all unedited text. All numerical
tokens in the edited paragraphs are preserved; the changes received a semantic
review for retained findings, methods, qualifications and citations.

A further contribution-focused pass reframes 43 paragraphs around measured
achievements and engineering decisions. The independent structural check verifies
both editorial stages against their preserved predecessors. All original numerical
tokens, tables, images, paragraph/run formatting and hyperlinks remain intact.
The abstract adds two existing results: the 64.07-percentage-point internal recall
gain and verified invariance across 8,622 pairs. Formal hypothesis decisions,
detection losses, false-positive rates, and methodological qualifications remain
explicit in the main text.

The author requested removal of two sentences about practitioner discussions
from the acknowledgements. An exact comparison with the preceding published
edition confirms this is the only manuscript change, with the remaining
acknowledgements, research content and formatting preserved.

Word SHA-256: `0ffa5fc27ef87b87ac80f84f9592fe2d362bf933395b36ed41b7fa6acb6a0637`

PDF SHA-256: `1751a4b286b3e9becb6462384a8552f2677bf61695990f2295d60f285879b3eb`

The machine-readable receipts are in `document-checks/` and `diagnostics/`.
The change ledger is an editorial record; it does not overwrite prior editions.
The diagnostic source and protocols retain their measured bytes and identities.

## Repository checks

Focused publication tests and a final package scan are recorded in
`document-checks/repository-checks.json`. They check the reading routes,
existing evidence and the new package; no claim is made that the complete
research software suite passes. The earlier [verification record](../2026-10-12/VERIFICATION.txt)
documents a pre-existing full-suite fixture incompatibility involving frozen
hashes after privacy projection. This revision does not change production
inference code or the original frozen research contracts.

The checks establish document integrity and reproducibility of the measured
calculations. They do not establish deployment suitability, causal provenance
of the collection convention, independent population replication or university
submission approval.
