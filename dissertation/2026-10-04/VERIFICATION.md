# Verification scope and chronology

## Scientific Evidence Retained from Completed Verification

- Original: [verification.json](aggregate-data/verification.json) records 238,159
  assertions, 7,340 file-hash comparisons, 250 successful owned operational exits,
  72 accepted retained cells and 53 newly completed cells. The complete reduction
  has 125 cells, 25 groups and 22 gates.
- D: [verification.json](aggregate-data/detection-D/verification.json) records
  saved-input/output arithmetic, source-order/label alignment, scheme invariance
  and domain-bootstrap uncertainty, with no new model execution by the verifier.
- S: [verification.json](aggregate-data/service-S/verification.json) records the
  complete 80 arms, all paired/group reductions, one measured error, sampled
  conditions and non-pooling of the interrupted schedule.
- Secondary: [secondary-verification.json](aggregate-data/secondary-verification.json)
  and [complete results](aggregate-data/complete-secondary-results.json) preserve
  the original declared secondary analyses and their qualifications.

These are historical retained verification reports, not new access to private
inputs during GitHub publication. Their hash references do not make every named
private input publicly available.

## Document Checks

The checked citation edition contains 150 PDF pages, 27 tables, four figures,
58 references and 113 navigation targets. Its
[layout report](document-checks/layout-audit/verification.json) verifies all 302
table data rows and 1,953 rendered paragraph/cell fragments. Targeted visual
inspection is documented [separately](document-checks/visual-review/inspection.txt);
automated checks are not represented as a visual review.

The [citation verification](document-checks/citation-verification.json) is a public
projection: private absolute input paths are replaced by basenames, while hashes
and findings are unchanged. Its original Word hash describes the sealed citation
edition. The public Word derivative's separate
[record](provenance/publication-copy.json) verifies unchanged visible text, table
XML and media bytes after removing template annotations and local-file links.

## Frozen-Source Test Failure and Publication Correction

The follow-up freeze's historical full-suite run had **13,016 passed, three
failed and two skipped**. The failures were the HTTP interruption test's three
parameterizations: its mock client lacked the `aclose()` lifecycle method now
used by the replay source. The historical failure and separate supplemental
check are retained in
[regression-verification-v1.md](provenance/followup/regression-verification-v1.md).

The publication checkout adds only that lifecycle method to the test double.
No measured runtime code, thresholds or reductions change. The local publication
check reproduces the old three failures before applying the correction; the
original frozen checkout remains unchanged. Later passing tests do not rewrite
the historical full-suite outcome.

## Publication Checks

The first expanded local publication check had 298 passing tests and one failing
service-fixture test. Isolated reproduction traced startup failure to the new
disposable environment's NumPy Accelerate wheel; the existing gate requires
OpenBLAS. The already hash-pinned macOS wheel was selected without changing
source, lock or thresholds. This local environment failure remains recorded;
it is separate from the older `aclose()` test-double issue.

After selecting the contract-pinned OpenBLAS wheel, the corrected focused run
passed **299 tests in 36.73 seconds** across 15 test files. Ruff check passed,
Ruff format reported 1,111 files already formatted, the dependency lock check
passed, the source distribution and wheel built, and CLI help exited
successfully. The [publication test receipt](provenance/publication-tests.json)
lists the exact focused selection and scope. This is not a new full-suite run
or a repetition of the scientific measurements.

The package manifest and publication tests verify file coverage and hashes,
all original adjudications, schedule completeness/accounting, data separation,
public-document annotation/link boundaries and current reader-facing links.
The publication audit additionally inspects selected text, Office ZIP parts and
PDF metadata/links for private paths and common credential patterns. It verifies
that original measured implementation files and frozen contracts are unchanged.

The package's `.gitattributes` preserves the hashed bytes on checkout rather
than converting line endings. Retained CSVs use CRLF; PDFs and Office files are
binary. Two byte-identical historical supplements retain their original final
blank line. These format-specific attributes resolve the staged whitespace
check without editing retained evidence or weakening checks on new prose/code.

The exact publication commit's full CI result is available in
[GitHub Actions](https://github.com/KrtiT/automated-phishing-detection-public/actions).
Do not infer a passing full suite from a passing package-only check or a commit
existing on GitHub. This record claims no institutional clearance, manuscript
length exception, committee-role confirmation or detector score.
