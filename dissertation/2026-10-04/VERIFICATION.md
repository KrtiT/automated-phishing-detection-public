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

Those reports preserve the original verification events. The later
[retained-record release](../../research-archive/2026-10-04/README.md) now makes
their scientific inputs publicly inspectable within its explicit inventory.
Private execution capability frames and full host/process logs remain excluded
or projected; their original hashes do not make the excluded bytes public.

## Public Saved-Observation Recomputation

The [new recomputation receipt](../../research-archive/2026-10-04/recomputation.json)
records a successful read-only reduction from materialized archive files:
125 cells, 25 groups, 1,243,505 original requests and 1,901 errors; 22 original
checks (nine pass, thirteen fail); 153 secondary population/model metric rows,
430 calibration bins and 396 monitor-window boundary checks; the 8,622-row D
comparison; and all 80 S arms/40 pairs/800,000 attempts with one error.
It authenticates the 220 files read by that arithmetic separately from the
archive verifier's complete file/blob accounting.

The original H1–H3 decisions and later D/S conjunctions are unchanged.
This recomputation uses saved observations and retained verifier algorithms;
it is not new prediction, training, timing, private execution-authorization
validation or an independent empirical replication. Historical controls whose
consumed-label provenance was incomplete remain qualified, not repaired by
publication. The release's archive checks and declared projection/exclusion
rules are documented in its own guide.

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

## Retained-Record Publication Checks

The final archive publication check passed 342 focused tests across 17
files, including archive safety, inventory authentication, saved-observation
input checks and dissertation consistency. Ruff check and formatting, the lock
check, CLI help and package build also passed. The new
[receipt](../../research-archive/2026-10-04/publication-checks.json) records this
event separately from the earlier 299-test publication check.

An independent code review identified four acceptance gaps in the first archive
tooling: unanchored inventory identities, unchecked prediction-file reopens,
incomplete secondary-coverage acceptance and materialization path collisions.
Regression tests reproduced these defects before the fixes. The final tooling
binds the exact inventory set to the committed catalog, reduces authenticated
prediction bytes, checks secondary/monitor coverage and rejects release-wide
path collisions before writing. The complete archive materialization and
scientific recomputation were then repeated successfully. These were verifier
safeguard defects, not evidence that the retained measurements were incorrect.

All nine archives were verified and materialized into a new destination before
recomputation. All 73,234 original source identities were rechecked. The
[content audit](../../research-archive/2026-10-04/content-audit.json) records the
bounded scan and its reviewed corpus-string false positives; it does not claim
that pattern matching can detect every possible secret. Published secondary,
monitor, D and S scientific objects were also compared with their authenticated
retained counterparts and matched. The original open Word file and the frozen
runtime, contracts and dependency lock remain unchanged.
