# v2 Legacy Preservation

This directory preserves selected implementation, data, test, documentary,
and reporting records from the `v2-clean` research phase.

## Provenance

| Field | Value |
| --- | --- |
| Source branch | `v2-clean` |
| Exact source commit | `a5eceecf21ad5ce29c4ab8f8d4de0edc8b73b240` |
| Local preservation tag | `legacy-v2-clean-2026-08-16` |
| Archive date | 2026-08-16 |
| Status | Historical and exploratory; superseded for future hypothesis decisions |

No file content or Git history was deleted during preservation. The selected
code, data, tests, documents, and reports relocated into this directory retain
their original bytes and filenames at corresponding paths under `legacy/v2/`.
The tag and exact commit retain the complete v2 snapshot, including files not
relocated here. Historical relative links should be read under `legacy/v2/` or
against the tagged tree.

## Basis for Supersession

The current release does not include frozen raw or processed inputs or saved
model bundles. Prior preprocessing accepted explicit inputs and also allowed
generated fallback data. Future confirmatory work requires stronger
source/domain separation and shortcut controls, as well as reconciliation of
report and threshold versions. Legacy endpoint timings omit raw URL feature
extraction, and the associated manuscript's transformer, Gaussian mixture
model (GMM), and distillation questions were not directly tested.

These limits constrain how the earlier work can be interpreted; they do not
remove it from the record. The old results remain visible for provenance and
audit and must not be cited as confirmatory v3 evidence.
