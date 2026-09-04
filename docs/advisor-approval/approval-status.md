# Research Status

**Status:** The August 20 advisor report directed continuation of the
three-question design with objective, reproducible outcome criteria. The
PhiUSIIL development split and aggregate preparation record are complete. The
current technical milestone is frozen raw-URL feature extraction and
`length-only` and `Logistic-L1` baseline implementation using only the train
and validation partitions. No confirmatory v3 result has been produced, H1,
H2, and H3 remain undecided, and no PhishVN record has been accessed.

| Field | Value |
|---|---|
| Protocol | `docs/advisor-approval/2026-08-16-realignment-matrix.md` |
| Protocol version | `1.3` |
| Protocol date | `2026-09-03` |
| Protocol SHA-256 | `bf38ed9eb69acfb4ccef0e647a9e45140f7c4ee9274e9d7ded2a966fcd3cd0fb` |
| Development source schema | `2` |
| Source-freeze release tag | `phiusiil-development-v1` |
| Source-freeze release | [`phiusiil-development-v1`](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/phiusiil-development-v1) |
| Development preparation | `complete` |
| Current technical milestone | Frozen raw-URL feature extraction; `length-only` and `Logistic-L1` baseline implementation using only the train and validation partitions. |
| Current hypothesis status | H1 `undecided`; H2 `undecided`; H3 `undecided` |
| Legacy base tag | `legacy-v2-clean-2026-08-16` |
| Legacy base SHA | `a5eceecf21ad5ce29c4ab8f8d4de0edc8b73b240` |
| Advisor report date | `2026-08-20` |
| Advisor report SHA-256 | `c4be5b5d5c4eaf89a2494a521522d124aa45351ee8d08dda2f41b01b961a0b7a` |
| Direction | Proceed with the three-question design and make the outcome criteria objective and reproducible. |

## Current Controls

- PhiUSIIL is the only development source. Its native labels are mapped by an
  exact rule, and no researcher assigns or changes a row's outcome.
- Source bytes, the Public Suffix List, the preparation algorithm, and every
  generated split are identified by SHA-256.
- Registrable domains are assigned to one development split by the frozen,
  label-blind rule in protocol v1.3.
- The aggregate [preparation record](../../reports/phiusiil-preparation-summary.json)
  under source schema version 2
  reports source and output hashes, split counts, class counts, and every
  quarantine reason without publishing row-level data.
- The [`phiusiil-development-v1`](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/phiusiil-development-v1)
  GitHub Release is the source-freeze record for the completed preparation
  milestone. It keeps the exact licensed UCI archive outside Git history and
  binds it to the archive and CSV checksums in `data/sources.json`.
- PhishVN is reserved for one external evaluation after the protocol, data
  pipeline, models, thresholds, and analysis code are frozen.
- H1, H2, and H3 remain undecided until their stated evidence and gates have
  been evaluated. Exploratory v2 results do not decide them.

## Change Record

| Date | Version | Change |
|---|---|---|
| 2026-09-01 | 1.1 | Defined dataset-specific label mappings, quarantine rules, PhishVN v4 evidence strata, the H3 request-error construct, future-only routing, and the bounded contribution. |
| 2026-09-03 | 1.2 | Recorded active implementation under the August 20 direction and froze the PhiUSIIL canonicalization and label-blind domain-allocation algorithms before the development-data run. |
| 2026-09-03 | 1.3 | Clarified the bounded contribution claim, source-derived label provenance, and study-defined target basis; recorded completed preparation and the next baseline milestone. No experiment was run, and no RQ/H method or decision rule changed. |

Any change to a research question, hypothesis, evidence designation, method,
or decision rule increments the protocol version and records a new hash before
the affected analysis runs.
