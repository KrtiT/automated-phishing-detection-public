# Research Status

**Status:** The August 20 advisor report directed continuation of the
three-question design with objective, reproducible outcome criteria. The
PhiUSIIL development split and aggregate preparation record are complete. The
RQ1 baseline contract and pure raw-URL feature extractor are frozen and
implemented. The prescribed training-only fit of the two logistic baselines and
validation-only threshold selection are in progress. No confirmatory v3 result
has been produced, H1, H2, and H3 remain undecided, and no PhishVN record has
been accessed.

| Field | Value |
|---|---|
| Protocol | `docs/advisor-approval/2026-08-16-realignment-matrix.md` |
| Protocol version | `1.4` |
| Protocol date | `2026-09-03` |
| Protocol SHA-256 | `2c2956e7cf958f9d2d948a2b1b665e214e12b175d84e4b73c766cc0a6e3be4de` |
| RQ1 baseline contract | `rq1-baselines-v1` |
| RQ1 baseline contract SHA-256 | `594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4` |
| Development source schema | `2` |
| Source-freeze release tag | `phiusiil-development-v1` |
| Source-freeze release | [`phiusiil-development-v1`](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/phiusiil-development-v1) |
| Development preparation | `complete` |
| Current technical milestone | RQ1 baseline contract and pure feature extractor complete; training-only fitting and validation-only threshold selection are `in_progress`. |
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
  label-blind rule in the protocol.
- `rq1-baselines-v1` freezes the 25 raw-URL features, their order and
  denominators, the shared logistic pipeline, partition use, and threshold
  rule before fitting. Its SHA-256 is recorded above.
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

## Execution Audit

On 2026-09-03, a broad local repository text search displayed row content from
the ignored PhiUSIIL `group_test.jsonl` file. This is recorded as analyst
access. The file was not accepted or read by the baseline command, no
group-test prediction or metric was produced, and the exposure did not change
the frozen contract, implementation, threshold rule, or hypothesis status. The
detailed record and mitigation are in the
[research evidence outline](../research-evidence-outline.md). No PhishVN record
was accessed.

When generated, `access.group_test_accessed=false` in a baseline artifact
describes only the `fit-baselines` process input boundary. It does not negate
the analyst access recorded here.

The protocol matrix remains the byte-for-byte v1.4 freeze record. Its embedded
status describes the state at freeze time; current execution status is
maintained here and in the evidence outline.

## Change Record

| Date | Version | Change |
|---|---|---|
| 2026-09-01 | 1.1 | Defined dataset-specific label mappings, quarantine rules, PhishVN v4 evidence strata, the H3 request-error construct, future-only routing, and the bounded contribution. |
| 2026-09-03 | 1.2 | Recorded active implementation under the August 20 direction and froze the PhiUSIIL canonicalization and label-blind domain-allocation algorithms before the development-data run. |
| 2026-09-03 | 1.3 | Clarified the bounded contribution claim, source-derived label provenance, and study-defined target basis; recorded completed preparation and the next baseline milestone. No experiment was run, and no RQ/H method or decision rule changed. |
| 2026-09-03 | 1.4 | Froze the exact RQ1 feature vector, shared logistic configuration, partition use, convergence handling, score meaning, and validation threshold algorithm before fitting. The feature extractor was implemented and tested; no model was fitted, no threshold was selected, and no RQ/H decision rule changed. |

Any change to a research question, hypothesis, evidence designation, method,
or decision rule increments the protocol version and records a new hash before
the affected analysis runs.
