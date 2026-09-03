# Research Status

**Status:** Active implementation under the August 20 written direction. The
PhiUSIIL development split and aggregate preparation record are complete; no
confirmatory v3 result has been produced.

| Field | Value |
|---|---|
| Protocol | `docs/advisor-approval/2026-08-16-realignment-matrix.md` |
| Protocol version | `1.2` |
| Protocol date | `2026-09-03` |
| Protocol SHA-256 | `ee0bdc75a0367b4ad1f745b3575a9843373519f9920cae39d0a5808507be1cd7` |
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
  label-blind rule in protocol v1.2.
- The aggregate [preparation record](../../reports/phiusiil-preparation-summary.json)
  reports source and output hashes, split counts, class counts, and every
  quarantine reason without publishing row-level data.
- PhishVN is reserved for one external evaluation after the protocol, data
  pipeline, models, thresholds, and analysis code are frozen.
- H1, H2, and H3 remain undecided until their stated evidence and gates have
  been evaluated. Exploratory v2 results do not decide them.

## Change Record

| Date | Version | Change |
|---|---|---|
| 2026-09-01 | 1.1 | Defined dataset-specific label mappings, quarantine rules, PhishVN v4 evidence strata, the H3 request-error construct, future-only routing, and the bounded contribution. |
| 2026-09-03 | 1.2 | Recorded active implementation under the August 20 direction and froze the PhiUSIIL canonicalization and label-blind domain-allocation algorithms before the development-data run. |

Any change to a research question, hypothesis, evidence designation, method,
or decision rule increments the protocol version and records a new hash before
the affected analysis runs.
