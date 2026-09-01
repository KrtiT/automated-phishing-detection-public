# Advisor Approval Status

**Status:** August 20 advisor direction received; formal v1.1 approval pending before protocol freeze.

| Field | Value |
|---|---|
| Matrix | `docs/advisor-approval/2026-08-16-realignment-matrix.md` |
| Matrix version | `1.1` |
| Matrix date | `2026-09-01` |
| Matrix SHA-256 | `572bc24a4d2638966bfdc0db2c625610d0f718ca21b928314b985cc0e31de736` |
| Legacy base tag | `legacy-v2-clean-2026-08-16` |
| Legacy base SHA | `a5eceecf21ad5ce29c4ab8f8d4de0edc8b73b240` |
| Report date | `2026-08-20` |
| Report SHA-256 | `c4be5b5d5c4eaf89a2494a521522d124aa45351ee8d08dda2f41b01b961a0b7a` |
| Sent | |
| Response | The August 20 written report gave conditional direction to proceed in the presented direction while requiring objective, reproducible criteria for the phishing and legitimate reference classifications rather than labels assigned or changed by an individual. |
| Decision | Proceed in the presented direction with objective and reproducible label criteria. Formal written approval of v1.1 remains pending before protocol freeze. |

## Approval Gates

- No access to any PhishVN v4 record before formal written approval of v1.1 and protocol freeze.
- After formal written approval of v1.1 and protocol freeze, verify source schema, provenance, and label encoding before processing any record; freeze deterministic mapping and exclusion counts before examining model predictions or inferential results.
- No confirmatory claim before the frozen external evaluation is complete.
- No hypothesis-driven confirmatory experiment before formal written approval of v1.1 and protocol freeze.

## Change Control

Every edit made after the matrix is sent must be recorded below and followed by a new matrix hash. A material change to a research question, hypothesis, method, evidence designation, or decision rule requires a version increment, renewed formal written advisor approval, and protocol re-freeze before test access. An editorial correction that does not alter interpretation does not reopen approval, but it still requires a dated entry and updated hash.

| Change date | Matrix version | Material change | Formal approval status |
|---|---|---|---|
| 2026-09-01 | 1.1 | Material: defined dataset-scoped mechanical outcome mapping and quarantine rules; proposed PhishVN v4 and evidence strata; defined the H3 request-error construct; clarified target bases, observed certified-registry FPR and Tranco control alert-rate gates, future-only replay, and the bounded contribution. | Formal written approval pending |
