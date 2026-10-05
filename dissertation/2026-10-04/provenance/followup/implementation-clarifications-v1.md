# Implementation clarifications before fitting or scoring

October 1, 2026, after the metadata/overlap-only admission and before any new model
fit or benchmark predictions. These details instantiate, rather than change, the
scientific comparisons in `comparison-specification-v1.md`.

- Prediction for both models is singleton sklearn inference, with the existing
  independent float64 reference audit. Training-only scaling and validation-only
  calibration remain as specified. BLAS thread count is one for fitting/scoring.
- Scheme swapping preserves the original remainder and uses a lowercase opposite
  scheme. It is a modeling counterfactual, not a label for a fetched destination.
- Bootstrap domain order is lexicographic (`numpy.unique`); each replicate draws
  D domain indices uniformly with replacement using PCG64 seed 20261001. All rows
  in a selected domain receive that domain's multiplicity. Undefined denominators
  are counted and excluded only from the corresponding interval calculation.
- Calibration reporting uses ten fixed equal-width probability bins, with
  probability 1 in the final bin. A zero-alert precision is undefined, not zero.
- URL-overlap checking used both raw and publisher-normalized PhishVN fields
  across its complete retained publisher manifest. The 50,832 unparseable field
  instances are not 50,832 missing rows: many raw fields lack an absolute scheme
  while their paired publisher-normalized fields are parseable. Original malformed
  observations do not acquire guessed domains. The valid-domain exclusion scope
  and this limitation remain explicit.
- The first download failure is retained; the second, identified User-Agent
  request returned bytes matching the exact original publisher hash. The same
  dataset and all eligibility rules were retained. Admission retained 8,622 rows
  (4,651 publisher-phishing; 3,971 publisher-legitimate) after 2,808 quarantines.
  These counts were observed before predictions and do not reveal model efficacy.

No claim is made that publisher labels are independently adjudicated or that this
benchmark represents current prevalence or a fully independent feed mechanism.
