# Diagnostic data dictionary

All published observations are aggregates. Rates and AUC/AP use fractions;
`elapsed_seconds` measures local execution duration and is not a service benchmark.

| File or field | Meaning |
|---|---|
| `PROTOCOL.json` | Frozen specification for all 99 training-label permutations |
| `CONVERGENCE_PROTOCOL.json` | Frozen continuation specification for five saved controls |
| `feature-order.json` | Ordered 25-feature vector; index 20 is `is_https` |
| `preparation-receipt.json` | SHA-256 identities of partitions and numerical inputs |
| `saved-fit-diagnosis.json` | Five original controls, input-audit outcomes, class-format counts and fixed-score sensitivities |
| `additional-fits/*-started.json` | Per-fit pre-execution input, label, permutation, code and parameter hashes |
| `additional-fits/*-result.json` | The corresponding result; all 99 attempts are retained |
| `coefficients`, `intercept` | Logistic score on training-standardized features; feature order is fixed |
| `auc`, `ap` | True-label validation ROC AUC and average precision |
| `without_https_auc` | AUC after subtracting only standardized HTTPS × coefficient from the fixed logit |
| `randomized_validation_auc` | Same scores, independently permuted validation labels; seed 210000 + fit seed |
| `domain_equal_weighted_auc` | Each row weighted by inverse validation domain size, so each domain has total weight one |
| `score_sha256` | Ordered float64 little-endian validation probability digest |
| `consumed_labels_sha256` | Ordered int64 little-endian training-label digest at fitting |
| `kkt` | L1 logistic stationarity residuals in sum-loss units and after division by total sample weight |
| `convergence/*-result.json` | Original/refined AUC, stricter fitted state, iterations, residuals and warnings |
| `permutation-summary.csv` | One row per additional permutation; all 99 in seed order |
| `original-controls-summary.csv` | Five original controls, fixed-score removal and numerical continuation |
| `diagnostic-summary.json` | Statistics of the complete fixed experiment; quantiles are empirical, not population confidence intervals |
| `diagnostic-verification.json` | Retained-data independent AUC/AP calculations and exact score-digest checks |
| `public-aggregate-verification.json` | Public checks of complete schedules, source hashes and aggregate arithmetic |

No threshold is selected in the additional permutations. The original five
controls' threshold-selected TP/FP remain in Table 4.24 and are not independent
performance estimates. Original H1–H3 decisions and D/S acceptance rules are not
changed by this diagnosis. Raw URLs, domains, row identifiers, predictions and
cache arrays are not included in this edition.
