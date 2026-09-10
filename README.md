# Automated Phishing Detection for Frontier AI Inference

## Current Work

This branch contains the active research implementation governed by protocol
v1.7. The PhiUSIIL preparation milestone and aggregate record are complete.
The historical [`rq1-baselines-v1`](data/rq1-baseline-contract.json) contract is
preserved unchanged. Its prescribed fit stopped at the 5,000-iteration limit
with a convergence warning and produced no model or summary artifact. A later
local `tol=1e-4` observation is provenance-incomplete and is not research
evidence. The v1.5 training-only SAGA diagnostic has status
`stopped_platform_warning`.

The corrected v1.6 training-only diagnostic has status
`passed_training_only`. Both fresh runs produced the same iteration counts and
fitted-state hashes. The finite scoring outputs also matched an independent
full-probability-matrix calculation within the frozen tolerances. This
establishes numerical feasibility only.
Protocol v1.7 froze the separate
[`rq1-baselines-v2`](data/rq1-baseline-contract-v2.json) method before its
validation run. The single planned execution completed from clean commit
`7ae6c9af85e935c551468f590a7ba43441f58def` and has status
`completed_development_validation`. Its aggregate result is development
validation only, not confirmatory evidence. H1, H2, and H3 remain undecided, no
RQ/H decision rule changed, and no PhishVN record has been accessed. The public
[summary](reports/rq1-baseline-v2-summary.json) has SHA-256
`bf5b3a6f0fc705d26852da4dd0053c6111ffc3e500d7a2e95dfba5ad859b279c`.
The private `length-only` and `Logistic-L1` artifacts remain outside Git; the
summary pins them at SHA-256
`b8b92cfbe29160e769e5e7d80712becc8fc0680cdfd45a44b839ef9bada87799`
and `71a3e24a0283a31ba188bc7dd60b18c1b708370b9ca275d5ab1a1004680c968a`.
The group test remains analyst-exposed but model-unscored. The summary's
`access.group_test_accessed=false` records the baseline process input boundary;
it does not erase the separately recorded analyst exposure.

PhiUSIIL is a published collection of URLs from UCI dataset 967. Research
observations come only from the licensed source; the code does not assign
outcomes from personal judgment. Invented or synthetic URLs are unit-test
fixtures only; they are never research observations. The publisher's native
label is converted by one exact rule: `0` becomes local `is_phishing=1`, and
`1` becomes local `is_phishing=0`. Invalid labels, invalid URLs, and conflicting
canonical-URL groups are quarantined mechanically. Same-label duplicates retain
one deterministic record.

[`data/sources.json`](data/sources.json) is the source-of-truth manifest for
the recorded run. Its `contract_id` names the preparation rules; the manifest
hash and listed content hashes identify the exact source bytes. The
[`phiusiil-development-v1`](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/phiusiil-development-v1)
GitHub Release is the source-freeze record for the exact licensed UCI archive
outside Git history, tied to the archive and CSV checksums.

Retained registrable domains are assigned to train, validation, or group-test
once by the frozen hash rule in the
[research protocol](docs/advisor-approval/2026-08-16-realignment-matrix.md).
The procedure is label-blind and does not reroll or rebalance a split after its
class counts are known.

The baseline extractor accepts one raw URL and returns the contract's ordered
`float64` feature vector. It uses the untouched string for whole-URL counts,
`urlsplit` for component text, and the existing IDNA-normalized ASCII host for
host syntax. It does not accept labels, split membership, record or domain
identity, or publisher fields. Missing and invalid URLs stop extraction rather
than receiving imputed values.

## Reproduce the Preparation

Python 3.10 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked
mkdir -p data/raw/phiusiil data/processed reports

curl --fail --location \
  'https://archive.ics.uci.edu/static/public/967/phiusiil%2Bphishing%2Burl%2Bdataset.zip' \
  --output data/raw/phiusiil-phishing-url-dataset.zip
printf '%s  %s\n' \
  '0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a' \
  'data/raw/phiusiil-phishing-url-dataset.zip' \
  | shasum -a 256 --check
unzip -p data/raw/phiusiil-phishing-url-dataset.zip \
  PhiUSIIL_Phishing_URL_Dataset.csv \
  > data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv

curl --fail --location \
  'https://raw.githubusercontent.com/publicsuffix/list/0f1fa47ec45056a19c2fdcd32a08442de9715d12/public_suffix_list.dat' \
  --output data/raw/public_suffix_list.dat

uv run --locked phishing-research prepare-phiusiil \
  --csv data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv \
  --suffix-rules data/raw/public_suffix_list.dat \
  --source-spec data/sources.json \
  --output-dir data/processed/phiusiil-v1 \
  --summary data/processed/phiusiil-v1-summary.json

cmp data/processed/phiusiil-v1-summary.json \
  reports/phiusiil-preparation-summary.json
```

The command checks the pinned input hashes before parsing a record. Row-level
outputs and the reproduced summary remain local; `cmp` confirms that the run
matches the aggregate [preparation summary](reports/phiusiil-preparation-summary.json)
recorded in the repository. Run the software checks with
`uv run --locked pytest -q`.

The recorded run read 235,795 rows, retained 233,536 rows across 197,105
registrable domains, and quarantined 2,259 rows under the stated rules. The
summary gives the split, class, and quarantine counts together with the source
and output hashes.

## Historical v1.4 Baseline Attempt

The command below documents the stopped v1.4 attempt. It is reproducible from
implementation commit `e535586c6162a306a8dac7a5a6546f55dc09136f` after the
preparation record is reproduced. Current HEAD intentionally rejects the v1
contract because the active implementation is pinned to `rq1-baselines-v2`.

```bash
uv run --locked phishing-research fit-baselines \
  --train data/processed/phiusiil-v1/train.jsonl \
  --validation data/processed/phiusiil-v1/validation.jsonl \
  --preparation-summary reports/phiusiil-preparation-summary.json \
  --contract data/rq1-baseline-contract.json \
  --output-dir data/processed/rq1-baselines-v1 \
  --summary reports/rq1-baseline-summary.json
```

The v1.5 failure receipt is
[`reports/rq1-saga-convergence-v1-execution.json`](reports/rq1-saga-convergence-v1-execution.json).
The corrected v1.6 receipt is
[`reports/rq1-saga-convergence-v2-execution.json`](reports/rq1-saga-convergence-v2-execution.json).
It was produced from clean commit
`69a67d4e5cb81d49009d6a90e87f4c0c5f2cea87` by this training-only command:

```bash
uv run --locked python scripts/rq1_saga_convergence_diagnostic.py
```

## Recorded v2 Baseline Validation

The command below was executed once on 2026-09-04 from clean commit
`7ae6c9af85e935c551468f590a7ba43441f58def`, after its GitHub Actions checks
passed. It is retained here as the exact execution command:

```bash
uv run --locked phishing-research fit-baselines \
  --train data/processed/phiusiil-v1/train.jsonl \
  --validation data/processed/phiusiil-v1/validation.jsonl \
  --preparation-summary reports/phiusiil-preparation-summary.json \
  --contract data/rq1-baseline-contract-v2.json \
  --output-dir data/processed/rq1-baselines-v2 \
  --summary reports/rq1-baseline-v2-summary.json
```

The command accepts no group-test or external input. It verifies all four input
hashes before parsing, fits the scaler and both SAGA classifiers on training
data only, audits validation scoring against the independent numerical
reference, and then applies the unchanged validation threshold rule. Private
model records are portable JSON rather than pickle or joblib files. The public
summary contains aggregate counts, hashes, configuration, scoring-audit fields,
and validation metrics, but no URL, domain, record identifier, feature row, or
row score.

Both models completed below the 5,000-iteration limit and produced a threshold
under the frozen one-sided FPR-bound rule. These are validation-set operating
points, not internal-test or external results.

| Model | `n_iter` | Threshold | Validation recall | Observed validation FPR | One-sided 95% FPR upper bound |
|---|---:|---:|---:|---:|---:|
| `length-only` | 69 | 0.7612031186147 | 0.3214800576645843 | 0.006680191993666189 | 0.007702192373035135 |
| `Logistic-L1` | 4783 | 0.2670846328466124 | 0.9842223290084895 | 0.008758473947251225 | 0.009915480854183582 |

The full model recorded six allowlisted Accelerate scoring warnings. Its finite
decision values and full probability matrix matched the independent reference,
with maximum absolute differences of `2.1316282072803006e-14` and
`3.3306690738754696e-16`. The summary contains the exact counts and private
artifact hashes. Coefficients and sparsity are not interpreted as feature
importance.

## Frozen Transformer/Cascade Implementation

Protocol v1.8 keeps the prospective method frozen in
[`rq1-transformer-cascade-v1`](data/rq1-transformer-cascade-contract-v1.json),
SHA-256
`aeaa84534c4cadf0459cf6d2f010dc802684d4801cce563ce18242f36359fb54`.
The procedure code, tests, CLI, and private/public artifact publication code
are complete at final reviewed code commit
`0793ca3dbc36e49b561cd0ac74968a4644060426`, with status
`frozen_implemented_not_run`. No transformer fit, threshold calibration, or
cascade result exists. The commit records executable method code, not a
performance run. H1, H2, and H3 remain undecided. The group test remains
analyst-exposed but model-unscored, and no PhishVN record has been accessed.
Publication uses temporary paths and writes the public summary last as its
completion marker; caught in-process publication errors roll back both
destinations.

## Evidence Status

The [evidence outline](docs/research-evidence-outline.md) links each research
question to the artifacts and decision rules needed to answer it. Data
preparation is an input-control milestone, and the completed baseline run is a
development-validation milestone. Neither is a hypothesis result. H1, H2, and
H3 remain undecided. The [research basis](docs/research-basis.md) records the
source-provenance limits, ownership of the study-defined gates, closest prior
work, and narrow contribution boundary.

The complete v2 snapshot remains available at tag
`legacy-v2-clean-2026-08-16`, commit
`a5eceecf21ad5ce29c4ab8f8d4de0edc8b73b240`. Its exploratory results are kept
separate from the active evidence tree.

## Citation

If you use this repository, please cite it using `CITATION.cff` (GitHub will surface this automatically on the repo landing page).

## License

This project is released under the MIT License (see `LICENSE`).
