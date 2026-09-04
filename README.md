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
Protocol v1.7 freezes the separate
[`rq1-baselines-v2`](data/rq1-baseline-contract-v2.json) method before its
validation run; the rq1-baselines-v2 execution has status `frozen_not_run`. No
baseline model, threshold, or validation result has been accepted, and H1, H2,
and H3 remain undecided. No RQ/H decision rule changed, and no PhishVN record
has been accessed.

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

## Run the Frozen v2 Baselines

After the v1.7 protocol, contract, and implementation checkpoint is committed,
run the active baseline command once from that clean commit:

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

## Evidence Status

The [evidence outline](docs/research-evidence-outline.md) links each research
question to the artifacts and decision rules needed to answer it. Data
preparation is an input-control milestone, not a hypothesis result. H1, H2,
and H3 remain undecided. The [research basis](docs/research-basis.md) records
the source-provenance limits, ownership of the study-defined gates, closest
prior work, and narrow contribution boundary.

The complete v2 snapshot remains available at tag
`legacy-v2-clean-2026-08-16`, commit
`a5eceecf21ad5ce29c4ab8f8d4de0edc8b73b240`. Its exploratory results are kept
separate from the active evidence tree.

## Citation

If you use this repository, please cite it using `CITATION.cff` (GitHub will surface this automatically on the repo landing page).

## License

This project is released under the MIT License (see `LICENSE`).
