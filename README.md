# Automated Phishing Detection for Frontier AI Inference

## Current Work

This branch contains the active research implementation governed by protocol
v1.4. The PhiUSIIL preparation milestone and aggregate record are complete.
The [`rq1-baselines-v1`](data/rq1-baseline-contract.json) contract now freezes
the 25 raw-URL features, the two logistic baselines, and validation threshold
selection before model fitting. The pure feature extractor is implemented;
training-only fitting and validation-only threshold selection are next.

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
