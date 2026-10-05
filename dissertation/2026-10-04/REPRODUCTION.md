# Reproduction and access boundaries

## Publicly Checkable

- File identities: run `shasum -a 256 -c SHA256SUMS.txt` in this folder.
- Package coverage and published outcome inventories: from the repository root,
  run `uv run --locked --no-sync pytest -q tests/test_dissertation_publication.py`.
- Algorithm behavior on invented fixtures: the ordinary source test suite.
- Aggregate counts, gate arithmetic and the service paired-ratio reduction:
  CSV/JSON files in `aggregate-data/`, with their data dictionaries.
- The original measured code at `77d128377ce5b401437d7179f5cd78fb4294b72c`
  and follow-up measured code at `ef8ba5f0b357cf3dd60c4d663e6297d13334460c`.
  The later publication commit is not a new measurement revision.

## macOS Arm64 Test Environment

`uv sync --locked --python 3.10.19` can select a NumPy 2.2.6 wheel backed by
Apple Accelerate. The existing numerical-runtime gate instead requires NumPy's
`scipy-openblas` 0.3.29 backend; equal distribution versions alone do not satisfy
that gate. Use the wheel already pinned in the original GMM contract:

```sh
uv pip install --python .venv/bin/python --reinstall 'https://files.pythonhosted.org/packages/22/c2/4b9221495b2a132cc9d2eb862e21d42a009f5a60e45fc44b00118c174bff/numpy-2.2.6-cp310-cp310-macosx_11_0_arm64.whl#sha256=8e41fd67c52b86603a91c1a505ebaef50b3314de0213461c7a6e99c9a3beff90'
.venv/bin/python -c 'from automated_phishing_detection.gmm_monitor import _require_runtime; _require_runtime()'
```

Run through `.venv/bin/` or `uv run --locked --no-sync` after selecting that
wheel. Another environment sync may replace it. This is a disposable test
environment setup, not a modification of the frozen research environment or
authorization to execute research. Linux CI follows its locked Linux wheels.

## Requires Controlled Retained Inputs

Full source-to-result reproduction requires the exact licensed data versions,
retained row-level predictions, fitted states, manifests and the authorized
execution environment. They are intentionally absent from the public repository.
The public source manifest records origins and versions; it is not permission
to access or redistribute data. No original sources were reopened for publication.

The historical verifier reports document checks already performed against those
retained inputs. A reader can audit their algorithms and aggregate arithmetic,
but cannot independently recompute a domain bootstrap or pooled request-level
quantile from aggregate CSVs alone. Hash-only inventories do not close that gap.

## Analysis and Document Source

`analysis-scripts/` preserves source used to export, synthesize, format and check
the research package. These files retain controlled-workspace relative paths,
edition bindings and additional dependencies (for example document/PDF tools).
They are audit source, not a one-command reconstruction from public inputs.
`provenance/original/measured-source/` and `provenance/followup-source/` are
selected frozen source snapshots; Git history contains the full repository.

Historical Markdown links and source-inventory paths inside preserved records
refer to their original local package topology. They are not all public download
links. Use the current README, research story and dictionaries for this edition.
The sealed earlier editions and their full local logs remain preserved, not
duplicated as additional current manuscripts.

## Scientific and Software Verification Are Different

The publication test-double correction adds `aclose()` to the mock HTTP client
used in interruption tests. It matches the API exercised by the existing source;
no runtime/scoring/reduction code is changed from the follow-up measurement freeze.
The original full-suite failure record remains disclosed. A later passing test
does not rewrite the historical run or imply scientific support.

Read [VERIFICATION.md](VERIFICATION.md) for exact verification scope. Institutional
review, committee-role confirmation and academic-integrity clearance are separate
from repository checks.
