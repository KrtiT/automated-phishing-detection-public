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

## Public Retained Inputs and Recomputation

The [versioned research archive](../../research-archive/2026-10-04/README.md)
now provides the licensed publisher inputs, prepared partitions, fitted artifacts,
retained predictions and request records. Follow its download, checksum,
materialization and recomputation commands. Dataset licenses and changes are
documented separately; no publisher source was reopened for this publication.

`scripts/recompute_research.py` uses hash-authenticated saved observations to
recalculate domain bootstraps, confusion metrics, gate decisions, pooled request
quantiles and the D/S paired comparisons. This closes the prior aggregate-only
recomputation gap. Hash-only inventory entries alone cannot do so; excluded
private execution files are not required for the stated arithmetic.

Independent model retraining or new latency measurement is a different task.
It requires the frozen code/environment, appropriate data rights and a declared
execution design. Public metadata projections are not original private manifests,
and no download grants reusable execution authority or institutional approval.

## Analysis and Document Source

The current [evidence map](EVIDENCE_MAP.md) connects every displayed table,
figure and slide to its public sources. The repository-linked manuscript adds
the release citation and current publication scope; it does not change the
scientific results. Its [editorial ledger](document-checks/integration/editorial-ledger.json)
identifies all text changes from the sealed research-record commit.
`analysis-scripts/dissertation/integrate_repository_20261004.py` reconstructs
that bounded Word/Markdown/notes derivative from the local Git object at
`91995dd3fa6f0661d185999bab90a6fabb25d962`, using `python-docx` and `lxml`.
It takes the repository and an unused output directory as arguments; PDF export
and rendered-layout checks are separate. These document-tool dependencies do
not modify the frozen scientific runtime.

`analysis-scripts/` preserves source used to export, synthesize, format and check
the research package. These files retain controlled-workspace relative paths,
edition bindings and additional dependencies (for example document/PDF tools).
They are historical audit source, not a one-command reconstruction of every
document and experiment. The new public recomputation adapter deliberately
rebinds only the saved-observation checks to the materialized archive.
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

## Privacy-Projected Historical Reports

Three reports in the repository are now privacy-projected reading copies with
separate original and public SHA-256 identities in the
[privacy record](../../privacy/2026-10-04/publication.json). Their personal machine
paths are removed; scientific values and failures are unchanged. Frozen contracts
still bind the original bytes and reject these derivatives. Use the recorded
frozen revision and authorized retained originals for historical execution;
do not alter a pin to make a reading copy executable. Public saved-observation
recomputation against the nine research archives is unaffected.
