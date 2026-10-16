# Reproduce the control diagnosis

## Check the public evidence

Use the repository's Python 3.10 environment with NumPy/SciPy installed. The
aggregate check does not fit models or access row-level data:

```bash
python dissertation/2026-10-15/diagnostics/source/verify_public_diagnostics.py dissertation/2026-10-15/diagnostics
cd dissertation/2026-10-15
shasum -a 256 -c SHA256SUMS.txt
```

The manifest covers every file in this edition except itself. All 99 result
records are required. The aggregate check recalculates summary statistics and
checks source/protocol hashes. It cannot independently verify predictions without
the input partitions. `diagnostic-verification.json` records that separate local
check: it reconstructs every score, verifies its digest, and recalculates AUC by
weighted pair counting with exact tie handling and AP by cumulative rank counts.
It also verifies the five numerical continuations.

## Recreate the 99-fit experiment

Obtain the original licensed training and validation JSONL partitions using the
[research archive](../../research-archive/2026-10-04/README.md). Only
`data/processed/phiusiil-v1/train.jsonl` and `validation.jsonl` are needed. The
protocol requires these exact hashes:

- train: `575f2fb13a0766020e29d78bf8e633a185b381abde7060bdd1ed04cc4a5e38a0`
- validation: `970c6568a6400a1fc265b7809ef7bd9d1c297632799cbb88801313bc34ac415a`

The producer uses Python 3.10.19, NumPy 2.2.6 with OpenBLAS 0.3.29, SciPy 1.15.3
and scikit-learn 1.7.2. Follow the [OpenBLAS setup](../2026-10-04/REPRODUCTION.md#macos-arm64-test-environment).
The default macOS Accelerate wheel does not satisfy the runtime contract.
Feature code is from `ef6c4172d4320f81b154b0dea40911c3dc0015ae`.

The adapter below checks the partition, feature and label hashes, builds the
training-only scaler, and creates a new output directory. It was checked locally
against the original prepared arrays. Use absolute paths for the three input
locations and a new writable destination:

```bash
python /path/to/repository/dissertation/2026-10-15/diagnostics/source/prepare_public_rerun.py \
  --diagnostics /path/to/repository/dissertation/2026-10-15/diagnostics \
  --source /path/to/source-at-ef6c4172 \
  --data /path/to/data/processed/phiusiil-v1 \
  --output /path/to/new-control-run
python /path/to/new-control-run/diagnose_controls.py run --work /path/to/new-control-run
```

This runs all 99 seeds, with four processes and one numerical thread each.
It creates a local `private-cache/` containing derived row data; do not commit it.
The frozen producer retains each attempt and does not resume into an existing
`additional-fits/` directory. Preserve interrupted attempts; never replace a seed
based on its outcome. The adapter is a public replay convenience, distinct from
the byte-identical measured producer. Exact output bytes can depend on numerical
runtime; compare metrics, fitted state and warnings rather than timestamps.

## Recheck the five original P fits

`source/diagnose_controls.py prepare` additionally requires the original P
`fit-call-prepared.json`, model state and validation-prediction records. Those
fit receipts contain row identifiers and are retained privately. The published
saved-fit diagnosis records the checks; it does not grant access to those inputs.
`source/check_convergence.py` uses those same saved model files and the prepared
cache. The source is byte-identical to the measured continuation producer and
expects its work files alongside it. `source/summarize_diagnostics.py` expects
both the full original audit outputs and all 99 results in that same work layout.
Do not mistake an aggregate-only check for reconstruction of private fit receipts.

These checks describe one fixed dataset and estimator. They do not test future
traffic, establish independent campaign sampling, or certify that every possible
leakage path is absent. The protocols distinguish the diagnostic analyses from
the original hypothesis tests.
