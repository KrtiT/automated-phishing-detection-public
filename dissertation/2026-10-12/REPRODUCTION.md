# Reproduction and frozen evidence

## What is published

The package includes the manuscript and checked PDF, the seven-comment response,
16-slide deck, extended methods, all F/R/C/P aggregate outputs, corrected original
ledger, and 35 exact specification/code/test snapshots in `research-snapshot/`.
Those snapshots are byte-identical to the locally frozen profile. Their base
research code is commit `f77027b3963f4e7ee6b831b9f09ff04c71491001`; publishing a
later document commit does not change the producer revision.

[freeze-projection.json](provenance/freeze-projection.json) records all 506 bound
content hashes, the permitted snapshot locators and non-identifying runtime
fields. Private paths are replaced by opaque binding ordinals. The projection
is disclosed and is not the original execution profile or an authorization.
[freeze-identities.json](aggregate-data/freeze-identities.json) retains exact
profile and output identities. [preservation.json](provenance/preservation.json)
identifies the unchanged submitted manuscript and the revised source documents.

The new raw records, URLs, domains, row-level predictions, host observations and
execution capability/profile are retained privately. The separately published
[October 4 archive](../../research-archive/2026-10-04/README.md) remains the route
to the original licensed data, models, predictions, request records and freezes.
It is not replaced or silently extended by this package.

## Safe public checks

From the repository root, use the locked Python 3.10 environment described in the
root README. The tests below use public artifacts or invented fixtures; they do
not initiate research fitting, predictions, timing or visits to URL destinations.

```bash
uv run --locked --no-sync pytest -q tests/test_dissertation_revision.py
PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync pytest -q dissertation/2026-10-12/research-snapshot
```

Verify the entire edition independently:

```bash
cd dissertation/2026-10-12
shasum -a 256 -c SHA256SUMS.txt
```

The snapshot tests require the repository package importable in that environment.
The publication-only conftest relocates the ledger test's two public aggregate
fixture paths after checking their original pinned hashes. It changes no archived
source bytes, expected result, threshold or assertion. With a shared pre-existing
Python environment, also set `PYTHONPATH=src` to avoid importing an older checkout.
The archived runners deliberately retain their original relative local-tree
assumptions and guarded inputs. They are frozen source evidence, **not a turnkey
public execution bundle**. Moving snapshots does not recreate private inputs or
grant permission to repeat the one-run study. Do not run the measurement runners
to check a document or aggregate.

## What each verification establishes

The public package tests check hashes, complete schedules, all seeds, the
corrected ledger, document structures, current links and specified privacy
patterns. They do not independently reproduce private row-level results.

The local saved-evidence verifier separately checked 506 bound files and 258 host
observations and reconstructed P's consumed-label linkage, F's paired counts,
R's memberships/repairs/ranking and C's arms/counters/quantiles. It did not load
models to make new predictions, refit models or create new timings. Its retained
receipt and source are provided in `document-checks/` and `verification-source/`;
the latter still requires authorized retained originals and the original tree.
Not every secondary F feature delta or R source-date/FPR-bound summary was
independently recomputed by that verifier. Saved-observation arithmetic checking
is distinct from empirical replication on new data.

Rates in CSV are fractions unless a column names milliseconds or another unit.
Nulls and blank cells mean undefined/not applicable, never zero. See the
[data dictionary](DATA_DICTIONARY.txt) before comparing denominators. New R
recall-only decisions do not overwrite original H1–H3. C and S are different
workloads and measurement layers and must not be pooled.
