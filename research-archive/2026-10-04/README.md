# Retained research archive

This archive accompanies the [4 October dissertation review package](../../dissertation/2026-10-04/README.md).
It supplies the retained inputs and observations behind the published aggregates,
including the records of interrupted work. Large files live in a versioned GitHub
Release rather than in ordinary Git history.

**Release:** [research-record-2026-10-04](https://github.com/KrtiT/automated-phishing-detection-public/releases/tag/research-record-2026-10-04).
Use the committed `catalog.json` and `SHA256SUMS.txt` to identify the exact assets.
Do not combine this release with a later archive by filename alone.

The nine archives total 479,517,035 compressed bytes. Their inventories account
for 73,234 retained files: 72,754 exact files, 340 administrative projections
and 140 hash-only exclusions. The [archive verification](archive-verification.jsonl)
records every family; the [bounded content audit](content-audit.json) records
source-identity checks and the review of scanner matches. These checks are
distinct from the scientific recomputation below.

**Safety:** dataset URLs include phishing targets. Treat them as inert strings.
Do not open, crawl or resolve them. The verification commands below do none of
those things and do not load model weights or execute new research measurements.

## Download and verify

From a checkout of this release tag, after installing the locked Python environment:

```sh
DOWNLOAD_DIR="$(mktemp -d /tmp/tallam-research-download.XXXXXX)"
gh release download research-record-2026-10-04 \
  --repo KrtiT/automated-phishing-detection-public \
  --pattern '*.tar.gz' --dir "$DOWNLOAD_DIR"
cp research-archive/2026-10-04/SHA256SUMS.txt "$DOWNLOAD_DIR/"
(cd "$DOWNLOAD_DIR" && shasum -a 256 -c SHA256SUMS.txt)
.venv/bin/python -B scripts/research_archive.py verify \
  "$DOWNLOAD_DIR"/*.tar.gz \
  --catalog research-archive/2026-10-04/catalog.json \
  --materialize /tmp/tallam-research-retained
.venv/bin/python -B scripts/recompute_research.py \
  --archive-root /tmp/tallam-research-retained \
  --output /tmp/tallam-research-recomputation.json
```

Choose unused materialization/output paths. Materialization refuses to overwrite
existing evidence. The verifier requires exactly the catalog's archive set and
checks every archive and inventory digest before writing. Recomputation requires
the exact inventory set bound to the catalog in this tagged checkout, and records
those identities in its receipt. `-B` keeps Python bytecode caches out of the hashed audit-source
package. See [environment setup](../../dissertation/2026-10-04/REPRODUCTION.md#macos-arm64-test-environment)
for the measured macOS backend pin. Python and packages are needed for arithmetic;
neither a service process nor data-source credentials are needed.

## Locating the evidence

Paths below are relative to the materialized archive. Every source path has an
inventory entry with its original hash and public disposition.

| Family | Contents and authoritative locators |
|---|---|
| `development-data-and-models` | PhiUSIIL publisher ZIP/CSV and pinned PSL under `gwu_working/automated-phishing-detection-public/data/raw/`; all retained preparation variants under `data/processed/`; primary fitted Logistic-L1, length, transformer and GMM artifacts there; PhishVN open ZIP under `gwu_working/study-only-v1-reviewed-inputs/` |
| `secondary-development` | Drift/formatting/permutation/RF development, seed checkpoints and probes, including unsuccessful attempts. Accepted RF: `gwu_working/secondary-development-correction-v2-2026-09-23-attempt-1/random_forest/evidence/model.json` |
| `original-hold-and-interrupted-attempts` | Original September 28 hold plus September 29 attempts 1–3; these do not replace accepted observations |
| `original-attempt-4` | `gwu_working/study-urlnorm-v1-2026-09-30-attempt-4/`: prepared populations, internal/external `evidence/predictions.jsonl`, secondary and routing outputs, and the first 72 accepted cells |
| `original-continuation` | `gwu_working/study-series-v1-20261001-segment-2/`: cells 73–125 and complete study reductions; historical custody/inventory packages are retained separately |
| `followup-detection` | Under `dissertation/followup-20261001/`: both admission attempts, the exact benchmark CSV, accepted 8,622-row population, fitted states and paired predictions |
| `followup-service-interrupted` | `service-comparison-v1/` and its preservation inventory; 45 complete arms, excluded from the new primary schedule |
| `followup-service-complete` | `service-comparison-v2/`, full 80-arm request records and summaries, synthetic manifest, recovery authorization and preflight |
| `freeze-and-recovery-metadata` | Explicit study profiles, envelopes, seals, amendments, reservations, sampled conditions and recovery verifications, with private execution details projected |

The authoritative corrected PhiUSIIL partition directory is
`gwu_working/automated-phishing-detection-public/data/processed/phiusiil-v1-source-schema-v2-label-map-v1-20260903/`.
The other preparation variants are historical, not interchangeable current splits.
Frozen method contracts and source manifests remain in the repository's `data/`
and `docs/` directories at their recorded revisions. See
[provenance](PROVENANCE.md) and [source identities/licenses](LICENSES.md).

## What the recomputation verifies

The public adapter uses retained verifier algorithms with authenticated archive
inputs. It recalculates all 125 operational cells/25 groups, the original 22
gate decisions and saved-outcome contrasts, secondary population/model metrics,
calibration, low-FPR score curves and prevalence projections, and the D/S paired
comparisons. No fitted model or threshold is changed. The [saved report](recomputation.json) records
the exact scope and input count.

This is **recomputation from saved observations**, not an independent repeat of
training or timing. Monitor scores are checked against their frozen boundaries;
this does not refit monitors. Historical development-control limitations remain
reported, including the missing consumed-label digests for permutation controls.
The full source and retained fitted states support further inspection, but private
execution capability frames are not executable public authorization.

## Representation and exclusions

Each gzip tar contains `inventory.jsonl` plus content-addressed `blobs/<sha256>`.
Duplicate byte strings are stored once per family and restored to their original
paths by the verifier. No tar path is extracted directly; member types, paths,
sizes, identities and destination boundaries are checked first.

- **`exact`:** byte-identical public content, including licensed raw datasets,
  partitions, saved predictions, fitted weight arrays and request observations.
- **`administrative_projection`:** an explicitly changed JSON/JSONL container.
  Local absolute paths become archive locators or hashed private markers; private
  process commands and encoded execution frames become hash records. Some model
  metadata and source containers fall in this category because they embed local
  paths. They must not be described as byte-identical original manifests.
- **`hash_only`:** an inventoried private log or unstructured execution file. Its
  original identity is retained, but its bytes are not included. Such an entry is
  not a public replica of the excluded file.

Raw and public hashes are both supplied where content is projected. Original
internal manifests can therefore refer to an original hash different from a
projected file's public hash; the inventory records that distinction explicitly.
Private correspondence, credentials, unrelated employer material, downloaded
copyrighted papers and private/gated publisher mappings are outside this archive.
The release is the explicitly inventoried September–October retained study record,
not a claim to contain every file ever created during the dissertation.
