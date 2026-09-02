# Synthetic Protocol Preflight Design

**Date:** 2026-08-18
**Status:** Implemented

## Purpose

Build a small, standalone validator that proves the domain-grouping and split-manifest mechanics with synthetic records. The preflight is preparation work, not a v3 experiment and not research evidence.

## Scope

The preflight will:

- parse raw URLs without fetching them or performing DNS or WHOIS lookups;
- normalize hostnames to lowercase IDNA ASCII for grouping;
- derive registrable domains from an explicitly supplied, frozen public-suffix rule file;
- validate required manifest fields and accepted split and label values;
- reject invalid URLs, duplicate record identifiers, exact duplicate URLs, label conflicts, and registrable domains assigned to more than one split;
- print a deterministic JSON summary, including manifest and suffix-rule SHA-256 values; and
- run entirely against synthetic inputs in automated tests.

This phase will not:

- read PhiUSIIL, PhishVN, or any legacy repository dataset;
- assign domains to train, validation, or test partitions;
- implement or select the final 70/15/15 allocation algorithm;
- train, load, or score a model;
- calculate research metrics;
- alter the legacy preprocessing pipeline, API, model, dataset, or schema; or
- write a processed dataset, split manifest, or result artifact.

## Design

### Module and command

Add one flat module, `code/protocol_preflight.py`, following the repository's existing script convention. Its validation functions will be importable for unit tests and later pipeline integration. Its command-line interface will require both paths explicitly:

```bash
python code/protocol_preflight.py \
  --manifest /path/to/synthetic-manifest.json \
  --suffix-rules /path/to/synthetic-public-suffix-list.dat
```

There will be no default data path and no fallback fixture. A missing or invalid input will produce a concise error on standard error and a nonzero exit status.

### Manifest contract

The input is a JSON object with `schema_version` set to `1` and a `records` array. Each record has exactly these required fields:

- `record_id`: nonempty string, unique within the manifest;
- `url`: nonempty absolute URL string;
- `label`: integer `0` or `1`; and
- `split`: `train`, `validation`, or `test`.

The preflight derives domain groups; it does not trust a caller-supplied domain value. Exact repeated URL strings are rejected. Repeated URL strings with different labels are reported as label conflicts. Different subdomains may repeat within a split, but their registrable domain may not occur in another split.

### Host and suffix handling

URL parsing uses the Python standard library only. Hostnames are lowercased, a terminal dot is removed, and Unicode labels are converted with the built-in IDNA codec. Ports and user information do not affect the group key. A URL without a hostname, an invalid IDNA hostname, an invalid port, or an IP-literal hostname fails validation because it has no registrable-domain group under this protocol.

The suffix-rule reader supports ordinary, wildcard, and exception rules using Public Suffix List matching semantics. It performs no download or cache lookup. Tests provide a deliberately small synthetic rule file that covers a normal suffix, a multi-label suffix, a wildcard, and an exception. A complete frozen rule file is a later protocol input, not part of this change.

### Output

Successful validation prints one JSON object with:

- `status` equal to `valid`;
- total record and registrable-domain counts;
- per-split record and domain counts;
- the SHA-256 of the exact manifest bytes; and
- the SHA-256 of the exact suffix-rule bytes.

Keys are sorted so repeated runs produce the same output. The command does not write files.

### Error handling

Validation raises a purpose-specific exception at the first violated invariant. The CLI converts that exception, JSON decoding errors, and missing-file errors into a short message and exit status `2`. Unexpected programming errors are not suppressed.

## Test Strategy

Use `tests/test_protocol_preflight.py` for pure validation behavior and `tests/test_protocol_preflight_cli.py` for the command boundary. Both modules are hermetic. CLI tests create all JSON and suffix-rule files under `tmp_path`; no repository data path is opened.

Coverage includes:

- subdomains resolving to one registrable-domain group;
- lowercase, terminal-dot, multi-label suffix, IDNA, wildcard, and exception handling;
- stable successful CLI output and hashes;
- duplicate identifiers and exact duplicate URLs;
- conflicting labels for one URL;
- cross-split registrable-domain leakage;
- malformed JSON and missing required fields;
- invalid URLs, ports, hostnames, labels, and split names; and
- missing input files.

The focused test command is:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --isolated --no-project --python 3.10 \
  --with pytest==7.4.2 \
  python -m pytest -c /dev/null --rootdir=. -p no:cacheprovider -q \
  tests/test_protocol_preflight.py tests/test_protocol_preflight_cli.py
```

The test must first fail because the module is absent, then pass after the smallest implementation is added.

## Future Integration

After written approval and protocol freeze, a separate change may add the deterministic group allocator, the complete frozen suffix-rule input, and an explicit integration point before feature extraction. The existing validator will remain the boundary check. The legacy preprocessor's silent mock-data fallback must be removed before that integration can safely read research data.
