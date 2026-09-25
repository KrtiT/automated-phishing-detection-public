"""Decode caller-pinned publisher bytes without authorizing protected access.

Channels remain opaque provenance. Raw URLs, including bare domains, pass through
unchanged to the existing quarantine policy. No file is opened or model invoked.
"""

import csv
import io
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from hashlib import sha256

from ._phishvn_archive import (
    EXPECTED_FORMAT,
    PhishVNSourceError,
    PhishVNSourcePins,
    authenticated_members,
    require,
)
from .phishvn import NormalizedExternalRow

_SPLITS = ("train", "val", "test")
_FULL_TABLE = "data/dataset_url.csv"
_REQUIRED = frozenset({"id", "url", "source", "label", "tier", "split", "channel"})
_MAPPING = {
    ("tinnhiemmang", "gold", "phishing"): ("ncsc", "gold", "phishing"),
    ("tinnhiemmang", "silver", "phishing"): ("ncsc", "silver", "phishing"),
    ("tinnhiem_web", "gold", "benign"): ("trusted_registry", "certified", "legitimate"),
    ("tinnhiem_org", "gold", "benign"): ("trusted_registry", "certified", "legitimate"),
    ("chongluadao", "bronze", "phishing"): (
        "chongluadao_openphish",
        "bronze",
        "phishing",
    ),
    ("openphish", "bronze", "phishing"): (
        "chongluadao_openphish",
        "bronze",
        "phishing",
    ),
    ("tranco", "silver", "benign"): ("tranco", "control", "reference_negative"),
    ("tranco_vn", "silver", "benign"): ("tranco", "control", "reference_negative"),
}


@dataclass(frozen=True)
class DecodedPhishVNSource:
    rows: tuple[NormalizedExternalRow, ...] = field(repr=False)
    published_split_counts: dict[str, int]
    private_outputs: dict[str, bytes] = field(repr=False)
    public_summary: dict


def _json_bytes(value):
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def _csv_table(content):
    text = content.decode("utf-8")
    require(
        not text.startswith("\ufeff") and "\x00" not in text, "invalid_csv_encoding"
    )
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    header = tuple(next(reader, ()))
    require(
        bool(header)
        and all(header)
        and len(set(header)) == len(header)
        and _REQUIRED <= set(header),
        "invalid_csv_header",
    )
    rows = tuple(tuple(values) for values in reader)
    require(all(len(values) == len(header) for values in rows), "ragged_csv_record")
    return header, rows


def _tables(contents):
    names = (_FULL_TABLE, *(f"data/splits/url_{split}.csv" for split in _SPLITS))
    tables = {name: _csv_table(contents[name]) for name in names}
    header, full_rows = tables[_FULL_TABLE]
    require(
        all(value[0] == header for value in tables.values()), "source_table_mismatch"
    )
    split_rows = (row for name in names[1:] for row in tables[name][1])
    require(Counter(full_rows) == Counter(split_rows), "source_table_mismatch")
    return tables


def _mapping(row):
    key = (row["source"], row["tier"], row["label"])
    if not all(key):
        return (None, None, None), "missing_mapping_field"
    if key not in _MAPPING:
        return ("unmapped_publisher_tuple",) * 3, "undefined_mapping"
    return _MAPPING[key], "mapped"


def _normalized_row(row, split, position):
    (group, tier, designation), status = _mapping(row)
    normalized = NormalizedExternalRow(
        published_id=row["id"],
        source_split=split,
        file_position=position,
        raw_url=row["url"],
        source_group=group,
        source_class=designation,
        confidence_tier=tier,
        published_split=row["split"],
    )
    return normalized, status


def _decode_rows(tables):
    normalized, provenance = [], []
    counts = {
        status: 0 for status in ("mapped", "missing_mapping_field", "undefined_mapping")
    }
    for split in _SPLITS:
        name = f"data/splits/url_{split}.csv"
        header, rows = tables[name]
        for position, cells in enumerate(rows, start=1):
            row = dict(zip(header, cells, strict=True))
            normalized_row, status = _normalized_row(row, split, position)
            counts[status] += 1
            normalized.append(normalized_row)
            provenance.append(
                {
                    "source_member": name,
                    "file_position": position,
                    "cells": cells,
                    "normalization_status": status,
                }
            )
    return tuple(normalized), provenance, counts


def _identity(contents, pins):
    return {
        "schema_version": 1,
        "algorithm_id": "phishvn-publisher-decoder-v1",
        "expected_format": dict(EXPECTED_FORMAT),
        "input_archive": asdict(pins),
        "source_binding": "caller_supplied_pins_only",
        "protected_evaluation_authorized": False,
        "member_inventory": {
            name: {"sha256": sha256(content).hexdigest(), "size_bytes": len(content)}
            for name, content in contents.items()
        },
    }


def _result(contents, tables, pins):
    rows, provenance, mapping_counts = _decode_rows(tables)
    split_counts = {
        split: len(tables[f"data/splits/url_{split}.csv"][1]) for split in _SPLITS
    }
    identity = _identity(contents, pins)
    headers = {name: table[0] for name, table in tables.items()}
    private = {
        "publisher-source.json": _json_bytes(
            identity | {"headers": headers, "rows": provenance}
        )
    }
    summary = identity | {
        "published_split_counts": split_counts,
        "full_table_rows": len(tables[_FULL_TABLE][1]),
        "mapping_counts": mapping_counts,
        "private_sha256": {
            name: sha256(content).hexdigest() for name, content in private.items()
        },
    }
    return DecodedPhishVNSource(rows, split_counts, private, summary)


def decode_phishvn_archive(
    archive_bytes: bytes, *, pins: PhishVNSourcePins
) -> DecodedPhishVNSource:
    """Verify the complete archive before decoding all split rows without filtering.

    Required columns are structural checks. No channel, source or confidence
    value selects a row here; unknown mappings enter existing group quarantine.
    Expected-format metadata is not an authenticated release identity.
    """
    try:
        contents = authenticated_members(archive_bytes, pins)
        return _result(contents, _tables(contents), pins)
    except PhishVNSourceError:
        raise
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError, csv.Error):
        raise PhishVNSourceError("publisher_decode_failed") from None
