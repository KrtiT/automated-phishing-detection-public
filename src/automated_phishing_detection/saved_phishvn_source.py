"""Restore retained publisher split cells without source or model access.

Caller pins establish consistency only. The split snapshot cannot independently
prove original archive membership or full-table completeness; those remain tied
to the original hash-bound decoder execution.
"""

import json
import re
from dataclasses import asdict
from hashlib import sha256

from . import phishvn_source
from ._phishvn_archive import (
    EXPECTED_FORMAT,
    EXPECTED_MEMBERS,
    PhishVNSourceError,
    PhishVNSourcePins,
    require,
)
from .phishvn_source import DecodedPhishVNSource, _json_bytes

_SPLIT_MEMBERS = {
    split: f"data/splits/url_{split}.csv" for split in phishvn_source._SPLITS
}
_TABLE_MEMBERS = frozenset((phishvn_source._FULL_TABLE, *_SPLIT_MEMBERS.values()))
_ROW_FIELDS = {"source_member", "file_position", "cells", "normalization_status"}


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _document(content):
    require(type(content) is bytes, "invalid_saved_source_bytes")
    value = json.loads(content)
    require(type(value) is dict, "invalid_saved_source_document")
    require(_json_bytes(value) == content, "noncanonical_saved_source")
    return value


def _pin_identity(pins):
    require(type(pins) is PhishVNSourcePins, "invalid_source_pins")
    require(
        _digest(pins.archive_sha256)
        and type(pins.archive_size_bytes) is int
        and pins.archive_size_bytes > 0,
        "invalid_source_pins",
    )
    return asdict(pins)


def _inventory(value):
    require(
        type(value) is dict and set(value) == EXPECTED_MEMBERS,
        "saved_member_inventory_mismatch",
    )
    for metadata in value.values():
        require(
            type(metadata) is dict
            and set(metadata) == {"sha256", "size_bytes"}
            and _digest(metadata["sha256"])
            and type(metadata["size_bytes"]) is int
            and metadata["size_bytes"] >= 0,
            "invalid_saved_member_metadata",
        )
    return value


def _identity(source, pins):
    identity = {
        "schema_version": 1,
        "algorithm_id": "phishvn-publisher-decoder-v1",
        "expected_format": dict(EXPECTED_FORMAT),
        "input_archive": _pin_identity(pins),
        "source_binding": "caller_supplied_pins_only",
        "protected_evaluation_authorized": False,
    }
    require(
        set(source) == set(identity) | {"member_inventory", "headers", "rows"},
        "saved_source_layout_mismatch",
    )
    require(
        _json_bytes({name: source[name] for name in identity}) == _json_bytes(identity),
        "saved_source_identity_mismatch",
    )
    return identity | {"member_inventory": _inventory(source["member_inventory"])}


def _source_text(value):
    if type(value) is not str or "\x00" in value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeError:
        return False
    return True


def _headers(value):
    require(
        type(value) is dict and set(value) == _TABLE_MEMBERS,
        "saved_header_inventory_mismatch",
    )
    for header in value.values():
        require(
            type(header) is list
            and bool(header)
            and all(_source_text(name) and bool(name) for name in header)
            and len(set(header)) == len(header)
            and phishvn_source._REQUIRED <= set(header),
            "invalid_saved_csv_header",
        )
    require(
        all(header == value[phishvn_source._FULL_TABLE] for header in value.values()),
        "saved_header_mismatch",
    )
    return value


def _append_row(row, tables):
    require(type(row) is dict and set(row) == _ROW_FIELDS, "saved_row_layout_mismatch")
    require(
        type(row["source_member"]) is str and row["source_member"] in tables,
        "invalid_saved_source_member",
    )
    require(type(row["file_position"]) is int, "invalid_saved_file_position")
    header, rows = tables[row["source_member"]]
    require(
        type(row["cells"]) is list
        and len(row["cells"]) == len(header)
        and all(_source_text(cell) for cell in row["cells"]),
        "invalid_saved_cells",
    )
    rows.append(tuple(row["cells"]))


def _tables(source):
    headers = _headers(source["headers"])
    require(type(source["rows"]) is list, "invalid_saved_rows")
    tables = {name: (tuple(headers[name]), []) for name in _SPLIT_MEMBERS.values()}
    for row in source["rows"]:
        _append_row(row, tables)
    return tables


def _restore(source, publisher_source, publisher_summary, pins):
    identity = _identity(source, pins)
    tables = _tables(source)
    rows, provenance, mapping_counts = phishvn_source._decode_rows(tables)
    require(
        _json_bytes(provenance) == _json_bytes(source["rows"]),
        "saved_row_reconstruction_mismatch",
    )
    split_counts = {
        split: len(tables[name][1]) for split, name in _SPLIT_MEMBERS.items()
    }
    summary = identity | {
        "published_split_counts": split_counts,
        "full_table_rows": len(rows),
        "mapping_counts": mapping_counts,
        "private_sha256": {
            "publisher-source.json": sha256(publisher_source).hexdigest()
        },
    }
    require(_json_bytes(summary) == publisher_summary, "saved_summary_mismatch")
    return DecodedPhishVNSource(
        rows, split_counts, {"publisher-source.json": publisher_source}, summary
    )


def restore_phishvn_source(
    publisher_source: bytes, publisher_summary: bytes, *, pins: PhishVNSourcePins
) -> DecodedPhishVNSource:
    """Restore all split rows and the exact public projection from retained bytes.

    No archive/full-table bytes are reconstructed and no access is authorized.
    An observing caller must separately bind these bytes to its accepted decoder.
    """
    try:
        require(type(publisher_summary) is bytes, "invalid_saved_summary_bytes")
        return _restore(
            _document(publisher_source), publisher_source, publisher_summary, pins
        )
    except PhishVNSourceError:
        raise
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise PhishVNSourceError("saved_publisher_restore_failed") from None
