"""Disclosed publisher URL representation derived only from retained v1 records.

Parent cells and provenance remain exact. Caller-bound ancestry establishes
consistency, not original archive authenticity or protected-access authority.
"""

from dataclasses import replace
from hashlib import sha256

from . import saved_phishvn_source
from ._phishvn_archive import PhishVNSourceError, PhishVNSourcePins, require
from .phishvn_source import DecodedPhishVNSource, _json_bytes

_IDENTITY = {
    "schema_version": 2,
    "algorithm_id": "phishvn-publisher-urlnorm-v1",
    "representation": "publisher_url_norm_v1",
}
_ANCESTRY = ("parent_source_sha256", "parent_summary_sha256")
_PARENT_IDENTITY = {
    "schema_version": 1,
    "algorithm_id": "phishvn-publisher-decoder-v1",
}


def _selected_rows(parent, source):
    headers = source["headers"]
    require(
        all("url_norm" in header for header in headers.values()),
        "missing_publisher_url_norm",
    )
    return tuple(
        replace(
            row,
            raw_url=record["cells"][headers[record["source_member"]].index("url_norm")],
        )
        for row, record in zip(parent.rows, source["rows"], strict=True)
    )


def _authenticated_parent(parent):
    require(type(parent) is DecodedPhishVNSource, "invalid_publisher_parent")
    require(
        type(parent.private_outputs) is dict
        and set(parent.private_outputs) == {"publisher-source.json"},
        "invalid_publisher_parent_outputs",
    )
    source_bytes = parent.private_outputs["publisher-source.json"]
    summary_bytes = _json_bytes(parent.public_summary)
    source = saved_phishvn_source._document(source_bytes)
    pins = PhishVNSourcePins(**parent.public_summary["input_archive"])
    restored = saved_phishvn_source._restore(source, source_bytes, summary_bytes, pins)
    require(restored == parent, "publisher_parent_object_mismatch")
    return restored, source, source_bytes, summary_bytes


def _derived_records(source, summary, source_bytes, summary_bytes):
    metadata = _IDENTITY | {
        "parent_source_sha256": sha256(source_bytes).hexdigest(),
        "parent_summary_sha256": sha256(summary_bytes).hexdigest(),
    }
    derived_source = _json_bytes(source | metadata)
    derived_summary = (
        summary
        | metadata
        | {
            "private_sha256": {
                "publisher-source.json": sha256(derived_source).hexdigest()
            }
        }
    )
    return derived_source, derived_summary


def derive_publisher_url_norm(parent: DecodedPhishVNSource) -> DecodedPhishVNSource:
    """Select exact authenticated url_norm cells without changing retained cells."""
    try:
        restored, source, source_bytes, summary_bytes = _authenticated_parent(parent)
        rows = _selected_rows(restored, source)
        derived_source, summary = _derived_records(
            source, restored.public_summary, source_bytes, summary_bytes
        )
        return DecodedPhishVNSource(
            rows,
            restored.published_split_counts,
            {"publisher-source.json": derived_source},
            summary,
        )
    except PhishVNSourceError:
        raise
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise PhishVNSourceError("publisher_url_norm_derivation_failed") from None


def _metadata(source, summary):
    require(
        all(saved_phishvn_source._digest(source[name]) for name in _ANCESTRY),
        "invalid_publisher_parent_digests",
    )
    expected = _IDENTITY | {name: source[name] for name in _ANCESTRY}
    for document in (source, summary):
        require(
            _json_bytes({name: document[name] for name in expected})
            == _json_bytes(expected),
            "derived_publisher_identity_mismatch",
        )


def _parent_document(document):
    omitted = {"representation", *_ANCESTRY}
    return {
        name: value for name, value in document.items() if name not in omitted
    } | _PARENT_IDENTITY


def _parent_records(source, summary, source_bytes):
    _metadata(source, summary)
    require(
        summary["private_sha256"]
        == {"publisher-source.json": sha256(source_bytes).hexdigest()},
        "derived_publisher_source_hash_mismatch",
    )
    parent_source = _parent_document(source)
    parent_source_bytes = _json_bytes(parent_source)
    require(
        sha256(parent_source_bytes).hexdigest() == source["parent_source_sha256"],
        "publisher_parent_source_hash_mismatch",
    )
    parent_summary = _parent_document(summary) | {
        "private_sha256": {"publisher-source.json": source["parent_source_sha256"]}
    }
    parent_summary_bytes = _json_bytes(parent_summary)
    require(
        sha256(parent_summary_bytes).hexdigest() == source["parent_summary_sha256"],
        "publisher_parent_summary_hash_mismatch",
    )
    return parent_source, parent_source_bytes, parent_summary_bytes


def restore_derived_publisher(
    source_bytes: bytes, summary_bytes: bytes, *, pins: PhishVNSourcePins
) -> DecodedPhishVNSource:
    """Authenticate canonical parent records before selecting normalized inputs."""
    try:
        source = saved_phishvn_source._document(source_bytes)
        summary = saved_phishvn_source._document(summary_bytes)
        parent_source, parent_bytes, parent_summary = _parent_records(
            source, summary, source_bytes
        )
        parent = saved_phishvn_source._restore(
            parent_source, parent_bytes, parent_summary, pins
        )
        return DecodedPhishVNSource(
            _selected_rows(parent, source),
            parent.published_split_counts,
            {"publisher-source.json": source_bytes},
            summary,
        )
    except PhishVNSourceError:
        raise
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise PhishVNSourceError("publisher_url_norm_restore_failed") from None
