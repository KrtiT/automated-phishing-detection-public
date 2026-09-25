"""Exact retained replay bytes without source reads, sorting or resampling.

The expected full-payload digest comes from already accepted scientific evidence.
This codec establishes consistency, never process observation or access authority.
"""

import json
from hashlib import sha256

from . import evaluation_manifest as manifest
from . import fixed_cascade

_FIELDS = {"schema_version", "algorithm_id", "prevalence_basis_points", "records"}
_RECORD_FIELDS = frozenset(manifest.ManifestRecord.__dataclass_fields__)


class ReplayManifestCodecError(ValueError):
    """Symbolic rejection without private records or parser details."""


def _require(condition):
    if not condition:
        raise ReplayManifestCodecError("invalid_retained_replay_manifest")


def _prevalence(value):
    _require(type(value) is int and value in (10, 100, 500))


def _digest(value):
    _require(type(value) is str and manifest._SHA256.fullmatch(value) is not None)


def _validated(value):
    _require(type(value) is manifest.ReplayManifest)
    _prevalence(value.prevalence_basis_points)
    _digest(value.sha256)
    _require(type(value.records) is tuple and len(value.records) == 10000)
    records = manifest._validated_records(value.records)
    _require(
        sum(record.is_phishing for record in records) == value.prevalence_basis_points
    )
    content = manifest._manifest_bytes(value.prevalence_basis_points, records)
    _require(sha256(content).hexdigest() == value.sha256)
    return content


def encode_replay_manifest(value: manifest.ReplayManifest) -> bytes:
    """Encode the accepted full manifest using its original hash convention."""
    try:
        return _validated(value)
    except Exception:
        raise ReplayManifestCodecError("invalid_retained_replay_manifest") from None


def _decoded(content, expected_hash, prevalence):
    value = json.loads(
        content,
        object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
        parse_constant=fixed_cascade._reject_json_constant,
    )
    _require(type(value) is dict and set(value) == _FIELDS)
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    _require(value["algorithm_id"] == "replay-manifest-v1")
    _prevalence(value["prevalence_basis_points"])
    _require(value["prevalence_basis_points"] == prevalence)
    _require(type(value["records"]) is list and len(value["records"]) == 10000)
    _require(
        all(
            type(row) is dict and set(row) == _RECORD_FIELDS for row in value["records"]
        )
    )
    records = tuple(manifest.ManifestRecord(**row) for row in value["records"])
    result = manifest.ReplayManifest(prevalence, records, expected_hash)
    _require(_validated(result) == content)
    return result


def decode_replay_manifest(
    content: bytes, *, expected_sha256: str, expected_prevalence_basis_points: int
) -> manifest.ReplayManifest:
    """Authenticate before parsing and preserve every original row and its order."""
    try:
        _require(type(content) is bytes)
        _digest(expected_sha256)
        _prevalence(expected_prevalence_basis_points)
        _require(sha256(content).hexdigest() == expected_sha256)
        return _decoded(content, expected_sha256, expected_prevalence_basis_points)
    except Exception:
        raise ReplayManifestCodecError("invalid_retained_replay_manifest") from None
