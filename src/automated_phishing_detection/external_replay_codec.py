"""Restore accepted external replay bytes without re-preparing or selecting rows.

The independent digest binds the exact retained JSONL; it establishes consistency,
not original publisher authentication, process observation or execution authority.
"""

from hashlib import sha256

from . import _external_input_records as records
from . import _saved_external_inputs as saved
from . import fixed_cascade
from .phishvn import PreparedExternalRow

_FIELDS = frozenset(PreparedExternalRow.__dataclass_fields__)
_STRINGS = _FIELDS - {"file_position", "is_phishing"}


class ExternalReplayCodecError(ValueError):
    """Symbolic rejection without private rows or parser diagnostics."""


def _require(condition):
    if not condition:
        raise ExternalReplayCodecError("invalid_retained_external_replay_manifest")


def _record(value):
    row = saved._fields(value, _FIELDS)
    _require(all(type(row[name]) is str for name in _STRINGS))
    _require(type(row["file_position"]) is int and row["file_position"] > 0)
    label = row["is_phishing"]
    _require(label is None or (type(label) is int and label in (0, 1)))
    return PreparedExternalRow(**row)


def decode_external_replay_manifest(
    content: bytes, *, expected_sha256: str
) -> tuple[PreparedExternalRow, ...]:
    """Authenticate before parsing; preserve every role, raw URL and file position."""
    try:
        _require(type(content) is bytes)
        fixed_cascade._lowercase_sha256(expected_sha256, "expected_sha256")
        _require(sha256(content).hexdigest() == expected_sha256)
        rows = tuple(_record(value) for value in saved._rows(content))
        _require(len(rows) >= 1000)
        records._retained_records(rows)
        return rows
    except Exception:
        raise ExternalReplayCodecError(
            "invalid_retained_external_replay_manifest"
        ) from None
