"""One private failure sidecar, never a replacement for begun publication."""

import base64

from . import execution_receipt as receipt
from . import fixed_cascade
from ._internal_scientific_io import guard, validate_identity
from ._internal_scientific_protocol import (
    SCIENTIFIC_CHECKPOINT_ORDER,
    SCIENTIFIC_CHECKPOINT_PROTOCOL,
    ScientificCheckpointError,
    loads,
    require,
    validate_source_hashes,
)

_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "status",
        "reservation_sha256",
        "execution",
        "source_checkpoint_sha256",
        "stage",
        "producer",
        "checkpoints",
        "failure",
        "cleanup_failed",
    }
)
_STAGES = frozenset(
    {
        "public_preflight",
        "reservation",
        "model_loading",
        "suffix_rules",
        "source_csv",
        "source_reconstruction",
        "source_checkpoints",
        "partition",
        "scoring",
        "final_binding",
        "summary",
        "publication",
    }
)
_FAILURES = frozenset(
    {
        "execution_failed",
        "cancelled",
        "keyboard_interrupt",
        "system_exit",
        "interrupted",
    }
)


def _checkpoint_maps(value: dict) -> None:
    confirmed = value["confirmed_sha256"]
    pending = value["pending_checkpoint_bytes"]
    require(type(confirmed) is dict and type(pending) is dict)
    count = len(confirmed)
    require(count != 1 and set(confirmed) == set(SCIENTIFIC_CHECKPOINT_ORDER[:count]))
    for digest in confirmed.values():
        fixed_cascade._lowercase_sha256(digest, "scientific_checkpoint_hash")
    next_names = SCIENTIFIC_CHECKPOINT_ORDER[count : count + (2 if count == 0 else 1)]
    require(not pending or set(pending) == set(next_names))
    for content in pending.values():
        require(type(content) is str)
        require(
            base64.b64encode(base64.b64decode(content, validate=True)).decode()
            == content
        )
    if value["status"] == "complete":
        require(count == len(SCIENTIFIC_CHECKPOINT_ORDER) and not pending)


def _writer_progress(value: object, reservation_sha256: str) -> None:
    if value is None:
        return
    require(type(value) is dict)
    require(
        set(value)
        == {
            "schema_version",
            "protocol_id",
            "reservation_sha256",
            "status",
            "confirmed_sha256",
            "pending_checkpoint_bytes",
        }
    )
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["protocol_id"] == SCIENTIFIC_CHECKPOINT_PROTOCOL)
    require(value["reservation_sha256"] == reservation_sha256)
    require(value["status"] in ("pending", "failed", "complete"))
    _checkpoint_maps(value)


def _validate(content: bytes, attempt: receipt.Attempt) -> dict:
    value = loads(content)
    require(set(value) == _FIELDS)
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["protocol_id"] == SCIENTIFIC_CHECKPOINT_PROTOCOL)
    require(value["status"] == "failed")
    require(value["reservation_sha256"] == attempt.reservation_sha256)
    require(type(value["execution"]) is dict)
    require(
        value["execution"].get("scientific_checkpoint_protocol")
        == SCIENTIFIC_CHECKPOINT_PROTOCOL
    )
    require(type(value["stage"]) is str and value["stage"] in _STAGES)
    require(type(value["failure"]) is str and value["failure"] in _FAILURES)
    require(type(value["cleanup_failed"]) is bool)
    require(type(value["producer"]) is dict)
    _writer_progress(value["checkpoints"], attempt.reservation_sha256)
    if value["source_checkpoint_sha256"] is not None:
        validate_source_hashes(value["source_checkpoint_sha256"])
    return value


def retain_failure(attempt: receipt.Attempt, content: bytes) -> None:
    try:
        require(type(attempt) is receipt.Attempt)
        value = _validate(content, attempt)
        with receipt._attempt_directory(attempt) as directory:
            guard(directory)
            validate_identity(attempt, directory, value["execution"])
            receipt._install_record(directory, "failure-progress.json", content)
    except Exception:
        raise ScientificCheckpointError(
            "scientific_failure_progress_write_failed"
        ) from None
