"""Private external failure retention, distinct from the checkpoint protocol.

Snapshots preserve already available producer bytes without inferring counts,
replaying science, authorizing access or independently accepting evidence.
"""

import asyncio
import base64
import json
from contextlib import contextmanager

from . import _external_source_failure_io as failure_io
from . import execution_receipt as receipt
from ._checkpoint_codec import canonical_bytes
from ._external_checkpoint_protocol import SCIENTIFIC_ORDER
from ._external_source_checkpoints import ExternalCheckpointWriter
from .external_producer import ProducedExternal

PROTOCOL = "external-source-failure-v1"
_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "reservation_sha256",
    "execution",
    "stage",
    "failure",
    "cleanup_failed",
    "checkpoints",
    "producer_progress_base64",
    "completed",
}


ExternalSourceFailureError = failure_io.ExternalSourceFailureError
_require = failure_io.require


def failure_kind(error: BaseException) -> str:
    if isinstance(error, asyncio.CancelledError):
        return "cancelled"
    if isinstance(error, KeyboardInterrupt):
        return "keyboard_interrupt"
    if isinstance(error, SystemExit):
        return "system_exit"
    return "execution_failed" if isinstance(error, Exception) else "interrupted"


def _completed(value):
    if value is None:
        return
    _require(
        type(value) is dict and set(value) == {"composition", "private_outputs_base64"}
    )
    receipt._json_bytes(value["composition"], "composition")
    outputs = value["private_outputs_base64"]
    _require(type(outputs) is dict and set(outputs) == set(SCIENTIFIC_ORDER))
    for content in outputs.values():
        failure_io.base64_value(content)


def _header(record):
    _require(type(record["schema_version"]) is int and record["schema_version"] == 1)
    _require(record["protocol_id"] == PROTOCOL and record["status"] == "failed")
    _require(
        type(record["reservation_sha256"]) is str
        and receipt._SHA256.fullmatch(record["reservation_sha256"]) is not None
    )
    receipt._json_bytes(record["execution"], "execution")
    _require(
        type(record["stage"]) is str
        and receipt._SYMBOL.fullmatch(record["stage"]) is not None
    )
    _require(
        record["failure"] in failure_io.KINDS and type(record["cleanup_failed"]) is bool
    )


def _validate(content):
    try:
        _require(type(content) is bytes)
        record = json.loads(content)
        _require(type(record) is dict and set(record) == _KEYS)
        receipt._json_bytes(record, "external_failure")
        _require(content == canonical_bytes(record))
        _header(record)
        failure_io.checkpoint(record["checkpoints"], record["reservation_sha256"])
        _completed(record["completed"])
        if record["producer_progress_base64"] is not None:
            failure_io.base64_value(record["producer_progress_base64"])
        return record
    except Exception:
        raise ExternalSourceFailureError("invalid_external_source_failure") from None


def _produced(value):
    if value is None:
        return None
    _require(type(value) is ProducedExternal)
    _require(type(value.private_outputs) is dict)
    _require(all(type(content) is bytes for content in value.private_outputs.values()))
    return {
        "composition": value.public_summary,
        "private_outputs_base64": {
            name: base64.b64encode(content).decode("ascii")
            for name, content in value.private_outputs.items()
        },
    }


def retain_external_failure(attempt, content: bytes) -> None:
    """Create a private diagnostic sidecar without claiming finalization."""
    try:
        failure_io.retain(attempt, content, _validate(content))
    except Exception:
        raise ExternalSourceFailureError("invalid_external_source_failure") from None


class ExternalSourceFailureState:
    def __init__(self):
        self.writer = None
        self.produced = None
        self.original_error = None
        self.body_exited = False
        self.session_closed = False
        self._producer_progress = None

    @contextmanager
    def capture_body(self):
        try:
            yield
        except BaseException as error:
            self.original_error = error
            progress = BaseException.__dict__["__dict__"].__get__(error).get("progress")
            self._producer_progress = progress if type(progress) is bytes else None
            raise
        finally:
            self.body_exited = True

    def selected_error(self, error: BaseException) -> BaseException:
        if self.original_error is not None and not isinstance(
            self.original_error, Exception
        ):
            return self.original_error
        return error

    def snapshot(
        self, attempt, identity: dict, stage: str, error: BaseException
    ) -> bytes:
        try:
            _require(
                type(attempt) is receipt.Attempt and isinstance(error, BaseException)
            )
            _require(
                self.writer is None or type(self.writer) is ExternalCheckpointWriter
            )
            if self.writer is not None:
                _require(self.writer.attempt == attempt)
                _require(
                    canonical_bytes(self.writer.identity) == canonical_bytes(identity)
                )
            content = canonical_bytes(self._record(attempt, identity, stage, error))
            _validate(content)
            return content
        except Exception:
            raise ExternalSourceFailureError(
                "invalid_external_source_failure"
            ) from None

    def _record(self, attempt, identity, stage, error):
        original = self.original_error if self.original_error is not None else error
        checkpoints = (
            json.loads(self.writer.snapshot()) if self.writer is not None else None
        )
        progress = (
            base64.b64encode(self._producer_progress).decode("ascii")
            if self._producer_progress is not None
            else None
        )
        return {
            "schema_version": 1,
            "protocol_id": PROTOCOL,
            "status": "failed",
            "reservation_sha256": attempt.reservation_sha256,
            "execution": identity,
            "stage": stage,
            "failure": failure_kind(original),
            "cleanup_failed": self.body_exited
            and not self.session_closed
            and error is not self.original_error,
            "checkpoints": checkpoints,
            "producer_progress_base64": progress,
            "completed": _produced(self.produced),
        }
