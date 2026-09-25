"""One-attempt external retention; no retries, inference or acceptance authority."""

import base64
import json
from hashlib import sha256

from . import _external_checkpoint_io as checkpoint_io
from ._checkpoint_codec import canonical_bytes
from ._external_checkpoint_protocol import (
    PROTOCOL,
    PROVENANCE_ORDER,
    SCIENTIFIC_ORDER,
    ExternalCheckpointError,
    hashes,
    require,
    snapshot_mapping,
)
from .execution_receipt import Attempt

__all__ = [
    "PROVENANCE_ORDER",
    "SCIENTIFIC_ORDER",
    "ExternalCheckpointError",
    "ExternalCheckpointWriter",
]


class ExternalCheckpointWriter:
    def __init__(self, attempt: Attempt, *, identity: dict):
        try:
            require(type(attempt) is Attempt and type(identity) is dict)
            self.identity = json.loads(canonical_bytes(identity))
        except Exception:
            raise ExternalCheckpointError("invalid_external_checkpoint") from None
        self.attempt = attempt
        self._contents: dict[str, bytes] = {}
        self._confirmed: dict[str, str] = {}
        self._identities = None
        self._failed = False

    def _failure(self, error):
        self._failed = True
        if not isinstance(error, Exception):
            raise error from None
        raise ExternalCheckpointError("external_checkpoint_write_failed") from None

    def begin(self, provenance: dict[str, bytes]) -> None:
        try:
            require(not self._failed and not self._contents)
            self._contents = snapshot_mapping(provenance, PROVENANCE_ORDER)
            self._identities = checkpoint_io.start(
                self.attempt, self.identity, self._contents
            )
            self._confirmed.update(hashes(self._contents))
        except BaseException as error:
            self._failure(error)

    def __call__(self, name: str, content: bytes) -> None:
        try:
            require(not self._failed and self._identities is not None)
            require(type(name) is str and type(content) is bytes)
            position = len(self._contents) - len(PROVENANCE_ORDER)
            require(position < len(SCIENTIFIC_ORDER))
            require(name == SCIENTIFIC_ORDER[position])
            self._contents[name] = content
            checkpoint_io.append(
                self.attempt, self._identities, self._confirmed, name, content
            )
            self._confirmed[name] = sha256(content).hexdigest()
        except BaseException as error:
            self._failure(error)

    def complete(self, scientific_outputs: dict[str, bytes]) -> dict[str, bytes]:
        """Return copied retained bytes, without asserting worker exit or acceptance."""
        try:
            require(not self._failed and self._identities is not None)
            expected = snapshot_mapping(scientific_outputs, SCIENTIFIC_ORDER)
            require(len(self._confirmed) == len(PROVENANCE_ORDER + SCIENTIFIC_ORDER))
            require(
                all(self._contents[name] == value for name, value in expected.items())
            )
            checkpoint_io.validate(self.attempt, self._identities, self._confirmed)
            return dict(self._contents)
        except BaseException as error:
            self._failure(error)

    def snapshot(self) -> bytes:
        return canonical_bytes(
            {
                "schema_version": 1,
                "protocol_id": PROTOCOL,
                "reservation_sha256": self.attempt.reservation_sha256,
                "status": "failed"
                if self._failed
                else "complete"
                if len(self._confirmed) == len(PROVENANCE_ORDER + SCIENTIFIC_ORDER)
                else "pending",
                "confirmed_sha256": dict(self._confirmed),
                "pending_checkpoint_bytes": {
                    name: base64.b64encode(content).decode("ascii")
                    for name, content in self._contents.items()
                    if name not in self._confirmed
                },
            }
        )
