"""Retain one ordered scientific attempt; no retries or execution acceptance."""

import base64
from hashlib import sha256

from . import _internal_scientific_io as checkpoint_io
from ._checkpoint_codec import canonical_bytes
from ._internal_scientific_protocol import (
    SCIENTIFIC_CHECKPOINT_NAMES,
    SCIENTIFIC_CHECKPOINT_ORDER,
    SCIENTIFIC_CHECKPOINT_PROTOCOL,
    ScientificCheckpointError,
    completion_bytes,
    context_bytes,
    hashes,
    loads,
    require,
)
from .bound_secondary import SecondaryInferenceCounts
from .execution_receipt import Attempt
from .selective_inference import InferenceCounts

__all__ = [
    "SCIENTIFIC_CHECKPOINT_NAMES",
    "SCIENTIFIC_CHECKPOINT_ORDER",
    "SCIENTIFIC_CHECKPOINT_PROTOCOL",
    "ScientificCheckpointError",
    "ScientificCheckpointWriter",
    "retain_failure_progress",
]


class ScientificCheckpointWriter:
    def __init__(
        self,
        attempt: Attempt,
        *,
        identity: dict,
        source_checkpoint_sha256: dict[str, str],
        record_ids: tuple[str, ...],
    ) -> None:
        try:
            require(type(attempt) is Attempt)
            self.attempt = attempt
            self.identity = loads(canonical_bytes(identity))
            self.source_hashes = loads(canonical_bytes(source_checkpoint_sha256))
        except Exception:
            raise ScientificCheckpointError("invalid_scientific_checkpoint") from None
        self.record_ids = record_ids
        self.contents: dict[str, bytes] = {}
        self.confirmed: dict[str, str] = {}
        self.identities: tuple | None = None
        self.failed = False

    def _write(self, name: str, content: bytes) -> None:
        require(not self.failed and type(name) is str and type(content) is bytes)
        if not self.contents:
            require(name == "bindings.json")
            context = context_bytes(
                self.identity,
                self.attempt.reservation_sha256,
                self.source_hashes,
                content,
                self.record_ids,
            )
            self.contents.update({"context.json": context, name: content})
            self.identities = checkpoint_io.start(
                self.attempt, self.identity, self.contents
            )
            self.confirmed.update(hashes(self.contents))
            return
        require(len(self.contents) < len(SCIENTIFIC_CHECKPOINT_ORDER))
        require(name == SCIENTIFIC_CHECKPOINT_ORDER[len(self.contents)])
        self.contents[name] = content
        checkpoint_io.append(
            self.attempt, self.identities, self.confirmed, name, content
        )
        self.confirmed[name] = sha256(content).hexdigest()

    def __call__(self, name: str, content: bytes) -> None:
        try:
            require(name not in ("context.json", "completion.json"))
            self._write(name, content)
        except BaseException as error:
            self.failed = True
            if not isinstance(error, Exception):
                raise
            raise ScientificCheckpointError(
                "scientific_checkpoint_write_failed"
            ) from None

    def complete(
        self,
        *,
        inference_counts: InferenceCounts,
        secondary_inference_counts: SecondaryInferenceCounts,
    ) -> dict[str, str]:
        try:
            content = completion_bytes(
                self.contents,
                self.attempt.reservation_sha256,
                inference_counts,
                secondary_inference_counts,
            )
            self._write("completion.json", content)
            return dict(self.confirmed)
        except BaseException as error:
            self.failed = True
            if not isinstance(error, Exception):
                raise
            raise ScientificCheckpointError(
                "scientific_checkpoint_write_failed"
            ) from None

    def snapshot(self) -> bytes:
        return canonical_bytes(
            {
                "schema_version": 1,
                "protocol_id": SCIENTIFIC_CHECKPOINT_PROTOCOL,
                "reservation_sha256": self.attempt.reservation_sha256,
                "status": "failed"
                if self.failed
                else "complete"
                if len(self.confirmed) == len(SCIENTIFIC_CHECKPOINT_ORDER)
                else "pending",
                "confirmed_sha256": dict(self.confirmed),
                "pending_checkpoint_bytes": {
                    name: base64.b64encode(content).decode("ascii")
                    for name, content in self.contents.items()
                    if name not in self.confirmed
                },
            }
        )


def retain_failure_progress(attempt: Attempt, content: bytes) -> None:
    from ._internal_scientific_failure import retain_failure

    retain_failure(attempt, content)
