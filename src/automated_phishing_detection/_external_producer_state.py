"""Private one-attempt retention state; no filesystem writer or retry mechanism."""

import base64
from collections.abc import Callable
from dataclasses import dataclass, field

from ._external_primary_progress import _observed_counts
from .evaluation_producer import _json_bytes


@dataclass
class ExternalProgress:
    retain: Callable[[str, bytes], None] | None = field(repr=False)
    outputs: dict[str, bytes] = field(default_factory=dict, repr=False)
    stage: str = "preparation"
    failed_checkpoint: str | None = None
    retention_status: str = "not_attempted"

    def store(self, name: str, content: bytes) -> None:
        if type(content) is not bytes or name in self.outputs:
            raise ValueError("invalid_external_checkpoint")
        self.outputs[name] = content
        self.failed_checkpoint = name
        self.retention_status = "failed_or_ambiguous"
        if self.retain is not None:
            self.retain(name, content)
        self.failed_checkpoint = None
        self.retention_status = "retained"


def _phase_progress(error: BaseException) -> str | None:
    try:
        child = vars(error).get("progress")
        return child.decode("ascii") if type(child) is bytes else None
    except Exception:
        return None


def failure_bytes(
    state: ExternalProgress, session: object, error: BaseException
) -> bytes:
    counts, reason = _observed_counts(session)
    return _json_bytes(
        {
            "schema_version": 1,
            "status": "failed",
            "stage": state.stage,
            "failed_checkpoint": state.failed_checkpoint,
            "retention_status": state.retention_status,
            "retained_checkpoints_base64": {
                name: base64.b64encode(content).decode("ascii")
                for name, content in state.outputs.items()
            },
            "phase_progress_json": _phase_progress(error),
            "inference_counts": counts,
            "inference_counts_reason": reason,
        }
    )
