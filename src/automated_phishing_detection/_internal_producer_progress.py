"""Parent-owned private completed-prefix state; never a retry or resume interface."""

import base64
from dataclasses import asdict, dataclass, field

from . import fixed_cascade
from ._checkpoint_codec import canonical_bytes
from ._internal_producer_checkpoints import COLUMN_NAMES, PRODUCER_CHECKPOINT_NAMES
from .selective_inference import InferenceCounts


@dataclass
class InternalProgress:
    status: str = "not_started"
    stage: str = "input_validation"
    record_ids: tuple[str, ...] = field(default=(), repr=False)
    rows: list = field(default_factory=list, repr=False)
    columns: list = field(default_factory=list, repr=False)
    outputs: dict[str, bytes] = field(default_factory=dict, repr=False)
    started_primary_position: int | None = None
    next_expected_member: str | None = None
    inference_counts: InferenceCounts | None = field(default=None, repr=False)
    inference_counts_reason: str | None = "physical_counts_unavailable"
    failed_checkpoint: str | None = None
    retention_status: str = "not_attempted"

    def begin(self) -> None:
        if not fixed_cascade._matches_exactly(asdict(self), asdict(InternalProgress())):
            raise ValueError("internal_progress_must_be_fresh")
        self.status = "running"

    def observe_counts(self, scorer, *, suppress_interruptions=False) -> None:
        try:
            counts = scorer.counts
            if type(counts) is not InferenceCounts or any(
                type(value) is not int or value < 0 for value in asdict(counts).values()
            ):
                raise ValueError("invalid_internal_counts")
        except BaseException as error:
            self.inference_counts = None
            self.inference_counts_reason = "physical_counts_unavailable"
            if not isinstance(error, Exception) and not suppress_interruptions:
                raise
        else:
            self.inference_counts = counts
            self.inference_counts_reason = None

    def store(self, name: str, content: bytes, retain) -> None:
        index = len(self.outputs)
        if (
            index >= len(PRODUCER_CHECKPOINT_NAMES)
            or name != PRODUCER_CHECKPOINT_NAMES[index]
            or type(content) is not bytes
        ):
            raise ValueError("invalid_internal_checkpoint")
        self.outputs[name] = content
        self.failed_checkpoint = name
        self.retention_status = "failed_or_ambiguous"
        if retain is not None:
            retain(name, content)
        self.failed_checkpoint = None
        self.retention_status = "retained"

    def snapshot(self) -> bytes:
        payload = _snapshot(self)
        payload["retained_checkpoints_base64"] = {
            name: base64.b64encode(content).decode("ascii")
            for name, content in self.outputs.items()
        }
        return canonical_bytes(payload)


def _snapshot(state: InternalProgress) -> dict:
    incomplete = _incomplete_member(state)
    return {
        "schema_version": 1,
        "status": state.status,
        "stage": state.stage,
        "expected_record_ids": state.record_ids,
        "completed_primary_rows": [asdict(row) for row in state.rows],
        "completed_secondary_columns": [asdict(column) for column in state.columns],
        "started_primary_position": state.started_primary_position,
        "next_expected_member": state.next_expected_member,
        "unattempted_secondary_members": _unattempted(state),
        "incomplete_member_counts": None,
        "incomplete_member_counts_reason": "incomplete_member_counts_unavailable"
        if incomplete
        else None,
        "inference_counts": asdict(state.inference_counts)
        if state.inference_counts is not None
        else None,
        "inference_counts_reason": state.inference_counts_reason,
        "failed_checkpoint": state.failed_checkpoint,
        "retention_status": state.retention_status,
    }


def _unattempted(state: InternalProgress) -> tuple[str, ...]:
    index = len(state.columns)
    if _incomplete_member(state):
        index += 1
    return COLUMN_NAMES[index:]


def _incomplete_member(state: InternalProgress) -> bool:
    return (
        state.stage in ("secondary_scoring", "secondary_column_validation")
        and state.next_expected_member is not None
    )
