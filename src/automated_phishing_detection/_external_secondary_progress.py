"""Private completed checkpoints remain recoverable after ambiguous writer failure."""

from dataclasses import dataclass, field

from ._external_secondary_checkpoints import SECONDARY_CHECKPOINTS, CompletedColumn
from .evaluation_producer import _json_bytes


@dataclass
class SecondaryProgress:
    stage: str = "input_validation"
    primary_hash: str | None = None
    record_ids: tuple[str, ...] = field(default=(), repr=False)
    bindings: tuple[dict, ...] = field(default=(), repr=False)
    columns: list[CompletedColumn] = field(default_factory=list, repr=False)
    private_outputs: dict[str, bytes] = field(default_factory=dict, repr=False)
    current_member: str | None = None
    current_checkpoint: str | None = None
    next_unattempted: int = 0


def secondary_progress_bytes(state: SecondaryProgress) -> bytes:
    reason = (
        "available_in_completed_checkpoint"
        if state.current_member in state.private_outputs
        else "physical_counts_unavailable"
    )
    return _json_bytes(
        {
            "schema_version": 1,
            "phase": "external_secondary",
            "status": "failed",
            "stage": state.stage,
            "primary_scores_sha256": state.primary_hash,
            "expected_record_ids": state.record_ids,
            "completed_checkpoints": {
                name: content.decode("ascii")
                for name, content in state.private_outputs.items()
            },
            "current_member": state.current_member,
            "current_checkpoint": state.current_checkpoint,
            "current_member_physical_counts": None,
            "current_member_counts_reason": reason,
            "unattempted_members": SECONDARY_CHECKPOINTS[state.next_unattempted :],
            "retention_status": "failed_or_ambiguous"
            if state.stage == "retention"
            else None,
        }
    )
