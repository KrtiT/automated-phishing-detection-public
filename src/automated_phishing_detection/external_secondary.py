"""Run one fixed secondary phase with synchronous completed-column retention.

No source access or fitting is authorized. Failed or interrupted attempts expose
private immutable progress bytes and must never be retried or resumed.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import partial
from hashlib import sha256

from . import bound_secondary, fixed_cascade
from ._external_primary_progress import attach_failure_progress
from ._external_secondary_checkpoints import (
    SECONDARY_CHECKPOINTS,
    CompletedColumn,
    column_bytes,
    completion_bytes,
    member_bindings,
)
from ._external_secondary_inputs import (
    validate_primary_phase,
    validate_secondary_scoring,
)
from ._external_secondary_progress import SecondaryProgress, secondary_progress_bytes
from ._external_secondary_validation import require, validate_column
from .bound_secondary import BoundSecondary, SecondaryScoring
from .external_primary import ExternalPrimaryScores

RetainSecondary = Callable[[str, bytes], None]


class ExternalSecondaryError(ValueError):
    """A symbolic phase rejection with private completed-checkpoint progress."""

    def __init__(self, stage: str, progress: bytes) -> None:
        super().__init__(f"external_secondary_{stage}_failed")
        self.progress = progress


@dataclass(frozen=True)
class ProducedExternalSecondary:
    scoring: SecondaryScoring = field(repr=False)
    private_outputs: dict[str, bytes] = field(repr=False)


def _validate_inputs(
    primary: ExternalPrimaryScores,
    bound: BoundSecondary,
    retain: RetainSecondary | None,
) -> None:
    require(retain is None or callable(retain))
    validate_primary_phase(primary)
    _validate_binding(bound, dict(primary.thresholds)["logistic_l1"])


def _validate_binding(bound: BoundSecondary, stage1_threshold: float) -> None:
    bound_secondary._validate_bound(bound)
    require(stage1_threshold == bound.stage1_threshold)
    require(all(type(member.seed) is int for member in bound.seeds))
    require(
        all(
            sha256(member.model.artifact_bytes).hexdigest() == member.artifact_sha256
            for member in bound.tabular
        )
    )


def _store(
    state: SecondaryProgress, retain: RetainSecondary | None, name: str, content: bytes
) -> None:
    state.private_outputs[name] = content
    state.stage, state.current_checkpoint = "retention", name
    if retain is not None:
        retain(name, content)
    state.current_checkpoint = None


def _next_member(state: SecondaryProgress) -> None:
    index = len(state.columns)
    state.current_member = SECONDARY_CHECKPOINTS[index] if index < 12 else None
    state.next_unattempted = min(index + 1, 12)
    state.stage = "scoring"


def _retain_column(
    state: SecondaryProgress, retain: RetainSecondary | None, column: CompletedColumn
) -> None:
    state.stage = "column_validation"
    index = len(state.columns)
    validate_column(column, index, len(state.record_ids))
    content = column_bytes(
        state.primary_hash, state.record_ids, state.bindings[index], column
    )
    state.columns.append(column)
    state.next_unattempted = len(state.columns)
    _store(state, retain, SECONDARY_CHECKPOINTS[index], content)
    _next_member(state)


def _complete(
    state: SecondaryProgress, scoring: SecondaryScoring, retain: RetainSecondary | None
) -> ProducedExternalSecondary:
    state.stage = "completion_validation"
    require(len(state.columns) == len(SECONDARY_CHECKPOINTS))
    validate_secondary_scoring(scoring, len(state.record_ids))
    expected = bound_secondary._scoring_result(state.columns, len(state.record_ids))
    require(fixed_cascade._matches_exactly(asdict(scoring), asdict(expected)))
    receipt = completion_bytes(state.primary_hash, scoring, state.private_outputs)
    _store(state, retain, "secondary-completion.json", receipt)
    return ProducedExternalSecondary(scoring, dict(state.private_outputs))


def score_external_secondary(
    primary: ExternalPrimaryScores,
    bound: BoundSecondary,
    *,
    retain: RetainSecondary | None = None,
) -> ProducedExternalSecondary:
    """Score and retain each fixed complete column exactly once in frozen order."""
    state = SecondaryProgress()
    try:
        _validate_inputs(primary, bound, retain)
        state.primary_hash = sha256(primary.checkpoint_bytes).hexdigest()
        state.record_ids = tuple(row.record_id for row in primary.records)
        state.bindings = member_bindings(bound)
        _next_member(state)
        scoring = bound_secondary.score_bound_secondary(
            bound,
            tuple(row.raw_url for row in primary.records),
            tuple(score.stage1_probability for score in primary.scores),
            tuple(score.transformer_probability for score in primary.scores),
            on_completed_column=partial(_retain_column, state, retain),
        )
        return _complete(state, scoring, retain)
    except BaseException as exc:
        progress = secondary_progress_bytes(state)
        if isinstance(exc, Exception):
            raise ExternalSecondaryError(state.stage, progress) from None
        attach_failure_progress(
            exc, progress, f"external_secondary_{state.stage}_interrupted"
        )
        raise
