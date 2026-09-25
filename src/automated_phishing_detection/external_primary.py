"""Single external primary phase; pure supplied inputs confer no access authority.

Completion writes are ordered and never retried. On failure the caller must retain
the exception's private progress bytes without resuming inference or promoting the
attempt. This phase does not bind publisher provenance or finalize an experiment.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from hashlib import sha256

from . import evaluation_producer
from ._external_inputs import validate_prepared_external
from ._external_primary_progress import (
    PrimaryProgress,
    attach_failure_progress,
    primary_progress_bytes,
    primary_rows_bytes,
)
from .bound_external_runtime import BoundExternalSession
from .bound_runtime import BoundEvaluationSession, BoundSession
from .phishvn import PreparedExternal, PreparedExternalRow
from .primary_scores import PrimaryURLScores
from .selective_inference import InferenceCounts

RetainPrimary = Callable[[str, bytes], None]


class ExternalPrimaryError(ValueError):
    """A symbolic phase error carrying private, immutable failure progress."""

    def __init__(self, stage: str, progress: bytes) -> None:
        super().__init__(f"external_primary_{stage}_failed")
        self.progress = progress


@dataclass(frozen=True)
class ExternalPrimaryScores:
    records: tuple[PreparedExternalRow, ...] = field(repr=False)
    scores: tuple[PrimaryURLScores, ...] = field(repr=False)
    thresholds: tuple[tuple[str, float], ...]
    inference_counts: InferenceCounts
    checkpoint_bytes: bytes = field(repr=False)
    receipt_bytes: bytes = field(repr=False)


def _validate_session(session: BoundExternalSession) -> BoundSession:
    if (
        type(session) is not BoundExternalSession
        or type(session.evaluation) is not BoundEvaluationSession
        or type(session.evaluation.primary) is not BoundSession
    ):
        raise ValueError("invalid_external_session")
    primary = session.evaluation.primary
    primary.scorer._require_owner()
    evaluation_producer._expected_counts(primary.scorer, 0)
    return primary


def _score_rows(
    state: PrimaryProgress, primary: BoundSession, thresholds: dict
) -> InferenceCounts:
    state.stage = "scoring"
    for position, record in enumerate(state.records, 1):
        state.started_position = position
        score = evaluation_producer.score_primary_url(
            record.raw_url, primary, thresholds, position
        )
        state.scores.append(score)
    state.started_position = None
    return evaluation_producer._expected_counts(primary.scorer, len(state.records))


def _receipt(
    source_hash: str,
    checkpoint: bytes,
    thresholds: dict,
    counts: InferenceCounts,
) -> bytes:
    return evaluation_producer._json_bytes(
        {
            "schema_version": 1,
            "phase": "external_primary",
            "row_count": counts.completed_requests,
            "retained_test_sha256": source_hash,
            "primary_scores_sha256": sha256(checkpoint).hexdigest(),
            "thresholds": thresholds,
            "inference_counts": asdict(counts),
        }
    )


def _complete(
    source_hash: str,
    state: PrimaryProgress,
    thresholds: dict,
    counts: InferenceCounts,
    retain: RetainPrimary | None,
) -> ExternalPrimaryScores:
    state.stage = "retention"
    scores = tuple(state.scores)
    checkpoint = primary_rows_bytes(state.records, scores)
    receipt = _receipt(source_hash, checkpoint, thresholds, counts)
    if retain is not None:
        retain("primary-scores.jsonl", checkpoint)
        retain("primary-completion.json", receipt)
    return ExternalPrimaryScores(
        state.records, scores, tuple(thresholds.items()), counts, checkpoint, receipt
    )


def score_external_primary(
    prepared: PreparedExternal,
    session: BoundExternalSession,
    *,
    retain: RetainPrimary | None = None,
) -> ExternalPrimaryScores:
    """Score once in retained order, preserving progress even on interruption."""
    state = PrimaryProgress()
    try:
        if retain is not None and not callable(retain):
            raise ValueError("invalid_retention_callback")
        state.records = validate_prepared_external(prepared)
        source_hash = sha256(
            prepared.private_outputs["retained-test.jsonl"]
        ).hexdigest()
        primary = _validate_session(session)
        thresholds = evaluation_producer._thresholds(primary)
        counts = _score_rows(state, primary, thresholds)
        return _complete(source_hash, state, thresholds, counts, retain)
    except BaseException as exc:
        progress = primary_progress_bytes(state, session)
        if isinstance(exc, Exception):
            raise ExternalPrimaryError(state.stage, progress) from None
        attach_failure_progress(exc, progress, f"external_primary_{state.stage}_failed")
        raise
