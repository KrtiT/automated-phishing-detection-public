"""Private completed-prefix evidence, without retrying a failed retention writer."""

from dataclasses import asdict, dataclass, field

from .evaluation_producer import _json_bytes
from .phishvn import PreparedExternalRow
from .primary_scores import PrimaryURLScores
from .selective_inference import InferenceCounts


@dataclass
class PrimaryProgress:
    records: tuple[PreparedExternalRow, ...] | None = field(default=None, repr=False)
    scores: list[PrimaryURLScores] = field(default_factory=list, repr=False)
    stage: str = "input_validation"
    started_position: int | None = None


def attach_failure_progress(
    exception: BaseException, progress: bytes, reason: str
) -> None:
    """Keep interruption semantics; the parent must never publish raw tracebacks."""
    exception.progress = progress
    exception.args = (reason,)
    if isinstance(exception, SystemExit) and not (
        exception.code is None or isinstance(exception.code, int)
    ):
        exception.code = reason
    exception.__cause__ = None
    exception.__suppress_context__ = True


def primary_rows_bytes(
    records: tuple[PreparedExternalRow, ...], scores: tuple[PrimaryURLScores, ...]
) -> bytes:
    return b"".join(
        _json_bytes({"record": asdict(record), "primary": asdict(score)})
        for record, score in zip(records, scores)
    )


def _observed_counts(session: object) -> tuple[dict | None, str | None]:
    try:
        counts = session.evaluation.primary.scorer.counts
        if type(counts) is not InferenceCounts:
            raise ValueError()
        values = asdict(counts)
        if any(type(value) is not int or value < 0 for value in values.values()):
            raise ValueError()
        return values, None
    except Exception:
        return None, "physical_counts_unavailable"


def primary_progress_bytes(state: PrimaryProgress, session: object) -> bytes:
    counts, reason = _observed_counts(session)
    records = state.records or ()
    next_position = (state.started_position or len(state.scores)) + 1
    return _json_bytes(
        {
            "schema_version": 1,
            "status": "failed",
            "stage": state.stage,
            "expected_order_known": state.records is not None,
            "expected_record_ids": [record.record_id for record in records],
            "completed_rows": [
                {"record": asdict(record), "primary": asdict(score)}
                for record, score in zip(records, state.scores)
            ],
            "started_position": state.started_position,
            "unattempted_positions": list(range(next_position, len(records) + 1)),
            "inference_counts": counts,
            "inference_counts_reason": reason,
        }
    )
