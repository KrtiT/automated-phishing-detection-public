"""Compose one supplied external stream with durable-boundary callbacks.

This helper performs no source read, experiment reservation, process observation
or completion publication. Returned summaries are not official accepted evidence.
Failure progress is private; the caller retains it without retrying inference.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace

from . import evaluation_producer, external_metrics, external_replay
from ._external_preparation_outputs import preparation_outputs
from ._external_primary_progress import attach_failure_progress
from ._external_producer_state import ExternalProgress, failure_bytes
from ._external_public_summary import public_summary
from .bound_external_runtime import BoundExternalSession
from .bound_secondary import SecondaryScoring
from .external_primary import ExternalPrimaryScores, score_external_primary
from .external_replay import ReplayedExternal
from .external_secondary import score_external_secondary
from .phishvn import PreparedExternal


class ExternalProducerError(ValueError):
    """Symbolic composition failure carrying private retained predecessor bytes."""

    def __init__(self, stage: str, progress: bytes) -> None:
        super().__init__(f"external_producer_{stage}_failed")
        self.progress = progress


@dataclass(frozen=True)
class ProducedExternal:
    replay: ReplayedExternal = field(repr=False)
    private_outputs: dict[str, bytes] = field(repr=False)
    public_summary: dict


def _prepare(
    state: ExternalProgress, prepared: PreparedExternal, session: BoundExternalSession
) -> PreparedExternal:
    if state.retain is not None and not callable(state.retain):
        raise ValueError("invalid_retention_callback")
    outputs = preparation_outputs(prepared, session)
    snapshot = replace(
        prepared,
        private_outputs=dict(prepared.private_outputs),
        public_summary=json.loads(outputs["preparation-summary.json"]),
    )
    for name, content in outputs.items():
        state.store(name, content)
    return snapshot


def _all_scores(primary: ExternalPrimaryScores, secondary: SecondaryScoring) -> bytes:
    return b"".join(
        evaluation_producer._json_bytes(
            {
                "record": asdict(record),
                "primary": asdict(score),
                "secondary": asdict(members),
            }
        )
        for record, score, members in zip(
            primary.records, primary.scores, secondary.rows, strict=True
        )
    )


def _score(
    state: ExternalProgress, prepared: PreparedExternal, session: BoundExternalSession
) -> tuple[ExternalPrimaryScores, SecondaryScoring]:
    state.stage = "primary"
    primary = score_external_primary(prepared, session, retain=state.store)
    bindings = json.loads(state.outputs["bindings.json"])
    if dict(primary.thresholds) != bindings["thresholds"]:
        raise ValueError("changed_primary_operating_points")
    state.stage = "secondary"
    secondary = score_external_secondary(
        primary, session.evaluation.secondary, retain=state.store
    ).scoring
    state.store("all-scores.jsonl", _all_scores(primary, secondary))
    return primary, secondary


def _derived(
    state: ExternalProgress,
    primary: ExternalPrimaryScores,
    secondary: SecondaryScoring,
    session: BoundExternalSession,
) -> ReplayedExternal:
    state.stage = "replay"
    replay = external_replay.replay_external_scores(
        primary, secondary, session.drift.reference
    )
    state.store(
        "routing.json", evaluation_producer._json_bytes(asdict(replay.evidence.replay))
    )
    state.store(
        "monitors.json",
        evaluation_producer._json_bytes(
            [asdict(monitor) for monitor in replay.monitors]
        ),
    )
    state.store(
        "predictions.jsonl",
        b"".join(evaluation_producer._json_bytes(asdict(row)) for row in replay.rows),
    )
    return replay


def _summarize(
    state: ExternalProgress,
    prepared: PreparedExternal,
    replay: ReplayedExternal,
    primary: ExternalPrimaryScores,
    secondary: SecondaryScoring,
) -> ProducedExternal:
    state.stage = "summary"
    summary = external_metrics.summarize_external(replay.rows)
    state.store("secondary.json", evaluation_producer._json_bytes(summary))
    public = public_summary(
        prepared, replay, primary.inference_counts, secondary.counts, state.outputs
    )
    evaluation_producer._json_bytes(public)
    return ProducedExternal(replay, dict(state.outputs), public)


def produce_external_evidence(
    prepared: PreparedExternal,
    session: BoundExternalSession,
    *,
    retain: Callable[[str, bytes], None] | None = None,
) -> ProducedExternal:
    """Retain every completed predecessor; never retry or finalize a root attempt."""
    state = ExternalProgress(retain)
    try:
        snapshot = _prepare(state, prepared, session)
        primary, secondary = _score(state, snapshot, session)
        replay = _derived(state, primary, secondary, session)
        return _summarize(state, snapshot, replay, primary, secondary)
    except BaseException as exc:
        progress = failure_bytes(state, session, exc)
        if isinstance(exc, Exception):
            raise ExternalProducerError(state.stage, progress) from None
        attach_failure_progress(
            exc, progress, f"external_producer_{state.stage}_failed"
        )
        raise
