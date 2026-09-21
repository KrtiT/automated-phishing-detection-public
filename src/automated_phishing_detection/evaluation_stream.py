"""Validate saved external evidence, route the full stream, then select strata.

This pure boundary performs no I/O, source-schema adaptation, or model execution.
The future frozen runner must authenticate artifacts, record IDs, source-derived
labels, pinned-PSL metadata, and completeness/order of the original stream. These
checks establish structural consistency only, not that upstream provenance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import fixed_cascade, policy_replay, protocol_preflight
from .hypothesis_evaluation import SavedControls, SavedPopulation, WindowCounts
from .paired_evaluation import BinaryPrediction, EvaluationRecord
from .policy_replay import MonitorScore, PairedProbabilities, PolicyReplay

ROLES = ("gold", "certified", "tranco", "secondary")


class EvaluationStreamError(ValueError):
    """Saved evidence cannot form a complete, aligned external stream."""


@dataclass(frozen=True)
class OutcomeMetadata:
    """Prepared source-derived labels; secondary labels may be absent.

    Secondary labels, when present, must be publisher/source-derived, not model
    predictions. Establishing that fact and mapping source schemas is upstream.
    Tranco controls must remain unlabeled.
    """

    record_id: str
    registrable_domain: str
    is_phishing: int | None
    role: str


@dataclass(frozen=True)
class ExternalEvidence:
    populations: dict[str, SavedPopulation]
    controls: SavedControls
    external_windows: WindowCounts
    replay: PolicyReplay
    role_counts: dict[str, int]


def _validate_outcome(row: OutcomeMetadata) -> None:
    if type(row.role) is not str or row.role not in ROLES:
        raise EvaluationStreamError("unknown outcome role")
    label = row.is_phishing
    if label is not None and (type(label) is not int or label not in (0, 1)):
        raise EvaluationStreamError("is_phishing must be a binary integer or None")
    required_labels = {"gold": 1, "certified": 0, "tranco": None}
    if row.role in required_labels and label != required_labels[row.role]:
        raise EvaluationStreamError("outcome label does not match role")
    try:
        domain = protocol_preflight._ascii_domain(row.registrable_domain)
        protocol_preflight._reject_ip_literal(domain)
    except protocol_preflight.PreflightError as exc:
        raise EvaluationStreamError(f"invalid registrable domain: {exc}") from exc
    if domain != row.registrable_domain:
        raise EvaluationStreamError("registrable domain must already be canonical")


def _validate_inputs(
    metadata: Sequence[OutcomeMetadata],
    probabilities: Sequence[PairedProbabilities],
    monitor_scores: Sequence[MonitorScore],
    length_predictions: Sequence[BinaryPrediction],
) -> None:
    values = (metadata, probabilities, monitor_scores, length_predictions)
    if not all(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        for value in values
    ):
        raise EvaluationStreamError("inputs must be materialized ordered sequences")
    if len({len(value) for value in values}) != 1:
        raise EvaluationStreamError("metadata and saved prediction counts must match")

    # Reuse replay validation before routing, including finite scores and unique IDs.
    policy_replay._validate_inputs(probabilities, monitor_scores)
    for row, scores, length in zip(
        metadata, probabilities, length_predictions, strict=True
    ):
        if type(row) is not OutcomeMetadata or type(length) is not BinaryPrediction:
            raise EvaluationStreamError(
                "inputs must use typed metadata and predictions"
            )
        if (
            type(row.record_id) is not str
            or type(length.record_id) is not str
            or row.record_id != scores.record_id
            or length.record_id != scores.record_id
        ):
            raise EvaluationStreamError(
                "metadata and prediction record IDs or order differ"
            )
        if type(length.decision) is not int or length.decision not in (0, 1):
            raise EvaluationStreamError("length decision must be a binary integer")
        _validate_outcome(row)


def build_external_evidence(
    metadata: Sequence[OutcomeMetadata],
    probabilities: Sequence[PairedProbabilities],
    monitor_scores: Sequence[MonitorScore],
    length_predictions: Sequence[BinaryPrediction],
    *,
    stage1_threshold: float,
    transformer_threshold: float,
    half_width: float,
    monitor_boundary: float,
) -> ExternalEvidence:
    """Route every saved row once before selecting gold/certified/Tranco evidence.

    All four sequences must be typed, complete, and exactly aligned; no row is
    repaired, dropped, sorted, or relabeled. Secondary rows participate in routing
    and complete-window counts but never primary outcome metrics. Empty strata
    remain explicit so downstream zero denominators are not reported as successes.
    The returned routing masks are logical selection, not physical invocation or
    HTTP evidence. No protected-data readiness or source provenance is implied.
    """
    try:
        _validate_inputs(metadata, probabilities, monitor_scores, length_predictions)
        replay = policy_replay.replay_policy(
            probabilities,
            monitor_scores,
            stage1_threshold=stage1_threshold,
            transformer_threshold=transformer_threshold,
            half_width=half_width,
            monitor_boundary=monitor_boundary,
        )
    except (policy_replay.PolicyReplayError, fixed_cascade.FixedCascadeError) as exc:
        raise EvaluationStreamError(str(exc)) from exc

    role_indices: dict[str, list[int]] = {role: [] for role in ROLES}
    for index, row in enumerate(metadata):
        role_indices[row.role].append(index)

    populations = {}
    for role in ("gold", "certified"):
        indices = role_indices[role]
        predictions: dict[str, list[BinaryPrediction]] = {
            model: []
            for model in (
                "length_only",
                "logistic_l1",
                "transformer",
                "cascade",
                "policy",
            )
        }
        for index in indices:
            row = metadata[index]
            scores = probabilities[index]
            trace = replay.rows[index]
            decisions = {
                "length_only": length_predictions[index].decision,
                "logistic_l1": int(scores.stage1_probability >= stage1_threshold),
                "transformer": int(
                    scores.transformer_probability >= transformer_threshold
                ),
                "cascade": trace.fixed_decision,
                "policy": trace.policy_decision,
            }
            for model, decision in decisions.items():
                predictions[model].append(BinaryPrediction(row.record_id, decision))
        populations[role] = SavedPopulation(
            records=tuple(
                EvaluationRecord(
                    metadata[index].record_id,
                    metadata[index].registrable_domain,
                    metadata[index].is_phishing,
                )
                for index in indices
            ),
            predictions={model: tuple(rows) for model, rows in predictions.items()},
        )

    control_indices = role_indices["tranco"]
    controls = SavedControls(
        record_ids=tuple(metadata[index].record_id for index in control_indices),
        predictions={
            "cascade": tuple(
                BinaryPrediction(
                    metadata[index].record_id, replay.rows[index].fixed_decision
                )
                for index in control_indices
            ),
            "transformer": tuple(
                BinaryPrediction(
                    metadata[index].record_id,
                    int(
                        probabilities[index].transformer_probability
                        >= transformer_threshold
                    ),
                )
                for index in control_indices
            ),
        },
    )
    return ExternalEvidence(
        populations=populations,
        controls=controls,
        external_windows=WindowCounts(
            alert_windows=sum(window.alert for window in replay.windows),
            complete_windows=len(replay.windows),
        ),
        replay=replay,
        role_counts={role: len(indices) for role, indices in role_indices.items()},
    )
