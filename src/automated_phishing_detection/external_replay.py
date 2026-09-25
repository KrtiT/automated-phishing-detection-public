"""Join saved external scores after routing the complete stream, without inference."""

from dataclasses import dataclass, field

from . import evaluation_stream, external_monitors
from ._external_metrics_validation import validate_external
from ._external_secondary_inputs import (
    validate_primary_phase,
    validate_secondary_scoring,
)
from .bound_secondary import SecondaryScoredRow, SecondaryScoring
from .evaluation_stream import ExternalEvidence, OutcomeMetadata
from .external_evidence_types import ScoredExternalRow
from .external_primary import ExternalPrimaryScores
from .paired_evaluation import BinaryPrediction
from .phishvn import PreparedExternalRow
from .policy_replay import MonitorScore, PairedProbabilities, RoutingTrace
from .primary_scores import PrimaryURLScores
from .probe_replay import MonitorReplay, MonitorWindow
from .retained_drift import RetainedDriftReference


class ExternalReplayError(ValueError):
    """Symbolic rejection of inconsistent caller-retained scores or routing."""


@dataclass(frozen=True)
class ReplayedExternal:
    rows: tuple[ScoredExternalRow, ...] = field(repr=False)
    evidence: ExternalEvidence = field(repr=False)
    monitors: tuple[MonitorReplay, ...] = field(repr=False)


def _evidence(primary: ExternalPrimaryScores) -> ExternalEvidence:
    pairs = tuple(zip(primary.records, primary.scores, strict=True))
    thresholds = dict(primary.thresholds)
    return evaluation_stream.build_external_evidence(
        tuple(
            OutcomeMetadata(
                row.record_id, row.registrable_domain, row.is_phishing, row.role
            )
            for row in primary.records
        ),
        tuple(
            PairedProbabilities(
                row.record_id, score.stage1_probability, score.transformer_probability
            )
            for row, score in pairs
        ),
        tuple(
            MonitorScore(row.record_id, score.negative_log_likelihood)
            for row, score in pairs
        ),
        tuple(
            BinaryPrediction(row.record_id, score.length_decision)
            for row, score in pairs
        ),
        stage1_threshold=thresholds["logistic_l1"],
        transformer_threshold=thresholds["transformer"],
        half_width=thresholds["half_width"],
        monitor_boundary=thresholds["monitor_boundary"],
    )


def _aligned(
    record: PreparedExternalRow,
    score: PrimaryURLScores,
    trace: RoutingTrace,
    thresholds: dict,
) -> None:
    expected = (
        record.record_id,
        int(score.length_probability >= thresholds["length_only"]),
        int(score.stage1_probability >= thresholds["logistic_l1"]),
        int(score.transformer_probability >= thresholds["transformer"]),
        trace.fixed_decision,
        trace.logical_band,
        score.transformer_probability
        if trace.logical_band
        else score.stage1_probability,
    )
    observed = (
        trace.record_id,
        score.length_decision,
        score.stage1_decision,
        score.transformer_decision,
        score.cascade_decision,
        score.band_selected,
        score.cascade_probability,
    )
    if expected != observed:
        raise ExternalReplayError("invalid_external_replay")


def _join(
    record: PreparedExternalRow,
    score: PrimaryURLScores,
    secondary: SecondaryScoredRow,
    standardized: tuple[float, ...],
    trace: RoutingTrace,
    thresholds: dict,
) -> ScoredExternalRow:
    _aligned(record, score, trace, thresholds)
    _secondary_alignment(score, secondary)
    return ScoredExternalRow(
        record,
        score,
        secondary.tabular,
        secondary.seeds,
        standardized,
        score.transformer_probability
        if trace.logical_stage2_mask
        else score.stage1_probability,
        trace.policy_decision,
        trace.drift_override,
        trace.logical_stage2_mask,
    )


def _secondary_alignment(
    score: PrimaryURLScores, secondary: SecondaryScoredRow
) -> None:
    if secondary.seeds[0].transformer_probability != score.transformer_probability:
        raise ExternalReplayError("invalid_external_replay")
    for seed in secondary.seeds:
        probability, decision = (
            (seed.transformer_probability, seed.transformer_decision)
            if seed.band_selected
            else (score.stage1_probability, score.stage1_decision)
        )
        if (seed.cascade_probability, seed.cascade_decision) != (probability, decision):
            raise ExternalReplayError("invalid_external_replay")


def _gmm_monitor(evidence: ExternalEvidence, thresholds: dict) -> MonitorReplay:
    windows = tuple(
        MonitorWindow(
            window.start_position, window.end_position, window.score, window.alert
        )
        for window in evidence.replay.windows
    )
    return MonitorReplay(
        "gmm",
        thresholds["monitor_boundary"],
        None,
        windows,
        None if windows else "no_complete_256_row_window",
    )


def replay_external_scores(
    primary: ExternalPrimaryScores,
    secondary: SecondaryScoring,
    reference: RetainedDriftReference,
) -> ReplayedExternal:
    """Validate caller consistency, never source provenance or execution authority."""
    try:
        validate_primary_phase(primary)
        validate_secondary_scoring(secondary, len(primary.records))
        evidence = _evidence(primary)
        drift = external_monitors.replay_external_monitors(primary.scores, reference)
        thresholds = dict(primary.thresholds)
        rows = tuple(
            _join(record, score, members, standardized, trace, thresholds)
            for record, score, members, standardized, trace in zip(
                primary.records,
                primary.scores,
                secondary.rows,
                drift.standardized_features,
                evidence.replay.rows,
                strict=True,
            )
        )
        validate_external(rows)
        monitors = (_gmm_monitor(evidence, thresholds), *drift.monitors)
        return ReplayedExternal(rows, evidence, monitors)
    except Exception:
        raise ExternalReplayError("invalid_external_replay") from None
