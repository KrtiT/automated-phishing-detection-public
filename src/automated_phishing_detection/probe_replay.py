"""Bounded, label-free, in-memory secondary audit replay with no fitting or I/O.

The caller authenticates source allocation, accepted model identities, runtime,
and frozen references before using this composition. This module does not grant
research execution permission. Every URL receives both primary model scores;
routing masks describe logical selection, never avoided physical computation.
The three serialization probes are descriptive, not evidence of attack success.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from . import fixed_cascade, gmm_monitor, policy_replay, secondary_drift
from .secondary_probes import PERTURBATION_OPERATORS, perturb_url
from .url_features import extract_url_features

CONTRACT_ID = "secondary-seed-probe-v1"
STREAM_NAMES = ("original", *PERTURBATION_OPERATORS)
DETECTOR_NAMES = (
    "length",
    "logistic_l1",
    "transformer_42",
    "fixed_cascade",
    "gmm_policy",
)
MONITOR_NAMES = ("gmm", "mmd", "psi")


class ProbeReplayError(ValueError):
    """Replay stopped rather than repairing, skipping, or partially reporting."""


@dataclass(frozen=True)
class AuditInput:
    record_id: str
    validation_position: int
    raw_url: str


@dataclass(frozen=True)
class OperatingPoints:
    length_threshold: float
    stage1_threshold: float
    transformer_threshold: float
    half_width: float
    monitor_boundary: float


@dataclass(frozen=True)
class PrimaryScores:
    """Authoritative singleton scores, with exact input identity and audits."""

    record_id: str
    raw_url: str
    length_probability: float
    stage1_probability: float
    transformer_probability: float
    length_scoring_audit_json: str
    stage1_scoring_audit_json: str


@dataclass(frozen=True)
class ProbeMapping:
    record_id: str
    validation_position: int
    stream_position: int
    original_url: str
    output_url: str
    eligible: bool
    changed: bool


@dataclass(frozen=True)
class ProbeRow:
    """Private arrays use DETECTOR_NAMES and the existing feature order."""

    mapping: ProbeMapping
    probabilities: tuple[float, ...]
    decisions: tuple[int, ...]
    structural_features: tuple[float, ...]
    portable_monitor_probability: float
    negative_log_likelihood: float
    standardized_monitor_features: tuple[float, ...]
    length_scoring_audit_json: str
    stage1_scoring_audit_json: str
    logical_band: bool
    drift_override: bool
    logical_stage2_mask: bool


@dataclass(frozen=True)
class MonitorWindow:
    """Inclusive one-based stream positions; PSI retains all 26 feature scores."""

    start_position: int
    end_position: int
    score: float | None
    alert: bool | None
    feature_scores: tuple[float, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class MonitorReplay:
    name: str
    threshold: float | None
    calibration_window_count: int | None
    windows: tuple[MonitorWindow, ...]
    reason: str | None = None


@dataclass(frozen=True)
class ProbeStream:
    name: str
    rows: tuple[ProbeRow, ...]
    monitors: tuple[MonitorReplay, ...]


@dataclass(frozen=True)
class ProbeReplay:
    streams: tuple[ProbeStream, ...]

    @property
    def public_summary(self) -> dict:
        """A fresh aggregate-only view; private row identities never enter it."""
        return _summary(self.streams)


def make_primary_scorer(
    length_model, cascade_scorer
) -> Callable[[AuditInput], PrimaryScores]:
    """Adapt accepted loaded models and an already-open SelectiveCascade owner.

    No model/session is loaded or entered here. Identity and runtime binding
    belong to the caller; the counter delta enforces one paired singleton call.
    """
    from . import length_inference
    from .selective_inference import InferenceCounts, RequestScores

    def score(record):
        try:
            if type(record) is not AuditInput:
                raise ProbeReplayError("primary adapter needs typed AuditInput")
            before = cascade_scorer.counts
            if type(before) is not InferenceCounts or any(
                type(value) is not int or value < 0 for value in vars(before).values()
            ):
                raise ProbeReplayError("singleton inference counts are invalid")
            length_values, length_audit = (
                length_inference.score_length_only_authoritative(
                    length_model, (record.raw_url,)
                )
            )
            length = _singleton_probability(length_values, "length probability")
            scores = cascade_scorer.score_all(record.raw_url)
            if type(scores) is not RequestScores:
                raise ProbeReplayError(
                    "paired singleton scorer must return RequestScores"
                )
            flags = (
                scores.band_selected,
                scores.drift_override,
                scores.logical_stage2_selected,
                scores.transformer_evaluated,
            )
            if (
                any(type(value) is not bool for value in flags)
                or flags != (scores.band_selected, False, scores.band_selected, True)
                or type(scores.decision) is not int
                or scores.decision not in (0, 1)
                or type(scores.fixed_decision) is not int
                or scores.fixed_decision != scores.decision
            ):
                raise ProbeReplayError(
                    "paired singleton scorer returned selective or overridden results"
                )
            expected = InferenceCounts(
                before.transformer_forward_attempts + 1,
                before.successful_transformer_scores + 1,
                before.completed_requests + 1,
                before.failed_requests,
            )
            after = cascade_scorer.counts
            if (
                type(after) is not InferenceCounts
                or any(type(value) is not int for value in vars(after).values())
                or after != expected
            ):
                raise ProbeReplayError(
                    "paired singleton inference count coverage differs"
                )
            return PrimaryScores(
                record.record_id,
                record.raw_url,
                length,
                _probability(scores.stage1_probability, "stage1 probability"),
                _probability(scores.transformer_probability, "transformer probability"),
                _audit_json(json.dumps(length_audit, allow_nan=False)),
                _audit_json(json.dumps(scores.stage1_scoring_audit, allow_nan=False)),
            )
        except ProbeReplayError:
            raise
        except Exception as exc:
            raise ProbeReplayError(f"primary singleton scoring stopped: {exc}") from exc

    return score


def _number(value, field):
    return fixed_cascade._finite_number(value, field)


def _probability(value, field):
    value = _number(value, field)
    if not 0 <= value <= 1:
        raise ProbeReplayError(f"{field} must be in [0, 1]")
    return value


def _vector(values, width, field):
    return tuple(
        float(value) for value in gmm_monitor._finite_array(values, (width,), field)
    )


def _singleton_probability(values, field):
    value = _vector(values, 1, field)[0]
    return _probability(value, field)


def _audit_json(value):
    if type(value) is not str:
        raise ProbeReplayError("scoring audit must be a JSON object string")
    parsed = json.loads(
        value,
        object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
        parse_constant=fixed_cascade._reject_json_constant,
    )
    if type(parsed) is not dict:
        raise ProbeReplayError("scoring audit must be a JSON object")
    return json.dumps(
        parsed,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_operating_points(points):
    if type(points) is not OperatingPoints:
        raise ProbeReplayError("operating_points must be typed")
    for name in ("length_threshold", "stage1_threshold", "transformer_threshold"):
        fixed_cascade._threshold(getattr(points, name), name)
    if _number(points.half_width, "half_width") < 0:
        raise ProbeReplayError("half_width must be nonnegative")
    _number(points.monitor_boundary, "monitor_boundary")


def _validate_calibration(calibration):
    if type(calibration) is not secondary_drift.DriftCalibration:
        raise ProbeReplayError("calibration must be a frozen DriftCalibration")
    count = calibration.calibration_window_count
    if type(count) is not int or count < 0:
        raise ProbeReplayError("calibration window count must be a nonnegative integer")
    if calibration.reason is not None:
        if type(calibration.reason) is not str or not calibration.reason:
            raise ProbeReplayError("calibration reason must be nonempty")
    elif not count or calibration.threshold is None:
        raise ProbeReplayError("estimable calibration needs a threshold and windows")
    if calibration.threshold is not None:
        _number(calibration.threshold, "calibration threshold")


def _validate_references(mmd, psi):
    if (
        type(mmd) is not secondary_drift.MMDReference
        or type(psi) is not secondary_drift.PSIReference
    ):
        raise ProbeReplayError("frozen MMDReference and PSIReference are required")
    for reference in (mmd, psi):
        if reference.reason is not None and (
            type(reference.reason) is not str or not reference.reason
        ):
            raise ProbeReplayError("reference reason must be nonempty")
    if not all(
        type(values) is tuple
        for values in (mmd.values, mmd.domains, mmd.stable_ids, psi.features)
    ):
        raise ProbeReplayError("frozen reference arrays must be tuples")
    if not len(mmd.values) == len(mmd.domains) == len(mmd.stable_ids):
        raise ProbeReplayError("MMD reference identity coverage differs")
    for row in mmd.values:
        _vector(row, 26, "MMD reference")
    if mmd.bandwidth_squared is not None:
        _number(mmd.bandwidth_squared, "MMD bandwidth")
    if mmd.reason is None:
        if (
            len(mmd.values) != 256
            or _number(mmd.bandwidth_squared, "MMD bandwidth") <= 0
        ):
            raise ProbeReplayError(
                "estimable MMD reference needs 256 rows and a positive bandwidth"
            )
    if type(psi.training_row_count) is not int or psi.training_row_count < 0:
        raise ProbeReplayError("PSI training row count is invalid")
    if psi.reason is None and (len(psi.features) != 26 or psi.training_row_count == 0):
        raise ProbeReplayError(
            "estimable PSI reference needs 26 features and training rows"
        )
    for feature in psi.features:
        if type(feature) is not secondary_drift.PSIFeatureReference:
            raise ProbeReplayError("PSI feature reference must be typed")
        edges = _vector(
            feature.internal_edges, len(feature.internal_edges), "PSI edges"
        )
        if any(a >= b for a, b in zip(edges, edges[1:])):
            raise ProbeReplayError("PSI edges must strictly increase")
        if feature.constant is not None:
            _number(feature.constant, "PSI constant")
            if edges:
                raise ProbeReplayError(
                    "constant PSI reference must have no internal edges"
                )
        bins = 3 if feature.constant is not None else len(edges) + 1
        counts = feature.training_counts
        proportions = _vector(feature.training_proportions, bins, "PSI proportions")
        if (
            len(counts) != bins
            or any(type(n) is not int or n < 0 for n in counts)
            or sum(counts) != psi.training_row_count
        ):
            raise ProbeReplayError("PSI training count coverage differs")
        expected = tuple(
            (n + 0.5) / (psi.training_row_count + 0.5 * bins) for n in counts
        )
        if proportions != expected:
            raise ProbeReplayError("PSI proportions differ from frozen smoothed counts")


def _prepare_mappings(records):
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ProbeReplayError("records must be a materialized ordered sequence")
    streams = [[] for _ in STREAM_NAMES]
    seen, previous = set(), -1
    for position, row in enumerate(records, start=1):
        if type(row) is not AuditInput:
            raise ProbeReplayError("records must use label-free AuditInput")
        identity = row.record_id
        if (
            type(identity) is not str
            or not identity
            or any(c.isspace() or not c.isprintable() for c in identity)
            or identity in seen
        ):
            raise ProbeReplayError("record IDs must be unique nonempty stable strings")
        if (
            type(row.validation_position) is not int
            or row.validation_position <= previous
        ):
            raise ProbeReplayError(
                "zero-based validation positions must strictly increase"
            )
        seen.add(identity)
        previous = row.validation_position
        probes = perturb_url(row.raw_url)
        if tuple(probe.operator for probe in probes) != PERTURBATION_OPERATORS:
            raise ProbeReplayError("operator coverage differs from the frozen order")
        original = ProbeMapping(
            identity,
            row.validation_position,
            position,
            row.raw_url,
            row.raw_url,
            True,
            False,
        )
        streams[0].append(original)
        for stream, probe in zip(streams[1:], probes, strict=True):
            stream.append(
                replace(
                    original,
                    output_url=probe.output_url,
                    eligible=probe.eligible,
                    changed=probe.changed,
                )
            )
    return tuple(tuple(stream) for stream in streams)


def _score_mapping(mapping, scorer, portable_model, artifact, points):
    request = AuditInput(
        mapping.record_id, mapping.validation_position, mapping.output_url
    )
    scores = scorer(request)
    if type(scores) is not PrimaryScores:
        raise ProbeReplayError("primary scorer must return typed PrimaryScores")
    if scores.record_id != request.record_id or scores.raw_url != request.raw_url:
        raise ProbeReplayError("primary score identity or exact URL alignment differs")
    probabilities = tuple(
        _probability(getattr(scores, name), name)
        for name in (
            "length_probability",
            "stage1_probability",
            "transformer_probability",
        )
    )
    audits = tuple(
        _audit_json(value)
        for value in (
            scores.length_scoring_audit_json,
            scores.stage1_scoring_audit_json,
        )
    )
    length, stage1, transformer = probabilities
    features = _vector(extract_url_features(request.raw_url), 25, "structural features")
    portable = _singleton_probability(
        portable_model.score_urls((request.raw_url,)), "portable monitor probability"
    )
    monitor_features = (*features, portable)
    nll = _vector(
        gmm_monitor.score_feature_matrix((monitor_features,), artifact),
        1,
        "negative log likelihood",
    )[0]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        standardized = (
            np.asarray(monitor_features) - np.asarray(artifact["scaler"]["mean"])
        ) / np.asarray(artifact["scaler"]["scale"])
    standardized = _vector(standardized, 26, "standardized monitor features")
    band = abs(stage1 - points.stage1_threshold) <= points.half_width
    fixed_probability = transformer if band else stage1
    fixed_decision = (
        int(transformer >= points.transformer_threshold)
        if band
        else int(stage1 >= points.stage1_threshold)
    )
    return ProbeRow(
        mapping,
        (*probabilities, fixed_probability, fixed_probability),
        (
            int(length >= points.length_threshold),
            int(stage1 >= points.stage1_threshold),
            int(transformer >= points.transformer_threshold),
            fixed_decision,
            fixed_decision,
        ),
        features,
        portable,
        nll,
        standardized,
        *audits,
        band,
        False,
        band,
    )


def _secondary_monitor(name, reference, calibration, matrix, expected_ends):
    score_windows = (
        secondary_drift.mmd_window_scores
        if name == "mmd"
        else secondary_drift.psi_window_scores
    )
    scores = score_windows(reference, matrix)
    if (
        type(scores) is not secondary_drift.DriftWindowScores
        or scores.window_end_positions != expected_ends
    ):
        raise ProbeReplayError(f"{name} window coverage differs")
    if scores.reason is None:
        values = _vector(scores.scores, len(expected_ends), f"{name} window scores")
        if name == "psi":
            if len(scores.feature_scores) != len(expected_ends):
                raise ProbeReplayError("PSI feature score coverage differs")
            features = tuple(
                _vector(row, 26, "PSI feature scores") for row in scores.feature_scores
            )
            if any(
                value != max(row) for value, row in zip(values, features, strict=True)
            ):
                raise ProbeReplayError("PSI window scores differ from feature maxima")
        else:
            if scores.feature_scores:
                raise ProbeReplayError("MMD unexpectedly returned feature scores")
            features = ((),) * len(expected_ends)
    else:
        if (
            type(scores.reason) is not str
            or not scores.reason
            or scores.scores
            or scores.feature_scores
        ):
            raise ProbeReplayError(f"{name} unavailable score coverage differs")
        values, features = (None,) * len(expected_ends), ((),) * len(expected_ends)
    reason = reference.reason or calibration.reason or scores.reason
    windows = tuple(
        MonitorWindow(
            end - 255,
            end,
            value,
            value > calibration.threshold if reason is None else None,
            feature_scores,
            reason,
        )
        for end, value, feature_scores in zip(
            expected_ends, values, features, strict=True
        )
    )
    return MonitorReplay(
        name,
        calibration.threshold,
        calibration.calibration_window_count,
        windows,
        reason,
    )


def _replay_stream(
    name, rows, points, mmd_reference, mmd_calibration, psi_reference, psi_calibration
):
    route = policy_replay.replay_policy(
        tuple(
            policy_replay.PairedProbabilities(
                row.mapping.record_id, row.probabilities[1], row.probabilities[2]
            )
            for row in rows
        ),
        tuple(
            policy_replay.MonitorScore(
                row.mapping.record_id, row.negative_log_likelihood
            )
            for row in rows
        ),
        stage1_threshold=points.stage1_threshold,
        transformer_threshold=points.transformer_threshold,
        half_width=points.half_width,
        monitor_boundary=points.monitor_boundary,
    )
    if len(route.rows) != len(rows):
        raise ProbeReplayError("routing row coverage differs")
    routed = []
    for row, trace in zip(rows, route.rows, strict=True):
        if (
            trace.record_id != row.mapping.record_id
            or trace.fixed_decision != row.decisions[3]
            or trace.logical_band != row.logical_band
        ):
            raise ProbeReplayError(
                "routing identity or fixed cascade alignment differs"
            )
        probability = (
            row.probabilities[2] if trace.logical_stage2_mask else row.probabilities[1]
        )
        routed.append(
            replace(
                row,
                probabilities=(*row.probabilities[:4], probability),
                decisions=(*row.decisions[:4], trace.policy_decision),
                drift_override=trace.drift_override,
                logical_stage2_mask=trace.logical_stage2_mask,
            )
        )
    ends = tuple(range(256, len(rows) + 1, 64))
    if tuple(window.end_position for window in route.windows) != ends:
        raise ProbeReplayError("GMM window coverage differs")
    gmm_windows = tuple(
        MonitorWindow(
            window.start_position,
            window.end_position,
            _number(window.score, "GMM window score"),
            window.alert,
        )
        for window in route.windows
    )
    gmm = MonitorReplay(
        "gmm",
        points.monitor_boundary,
        None,
        gmm_windows,
        None if ends else "no_complete_256_row_window",
    )
    matrix = np.asarray(
        tuple(row.standardized_monitor_features for row in rows), dtype=np.float64
    ).reshape(len(rows), 26)
    monitors = (
        gmm,
        _secondary_monitor("mmd", mmd_reference, mmd_calibration, matrix, ends),
        _secondary_monitor("psi", psi_reference, psi_calibration, matrix, ends),
    )
    return ProbeStream(name, tuple(routed), monitors)


def replay_probes(
    records: Sequence[AuditInput],
    *,
    primary_scorer: Callable[[AuditInput], PrimaryScores],
    stage1_model,
    gmm_artifact: dict,
    operating_points: OperatingPoints,
    mmd_reference: secondary_drift.MMDReference,
    mmd_calibration: secondary_drift.DriftCalibration,
    psi_reference: secondary_drift.PSIReference,
    psi_calibration: secondary_drift.DriftCalibration,
    row_callback: Callable[[str, ProbeRow], None] | None = None,
    stream_callback: Callable[[ProbeStream], None] | None = None,
) -> ProbeReplay:
    """Score exactly four independent streams, in fixed order, or raise.

    ``primary_scorer`` supplies authoritative length/Logistic-L1/seed-42 scores.
    ``stage1_model.score_urls`` is only the accepted portable monitor scorer.
    Artifact/source binding and authorization remain the caller's responsibility.
    Supplied records, artifact dictionaries, and frozen references are unchanged.

    Optional callbacks run synchronously and exceptions stop replay immediately.
    ``row_callback(stream_name, row)`` receives each scored-row snapshot before
    the next score or any policy/window replay. Its policy routing fields are
    provisional: they mirror the fixed cascade with no drift override.
    ``stream_callback(stream)`` receives authoritative routing and windows before
    the next stream or final aggregation. Neither callback changes the result.
    """
    try:
        mappings = _prepare_mappings(records)
        _validate_operating_points(operating_points)
        _validate_references(mmd_reference, psi_reference)
        _validate_calibration(mmd_calibration)
        _validate_calibration(psi_calibration)
        artifact = gmm_monitor.load_gmm_artifact_bytes(
            gmm_monitor._canonical_json_bytes(gmm_artifact)
        )
        if not callable(primary_scorer) or not callable(
            getattr(stage1_model, "score_urls", None)
        ):
            raise ProbeReplayError("primary and portable scorers are required")
        for name, callback in (
            ("row_callback", row_callback),
            ("stream_callback", stream_callback),
        ):
            if callback is not None and not callable(callback):
                raise ProbeReplayError(f"{name} must be callable or None")
        streams = []
        for name, stream_mappings in zip(STREAM_NAMES, mappings, strict=True):
            rows = []
            for mapping in stream_mappings:
                row = _score_mapping(
                    mapping, primary_scorer, stage1_model, artifact, operating_points
                )
                if row_callback is not None:
                    row_callback(name, row)
                rows.append(row)
            stream = _replay_stream(
                name,
                tuple(rows),
                operating_points,
                mmd_reference,
                mmd_calibration,
                psi_reference,
                psi_calibration,
            )
            if stream_callback is not None:
                stream_callback(stream)
            streams.append(stream)
        result = ProbeReplay(tuple(streams))
        result.public_summary  # Fail stop if finite row scores overflow aggregation.
        return result
    except ProbeReplayError:
        raise
    except Exception as exc:
        raise ProbeReplayError(f"probe replay stopped: {exc}") from exc


def _mean(values):
    return (
        _number(math.fsum(values) / len(values), "aggregate mean") if values else None
    )


def _differences(original, transformed):
    deltas = tuple(
        _number(b - a, "paired difference")
        for a, b in zip(original, transformed, strict=True)
    )
    absolute = tuple(abs(value) for value in deltas)
    return {
        "count": len(deltas),
        "changed_score_count": sum(
            a != b for a, b in zip(original, transformed, strict=True)
        ),
        "mean_signed_delta": _mean(deltas),
        "mean_absolute_delta": _mean(absolute),
        "max_absolute_delta": max(absolute) if absolute else None,
    }


def _transitions(original, transformed):
    result = dict.fromkeys(("00", "01", "10", "11"), 0)
    for first, second in zip(original, transformed, strict=True):
        result[f"{int(first)}{int(second)}"] += 1
    return result


def _monitor_summary(monitor):
    estimable = monitor.reason is None
    scores = tuple(window.score for window in monitor.windows)
    count = sum(window.alert for window in monitor.windows) if estimable else None
    return {
        "window_count": len(monitor.windows),
        "threshold": monitor.threshold,
        "calibration_window_count": monitor.calibration_window_count,
        "score_mean": _mean(scores) if estimable else None,
        "alert_count": count,
        "alert_fraction": count / len(scores) if estimable and scores else None,
        "reason": monitor.reason,
    }


def _paired_monitors(original, transformed):
    reason = original.reason or transformed.reason
    return {
        "window_count": len(transformed.windows),
        "original_window_count": len(original.windows),
        "transformed_window_count": len(transformed.windows),
        "original_alert_count": None
        if reason
        else sum(w.alert for w in original.windows),
        "transformed_alert_count": None
        if reason
        else sum(w.alert for w in transformed.windows),
        "original_score_mean": None
        if reason
        else _mean(tuple(w.score for w in original.windows)),
        "transformed_score_mean": None
        if reason
        else _mean(tuple(w.score for w in transformed.windows)),
        "score_differences": None
        if reason
        else _differences(
            tuple(w.score for w in original.windows),
            tuple(w.score for w in transformed.windows),
        ),
        "alert_transitions": None
        if reason
        else _transitions(
            tuple(w.alert for w in original.windows),
            tuple(w.alert for w in transformed.windows),
        ),
        "reason": reason,
    }


def _summary(streams):
    original = streams[0]
    summaries = []
    for stream in streams:
        detectors, paired = {}, {}
        for index, name in enumerate(DETECTOR_NAMES):
            probabilities = tuple(row.probabilities[index] for row in stream.rows)
            decisions = tuple(row.decisions[index] for row in stream.rows)
            detectors[name] = {
                "score_mean": _mean(probabilities),
                "positive_decision_count": sum(decisions),
            }
            paired[name] = {
                "score_differences": _differences(
                    tuple(row.probabilities[index] for row in original.rows),
                    probabilities,
                ),
                "decision_transitions": _transitions(
                    tuple(row.decisions[index] for row in original.rows), decisions
                ),
            }
        summaries.append(
            {
                "name": stream.name,
                "row_count": len(stream.rows),
                "mapping_counts": {
                    "eligible": sum(row.mapping.eligible for row in stream.rows),
                    "changed": sum(row.mapping.changed for row in stream.rows),
                    "eligible_noop": sum(
                        row.mapping.eligible and not row.mapping.changed
                        for row in stream.rows
                    ),
                    "ineligible": sum(not row.mapping.eligible for row in stream.rows),
                },
                "detectors": detectors,
                "monitors": {
                    monitor.name: _monitor_summary(monitor)
                    for monitor in stream.monitors
                },
                "paired_with_original": {
                    "detectors": paired,
                    "monitors": {
                        monitor.name: _paired_monitors(reference, monitor)
                        for reference, monitor in zip(
                            original.monitors, stream.monitors, strict=True
                        )
                    },
                },
            }
        )
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "scope": "in_memory_descriptive_only",
        "streams": summaries,
    }
