"""Pure external MMD/PSI replay from caller-bound retained state and scores.

The caller binds accepted references, scalers and calibration before scoring.
This composition grants no execution authority and performs no fitting or I/O.
"""

from dataclasses import dataclass

import numpy as np

from . import probe_replay
from .primary_scores import PrimaryURLScores
from .probe_replay import MonitorReplay
from .retained_drift import RetainedDriftReference


class ExternalMonitorError(ValueError):
    """Symbolic rejection without private score or retained-state values."""


@dataclass(frozen=True)
class ExternalDrift:
    standardized_features: tuple[tuple[float, ...], ...]
    monitors: tuple[MonitorReplay, ...]


def _vector(values: tuple[float, ...], width: int) -> tuple[float, ...]:
    if type(values) is not tuple:
        raise ExternalMonitorError("invalid_external_monitor_state")
    return probe_replay._vector(values, width, "external_monitor_vector")


def _matrix(scores: tuple[PrimaryURLScores, ...]) -> np.ndarray:
    if type(scores) is not tuple:
        raise ExternalMonitorError("invalid_external_monitor_state")
    rows = []
    for score in scores:
        if type(score) is not PrimaryURLScores:
            raise ExternalMonitorError("invalid_external_monitor_state")
        features = _vector(score.features, 25)
        probability = probe_replay._probability(
            score.monitor_probability, "portable_monitor_probability"
        )
        rows.append((*features, probability))
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), 26)


def _standardize(
    scores: tuple[PrimaryURLScores, ...], reference: RetainedDriftReference
) -> np.ndarray:
    mean = np.asarray(_vector(reference.scaler_mean, 26), dtype=np.float64)
    scale = np.asarray(_vector(reference.scaler_scale, 26), dtype=np.float64)
    if np.any(scale <= 0):
        raise ExternalMonitorError("invalid_external_monitor_state")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        matrix = (_matrix(scores) - mean) / scale
    if matrix.shape != (len(scores), 26) or not np.all(np.isfinite(matrix)):
        raise ExternalMonitorError("invalid_external_monitor_state")
    return matrix


def replay_external_monitors(
    scores: tuple[PrimaryURLScores, ...], reference: RetainedDriftReference
) -> ExternalDrift:
    """Replay fixed MMD/PSI windows with the retained strict alert thresholds."""
    try:
        if type(reference) is not RetainedDriftReference:
            raise ExternalMonitorError("invalid_external_monitor_state")
        probe_replay._validate_references(reference.mmd, reference.psi)
        for calibration in (reference.mmd_calibration, reference.psi_calibration):
            probe_replay._validate_calibration(calibration)
        matrix = _standardize(scores, reference)
        expected_ends = tuple(range(256, len(scores) + 1, 64))
        monitors = tuple(
            probe_replay._secondary_monitor(
                name,
                getattr(reference, name),
                getattr(reference, f"{name}_calibration"),
                matrix,
                expected_ends,
            )
            for name in ("mmd", "psi")
        )
        return ExternalDrift(tuple(tuple(row) for row in matrix.tolist()), monitors)
    except Exception:
        raise ExternalMonitorError("invalid_external_monitor_state") from None
