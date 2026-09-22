"""Pure, development-only MMD/PSI comparators from already-standardized inputs.

The caller supplies training-only data in the existing 26-dimensional monitor
representation and preserves the original calibration/audit stream allocation.
These utilities neither fit a scaler nor access files, authorize execution, or
turn dependent windows into hypothesis tests. Reference and bin construction
must precede calibration; the resulting strict boundary is reused unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from numbers import Real

import numpy as np

from .protocol_preflight import PreflightError, _ascii_domain

FEATURE_COUNT = 26
WINDOW_SIZE = 256
WINDOW_STRIDE = 64
_REFERENCE_PREFIX = "secondary-mmd-reference-v1:20260816:"


class SecondaryDriftError(ValueError):
    """Inputs cannot be used under the frozen secondary drift method."""


@dataclass(frozen=True)
class DriftEstimate:
    value: float | None
    reason: str | None = None


@dataclass(frozen=True)
class MMDReference:
    values: tuple[tuple[float, ...], ...]
    domains: tuple[str, ...]
    stable_ids: tuple[str, ...]
    bandwidth_squared: float | None
    reason: str | None = None


@dataclass(frozen=True)
class PSIFeatureReference:
    internal_edges: tuple[float, ...]
    constant: float | None
    training_counts: tuple[int, ...]
    training_proportions: tuple[float, ...]


@dataclass(frozen=True)
class PSIReference:
    features: tuple[PSIFeatureReference, ...]
    training_row_count: int
    reason: str | None = None


@dataclass(frozen=True)
class PSIWindowScore:
    value: float | None
    feature_scores: tuple[float, ...]
    reason: str | None = None


@dataclass(frozen=True)
class DriftWindowScores:
    window_end_positions: tuple[int, ...]
    scores: tuple[float, ...]
    feature_scores: tuple[tuple[float, ...], ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class DriftCalibration:
    threshold: float | None
    calibration_window_count: int
    reason: str | None = None


@dataclass(frozen=True)
class DriftAudit:
    threshold: float | None
    calibration_window_count: int
    window_count: int
    alerts: tuple[bool, ...]
    alert_count: int | None
    alert_fraction: float | None
    reason: str | None = None


def _numeric_array(value, field: str) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=object)
        if any(
            not isinstance(item, Real) or isinstance(item, (bool, np.bool_))
            for item in raw.flat
        ):
            raise SecondaryDriftError(f"{field} must contain real numeric values")
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SecondaryDriftError(
            f"{field} must contain finite numeric values"
        ) from exc
    if not np.all(np.isfinite(array)):
        raise SecondaryDriftError(f"{field} must contain finite numeric values")
    return array


def _matrix(value) -> np.ndarray:
    matrix = _numeric_array(value, "standardized features")
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_COUNT:
        raise SecondaryDriftError("standardized features must be a 26-column matrix")
    return matrix


def _window(value) -> np.ndarray:
    matrix = _matrix(value)
    if len(matrix) != WINDOW_SIZE:
        raise SecondaryDriftError("an individual window must contain exactly 256 rows")
    return matrix


def _squared_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            differences = first[:, None, :] - second[None, :, :]
            distances = np.sum(differences * differences, axis=2, dtype=np.float64)
    except FloatingPointError as exc:
        raise SecondaryDriftError("squared pairwise distances are nonfinite") from exc
    return distances


def rbf_kernel(first, second, *, bandwidth_squared: float) -> np.ndarray:
    """Return a float64 C-contiguous RBF kernel, with no bandwidth selection."""
    first, second = _matrix(first), _matrix(second)
    if (
        isinstance(bandwidth_squared, (bool, np.bool_))
        or not isinstance(bandwidth_squared, Real)
        or not np.isfinite(bandwidth_squared)
        or bandwidth_squared <= 0
    ):
        raise SecondaryDriftError("bandwidth_squared must be finite and positive")
    distances = _squared_distances(first, second)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            kernel = np.exp(-distances / (2 * np.float64(bandwidth_squared)))
    except FloatingPointError as exc:
        raise SecondaryDriftError("RBF kernel computation is nonfinite") from exc
    return np.ascontiguousarray(kernel, dtype=np.float64)


def _training_identities(domains, stable_ids, row_count):
    for values in (domains, stable_ids):
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise SecondaryDriftError(
                "domains and stable IDs must be ordered sequences"
            )
        if len(values) != row_count:
            raise SecondaryDriftError("domains and stable IDs must align with rows")
    if any(not isinstance(value, str) or not value.strip() for value in stable_ids):
        raise SecondaryDriftError("stable IDs must be nonempty strings")
    if len(set(stable_ids)) != row_count:
        raise SecondaryDriftError("stable IDs must be unique")
    try:
        return tuple(_ascii_domain(domain) for domain in domains)
    except PreflightError as exc:
        raise SecondaryDriftError(f"invalid canonical domain: {exc}") from exc


def fit_mmd_reference(training_standardized, domains, stable_ids) -> MMDReference:
    """Choose the frozen training rows and positive-distance linear median.

    Stable IDs use lexicographic string ordering, independently of input order.
    Canonical domains use the existing preflight IDNA/lowercase convention.
    """
    matrix = _matrix(training_standardized)
    canonical = _training_identities(domains, stable_ids, len(matrix))
    rows_by_domain = {}
    for index, domain in enumerate(canonical):
        if (
            domain not in rows_by_domain
            or stable_ids[index] < stable_ids[rows_by_domain[domain]]
        ):
            rows_by_domain[domain] = index
    selected = tuple(
        sorted(
            rows_by_domain,
            key=lambda domain: (
                sha256((_REFERENCE_PREFIX + domain).encode("ascii")).digest(),
                domain,
            ),
        )[:WINDOW_SIZE]
    )
    positions = [rows_by_domain[domain] for domain in selected]
    values = tuple(tuple(float(value) for value in matrix[i]) for i in positions)
    ids = tuple(stable_ids[i] for i in positions)
    if len(selected) < WINDOW_SIZE:
        return MMDReference(
            values, selected, ids, None, "fewer_than_256_training_domains"
        )
    distances = _squared_distances(matrix[positions], matrix[positions])
    upper_triangle = distances[np.triu_indices(WINDOW_SIZE, k=1)]
    positive = upper_triangle[upper_triangle > 0]
    if not len(positive):
        return MMDReference(
            values, selected, ids, None, "no_positive_reference_distance"
        )
    bandwidth = float(np.quantile(positive, 0.5, method="linear"))
    return MMDReference(values, selected, ids, bandwidth)


def mmd_squared(reference: MMDReference, window_standardized) -> DriftEstimate:
    """Biased MMD squared, including diagonals and preserving float64 roundoff."""
    window = _window(window_standardized)
    if not isinstance(reference, MMDReference):
        raise SecondaryDriftError("reference must be an MMDReference")
    if reference.reason is not None:
        return DriftEstimate(None, reference.reason)
    training = _window(reference.values)
    rr = rbf_kernel(training, training, bandwidth_squared=reference.bandwidth_squared)
    ww = rbf_kernel(window, window, bandwidth_squared=reference.bandwidth_squared)
    rw = rbf_kernel(training, window, bandwidth_squared=reference.bandwidth_squared)
    value = float(
        np.mean(rr, dtype=np.float64)
        + np.mean(ww, dtype=np.float64)
        - 2 * np.mean(rw, dtype=np.float64)
    )
    return DriftEstimate(value)


def _bin_counts(values, internal_edges, constant):
    if constant is not None:
        bins = np.where(values < constant, 0, np.where(values == constant, 1, 2))
        count = 3
    else:
        # Search on internal edges is equivalent to using infinite outer edges.
        bins = np.searchsorted(internal_edges, values, side="right")
        count = len(internal_edges) + 1
    return np.bincount(bins, minlength=count)


def _proportions(counts, row_count):
    return (np.asarray(counts, dtype=np.float64) + 0.5) / (
        row_count + 0.5 * len(counts)
    )


def fit_psi_reference(training_standardized) -> PSIReference:
    """Freeze training deciles and smoothed proportions without a new scaler."""
    matrix = _matrix(training_standardized)
    if not len(matrix):
        return PSIReference((), 0, "no_training_rows")
    features = []
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for column in matrix.T:
                constant = float(column[0]) if np.all(column == column[0]) else None
                edges = (
                    ()
                    if constant is not None
                    else tuple(
                        float(value)
                        for value in np.unique(
                            np.quantile(column, np.arange(1, 10) / 10, method="linear")
                        )
                    )
                )
                counts = _bin_counts(column, edges, constant)
                features.append(
                    PSIFeatureReference(
                        edges,
                        constant,
                        tuple(int(value) for value in counts),
                        tuple(
                            float(value) for value in _proportions(counts, len(matrix))
                        ),
                    )
                )
    except FloatingPointError as exc:
        raise SecondaryDriftError("PSI training quantiles are nonfinite") from exc
    return PSIReference(tuple(features), len(matrix))


def psi_window_score(reference: PSIReference, window_standardized) -> PSIWindowScore:
    """Retain all 26 PSI scores and their maximum, using frozen training bins."""
    window = _window(window_standardized)
    if not isinstance(reference, PSIReference):
        raise SecondaryDriftError("reference must be a PSIReference")
    if reference.reason is not None:
        return PSIWindowScore(None, (), reference.reason)
    if len(reference.features) != FEATURE_COUNT:
        raise SecondaryDriftError("PSI reference must contain exactly 26 features")
    scores = []
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for column, feature in zip(window.T, reference.features, strict=True):
                p = np.asarray(feature.training_proportions, dtype=np.float64)
                counts = _bin_counts(column, feature.internal_edges, feature.constant)
                q = _proportions(counts, len(window))
                score = float(np.sum((q - p) * np.log(q / p), dtype=np.float64))
                if not np.isfinite(score):
                    raise SecondaryDriftError("PSI score is nonfinite")
                scores.append(score)
    except FloatingPointError as exc:
        raise SecondaryDriftError("PSI score computation is nonfinite") from exc
    return PSIWindowScore(max(scores), tuple(scores))


def _window_scores(reference, stream_standardized, score_window):
    matrix = _matrix(stream_standardized)
    ends = tuple(range(WINDOW_SIZE, len(matrix) + 1, WINDOW_STRIDE))
    if not ends:
        return DriftWindowScores((), (), reason="no_complete_256_row_window")
    results = tuple(
        score_window(reference, matrix[end - WINDOW_SIZE : end]) for end in ends
    )
    if results[0].reason is not None:
        return DriftWindowScores(ends, (), reason=results[0].reason)
    feature_scores = (
        tuple(result.feature_scores for result in results)
        if isinstance(results[0], PSIWindowScore)
        else ()
    )
    return DriftWindowScores(
        ends, tuple(result.value for result in results), feature_scores
    )


def mmd_window_scores(
    reference: MMDReference, stream_standardized
) -> DriftWindowScores:
    """Score complete original-order 256-row windows at stride 64."""
    if not isinstance(reference, MMDReference):
        raise SecondaryDriftError("reference must be an MMDReference")
    return _window_scores(reference, stream_standardized, mmd_squared)


def psi_window_scores(
    reference: PSIReference, stream_standardized
) -> DriftWindowScores:
    """Score complete windows and retain feature scores in the original order."""
    if not isinstance(reference, PSIReference):
        raise SecondaryDriftError("reference must be a PSIReference")
    return _window_scores(reference, stream_standardized, psi_window_score)


def _scores(value):
    scores = _numeric_array(value, "window scores")
    if scores.ndim != 1:
        raise SecondaryDriftError("window scores must be a finite vector")
    return scores


def calibrate_window_scores(calibration_scores) -> DriftCalibration:
    """Freeze the existing calibration stream's linear 95th-percentile boundary."""
    scores = _scores(calibration_scores)
    if not len(scores):
        return DriftCalibration(None, 0, "no_calibration_windows")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            threshold = float(np.quantile(scores, 0.95, method="linear"))
    except FloatingPointError as exc:
        raise SecondaryDriftError("calibration threshold is nonfinite") from exc
    return DriftCalibration(threshold, len(scores))


def audit_window_scores(window_scores, calibration: DriftCalibration) -> DriftAudit:
    """Apply one unchanged strict boundary; report no primary H2 gate result."""
    scores = _scores(window_scores)
    if not isinstance(calibration, DriftCalibration):
        raise SecondaryDriftError("calibration must be a DriftCalibration")
    reason = calibration.reason or ("no_audit_windows" if not len(scores) else None)
    if reason is not None:
        return DriftAudit(
            calibration.threshold,
            calibration.calibration_window_count,
            len(scores),
            (),
            None,
            None,
            reason,
        )
    threshold = _numeric_array(calibration.threshold, "calibration threshold")
    if threshold.ndim != 0:
        raise SecondaryDriftError("calibration threshold must be a finite scalar")
    alerts = tuple(bool(value) for value in scores > threshold)
    count = sum(alerts)
    return DriftAudit(
        float(threshold),
        calibration.calibration_window_count,
        len(scores),
        alerts,
        count,
        count / len(scores),
    )
