"""Pure retained external MMD/PSI composition on invented primary scores."""

import importlib.util
import math
from dataclasses import FrozenInstanceError, replace
from types import ModuleType

import numpy as np
import pytest
from external_monitor_fixtures import monitors as monitors
from external_monitor_fixtures import reference as reference
from external_monitor_fixtures import scores, unavailable

from automated_phishing_detection import secondary_drift as drift
from automated_phishing_detection.retained_drift import (
    RetainedDriftReference,
    load_retained_drift_reference,
)


def test_external_monitor_composition_is_available() -> None:
    assert importlib.util.find_spec("automated_phishing_detection.external_monitors")


@pytest.mark.parametrize("count", [0, 255, 256, 319, 320])
def test_retains_every_complete_window_in_fixed_monitor_order(
    monitors: ModuleType, reference: RetainedDriftReference, count: int
) -> None:
    result = monitors.replay_external_monitors(scores(count), reference)
    assert len(result.standardized_features) == count
    assert [monitor.name for monitor in result.monitors] == ["mmd", "psi"]
    expected_ends = tuple(range(256, count + 1, 64))
    for monitor in result.monitors:
        assert tuple(window.end_position for window in monitor.windows) == expected_ends
        assert tuple(window.start_position for window in monitor.windows) == tuple(
            end - 255 for end in expected_ends
        )
        assert (
            monitor.threshold
            == getattr(reference, f"{monitor.name}_calibration").threshold
        )
        assert monitor.calibration_window_count == 2
        assert monitor.reason == (
            None if expected_ends else "no_complete_256_row_window"
        )


def test_scaling_uses_float64_subtract_then_divide_and_portable_probability(
    monitors: ModuleType, reference: RetainedDriftReference
) -> None:
    features = tuple(float(index) / 7 for index in range(25))
    rows = (replace(scores(1)[0], features=features, monitor_probability=0.25),)
    mean = tuple(float(index) / 11 for index in range(26))
    scale = tuple(float(index + 1) / 3 for index in range(26))
    reference = replace(reference, scaler_mean=mean, scaler_scale=scale)
    result = monitors.replay_external_monitors(rows, reference)
    expected = (np.asarray([(*features, 0.25)], dtype=np.float64) - mean) / scale
    np.testing.assert_array_equal(np.asarray(result.standardized_features), expected)
    assert result.standardized_features[0][-1] != (0.875 - mean[-1]) / scale[-1]
    with pytest.raises(FrozenInstanceError):
        result.monitors = ()


def test_real_retained_mmd_and_all_psi_features_match_known_answers(
    monitors: ModuleType, reference: RetainedDriftReference
) -> None:
    result = monitors.replay_external_monitors(scores(256), reference)
    mmd, psi = (monitor.windows[0] for monitor in result.monitors)
    assert mmd.score == pytest.approx(0.5 * (1.0 - math.exp(-0.5)), abs=1e-15)
    training = np.asarray([0.5, 128.5, 0.5, 128.5]) / 258.0
    current = np.asarray([0.5, 256.5, 0.5, 0.5]) / 258.0
    expected_psi = float(np.sum((current - training) * np.log(current / training)))
    assert psi.feature_scores == pytest.approx((expected_psi,) + (0.0,) * 25)
    assert psi.score == max(psi.feature_scores)
    assert mmd.alert is False
    assert psi.alert is True


def test_windows_follow_original_membership_and_threshold_equality_is_not_alert(
    monitors: ModuleType, reference: RetainedDriftReference
) -> None:
    rows = scores(256) + tuple(
        replace(row, features=(2.0,) + (0.0,) * 24) for row in scores(64)
    )
    baseline = monitors.replay_external_monitors(rows, reference)
    for monitor in baseline.monitors:
        reference = replace(
            reference,
            **{
                f"{monitor.name}_calibration": drift.DriftCalibration(
                    monitor.windows[0].score, 2
                )
            },
        )
    result = monitors.replay_external_monitors(rows, reference)
    kernel_one, kernel_two = math.exp(-0.5), math.exp(-2.0)
    expected_second = (0.5 + 0.5 * kernel_one) + (0.625 + 0.375 * kernel_two)
    expected_second -= 2 * (0.375 + 0.125 * kernel_two + 0.5 * kernel_one)
    assert result.monitors[0].windows[1].score == pytest.approx(expected_second)
    for monitor in result.monitors:
        assert monitor.windows[0].alert is False
        assert monitor.windows[1].alert == (
            monitor.windows[1].score > monitor.threshold
        )


@pytest.mark.parametrize("count", [0, 255, 256, 320])
@pytest.mark.parametrize("name", ["mmd", "psi"])
@pytest.mark.parametrize("source", ["reference", "calibration"])
def test_unavailable_state_retains_reasons_positions_and_null_alerts(
    monitors: ModuleType,
    reference: RetainedDriftReference,
    count: int,
    name: str,
    source: str,
) -> None:
    reference, reason = unavailable(reference, name, source)
    result = monitors.replay_external_monitors(scores(count), reference)
    monitor = next(item for item in result.monitors if item.name == name)
    assert monitor.reason == reason
    assert tuple(window.end_position for window in monitor.windows) == tuple(
        range(256, count + 1, 64)
    )
    assert all(
        window.alert is None and window.reason == reason for window in monitor.windows
    )
    assert all(
        (window.score is None) == (source == "reference") for window in monitor.windows
    )


def test_loader_accepted_unavailable_mmd_is_preserved(monitors: ModuleType) -> None:
    from test_retained_drift import _snapshots

    retained = load_retained_drift_reference(**_snapshots(training_count=254))
    result = monitors.replay_external_monitors(scores(320), retained)
    assert result.monitors[0].reason == "fewer_than_256_training_domains"
    assert tuple(window.score for window in result.monitors[0].windows) == (None, None)
    assert result.monitors[1].reason is None
