"""External drift rejects malformed caller state without exposing its values."""

import builtins
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType

import pytest
from external_monitor_fixtures import monitors as monitors
from external_monitor_fixtures import reference as reference
from external_monitor_fixtures import scores

from automated_phishing_detection import gmm_monitor, probe_replay
from automated_phishing_detection import secondary_drift as drift
from automated_phishing_detection.retained_drift import RetainedDriftReference


@pytest.mark.parametrize("value", [[], (object(),), (asdict(scores(1)[0]),)])
def test_requires_typed_immutable_score_sequence(
    monitors: ModuleType, reference: RetainedDriftReference, value: object
) -> None:
    with pytest.raises(monitors.ExternalMonitorError):
        monitors.replay_external_monitors(value, reference)


@pytest.mark.parametrize("value", [None, {}, object()])
def test_requires_retained_reference_even_for_empty_input(
    monitors: ModuleType, value: object
) -> None:
    with pytest.raises(monitors.ExternalMonitorError):
        monitors.replay_external_monitors((), value)


@pytest.mark.parametrize(
    "features",
    [
        (0.0,) * 24,
        [0.0] * 25,
        (True,) * 25,
        (float("nan"),) * 25,
        (float("inf"),) * 25,
        ("private-url",) * 25,
    ],
)
def test_invalid_structural_features_are_rejected(
    monitors: ModuleType, reference: RetainedDriftReference, features: object
) -> None:
    rows = (replace(scores(1)[0], features=features),)
    with pytest.raises(monitors.ExternalMonitorError):
        monitors.replay_external_monitors(rows, reference)


@pytest.mark.parametrize(
    "value", [True, -0.01, 1.01, float("nan"), float("inf"), "private-value"]
)
def test_invalid_portable_probability_is_rejected(
    monitors: ModuleType, reference: RetainedDriftReference, value: object
) -> None:
    rows = (replace(scores(1)[0], monitor_probability=value),)
    with pytest.raises(monitors.ExternalMonitorError):
        monitors.replay_external_monitors(rows, reference)


@pytest.mark.parametrize("field", ["scaler_mean", "scaler_scale"])
@pytest.mark.parametrize(
    "value",
    [(1.0,) * 25, [1.0] * 26, (True,) * 26, (float("nan"),) * 26, (float("inf"),) * 26],
)
def test_invalid_scaler_is_rejected_even_for_empty_input(
    monitors: ModuleType, reference: RetainedDriftReference, field: str, value: object
) -> None:
    with pytest.raises(monitors.ExternalMonitorError):
        monitors.replay_external_monitors((), replace(reference, **{field: value}))


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_scaler_scale_must_be_positive(
    monitors: ModuleType, reference: RetainedDriftReference, scale: float
) -> None:
    with pytest.raises(monitors.ExternalMonitorError):
        monitors.replay_external_monitors(
            (), replace(reference, scaler_scale=(scale,) * 26)
        )


@pytest.mark.parametrize("kind", ["subtract", "divide", "conversion"])
def test_overflow_is_symbolic_and_suppresses_private_exception_context(
    monitors: ModuleType, reference: RetainedDriftReference, kind: str
) -> None:
    value = 10**1000 if kind == "conversion" else 1e308
    rows = (replace(scores(1)[0], features=(value,) * 25),)
    if kind == "subtract":
        reference = replace(reference, scaler_mean=(-1e308,) * 26)
    if kind == "divide":
        reference = replace(reference, scaler_scale=(1e-308,) * 26)
    with pytest.raises(monitors.ExternalMonitorError) as captured:
        monitors.replay_external_monitors(rows, reference)
    assert str(captured.value) == "invalid_external_monitor_state"
    assert captured.value.__suppress_context__ is True
    assert captured.value.__cause__ is None


@pytest.mark.parametrize("name", ["mmd_calibration", "psi_calibration"])
def test_invalid_calibration_fails_before_short_window_evaluation(
    monitors: ModuleType, reference: RetainedDriftReference, name: str
) -> None:
    invalid = drift.DriftCalibration(float("nan"), 2)
    with pytest.raises(monitors.ExternalMonitorError):
        monitors.replay_external_monitors((), replace(reference, **{name: invalid}))


@pytest.mark.parametrize("name", ["mmd", "psi"])
def test_invalid_reference_fails_before_short_window_evaluation(
    monitors: ModuleType, reference: RetainedDriftReference, name: str
) -> None:
    with pytest.raises(monitors.ExternalMonitorError):
        monitors.replay_external_monitors((), replace(reference, **{name: object()}))


@pytest.mark.parametrize(
    "output",
    [
        drift.DriftWindowScores((255,), (1.0,), ((1.0,) * 26,)),
        drift.DriftWindowScores((256,), (1.0,), ((1.0,) * 25,)),
        drift.DriftWindowScores((256,), (2.0,), ((1.0,) * 26,)),
        drift.DriftWindowScores((256,), (1.0,), ((float("nan"),) * 26,)),
        drift.DriftWindowScores((256,), (1.0,), ((1.0,) * 26,), "unavailable"),
    ],
)
def test_psi_rejects_coverage_nonfinite_maximum_and_unavailable_corruption(
    monitors: ModuleType,
    reference: RetainedDriftReference,
    monkeypatch: pytest.MonkeyPatch,
    output: drift.DriftWindowScores,
) -> None:
    monkeypatch.setattr(drift, "psi_window_scores", lambda *args: output)
    with pytest.raises(monitors.ExternalMonitorError):
        monitors.replay_external_monitors(scores(256), reference)


def test_monitor_errors_hide_arbitrary_private_values(
    monitors: ModuleType,
    reference: RetainedDriftReference,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object) -> None:
        raise RuntimeError("private-source-value")

    monkeypatch.setattr(drift, "mmd_window_scores", fail)
    with pytest.raises(monitors.ExternalMonitorError) as captured:
        monitors.replay_external_monitors(scores(256), reference)
    assert str(captured.value) == "invalid_external_monitor_state"
    assert captured.value.__suppress_context__ is True


def test_no_io_fit_recalibration_transformed_replay_or_mutation(
    monitors: ModuleType,
    reference: RetainedDriftReference,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = scores(256)
    before = asdict(reference), tuple(asdict(row) for row in rows)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("external monitor composition attempted forbidden work")

    with monkeypatch.context() as guard:
        for target, name in (
            (builtins, "open"),
            (Path, "open"),
            (Path, "read_bytes"),
            (gmm_monitor.StandardScaler, "fit"),
            (gmm_monitor.GaussianMixture, "fit"),
            (drift, "fit_mmd_reference"),
            (drift, "fit_psi_reference"),
            (drift, "calibrate_window_scores"),
            (probe_replay, "replay_probes"),
        ):
            guard.setattr(target, name, forbidden)
        result = monitors.replay_external_monitors(rows, reference)
    assert before == (asdict(reference), tuple(asdict(row) for row in rows))
    assert result.monitors[0].windows[0].score is not None
