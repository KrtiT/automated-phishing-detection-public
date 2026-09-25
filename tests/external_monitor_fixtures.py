"""Invented, caller-bound monitor inputs without protected data or model reads."""

import importlib
from dataclasses import replace
from types import ModuleType

import pytest
from test_retained_drift import _snapshots

from automated_phishing_detection import secondary_drift as drift
from automated_phishing_detection.primary_scores import PrimaryURLScores
from automated_phishing_detection.retained_drift import (
    RetainedDriftReference,
    load_retained_drift_reference,
)
from automated_phishing_detection.selective_inference import InferenceCounts


@pytest.fixture
def monitors() -> ModuleType:
    name = "automated_phishing_detection.external_monitors"
    assert importlib.util.find_spec(name), "missing external monitor composition"
    return importlib.import_module(name)


@pytest.fixture
def reference() -> RetainedDriftReference:
    return load_retained_drift_reference(**_snapshots())


def unavailable(
    reference: RetainedDriftReference, name: str, source: str
) -> tuple[RetainedDriftReference, str]:
    reason = "no_calibration_windows"
    if source == "calibration":
        value = drift.DriftCalibration(None, 0, reason)
        return replace(reference, **{f"{name}_calibration": value}), reason
    if name == "mmd":
        reason = "fewer_than_256_training_domains"
        value = drift.MMDReference((), (), (), None, reason)
    else:
        reason = "no_training_rows"
        value = drift.PSIReference((), 0, reason)
    return replace(reference, **{name: value}), reason


def scores(count: int) -> tuple[PrimaryURLScores, ...]:
    row = PrimaryURLScores(
        features=(0.0,) * 25,
        length_probability=0.125,
        stage1_probability=0.875,
        transformer_probability=0.25,
        cascade_probability=0.875,
        length_decision=0,
        stage1_decision=1,
        transformer_decision=0,
        cascade_decision=1,
        band_selected=False,
        monitor_probability=0.0,
        negative_log_likelihood=2.0,
        length_scoring_audit_json="{}",
        stage1_scoring_audit_json="{}",
        inference_counts=InferenceCounts(1, 1, 1, 0),
    )
    return tuple(replace(row) for _ in range(count))
