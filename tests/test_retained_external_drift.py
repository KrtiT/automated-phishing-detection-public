"""Saved external drift restoration authenticates exact public bytes before parsing."""

import builtins
import importlib.util
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from retained_external_drift_fixtures import chain as chain
from retained_external_drift_fixtures import restorer as restorer

from automated_phishing_detection import bound_drift, secondary_development
from automated_phishing_detection.retained_drift import load_retained_drift_reference


def test_byte_only_restorer_exists() -> None:
    assert importlib.util.find_spec(
        "automated_phishing_detection.retained_external_drift"
    )


def test_restores_real_retained_loader_state_and_exact_public_bytes(
    restorer: ModuleType, chain: SimpleNamespace
) -> None:
    expected = load_retained_drift_reference(**chain.data)
    result = restorer.restore_external_drift(*chain.arguments)
    assert type(result) is bound_drift.BoundDrift
    assert result.reference == expected
    assert result.training_reference_bytes is chain.arguments[0]
    assert result.validation_audit_bytes is chain.arguments[1]
    assert result.public_inputs == chain.public
    with pytest.raises(FrozenInstanceError):
        result.public_inputs = ()
    assert "train-" not in repr(result)


def test_bound_drift_three_argument_construction_remains_compatible(
    chain: SimpleNamespace,
) -> None:
    reference = load_retained_drift_reference(**chain.data)
    legacy = bound_drift.BoundDrift(reference, *chain.arguments[:2])
    assert legacy.public_inputs == ()


def test_unaccepted_report_is_rejected_before_any_parse_or_retained_load(
    restorer: ModuleType, chain: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("unaccepted report reached parsing or retained loading")

    monkeypatch.setattr(secondary_development, "_json", forbidden)
    public = ((chain.public[0][0], b"private malformed report"), *chain.public[1:])
    with pytest.raises(bound_drift.BoundDriftError) as caught:
        restorer.restore_external_drift(*chain.arguments[:2], public)
    assert "private" not in str(caught.value)


def test_restoration_performs_no_io_model_access_or_fitting(
    restorer: ModuleType, chain: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automated_phishing_detection import gmm_monitor, secondary_drift

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("retained restoration attempted forbidden work")

    with monkeypatch.context() as guard:
        for target, name in (
            (builtins, "open"),
            (Path, "open"),
            (Path, "read_bytes"),
            (Path, "read_text"),
            (bound_drift, "_read_file_once"),
            (secondary_development, "_accepted_states"),
            (gmm_monitor.GaussianMixture, "fit"),
            (gmm_monitor.StandardScaler, "fit"),
            (secondary_drift, "fit_mmd_reference"),
            (secondary_drift, "fit_psi_reference"),
        ):
            guard.setattr(target, name, forbidden)
        result = restorer.restore_external_drift(*chain.arguments)
    assert (
        result.reference.training_reference_sha256
        == chain.data["expected_reference_sha256"]
    )
