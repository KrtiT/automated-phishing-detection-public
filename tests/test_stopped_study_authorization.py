"""Invented failed roots prove metadata consistency without live authority."""

import importlib.util
from dataclasses import FrozenInstanceError

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_authorization_fixtures import make_stopped, verify
from study_run_record_fixtures import prepared

from automated_phishing_detection import _adopted_study_cell_evidence as evidence
from automated_phishing_detection import _operational_process_records as process
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["candidates", "manifests", "prepared"]


def test_stopped_authorization_api_exists():
    assert (
        importlib.util.find_spec(
            "automated_phishing_detection.stopped_study_authorization"
        )
        is not None
    )


@pytest.mark.parametrize("prefix", [1, 72, 124])
@pytest.mark.parametrize("stopped_admissions", [0, 1, 2])
def test_complete_prefix_returns_only_immutable_metadata(
    prepared, manifests, prefix, stopped_admissions
):
    case = make_stopped(
        prepared, manifests, prefix=prefix, stopped_admissions=stopped_admissions
    )
    result = verify(case)
    assert result.accepted_ordinals == tuple(range(1, prefix + 1))
    assert result.stopped_ordinal == prefix + 1
    assert type(result.parent_pid) is int and result.parent_pid > 0
    assert result.reservation_sha256 == case.snapshot.reservation_sha256
    assert result.profile_sha256 == case.authorization.profile_sha256
    assert result.envelope_sha256 == case.authorization.envelope_sha256
    assert result.execution_bytes == canonical_bytes(case.execution)
    assert result.profile_bytes == case.authorization.profile_bytes
    assert result.source_results_bytes == case.contents["source-results.json"]
    assert result.accounting_bytes == case.contents["study-accounting.json"]
    assert result.snapshot_sha256 == tuple(sorted(case.pins.items()))
    with pytest.raises(FrozenInstanceError):
        result.parent_pid = 42


def test_historical_bytes_never_construct_live_observations(
    prepared, manifests, monkeypatch
):
    case = make_stopped(prepared, manifests)

    def forbidden(*args, **kwargs):
        pytest.fail("historical bytes cannot create observations")

    monkeypatch.setattr(process, "ProcessObservation", forbidden)
    monkeypatch.setattr(evidence, "ProcessObservation", forbidden)
    monkeypatch.setattr(evidence, "validate", forbidden)
    verify(case)
