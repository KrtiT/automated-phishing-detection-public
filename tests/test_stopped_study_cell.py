"""Invented service-only cancellations confer no execution or scientific authority."""

import importlib.util
from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_cell_fixtures import make_cell, verify_cell
from study_run_record_fixtures import prepared

__all__ = ["candidates", "manifests", "prepared"]


def test_stopped_cell_history_api_exists():
    assert (
        importlib.util.find_spec("automated_phishing_detection.stopped_study_cell")
        is not None
    )


@pytest.mark.parametrize("ordinal", [2, 73, 91, 125])
@pytest.mark.parametrize("readiness", [False, True])
def test_exact_cancelled_service_history_retains_all_bytes(
    prepared, manifests, ordinal, readiness
):
    case = make_cell(prepared, manifests, ordinal=ordinal, readiness=readiness)
    result = verify_cell(case)
    assert result.root_reservation_sha256 == case.root.snapshot.reservation_sha256
    assert (
        result.cell_reservation_sha256
        == sha256(case.values["reservation.json"]).hexdigest()
    )
    assert result.cell_ordinal == ordinal
    assert result.service_pid == 888
    assert result.service_command_sha256 == "a" * 64
    assert result.cell_binding_sha256 == sha256(case.inputs["binding.json"]).hexdigest()
    assert result.root_payloads == case.root.snapshot.payloads
    assert result.attempt_payloads == case.attempt_payloads
    assert result.input_payloads == case.input_payloads
    assert result.root_snapshot_sha256 == tuple(sorted(case.root.pins.items()))
    assert result.attempt_snapshot_sha256 == tuple(sorted(case.attempt_pins.items()))
    assert result.input_snapshot_sha256 == tuple(sorted(case.input_pins.items()))
    with pytest.raises(FrozenInstanceError):
        result.service_pid = 999
