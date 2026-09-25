"""The preparation entry never predicts or promotes source capacity to authority."""

import json
from dataclasses import FrozenInstanceError, fields
from hashlib import sha256

import pytest
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)

__all__ = ["inputs", "preparation_api", "preparation_case", "runner"]


def test_paths_have_no_model_or_scoring_parameters(preparation_api):
    assert [
        member.name for member in fields(preparation_api.StudyPreparationPaths)
    ] == ["source_csv", "suffix_rules", "archive", "attempt"]


def test_closed_gate_rejects_before_paths(
    preparation_api, preparation_case, monkeypatch
):
    binding = preparation_case.binding
    monkeypatch.setattr(
        preparation_api, "bind_execution", lambda *args, **kwargs: binding
    )

    def forbidden(*args, **kwargs):
        pytest.fail("closed entry inspected preparation inputs")

    monkeypatch.setattr(preparation_api, "_run_bound_preparation", forbidden)
    with pytest.raises(
        preparation_api.StudyPreparationError, match="pre_access_freeze"
    ):
        preparation_api.run_study_preparation(
            binding.root,
            expected_revision=binding.revision,
            expected_contract_sha256=binding.contract_sha256,
            paths=object(),
        )


def test_complete_preparation_retains_no_scientific_result(
    preparation_api, preparation_case
):
    case = preparation_case
    result = preparation_api._run_bound_preparation(case.binding, case.paths)
    assert len(result.payloads) == 12
    assert len(case.session.primary.scorer.urls) == 0
    assert "enter" not in case.events
    assert set(case.paths.attempt.iterdir()) == {
        case.paths.attempt / name
        for name in ("reservation.json", *dict(result.payloads))
    }
    for name, content in result.payloads:
        assert (case.paths.attempt / name).read_bytes() == content
    receipt = json.loads(result.payload("preparation-complete.json"))
    assert receipt["status"] == "preparation_only"
    assert receipt["protected_evaluation_authorized"] is False
    assert receipt["scoring_authorized"] is False
    assert receipt["input_sha256"] == {
        name: sha256(content).hexdigest() for name, content in result.payloads[:-1]
    }
    assert json.loads(result.payload("feasibility.json"))["shortages"]


def test_snapshot_is_immutable(preparation_api, preparation_case):
    result = preparation_api._run_bound_preparation(
        preparation_case.binding, preparation_case.paths
    )
    assert type(result.payloads) is tuple
    with pytest.raises(FrozenInstanceError):
        result.payloads = ()
    with pytest.raises(KeyError):
        result.payload("not-present")
