"""The retained adapter feeds the existing producer and new reservation receipts."""

import json

import pytest
from prepared_internal_fixtures import forbid_preparation, restored_case
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)
from test_prepared_internal_runner import module

__all__ = ["inputs", "preparation_api", "preparation_case", "runner"]


def test_retained_scoring_has_no_original_paths_or_second_preparation(
    preparation_api, preparation_case, inputs, monkeypatch
):
    api = module()
    assert hasattr(api, "_run_bound_prepared_internal"), "missing retained scoring seam"
    case = restored_case(preparation_api, preparation_case, inputs)
    for filename in (
        case.original.source_csv,
        case.original.suffix_rules,
        preparation_case.paths.archive,
    ):
        filename.unlink()
    forbid_preparation(monkeypatch)
    result = api._run_bound_prepared_internal(
        case.binding, case.paths, case.preparation
    )
    assert result == case.paths.public_summary
    public = json.loads(result.read_bytes())
    assert public["execution"]["source_interface"] == "retained_study_preparation_v1"
    assert public["offline_inference_counts"]["failed_requests"] == 0
    assert len(case.preparation.internal.records) == public["row_count"]


def test_scoring_receipt_is_new_while_preparation_receipt_stays_exact(
    preparation_api, preparation_case, inputs
):
    api = module()
    assert hasattr(api, "_run_bound_prepared_internal"), "missing retained scoring seam"
    case = restored_case(preparation_api, preparation_case, inputs)
    original = case.preparation.payload("source-reconstruction.json")
    api._run_bound_prepared_internal(case.binding, case.paths, case.preparation)
    receipt = json.loads(
        (case.paths.attempt / "checkpoints/source-reconstruction.json").read_bytes()
    )
    assert (
        receipt["execution"]
        == json.loads((case.paths.attempt / "reservation.json").read_bytes())[
            "identity"
        ]
    )
    assert receipt["reservation_sha256"] != case.preparation.reservation_sha256
    assert receipt["reconstruction"] == json.loads(original)["reconstruction"]
    assert (
        case.paths.preparation / "source-reconstruction.json"
    ).read_bytes() == original
    for name in ("group_test.jsonl", "source-overlap.json"):
        assert (
            case.paths.attempt / "checkpoints" / name
        ).read_bytes() == case.preparation.payload(name)


@pytest.mark.parametrize(
    "entry",
    ["run_prepared_internal_evaluation", "run_prepared_internal_process_with_evidence"],
)
def test_candidate_profile_gate_rejects_before_transport(inputs, monkeypatch, entry):
    from automated_phishing_detection.execution_preflight import ExecutionBinding

    api = module()
    binding, unused_paths, unused_session, unused_events = inputs
    monkeypatch.setattr(api, "bind_execution", lambda *args, **kwargs: binding)
    monkeypatch.setattr(
        ExecutionBinding, "protected_evaluation_ready", property(lambda self: True)
    )
    from types import SimpleNamespace

    monkeypatch.setattr(
        api,
        "bound_preparation_context",
        lambda _: (None, None, None, SimpleNamespace(protected_evaluation_ready=False)),
    )
    with pytest.raises(api.SourceExecutionError, match="pre_access_freeze_incomplete"):
        getattr(api, entry)(
            binding.root,
            expected_revision=binding.revision,
            expected_contract_sha256=binding.contract_sha256,
            paths=object(),
            expected_preparation_reservation_sha256="a" * 64,
            expected_preparation_completion_sha256="b" * 64,
        )
