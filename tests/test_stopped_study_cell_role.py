"""Service declaration retains exact original model and process bindings."""

import json

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_cell_fixtures import (
    cell_history,
    change_record,
    invented_cell_history,
    verify_cell,
)
from study_run_record_fixtures import prepared

__all__ = [
    "candidates",
    "manifests",
    "prepared",
    "cell_history",
    "invented_cell_history",
]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("protocol", "other"),
        ("role", "client"),
        ("binding_sha256", "0" * 64),
        ("pid", 999),
        ("command_sha256", "0" * 64),
        ("base_url", "http://127.0.0.1:54322"),
        ("workload", "transformer_only"),
        ("unknown", True),
    ],
)
def test_service_role_identity_is_exact(cell_history, field, value):
    change_record(cell_history, "service-role.json", **{field: value})
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize(
    "artifact",
    [
        "cascade.json",
        "gmm.json",
        "length-only.json",
        "logistic-l1.json",
        "transformer-weights.npz",
        "transformer.json",
        "vocabulary.json",
    ],
)
def test_model_identity_cannot_be_substituted(cell_history, artifact):
    value = json.loads(cell_history.values["service-role.json"])
    value["artifact_hashes"][artifact] = "0" * 64
    change_record(cell_history, "service-role.json", **value)
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize(
    "threshold",
    ["length_only", "logistic_l1", "transformer", "half_width", "monitor_boundary"],
)
def test_original_thresholds_remain_exact(cell_history, threshold):
    value = json.loads(cell_history.values["service-role.json"])
    value["thresholds"][threshold] += 0.001
    change_record(cell_history, "service-role.json", **value)
    with pytest.raises(ValueError):
        verify_cell(cell_history)
