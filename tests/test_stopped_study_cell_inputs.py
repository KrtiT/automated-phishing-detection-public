"""Rehashed supplied input identities still match the original stopped cell."""

import json

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_cell_fixtures import (
    cell_history,
    change_record,
    invented_cell_history,
    refresh_cell,
    verify_cell,
)
from stopped_study_cell_mutation_fixtures import (
    change_input,
    change_manifest,
    rebind_inputs,
)
from study_run_record_fixtures import prepared

from automated_phishing_detection._checkpoint_codec import canonical_bytes

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
        ("kind", "other"),
        ("unknown", True),
        ("root_reservation_sha256", "0" * 64),
        ("accepted_inputs_sha256", "0" * 64),
        ("manifest_sha256", "0" * 64),
    ],
)
def test_rehashed_descriptor_keeps_root_and_source_identity(cell_history, field, value):
    change_input(cell_history, "descriptor.json", **{field: value})
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize(
    "field,value", [("ordinal", 3), ("run_index", 3), ("concurrency", 8)]
)
def test_descriptor_schedule_cannot_change(cell_history, field, value):
    value_cell = json.loads(cell_history.inputs["descriptor.json"])["cell"]
    value_cell[field] = value
    change_input(cell_history, "descriptor.json", cell=value_cell)
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("kind", "other"),
        ("unknown", True),
        ("descriptor_sha256", "0" * 64),
        ("cell_reservation_sha256", "0" * 64),
    ],
)
def test_binding_schema_and_reservation_are_closed(cell_history, field, value):
    binding = json.loads(cell_history.inputs["binding.json"]) | {field: value}
    cell_history.inputs["binding.json"] = canonical_bytes(binding)
    refresh_cell(cell_history)
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize("change", ["short", "source", "extra", "order"])
def test_rehashed_manifest_must_pass_original_codec_and_source_checks(
    cell_history, change
):
    value = json.loads(cell_history.inputs["manifest"])
    if change == "short":
        value["records"].pop()
    elif change == "source":
        fields = value["records"][0]["record_id"].split(":")
        fields[1] = "b" * 64
        value["records"][0]["record_id"] = ":".join(fields)
    elif change == "extra":
        value["unknown"] = True
    else:
        value["records"][0] = value["records"][1]
    change_manifest(cell_history, value)
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize(
    "field,value",
    [
        ("directory", "/invented/other-attempt"),
        ("status", "failed"),
        ("unknown", True),
    ],
)
def test_rehashed_reservation_keeps_original_approved_path(cell_history, field, value):
    change_record(cell_history, "reservation.json", **{field: value})
    rebind_inputs(cell_history)
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize(
    "field",
    [
        "revision",
        "runtime_sha256",
        "source_spec_sha256",
        "root_reservation_sha256",
        "operational_profile_sha256",
    ],
)
def test_rehashed_reservation_preserves_source_execution(cell_history, field):
    reservation = json.loads(cell_history.values["reservation.json"])
    reservation["identity"][field] = "0" * (40 if field == "revision" else 64)
    change_record(cell_history, "reservation.json", **reservation)
    rebind_inputs(cell_history)
    with pytest.raises(ValueError):
        verify_cell(cell_history)
