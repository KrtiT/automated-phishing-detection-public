"""Role declarations close every primary identity field without model I/O."""

import json
from dataclasses import replace

import pytest
from operational_runtime_fixtures import ARTIFACTS, THRESHOLDS, inputs, primary, records
from test_operational_role_records import context

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["records"]


@pytest.mark.parametrize("name", ARTIFACTS + THRESHOLDS)
def test_loaded_primary_must_equal_every_saved_identity(records, name):
    value = primary()
    category = "artifact_hashes" if name in ARTIFACTS else "thresholds"
    value[category][name] = "f" * 64 if name in ARTIFACTS else 0.75
    with pytest.raises(records.OperationalRoleError):
        records.build_service_role(inputs(), context(records), primary=value)


@pytest.mark.parametrize(
    "name,value",
    [
        ("thresholds", {}),
        ("artifact_hashes", {}),
        ("length_only", True),
        ("logistic_l1", "0.5"),
        ("transformer", -0.1),
        ("transformer", 1.1),
        ("half_width", -0.1),
        ("monitor_boundary", float("inf")),
        ("monitor_boundary", float("nan")),
        ("cascade.json", "F" * 64),
        ("gmm.json", "f" * 63),
        ("extra", {}),
    ],
)
def test_even_matching_malformed_primary_is_rejected(records, name, value):
    expected = primary()
    target = (
        expected
        if name not in ARTIFACTS + THRESHOLDS
        else expected["artifact_hashes" if name in ARTIFACTS else "thresholds"]
    )
    target[name] = value
    cell = inputs()
    accepted = json.loads(cell.accepted_bytes) | {"primary": expected}
    cell = replace(cell, accepted_bytes=json.dumps(accepted).encode())
    with pytest.raises(records.OperationalRoleError):
        records.build_service_role(cell, context(records), primary=expected)


@pytest.mark.parametrize("role", ["service", "client"])
def test_verifier_checks_actual_parent_context_not_saved_claims(records, role):
    cell, actual = inputs(), context(records)
    content = (
        records.build_service_role(cell, actual, primary=primary())
        if role == "service"
        else records.build_client_role(cell, actual)
    )
    changed = replace(actual, command=actual.command + ("--different",))
    with pytest.raises(records.OperationalRoleError):
        records.verify_role_record(content, inputs=cell, context=changed, role=role)


def test_primary_views_cannot_rewrite_previously_retained_role(records):
    cell, actual = inputs(), context(records)
    content = records.build_service_role(cell, actual, primary=primary())
    cell.primary["thresholds"]["length_only"] = 0.99
    assert records.build_service_role(cell, actual, primary=primary()) == content
    assert canonical_bytes(json.loads(content)) == content
