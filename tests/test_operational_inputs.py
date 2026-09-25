"""Pure same-parent input projection establishes consistency, never authority."""

import importlib
import importlib.util
import json
from dataclasses import FrozenInstanceError, replace

import pytest
from operational_input_fixtures import (
    build,
    candidates,
    case,
    change_payload,
    digest,
    manifests,
)

__all__ = ["candidates", "manifests", "case"]


def module():
    name = "automated_phishing_detection.operational_inputs"
    assert importlib.util.find_spec(name), "missing accepted operational inputs"
    return importlib.import_module(name)


def test_accepted_input_api_exists():
    api = module()
    assert callable(api.build_accepted_inputs)
    assert api.AcceptedOperationalInputs.__dataclass_params__.frozen


def test_projects_exact_observed_metadata_without_copying_source_objects(case):
    accepted = build(module(), case)
    value = json.loads(accepted.metadata_bytes)
    assert set(value) == {
        "schema_version",
        "kind",
        "root_reservation_sha256",
        "execution",
        "operational_profile_sha256",
        "primary",
        "internal",
        "external",
    }
    assert len(value["internal"]["snapshot_sha256"]) == 35
    assert len(value["external"]["snapshot_sha256"]) == 76
    assert accepted.internal is case.internal and accepted.external is case.external
    assert value["external"]["execution"]["source_profile_sha256"] == digest(
        case.external.snapshot.profile_bytes
    )
    assert len(value["primary"]["artifact_hashes"]) == 7
    assert len(value["primary"]["thresholds"]) == 5
    with pytest.raises(FrozenInstanceError):
        accepted.metadata_bytes = b"changed"
    assert "internal=" not in repr(accepted)


@pytest.mark.parametrize("side", ["internal", "external"])
@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "mutable"])
def test_rejects_inexact_snapshot_inventory(case, side, change):
    observed = getattr(case, side)
    original = observed.snapshot.payloads
    payloads = {
        "missing": original[:-1],
        "extra": (*original, ("unexpected", b"value")),
        "duplicate": (*original, original[0]),
        "mutable": list(original),
    }[change]
    changed = replace(observed, snapshot=replace(observed.snapshot, payloads=payloads))
    arguments = (
        (changed, case.external) if side == "internal" else (case.internal, changed)
    )
    with pytest.raises(ValueError, match="invalid_operational_inputs"):
        module().build_accepted_inputs(
            *arguments,
            binding=case.binding,
            root_reservation_sha256=case.reservation,
            operational_profile_sha256=case.profile,
        )


@pytest.mark.parametrize("field", ["artifact_hashes", "thresholds"])
def test_rejects_saved_model_identity_difference(case, field):
    def mutate(value):
        name = next(iter(value[field]))
        value[field][name] = "0" * 64 if field == "artifact_hashes" else True
        return value

    changed = change_payload(case.external, "attempt/evidence/bindings.json", mutate)
    with pytest.raises(ValueError, match="invalid_operational_inputs"):
        module().build_accepted_inputs(
            case.internal,
            changed,
            binding=case.binding,
            root_reservation_sha256=case.reservation,
            operational_profile_sha256=case.profile,
        )
