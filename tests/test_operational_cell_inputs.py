"""Closed cell restoration binds exact retained bytes without source access."""

import importlib
import importlib.util
from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest
from operational_input_fixtures import build, candidates, case, manifests
from test_operational_inputs import module as inputs_module

__all__ = ["candidates", "manifests", "case"]


def module():
    name = "automated_phishing_detection.operational_cell_inputs"
    assert importlib.util.find_spec(name), "missing operational cell inputs"
    return importlib.import_module(name)


def test_cell_input_records_are_frozen():
    api = module()
    from automated_phishing_detection.operational_schedule import cell_for_ordinal

    value = api.RestoredOperationalCell(b"a", b"d", b"b", b"m", cell_for_ordinal(1), ())
    with pytest.raises(FrozenInstanceError):
        value.manifest_bytes = b"changed"
    assert "accepted_bytes" not in repr(value)
    assert "requests" not in repr(value)


@pytest.mark.parametrize("ordinal", [1, 31, 61, 91, 121])
def test_restore_binds_full_manifest_and_immutable_requests(case, ordinal):
    api = module()
    from automated_phishing_detection.operational_schedule import cell_for_ordinal

    accepted = build(inputs_module(), case)
    cell = cell_for_ordinal(ordinal)
    payloads = api.build_cell_descriptor(accepted, cell)
    binding = api.bind_cell_descriptor(
        payloads.descriptor_bytes, cell_reservation_sha256="8" * 64
    )
    restored = api.restore_cell_inputs(
        accepted.metadata_bytes,
        payloads.descriptor_bytes,
        binding,
        payloads.manifest_bytes,
        expected_binding_sha256=sha256(binding).hexdigest(),
        expected_cell_reservation_sha256="8" * 64,
    )
    assert restored.cell == cell
    assert restored.requests == payloads.requests
    assert type(restored.requests) is tuple
    assert restored.manifest_sha256 == sha256(payloads.manifest_bytes).hexdigest()
    assert restored.binding_sha256 == sha256(binding).hexdigest()
    restored.primary["artifact_hashes"].clear()
    restored.execution.clear()
    assert len(restored.primary["artifact_hashes"]) == 7
    assert len(restored.execution) == 4
    assert restored.operational_profile_sha256 == case.profile
