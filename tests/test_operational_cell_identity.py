"""Cell reservation identity precedes its binding and command digests."""

import importlib
import importlib.util
import json
from hashlib import sha256

import pytest
from operational_input_fixtures import build, candidates, case, manifests

from automated_phishing_detection import operational_inputs
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_cell_protocol import PROTOCOL
from automated_phishing_detection.operational_cell_inputs import build_cell_descriptor
from automated_phishing_detection.operational_schedule import planned_cells

__all__ = ["candidates", "case", "manifests"]


def api():
    name = "automated_phishing_detection.operational_cell_acceptance"
    assert importlib.util.find_spec(name), "missing operational cell acceptance"
    return importlib.import_module(name)


def test_identity_binds_accepted_execution_without_future_binding(case):
    accepted = build(operational_inputs, case)
    descriptor = build_cell_descriptor(accepted, planned_cells()[20]).descriptor_bytes
    metadata = json.loads(accepted.metadata_bytes)
    expected = {
        "kind": "operational_cell",
        "protocol": PROTOCOL,
        **metadata["execution"],
        "operational_profile_sha256": case.profile,
        "root_reservation_sha256": case.reservation,
        "descriptor_sha256": sha256(descriptor).hexdigest(),
    }
    assert api().cell_identity(accepted, descriptor) == expected


@pytest.mark.parametrize("field", ["accepted_inputs_sha256", "root_reservation_sha256"])
def test_identity_rejects_descriptor_from_another_parent(case, field):
    accepted = build(operational_inputs, case)
    descriptor = build_cell_descriptor(accepted, planned_cells()[20]).descriptor_bytes
    value = json.loads(descriptor)
    value[field] = "0" * 64
    with pytest.raises(api().OperationalCellAcceptanceError):
        api().cell_identity(accepted, canonical_bytes(value))


def test_identity_rejects_noncanonical_descriptor(case):
    accepted = build(operational_inputs, case)
    descriptor = build_cell_descriptor(accepted, planned_cells()[20]).descriptor_bytes
    with pytest.raises(api().OperationalCellAcceptanceError):
        api().cell_identity(accepted, descriptor + b"\n")
