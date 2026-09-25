from contextlib import contextmanager
from hashlib import sha256

import pytest
from operational_input_fixtures import build, candidates, case, manifests

from automated_phishing_detection import operational_cell_inputs as cells
from automated_phishing_detection import operational_input_transport as transport
from automated_phishing_detection import operational_inputs as inputs
from automated_phishing_detection.operational_schedule import cell_for_ordinal

__all__ = ["accepted", "candidates", "case", "manifests"]


@pytest.fixture(scope="module")
def accepted(case):
    return build(inputs, case)


def payloads(accepted, ordinal):
    selected = cells.build_cell_descriptor(accepted, cell_for_ordinal(ordinal))
    binding = cells.bind_cell_descriptor(
        selected.descriptor_bytes, cell_reservation_sha256="8" * 64
    )
    return {
        "accepted-inputs.json": accepted.metadata_bytes,
        "descriptor.json": selected.descriptor_bytes,
        "binding.json": binding,
        "manifest": selected.manifest_bytes,
    }


def expectations(contents):
    return {
        "expected_binding_sha256": sha256(contents["binding.json"]).hexdigest(),
        "expected_cell_reservation_sha256": "8" * 64,
    }


@contextmanager
def retain_inputs(tmp_path, contents):
    root, cell = tmp_path.resolve() / "accepted-inputs", tmp_path.resolve() / "cell-001"
    with transport.retain_operational_root_inputs(
        root, accepted_inputs=contents["accepted-inputs.json"]
    ):
        with transport.retain_operational_cell_inputs(
            cell,
            descriptor=contents["descriptor.json"],
            binding=contents["binding.json"],
            manifest=contents["manifest"],
        ):
            yield root, cell
