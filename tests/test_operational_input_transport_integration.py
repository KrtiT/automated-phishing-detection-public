from dataclasses import FrozenInstanceError

import pytest
from operational_transport_integration_fixtures import (
    accepted,
    candidates,
    case,
    expectations,
    manifests,
    payloads,
    retain_inputs,
)

from automated_phishing_detection import operational_cell_inputs as cells
from automated_phishing_detection import operational_input_transport as transport
from automated_phishing_detection.operational_schedule import cell_for_ordinal

__all__ = ["accepted", "candidates", "case", "manifests"]


@pytest.mark.parametrize("ordinal", [1, 31, 61, 91, 121])
def test_actual_pure_restore_over_held_invented_files(tmp_path, accepted, ordinal):
    contents = payloads(accepted, ordinal)
    with retain_inputs(tmp_path, contents) as paths:
        with transport.hold_operational_inputs(
            *paths, **expectations(contents)
        ) as result:
            assert type(result) is cells.RestoredOperationalCell
            assert result.cell == cell_for_ordinal(ordinal)
            assert result.accepted_bytes == contents["accepted-inputs.json"]
            assert result.descriptor_bytes == contents["descriptor.json"]
            assert result.binding_bytes == contents["binding.json"]
            assert result.manifest_bytes == contents["manifest"]
            assert len(result.requests) >= 1000
            original = result.primary
            result.primary.clear()
            assert result.primary == original
            with pytest.raises(FrozenInstanceError):
                result.manifest_bytes = b"mutated"


@pytest.mark.parametrize(
    "name", ["accepted-inputs.json", "descriptor.json", "binding.json", "manifest"]
)
def test_authenticated_physical_files_still_require_pure_links(
    tmp_path, accepted, name
):
    contents = payloads(accepted, 121)
    contents[name] += b" "
    with retain_inputs(tmp_path, contents) as paths:
        with pytest.raises(transport.OperationalInputTransportError):
            with transport.hold_operational_inputs(*paths, **expectations(contents)):
                pytest.fail("broken canonical payload or hash link yielded")


def test_independent_actual_reservation_not_binding_claim(tmp_path, accepted):
    contents = payloads(accepted, 121)
    expected = {**expectations(contents), "expected_cell_reservation_sha256": "9" * 64}
    with retain_inputs(tmp_path, contents) as paths:
        with pytest.raises(transport.OperationalInputTransportError):
            with transport.hold_operational_inputs(*paths, **expected):
                pytest.fail("disk reservation replaced actual parent expectation")


def test_same_root_leaf_remains_held_across_distinct_cell_inputs(tmp_path, accepted):
    first, second = payloads(accepted, 1), payloads(accepted, 121)
    root = tmp_path.resolve() / "accepted-inputs"
    with transport.retain_operational_root_inputs(
        root, accepted_inputs=accepted.metadata_bytes
    ):
        for ordinal, contents in ((1, first), (121, second)):
            cell = tmp_path.resolve() / f"cell-{ordinal:03d}"
            with transport.retain_operational_cell_inputs(
                cell,
                descriptor=contents["descriptor.json"],
                binding=contents["binding.json"],
                manifest=contents["manifest"],
            ):
                with transport.hold_operational_inputs(
                    root, cell, **expectations(contents)
                ) as result:
                    assert result.cell.ordinal == ordinal
    assert set(path.name for path in root.iterdir()) == {"accepted-inputs.json"}
