"""Authenticate supplied cell inputs without loading models or granting access."""

from hashlib import sha256
from pathlib import PurePath

from . import _operational_cell_records as cells
from . import _study_run_schema as schema
from . import execution_receipt as receipt
from ._checkpoint_codec import canonical_bytes
from ._operational_cell_protocol import PROTOCOL


def _reservation(attempt, root, inputs, metadata, cell):
    identity = {
        "kind": "operational_cell",
        "protocol": PROTOCOL,
        **metadata["execution"],
        "operational_profile_sha256": metadata["operational_profile_sha256"],
        "root_reservation_sha256": metadata["root_reservation_sha256"],
        "descriptor_sha256": sha256(inputs["descriptor.json"]).hexdigest(),
    }
    profile = schema.load(root.profile_bytes)
    directory = (
        PurePath(profile["paths"]["cells-dir"]) / f"cell-{cell.ordinal:03d}-attempt"
    )
    expected = dict(
        schema_version=1, status="reserved", directory=str(directory), identity=identity
    )
    schema.require(
        attempt["attempt/reservation.json"] == receipt._json_bytes(expected, "cell")
    )


def _finalization(attempt, reservation):
    common = dict(schema_version=1, reservation_sha256=reservation)
    values = {
        "finalize.claim": common | dict(operation="failure"),
        "outcome.json": common
        | dict(status="failed", stage="observation", error_type="cancelled"),
    }
    for name, value in values.items():
        schema.require(attempt[f"attempt/{name}"] == receipt._json_bytes(value, "cell"))


def authenticate_inputs(root, attempt, inputs, frame):
    metadata = schema.load(root.source_results_bytes)["accepted_inputs"]
    reservation = sha256(attempt["attempt/reservation.json"]).hexdigest()
    cell, described, accepted = cells.authenticate(
        canonical_bytes(metadata),
        inputs["descriptor.json"],
        inputs["binding.json"],
        expected_binding_sha256=frame.cell_binding_sha256,
        expected_cell_reservation_sha256=reservation,
    )
    schema.require(cell.ordinal == root.stopped_ordinal)
    cells.restored_rows(inputs["manifest"], cell, described, accepted)
    _reservation(attempt, root, inputs, metadata, cell)
    _finalization(attempt, reservation)
    return cell, metadata, reservation
