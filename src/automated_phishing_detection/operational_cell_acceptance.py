"""Same-parent operational evidence consistency, never portable process authority."""

from dataclasses import asdict
from hashlib import sha256

from . import _operational_cell_publication_records as publication
from . import _operational_cell_records as cell_records
from . import _operational_input_schema as schema
from . import execution_receipt as receipt
from ._checkpoint_codec import canonical_bytes
from ._operational_cell_process_records import verify_process_records
from ._operational_cell_protocol import PROTOCOL, SNAPSHOT_NAMES, WORKING_NAMES
from ._operational_cell_results import (
    VerifiedOperationalCell,
    VerifiedOperationalWorking,
)
from ._operational_cell_run_records import verify_run
from .operational_cell_inputs import (
    RestoredOperationalCell,
    bind_cell_descriptor,
    build_cell_descriptor,
)
from .operational_inputs import AcceptedOperationalInputs

__all__ = [
    "OperationalCellAcceptanceError",
    "VerifiedOperationalCell",
    "VerifiedOperationalWorking",
    "cell_identity",
    "verify_working_cell",
    "build_cell_public",
    "verify_published_cell",
]


class OperationalCellAcceptanceError(ValueError):
    """Symbolic rejection without private source rows or diagnostics."""


def cell_identity(accepted, descriptor_bytes) -> dict:
    """Bind the descriptor before reserving, without a future binding-hash cycle."""
    try:
        schema.require(type(accepted) is AcceptedOperationalInputs)
        described = schema.loads(descriptor_bytes)
        cell_records.descriptor(described)
        metadata = schema.authenticated(
            accepted.metadata_bytes, described["accepted_inputs_sha256"]
        )
        schema.validate_metadata(metadata)
        schema.require(
            described["root_reservation_sha256"] == metadata["root_reservation_sha256"]
        )
        return {
            "kind": "operational_cell",
            "protocol": PROTOCOL,
            **metadata["execution"],
            "operational_profile_sha256": metadata["operational_profile_sha256"],
            "root_reservation_sha256": metadata["root_reservation_sha256"],
            "descriptor_sha256": sha256(descriptor_bytes).hexdigest(),
        }
    except Exception:
        raise OperationalCellAcceptanceError("invalid_operational_cell") from None


def _inputs(inputs, accepted, reservation):
    schema.require(type(inputs) is RestoredOperationalCell)
    schema.require(inputs.accepted_bytes == accepted.metadata_bytes)
    selected = build_cell_descriptor(accepted, inputs.cell)
    schema.require(selected.descriptor_bytes == inputs.descriptor_bytes)
    schema.require(selected.manifest_bytes == inputs.manifest_bytes)
    schema.require(
        type(inputs.requests) is tuple and selected.requests == inputs.requests
    )
    expected = bind_cell_descriptor(
        inputs.descriptor_bytes, cell_reservation_sha256=reservation
    )
    schema.require(inputs.binding_bytes == expected)


def _working(payloads, inputs, accepted, reservation):
    _inputs(inputs, accepted, reservation)
    summary = verify_run(dict(payloads), inputs, accepted)
    return VerifiedOperationalWorking(
        payloads, inputs, accepted, canonical_bytes(summary), reservation
    )


def verify_working_cell(
    payloads,
    *,
    attempt,
    expected_identity,
    inputs,
    accepted,
    observation,
    service_command,
    client_command,
    expected_deadlines,
) -> VerifiedOperationalWorking:
    """Join complete evidence to independently retained parent expectations."""
    try:
        values = publication.inventory(payloads, WORKING_NAMES)
        identity = cell_identity(accepted, inputs.descriptor_bytes)
        schema.same(identity, expected_identity)
        publication.verify_reservation(values["reservation.json"], attempt, identity)
        verify_process_records(
            values,
            inputs=inputs,
            observation=observation,
            reservation=attempt.reservation_sha256,
            service_command=service_command,
            client_command=client_command,
            expected_deadlines=expected_deadlines,
        )
        return _working(payloads, inputs, accepted, attempt.reservation_sha256)
    except Exception:
        raise OperationalCellAcceptanceError("invalid_operational_cell") from None


def build_cell_public(working, *, reservation_sha256) -> dict:
    """Project aggregates only; publication neither authorizes nor accepts a study."""
    try:
        schema.require(type(working) is VerifiedOperationalWorking)
        schema.digest(reservation_sha256)
        schema.require(reservation_sha256 == working.reservation_sha256)
        identity = cell_identity(working.accepted, working.inputs.descriptor_bytes)
        return {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "operational_evidence_published",
            "execution": identity | {"reservation_sha256": reservation_sha256},
            "cell": asdict(working.inputs.cell),
            "summary": working.summary,
            "private_sha256": publication.private_hashes(working),
        }
    except Exception:
        raise OperationalCellAcceptanceError("invalid_operational_cell") from None


def verify_published_cell(
    payloads, *, working, expected_public_bytes
) -> VerifiedOperationalCell:
    """Join exact saved copies to the same parent's already checked working bytes."""
    try:
        schema.require(type(working) is VerifiedOperationalWorking)
        schema.require(type(expected_public_bytes) is bytes)
        public = build_cell_public(
            working, reservation_sha256=working.reservation_sha256
        )
        schema.require(
            expected_public_bytes == receipt._json_bytes(public, "public_summary")
        )
        values = publication.inventory(payloads, SNAPSHOT_NAMES)
        publication.verify_completion(values, working, expected_public_bytes)
        return VerifiedOperationalCell(payloads, working)
    except Exception:
        raise OperationalCellAcceptanceError("invalid_operational_cell") from None
