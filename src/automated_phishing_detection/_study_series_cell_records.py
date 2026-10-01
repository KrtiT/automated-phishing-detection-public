"""Project current identity and rejoin inputs to actual parent expectations."""

from . import _operational_cell_records as cells
from . import _operational_input_schema as schema
from . import _study_series_input_context as context
from ._operational_cell_protocol import PROTOCOL
from ._study_history_snapshot_records import digest
from .operational_cell_inputs import RestoredOperationalCell
from .study_series_inputs import SeriesOperationalCell, restore_series_cell_inputs


def identity(metadata_bytes, descriptor_bytes, metadata_pin):
    metadata = schema.authenticated(metadata_bytes, metadata_pin)
    schema.keys(metadata, context.FIELDS)
    schema.require(
        type(metadata["schema_version"]) is int and metadata["schema_version"] == 1
    )
    schema.require(metadata["kind"] == context.KIND)
    schema.shared_execution(metadata["execution"])
    schema.validate_primary(metadata["primary"])
    schema.validate_metadata(metadata["origin"])
    for name in context.FIELDS:
        if name.endswith("_sha256"):
            schema.digest(metadata[name])
    described = schema.loads(descriptor_bytes)
    cells.descriptor(described)
    schema.require(described["accepted_inputs_sha256"] == metadata_pin)
    schema.require(
        described["root_reservation_sha256"] == metadata["root_reservation_sha256"]
    )
    return {
        "kind": "operational_cell",
        "protocol": PROTOCOL,
        **metadata["execution"],
        "operational_profile_sha256": metadata["operational_profile_sha256"],
        "root_reservation_sha256": metadata["root_reservation_sha256"],
        "descriptor_sha256": digest(descriptor_bytes),
    }


def restore(inputs, reservation, source_context, pins):
    schema.require(type(inputs) is SeriesOperationalCell)
    original = inputs.computational
    schema.require(type(original) is RestoredOperationalCell)
    profile, internal, external = source_context
    restored = restore_series_cell_inputs(
        original.accepted_bytes,
        profile,
        internal,
        external,
        original.descriptor_bytes,
        original.binding_bytes,
        original.manifest_bytes,
        expected_metadata_sha256=pins[0],
        expected_profile_sha256=pins[1],
        expected_binding_sha256=original.binding_sha256,
        expected_cell_reservation_sha256=reservation,
    )
    schema.require(restored == inputs)
    return restored
