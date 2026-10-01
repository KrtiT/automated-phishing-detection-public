"""Reuse original manifest and descriptor kernels under truthful series metadata."""

from . import _operational_cell_records as records
from . import _operational_input_schema as schema
from . import _study_history_cell_inputs as history
from .http_replay import ReplayRequest
from .operational_cell_inputs import CellDescriptorPayloads, RestoredOperationalCell
from .operational_schedule import cell_for_ordinal


def view(metadata):
    return {
        "root_reservation_sha256": metadata["root_reservation_sha256"],
        "internal": metadata["origin"]["internal"],
        "external": metadata["origin"]["external"],
    }


def selection(internal, external, cell):
    return (
        history._external_manifest(external)
        if cell.workload == "shift_period"
        else history._internal_manifest(internal, cell)
    )


def _cell(profile, ordinal):
    cell = cell_for_ordinal(ordinal)
    schema.require(
        profile["segment"]["start_ordinal"]
        <= ordinal
        <= profile["segment"]["end_ordinal"]
    )
    return cell


def descriptor(metadata_bytes, metadata, profile, internal, external, ordinal):
    cell = _cell(profile, ordinal)
    content, rows = selection(internal, external, cell)
    described = records.encode_descriptor(
        metadata_bytes, view(metadata), cell, content, rows
    )
    return CellDescriptorPayloads(
        described,
        content,
        tuple(ReplayRequest(row.record_id, row.raw_url) for row in rows),
    )


def _bound_cell(descriptor_bytes, binding_bytes, binding_pin, reservation):
    schema.digest(reservation)
    bound = schema.authenticated(binding_bytes, binding_pin)
    records.binding(bound)
    schema.require(bound["cell_reservation_sha256"] == reservation)
    described = schema.authenticated(descriptor_bytes, bound["descriptor_sha256"])
    return records.descriptor(described)


def restore(
    metadata_bytes,
    metadata,
    profile,
    internal,
    external,
    descriptor_bytes,
    binding_bytes,
    manifest_bytes,
    binding_pin,
    reservation,
):
    cell = _bound_cell(descriptor_bytes, binding_bytes, binding_pin, reservation)
    expected = descriptor(
        metadata_bytes, metadata, profile, internal, external, cell.ordinal
    )
    schema.require(
        type(manifest_bytes) is bytes and manifest_bytes == expected.manifest_bytes
    )
    schema.require(descriptor_bytes == expected.descriptor_bytes)
    return RestoredOperationalCell(
        metadata_bytes,
        descriptor_bytes,
        binding_bytes,
        manifest_bytes,
        cell,
        expected.requests,
    )
