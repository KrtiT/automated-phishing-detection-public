"""Verify the entire original prefix without opening models or original data."""

import json
from pathlib import Path

from . import _study_execution_schema as schema
from ._adopted_study_records import decoded
from ._study_history_snapshot_cells import cell_snapshot
from ._study_history_snapshot_records import digest
from ._study_series_history_context import payloads, pins
from ._study_series_index_refs import cell_locations
from .study_history_cell import verify_historical_cell_science


def _inputs(reader, origin, entry, locations):
    descriptor, binding = (
        decoded(entry[name]) for name in ("descriptor_bytes", "binding_bytes")
    )
    for name, content in (("descriptor.json", descriptor), ("binding.json", binding)):
        schema.require(
            reader.read(locations.cell_input_directory / name, digest(content))
            == content
        )
    reader.read(
        locations.cell_input_directory / "manifest",
        json.loads(descriptor)["manifest_sha256"],
    )
    return dict(
        accepted_metadata_bytes=origin.metadata_bytes,
        descriptor_bytes=descriptor,
        binding_bytes=binding,
        expected_descriptor_sha256=digest(descriptor),
        expected_binding_sha256=digest(binding),
    )


def _cell(reader, selected, origin, sources):
    ordinal = selected["ordinal"]
    projection = origin.scientific["cells"][ordinal - 1]
    acceptance = origin.accounting["authorization_ledger"]["cell_acceptances"][
        ordinal - 1
    ]
    values = tuple(payloads(reader, selected["payloads"]).items())
    expected = pins(selected["payloads"])
    reservation = selected["reservation_sha256"]
    cell_snapshot(
        (ordinal, reservation, values),
        expected,
        acceptance,
        projection,
        origin.authority,
    )
    locations = cell_locations(origin.profile["paths"], ordinal)
    return verify_historical_cell_science(
        values,
        expected_snapshot_sha256=expected,
        **_inputs(reader, origin, acceptance, locations),
        internal_snapshot=sources[0],
        external_snapshot=sources[1],
        expected_cell_reservation_sha256=reservation,
        expected_attempt_directory=str(locations.attempt),
    )


def restore_cells(reader, value, origin, sources):
    entries = value["accepted_cells"]
    schema.require(
        tuple(entry["ordinal"] for entry in entries)
        == origin.authority.accepted_ordinals
    )
    path = Path(origin.profile["paths"]["accepted-inputs-dir"]) / "accepted-inputs.json"
    schema.require(
        reader.read(path, digest(origin.metadata_bytes)) == origin.metadata_bytes
    )
    return tuple(_cell(reader, entry, origin, sources) for entry in entries)
