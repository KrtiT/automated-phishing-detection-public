"""Closed mixed-origin inputs; upstream scientific verification remains required."""

from . import _operational_cell_records as cells
from . import _operational_input_schema as schema
from . import _study_series_input_cells as selection
from . import _study_series_input_context as context
from ._checkpoint_codec import canonical_bytes
from ._operational_cell_protocol import SNAPSHOT_NAMES
from ._operational_cell_publication_records import inventory
from .http_replay import ReplayRequest
from .operational_cell_inputs import RestoredOperationalCell
from .operational_schedule import planned_cells, validate_cell
from .study_history_cell import HistoricalCellScience
from .study_series_cell import SeriesCellScience
from .study_series_inputs import SeriesOperationalCell


def authenticate(buffers, pins, internal, external):
    original = schema.authenticated(buffers[0], pins[0])
    schema.validate_metadata(original)
    current, profile = context.authenticate(
        buffers[1],
        buffers[2],
        internal,
        external,
        expected_metadata_sha256=pins[1],
        expected_profile_sha256=pins[2],
    )
    schema.require(canonical_bytes(current["origin"]) == buffers[0])
    return original, current, profile


def _record(value, expected_type, original_bytes):
    schema.require(type(value) is expected_type)
    inventory(value.payloads, SNAPSHOT_NAMES)
    schema.loads(value.summary_bytes)
    schema.digest(value.reservation_sha256)
    restored = value.inputs
    if expected_type is SeriesCellScience:
        schema.require(type(restored) is SeriesOperationalCell)
        schema.require(
            type(restored.origin_metadata_bytes) is bytes
            and restored.origin_metadata_bytes == original_bytes
        )
        restored = restored.computational
    schema.require(type(restored) is RestoredOperationalCell)
    return restored


def _selected(pool, content, cell, internal, external):
    key = (content, cell.prevalence_basis_points)
    if key not in pool:
        manifest, rows = selection.selection(internal, external, cell)
        requests = tuple(ReplayRequest(row.record_id, row.raw_url) for row in rows)
        pool[key] = manifest, rows, requests
    return pool[key]


def _inputs(restored, cell, selected, identity, reservation):
    content, metadata = identity
    manifest, rows, requests = selected
    schema.require(validate_cell(restored.cell) == cell)
    schema.require(
        type(restored.accepted_bytes) is bytes and restored.accepted_bytes == content
    )
    schema.require(
        type(restored.manifest_bytes) is bytes and restored.manifest_bytes == manifest
    )
    schema.require(type(restored.requests) is tuple and restored.requests == requests)
    schema.require(all(type(request) is ReplayRequest for request in restored.requests))
    descriptor = cells.encode_descriptor(content, metadata, cell, manifest, rows)
    schema.require(
        type(restored.descriptor_bytes) is bytes
        and restored.descriptor_bytes == descriptor
    )
    binding = schema.loads(restored.binding_bytes)
    cells.binding(binding)
    schema.authenticated(descriptor, binding["descriptor_sha256"])
    schema.require(binding["cell_reservation_sha256"] == reservation)


def prepare(historical, fresh, buffers, pins, internal, external):
    schema.require(type(historical) is tuple and type(fresh) is tuple)
    schema.require(len(historical) + len(fresh) == 125)
    original, current, profile = authenticate(buffers, pins, internal, external)
    schema.require(len(historical) == profile["segment"]["start_ordinal"] - 1)
    values, reservations, pool = historical + fresh, set(), {}
    for index, (value, cell) in enumerate(zip(values, planned_cells(), strict=True)):
        old = index < len(historical)
        restored = _record(
            value, HistoricalCellScience if old else SeriesCellScience, buffers[0]
        )
        schema.require(value.reservation_sha256 not in reservations)
        reservations.add(value.reservation_sha256)
        content = buffers[0] if old else buffers[1]
        selected = _selected(pool, content, cell, internal, external)
        identity = content, original if old else selection.view(current)
        _inputs(restored, cell, selected, identity, value.reservation_sha256)
    return values


def match_summaries(operational, values):
    summaries = tuple(
        summary for group in operational["groups"] for summary in group["run_summaries"]
    )
    for value, summary in zip(values, summaries, strict=True):
        actual = {
            name: member for name, member in summary.items() if name != "cell_ordinal"
        }
        schema.require(canonical_bytes(actual) == value.summary_bytes)
