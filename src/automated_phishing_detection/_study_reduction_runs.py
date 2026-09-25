"""Restore retained complete runs against the one accepted source context."""

import json
from dataclasses import asdict

from . import _operational_input_schema as schema
from ._checkpoint_codec import canonical_bytes
from ._operational_cell_run_records import decode_run
from .operational_cell_acceptance import cell_identity
from .operational_cell_inputs import RestoredOperationalCell, build_cell_descriptor


def _inputs(accepted, record, selected):
    descriptor = canonical_bytes(
        json.loads(selected.descriptor_bytes) | {"cell": asdict(record.cell)}
    )
    schema.require(record.descriptor_bytes == descriptor)
    identity = cell_identity(accepted, descriptor)
    reservation = json.loads(record.binding_bytes)["cell_reservation_sha256"]
    schema.require(
        json.loads(record.public_bytes)["execution"]
        == identity | {"reservation_sha256": reservation}
    )
    return RestoredOperationalCell(
        accepted.metadata_bytes,
        descriptor,
        record.binding_bytes,
        selected.manifest_bytes,
        record.cell,
        selected.requests,
    )


def restore_runs(accepted, slots):
    manifests, runs = {}, []
    for slot in slots:
        record = slot.accepted
        key = record.cell.prevalence_basis_points
        if key not in manifests:
            manifests[key] = build_cell_descriptor(accepted, record.cell)
        inputs = _inputs(accepted, record, manifests[key])
        runs.append(decode_run(record.run_bytes, inputs))
    return tuple(runs)


def match_summaries(operational, slots):
    summaries = tuple(
        summary for group in operational["groups"] for summary in group["run_summaries"]
    )
    for slot, summary in zip(slots, summaries, strict=True):
        expected = json.loads(slot.accepted.public_bytes)["summary"]
        actual = {
            name: member for name, member in summary.items() if name != "cell_ordinal"
        }
        schema.require(canonical_bytes(actual) == canonical_bytes(expected))
