"""Rejoin immutable source science to original metadata before exact selection."""

from . import _internal_handoff_validation as internal
from . import _operational_cell_records as cell_records
from . import _operational_input_schema as schema
from . import _study_history_snapshot_records as records
from ._checkpoint_codec import canonical_bytes
from ._external_completion_records import _LOGICAL_NAMES
from .evaluation_producer import ManifestOutcome
from .external_replay_codec import decode_external_replay_manifest
from .external_source_handoff import VerifiedExternalSnapshot
from .internal_source_handoff import VerifiedInternalSnapshot
from .operational_cell_inputs import restore_cell_inputs
from .operational_inputs import _primary
from .replay_manifest_codec import encode_replay_manifest


def _source(snapshot, names, expected):
    values = records.authenticate(
        snapshot.payloads,
        names,
        expected["snapshot_sha256"],
        expected["snapshot_sha256"],
    )
    public = schema.loads(values["public-summary.json"], canonical=False)
    schema.same(public["execution"], expected["execution"])
    return values


def sources(first, second, metadata):
    schema.require(type(first) is VerifiedInternalSnapshot)
    schema.require(type(second) is VerifiedExternalSnapshot)
    original = _source(first, internal.SNAPSHOT_NAMES, metadata["internal"])
    external = _source(second, _LOGICAL_NAMES, metadata["external"])
    primary = _primary(
        original["attempt/evidence/bindings.json"],
        external["attempt/evidence/bindings.json"],
    )
    schema.same(primary, metadata["primary"])
    schema.require(type(second.profile_bytes) is bytes)
    schema.require(
        records.digest(second.profile_bytes)
        == metadata["external"]["execution"]["source_profile_sha256"]
    )
    schema.same(schema.loads(second.profile_bytes)["execution"], metadata["execution"])
    for directory in ("checkpoints", "evidence"):
        for name, content in (
            ("internal-source-handoff.json", canonical_bytes(metadata["internal"])),
            ("internal-source-overlap.json", original[internal.OVERLAP_NAME]),
        ):
            schema.require(external[f"attempt/{directory}/{name}"] == content)


def _internal_manifest(snapshot, cell):
    schema.require(type(snapshot.manifest_outcomes) is tuple)
    outcomes = {}
    for member in snapshot.manifest_outcomes:
        schema.require(type(member) is tuple and len(member) == 2)
        prevalence, outcome = member
        schema.require(type(prevalence) is int and prevalence not in outcomes)
        outcomes[prevalence] = outcome
    schema.keys(outcomes, {10, 100, 500})
    outcome = outcomes[cell.prevalence_basis_points]
    schema.require(type(outcome) is ManifestOutcome and outcome.status == "prepared")
    schema.require(
        outcome.insufficient_label is outcome.required is outcome.available is None
    )
    content = encode_replay_manifest(outcome.manifest)
    schema.require(
        outcome.manifest.prevalence_basis_points == cell.prevalence_basis_points
    )
    return content, outcome.manifest.records


def _external_manifest(snapshot):
    schema.require(type(snapshot.rows) is tuple)
    content = snapshot.payload("attempt/evidence/retained-test.jsonl")
    rows = decode_external_replay_manifest(
        content, expected_sha256=records.digest(content)
    )
    schema.require(tuple(row.record for row in snapshot.rows) == rows)
    return content, rows


def restore(accepted, first, second, descriptor, binding, binding_pin, reservation):
    cell, _, metadata = cell_records.authenticate(
        accepted,
        descriptor,
        binding,
        expected_binding_sha256=binding_pin,
        expected_cell_reservation_sha256=reservation,
    )
    sources(first, second, metadata)
    content, rows = (
        _external_manifest(second)
        if cell.workload == "shift_period"
        else _internal_manifest(first, cell)
    )
    schema.require(
        descriptor
        == cell_records.encode_descriptor(accepted, metadata, cell, content, rows)
    )
    return restore_cell_inputs(
        accepted,
        descriptor,
        binding,
        content,
        expected_binding_sha256=binding_pin,
        expected_cell_reservation_sha256=reservation,
    )
