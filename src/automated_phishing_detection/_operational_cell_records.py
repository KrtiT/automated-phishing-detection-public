"""Exact acyclic cell schemas; command and output hashes are deliberately absent."""

from dataclasses import asdict
from hashlib import sha256

from . import _operational_input_schema as schema
from ._checkpoint_codec import canonical_bytes
from .external_replay_codec import decode_external_replay_manifest
from .operational_schedule import OperationalCell, validate_cell
from .replay_manifest_codec import decode_replay_manifest


def descriptor(value):
    schema.keys(
        value,
        {
            "schema_version",
            "kind",
            "root_reservation_sha256",
            "cell",
            "manifest_sha256",
            "accepted_inputs_sha256",
        },
    )
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(value["kind"] == "operational-cell-descriptor-v1")
    for name in (
        "root_reservation_sha256",
        "manifest_sha256",
        "accepted_inputs_sha256",
    ):
        schema.digest(value[name])
    schema.keys(value["cell"], OperationalCell.__dataclass_fields__)
    return validate_cell(OperationalCell(**value["cell"]))


def binding(value):
    schema.keys(
        value,
        {"schema_version", "kind", "descriptor_sha256", "cell_reservation_sha256"},
    )
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(value["kind"] == "operational-cell-binding-v1")
    schema.digest(value["descriptor_sha256"])
    schema.digest(value["cell_reservation_sha256"])


def authenticate(
    accepted_bytes,
    descriptor_bytes,
    binding_bytes,
    *,
    expected_binding_sha256,
    expected_cell_reservation_sha256,
):
    schema.digest(expected_cell_reservation_sha256)
    bound = schema.authenticated(binding_bytes, expected_binding_sha256)
    binding(bound)
    schema.require(bound["cell_reservation_sha256"] == expected_cell_reservation_sha256)
    described = schema.authenticated(descriptor_bytes, bound["descriptor_sha256"])
    cell = descriptor(described)
    accepted = schema.authenticated(accepted_bytes, described["accepted_inputs_sha256"])
    schema.validate_metadata(accepted)
    schema.require(
        described["root_reservation_sha256"] == accepted["root_reservation_sha256"]
    )
    return cell, described, accepted


def _manifest_identity(cell, described, accepted):
    if cell.workload == "shift_period":
        expected = accepted["external"]["snapshot_sha256"][
            "attempt/evidence/retained-test.jsonl"
        ]
        schema.require(described["manifest_sha256"] == expected)


def _source_identity(cell, rows, accepted):
    if cell.workload != "shift_period":
        source = accepted["internal"]["execution"]["source_csv_sha256"]
        schema.require(all(row.record_id.split(":")[1] == source for row in rows))


def encode_descriptor(accepted_bytes, metadata, cell, content, rows):
    value = {
        "schema_version": 1,
        "kind": "operational-cell-descriptor-v1",
        "root_reservation_sha256": metadata["root_reservation_sha256"],
        "cell": asdict(cell),
        "manifest_sha256": sha256(content).hexdigest(),
        "accepted_inputs_sha256": sha256(accepted_bytes).hexdigest(),
    }
    _manifest_identity(cell, value, metadata)
    _source_identity(cell, rows, metadata)
    return canonical_bytes(value)


def restored_rows(content, cell, described, accepted):
    options = {"expected_sha256": described["manifest_sha256"]}
    _manifest_identity(cell, described, accepted)
    if cell.workload == "shift_period":
        return decode_external_replay_manifest(content, **options)
    manifest = decode_replay_manifest(
        content,
        expected_prevalence_basis_points=cell.prevalence_basis_points,
        **options,
    )
    _source_identity(cell, manifest.records, accepted)
    return manifest.records
