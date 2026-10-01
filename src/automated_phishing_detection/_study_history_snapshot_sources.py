"""Validate accepted source receipt structure while keeping science opaque."""

from . import _internal_handoff_validation as internal
from . import _study_history_snapshot_records as records
from . import _study_run_schema as schema
from ._checkpoint_codec import canonical_bytes
from ._external_completion_files import PAYLOAD_NAMES
from ._external_completion_records import _LOGICAL_NAMES
from ._internal_scientific_protocol import (
    SCIENTIFIC_OUTPUT_NAMES,
    SOURCE_CHECKPOINT_NAMES,
)
from .source_completion import _PUBLIC_NAMES

EXTERNAL_PUBLIC = frozenset(
    {
        "schema_version",
        "status",
        "source_binding",
        "protected_evaluation_authorized",
        "execution",
        "source_profile",
        "publisher",
        "composition",
        "checkpoint_sha256",
        "private_sha256",
    }
)


def internal_snapshot(payloads, expected, metadata, profile):
    values = records.authenticate(
        payloads, internal.SNAPSHOT_NAMES, expected, metadata["snapshot_sha256"]
    )
    execution = metadata["execution"]
    records.reservation(values, profile["paths"]["internal-attempt"], execution)
    private = records.outputs(
        values, SCIENTIFIC_OUTPUT_NAMES, "scientific-checkpoints/"
    )
    public = records.public_record(
        values, execution, private, _PUBLIC_NAMES, "internal_evidence_published", 4
    )
    schema.require(public["source_binding"] == "authenticated_public_preparation")
    schema.require(public["protected_evaluation_authorized"] is False)
    checkpoints = {
        name: values[f"attempt/checkpoints/{name}"] for name in SOURCE_CHECKPOINT_NAMES
    }
    schema.same(public["checkpoint_sha256"], records.hashes(checkpoints))
    for name, relative in internal.SOURCE_LINKS.items():
        schema.require(records.digest(values[relative]) == execution[name])
    records.finalization(values, execution, private)
    return values


def external_snapshot(payloads, expected, metadata, original, original_values, profile):
    values = records.authenticate(
        payloads, _LOGICAL_NAMES, expected, metadata["snapshot_sha256"]
    )
    execution = metadata["execution"]
    records.reservation(values, profile["paths"]["external-attempt"], execution)
    private = records.outputs(values, PAYLOAD_NAMES, "checkpoints/")
    public = records.public_record(
        values, execution, private, EXTERNAL_PUBLIC, "external_evidence_published", 1
    )
    _external_public(public, execution, private)
    schema.require(private["internal-source-handoff.json"] == canonical_bytes(original))
    schema.require(
        private["internal-source-overlap.json"]
        == original_values[internal.OVERLAP_NAME]
    )
    records.finalization(values, execution, private)
    return values


def _external_public(public, execution, private):
    schema.require(
        public["source_binding"]
        == "authenticated_retained_preparation_with_parent_declared_internal_handoff"
    )
    schema.require(public["protected_evaluation_authorized"] is False)
    schema.same(public["checkpoint_sha256"], records.hashes(private))
    profile = public["source_profile"]
    schema.require(type(profile) is dict)
    schema.require(
        records.digest(canonical_bytes(profile)) == execution["source_profile_sha256"]
    )
    schema.same(
        profile["execution"], {name: execution[name] for name in schema.EXECUTION}
    )
