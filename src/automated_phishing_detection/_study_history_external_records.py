"""Original external byte joins, without IO or observing-parent construction."""

from pathlib import PurePosixPath

from . import _operational_input_schema as schema
from . import _study_history_snapshot_records as records
from ._checkpoint_codec import canonical_bytes
from ._external_completion_files import PAYLOAD_NAMES
from ._external_completion_records import _LOGICAL_NAMES
from ._external_preparation_outputs import _PUBLIC_INPUTS
from ._external_records_validation import _CANDIDATE
from ._study_history_snapshot_sources import EXTERNAL_PUBLIC
from .internal_external_handoff import verify_internal_handoff


def snapshot(payloads, expected):
    schema.keys(payloads, _LOGICAL_NAMES)
    return records.authenticate(
        tuple(sorted(payloads.items())), _LOGICAL_NAMES, expected, expected
    )


def handoff(execution, hashes, handoff_bytes, overlap_bytes):
    verify_internal_handoff(
        handoff_bytes,
        overlap_bytes,
        expected_handoff_sha256=execution["internal_handoff_sha256"],
    )
    original = schema.loads(handoff_bytes)["execution"]
    schema._external_execution(execution, original, hashes)
    schema.shared_execution({name: execution[name] for name in schema.EXECUTION_FIELDS})
    schema.require(
        execution["internal_overlap_sha256"] == records.digest(overlap_bytes)
    )
    schema.require(
        execution["internal_reservation_sha256"] == original["reservation_sha256"]
    )
    return original


def profile(content, execution):
    schema.require(records.digest(content) == execution["source_profile_sha256"])
    value = schema.loads(content)
    schema.same({name: value[name] for name in _CANDIDATE}, _CANDIDATE)
    schema.same(
        value["execution"], {name: execution[name] for name in schema.EXECUTION_FIELDS}
    )
    publisher = value["publisher"]["expected_format"]
    schema.require(publisher["archive_sha256"] == execution["archive_sha256"])
    schema.require(type(publisher["archive_size_bytes"]) is int)
    schema.require(publisher["archive_size_bytes"] == execution["archive_size_bytes"])
    schema.require(
        value["public_suffix_list"]["sha256"] == execution["suffix_rules_sha256"]
    )
    return value


def public_sources(outputs, pins, execution, original):
    schema.keys(pins, {name for name, unused in _PUBLIC_INPUTS})
    for name, retained in _PUBLIC_INPUTS:
        schema.digest(pins[name])
        schema.require(records.digest(outputs[retained]) == pins[name])
    schema.require(pins["data/sources.json"] == execution["source_spec_sha256"])
    schema.require(
        pins["reports/phiusiil-preparation-summary.json"]
        == original["preparation_summary_sha256"]
    )


def publication(values, execution, candidate, attempt_directory):
    schema.require(type(attempt_directory) is str)
    path = PurePosixPath(attempt_directory)
    schema.require(
        path.is_absolute() and str(path) == attempt_directory and ".." not in path.parts
    )
    records.reservation(values, attempt_directory, execution)
    outputs = records.outputs(values, PAYLOAD_NAMES, "checkpoints/")
    public = records.public_record(
        values, execution, outputs, EXTERNAL_PUBLIC, "external_evidence_published", 1
    )
    _public(public, execution, candidate, outputs)
    records.finalization(values, execution, outputs)
    return public, outputs


def _public(public, execution, candidate, outputs):
    prepared = execution["source_interface"] == "retained_study_preparation_v1"
    expected = (
        "authenticated_retained_preparation_with_parent_declared_internal_handoff"
        if prepared
        else "authenticated_publisher_with_parent_declared_internal_handoff"
    )
    schema.require(public["source_binding"] == expected)
    schema.require(public["protected_evaluation_authorized"] is False)
    schema.same(public["checkpoint_sha256"], records.hashes(outputs))
    schema.same(public["source_profile"], candidate)
    schema.require(
        canonical_bytes(public["publisher"]) == outputs["publisher-summary.json"]
    )
