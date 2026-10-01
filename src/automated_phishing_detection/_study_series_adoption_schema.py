"""Closed field sets and pure metadata primitives for the new series route."""

from hashlib import sha256
from pathlib import PurePosixPath

from . import _study_execution_schema as legacy
from ._checkpoint_codec import canonical_bytes

FIELDS = {
    "profile": "schema_version profile_id series_id policy_sha256 amendment_sha256 execution origin components source_artifact_scope transition scientific_pins history paths segment invocation operator",
    "execution": "revision contract_sha256 runtime_sha256",
    "origin": "attempt_ordinal profile revision profile_sha256 envelope_sha256 policy_sha256 method_sha256 root_reservation_sha256 preparation_reservation_sha256 preparation_complete_sha256",
    "components": "original_external original_operational current_external current_operational",
    "transition": "unchanged_sha256 added_sha256",
    "scientific_pins": "source_spec_sha256 primary_metadata_sha256 internal_bindings_sha256 external_bindings_sha256 schedule_sha256 original_h2_audit_sha256",
    "history": "index_path index_sha256 eligible_prefix_review_path eligible_prefix_review_sha256 exposure_record_path exposure_record_sha256",
    "paths": "repo_root series_attempt series_public_summary segment_attempt segment_public_summary historical_inputs_dir cells_dir physical_records_dir",
    "segment": "ordinal session_id start_ordinal end_ordinal predecessor_accounting_sha256 session_requirements operator_commitment",
    "invocation": "script arguments",
    "arguments": "repo-root expected-revision",
    "envelope": "schema_version envelope_id profile operator_directive technical_rebind decisions revoked",
    "directive": "schema_version operator decision decision_text question_text recorded_at reviewed_proposal_text reviewed_proposal_sha256 scope development_directive_sha256 advisor_decision_obtained personally_reviewed_future_code_or_profile_hash_claimed",
    "rebind": "recorded_by recorded_at directive_sha256 profile_sha256 history_index_sha256 implementation_review_sha256 verification_evidence_sha256 scope advisor_decision_obtained",
    "decisions": "amendment historical_access segment_execution",
    "decision": "operator status directive_sha256 subject_kind subject_sha256",
}
EXECUTION_SCOPE = "checkpointed_series_v1_one_segment_execution"


def closed(value, kind):
    legacy.closed(value, FIELDS[kind].split())


def digest(value):
    return sha256(canonical_bytes(value)).hexdigest()


def ordinal(value, lower, upper):
    legacy.require(type(value) is int and lower <= value <= upper)


def pins(value):
    for name, member in value.items():
        if name.endswith("_sha256"):
            legacy.digest(member)


def file_map(value):
    legacy.require(type(value) is dict and value)
    for name, member in value.items():
        legacy.text(name)
        path = PurePosixPath(name)
        legacy.require(not path.is_absolute() and ".." not in path.parts)
        legacy.require(str(path) == name and name != "." and "\\" not in name)
        legacy.digest(member)


def disjoint(first, second):
    legacy.require(
        not first.is_relative_to(second) and not second.is_relative_to(first)
    )


def version(value):
    ordinal(value, 1, 1)
