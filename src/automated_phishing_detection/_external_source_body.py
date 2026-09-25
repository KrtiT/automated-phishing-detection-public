"""Reserved source-to-checkpoint composition under one numerical owner."""

from dataclasses import dataclass, field, fields
from hashlib import sha256

from . import execution_receipt, phishvn, protocol_preflight
from ._external_source_checkpoints import ExternalCheckpointWriter
from ._external_source_profile import (
    CandidateExternalProfile,
    resolve_external_source_profile,
)
from ._external_source_records import (
    ExternalRunPaths,
    ExternalSourceExecutionError,
    external_identity,
)
from .bound_drift import DriftArtifactPaths
from .bound_models import ArtifactPaths
from .bound_secondary import SecondaryArtifactPaths
from .execution_preflight import ExecutionBinding
from .external_producer import produce_external_evidence
from .external_source_failure import ExternalSourceFailureState
from .external_source_provenance import PROVENANCE_NAMES, build_external_provenance
from .internal_external_handoff import InternalHandoffPayloads, verify_internal_handoff
from .phishvn_source import decode_phishvn_archive
from .source_runner import _read_file_once


@dataclass(repr=False)
class ExternalRun:
    binding: ExecutionBinding
    paths: ExternalRunPaths
    handoff: InternalHandoffPayloads
    failures: ExternalSourceFailureState = field(
        default_factory=ExternalSourceFailureState
    )
    stage: str = "public_preflight"
    attempt: execution_receipt.Attempt | None = None
    profile: CandidateExternalProfile | None = None
    identity: dict | None = None
    outputs: dict[str, bytes] | None = None
    overlap_domains: frozenset[str] = frozenset()
    publishing: bool = False


def _output_paths(binding, paths):
    if type(paths) is not ExternalRunPaths:
        raise ExternalSourceExecutionError("invalid_external_run_paths")
    expected = (
        (paths.artifacts, ArtifactPaths),
        (paths.secondary_artifacts, SecondaryArtifactPaths),
        (paths.drift_artifacts, DriftArtifactPaths),
    )
    if any(type(value) is not kind for value, kind in expected):
        raise ExternalSourceExecutionError("invalid_external_run_paths")
    supplied = [paths.archive, paths.suffix_rules, paths.attempt, paths.public_summary]
    for group in (paths.artifacts, paths.secondary_artifacts, paths.drift_artifacts):
        supplied.extend(getattr(group, member.name) for member in fields(group))
    for path in supplied:
        execution_receipt._absolute_path(path)
    attempt = paths.attempt.absolute()
    public = paths.public_summary.absolute()
    if public.is_relative_to(attempt):
        raise ExternalSourceExecutionError("public_summary_inside_attempt")
    for path in (attempt, public):
        if path.is_relative_to(binding.root):
            raise ExternalSourceExecutionError("outputs_must_be_outside_checkout")
        with execution_receipt._directory(path.parent) as parent:
            execution_receipt._require_absent(parent, path.name)


def preflight(state):
    state.profile = resolve_external_source_profile(state.binding)
    state.identity = external_identity(state.binding, state.profile, state.handoff)
    state.overlap_domains = verify_internal_handoff(
        state.handoff.handoff_bytes,
        state.handoff.overlap_bytes,
        expected_handoff_sha256=sha256(state.handoff.handoff_bytes).hexdigest(),
    )
    _output_paths(state.binding, state.paths)
    state.stage = "reservation"
    state.attempt = execution_receipt.reserve_attempt(
        state.paths.attempt, identity=state.identity
    )


def _source(state):
    state.stage = "suffix_rules"
    suffix = _read_file_once(state.paths.suffix_rules)
    if sha256(suffix).hexdigest() != state.profile.suffix_rules_sha256:
        raise ExternalSourceExecutionError("suffix_hash_mismatch")
    state.stage = "publisher_archive"
    archive = _read_file_once(state.paths.archive)
    state.stage = "source_reconstruction"
    decoded = decode_phishvn_archive(archive, pins=state.profile.archive_pins)
    state.stage = "preparation"
    prepared = phishvn.prepare_external_rows(
        decoded.rows,
        published_split_counts=decoded.published_split_counts,
        test_split="test",
        suffix_rules=protocol_preflight.parse_suffix_rules(suffix.decode("utf-8")),
        phiusiil_domains=state.overlap_domains,
    )
    return decoded, prepared, suffix


def produce(state, session):
    decoded, prepared, suffix = _source(state)
    state.stage = "source_checkpoints"
    provenance = build_external_provenance(
        decoded,
        prepared,
        suffix_rules=suffix,
        internal_handoff=state.handoff.handoff_bytes,
        internal_overlap=state.handoff.overlap_bytes,
        execution=state.identity,
        reservation_sha256=state.attempt.reservation_sha256,
    )
    if set(provenance) != PROVENANCE_NAMES:
        raise ExternalSourceExecutionError("invalid_external_provenance_inventory")
    state.failures.writer = ExternalCheckpointWriter(
        state.attempt, identity=state.identity
    )
    state.failures.writer.begin(provenance)
    state.stage = "scoring"
    state.failures.produced = produce_external_evidence(
        prepared, session, retain=state.failures.writer
    )
    state.stage = "checkpoint_completion"
    state.outputs = state.failures.writer.complete(
        state.failures.produced.private_outputs
    )
