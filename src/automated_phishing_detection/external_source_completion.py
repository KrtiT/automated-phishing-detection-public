"""Verify retained external provenance and science after an owned worker exit.

The trusted observing parent supplies its actual worker observation and retained
internal expectations. Caller-constructed objects cannot establish that authority.
No original archive, internal CSV, PSL or model path is read by this verifier.
"""

from hashlib import sha256

from . import source_runner
from ._checkpoint_codec import canonical_bytes
from ._external_completion_files import snapshot_external_files
from ._external_completion_records import (
    ExternalCompletionVerificationError,
    authenticate_external_records,
)
from ._external_preparation_outputs import _PUBLIC_INPUTS
from ._external_provenance_payloads import PREPARED_NAMES, PROVENANCE_NAMES
from ._external_source_profile import resolve_external_source_profile
from ._external_source_records import ExternalRunPaths, PreparedExternalRunPaths
from ._prepared_external_io import io_scope, snapshot_prepared_external_files
from ._saved_external_bindings import PRIVATE_OUTPUTS
from .bound_drift import DriftArtifactPaths
from .bound_models import ArtifactPaths
from .bound_secondary import SecondaryArtifactPaths
from .execution_preflight import ExecutionBinding, recheck_binding
from .external_source_handoff import VerifiedExternalSnapshot, freeze_external_snapshot
from .external_source_provenance import verify_external_provenance
from .internal_external_handoff import InternalHandoffPayloads
from .owned_worker import WorkerObservation
from .saved_external_evidence import reconstruct_external_evidence

__all__ = ["ExternalCompletionVerificationError", "verify_external_completion_snapshot"]


def _require(condition):
    if not condition:
        raise ExternalCompletionVerificationError("invalid_external_completion")


def _paths(binding, paths, preparation):
    expected_type = (
        ExternalRunPaths if preparation is None else PreparedExternalRunPaths
    )
    _require(type(paths) is expected_type)
    _require(type(paths.artifacts) is ArtifactPaths)
    _require(type(paths.secondary_artifacts) is SecondaryArtifactPaths)
    _require(type(paths.drift_artifacts) is DriftArtifactPaths)
    attempt = source_runner.execution_receipt._absolute_path(paths.attempt)
    public = source_runner.execution_receipt._absolute_path(paths.public_summary)
    _require(not attempt.is_relative_to(binding.root))
    _require(not public.is_relative_to(binding.root))
    _require(not public.is_relative_to(attempt))


def _provenance(records, binding, profile, handoff):
    outputs = records.private_outputs
    pins = dict(binding.source_hashes)
    for relative, retained in _PUBLIC_INPUTS:
        _require(
            relative in pins and sha256(outputs[retained]).hexdigest() == pins[relative]
        )
    verify_external_provenance(
        {name: outputs[name] for name in PROVENANCE_NAMES},
        {name: outputs[name] for name in PREPARED_NAMES},
        pins=profile.archive_pins,
        expected_handoff=handoff.handoff_bytes,
        expected_overlap=handoff.overlap_bytes,
        execution=records.identity,
        reservation_sha256=records.reservation_sha256,
        suffix_rules_sha256=profile.suffix_rules_sha256,
    )


def _verify(binding, paths, handoff, preparation):
    with io_scope(preparation):
        profile = resolve_external_source_profile(binding)
        _paths(binding, paths, preparation)
    reader = (
        snapshot_external_files
        if preparation is None
        else snapshot_prepared_external_files
    )
    with reader(paths.attempt, paths.public_summary) as files:
        records = authenticate_external_records(
            files,
            paths.attempt,
            binding=binding,
            profile=profile,
            handoff=handoff,
            preparation=preparation,
        )
        _provenance(records, binding, profile, handoff)
        replay = reconstruct_external_evidence(
            {name: records.private_outputs[name] for name in PRIVATE_OUTPUTS},
            canonical_bytes(records.public["composition"]),
        )
        with io_scope(preparation):
            recheck_binding(binding)
        return freeze_external_snapshot(files, profile.canonical_bytes, replay)


def verify_external_completion_snapshot(
    binding: ExecutionBinding,
    paths: ExternalRunPaths,
    *,
    expected_handoff: InternalHandoffPayloads,
    worker: WorkerObservation,
    command: tuple[str, ...],
    expected_preparation=None,
) -> VerifiedExternalSnapshot:
    """Retain accepted saved bytes in the same parent that observed the worker."""
    try:
        source_runner._require_successful_worker(worker, command)
        return _verify(binding, paths, expected_handoff, expected_preparation)
    except Exception:
        raise ExternalCompletionVerificationError(
            "invalid_external_completion"
        ) from None
