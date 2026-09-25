"""Same-parent source composition behind two independently closed access gates.

Only the parent running both workers supplies the retained internal expectations.
Caller-created records and historical handoff files cannot establish this lineage.
This pair supplies neither operational observations nor final study decisions.
"""

import sys
from dataclasses import dataclass, fields
from pathlib import Path

from . import source_runner
from ._external_process_context import ExternalObservationState
from ._external_source_profile import resolve_external_source_profile
from ._external_source_records import (
    ExternalRunPaths,
    ExternalSourceExecutionError,
    external_identity,
)
from .bound_drift import DriftArtifactPaths
from .bound_models import ArtifactPaths
from .bound_secondary import SecondaryArtifactPaths
from .execution_preflight import bind_execution
from .external_source_completion import verify_external_completion_snapshot
from .external_source_handoff import ObservedExternalCompletion
from .internal_external_handoff import build_internal_handoff
from .internal_handoff_transport import retain_internal_handoff
from .internal_process_handoff import ObservedInternalCompletion
from .owned_worker import observe_worker
from .source_runner import InternalRunPaths


@dataclass(frozen=True)
class ObservedSourceCompletion:
    internal: ObservedInternalCompletion
    external: ObservedExternalCompletion


def _worker_options(paths):
    if (
        type(paths) is not ExternalRunPaths
        or type(paths.artifacts) is not ArtifactPaths
        or type(paths.secondary_artifacts) is not SecondaryArtifactPaths
        or type(paths.drift_artifacts) is not DriftArtifactPaths
    ):
        raise ExternalSourceExecutionError("invalid_external_run_paths")
    return (
        ("archive", paths.archive),
        ("suffix-rules", paths.suffix_rules),
        *(
            (member.name.replace("_", "-"), getattr(group, member.name))
            for group in (
                paths.artifacts,
                paths.secondary_artifacts,
                paths.drift_artifacts,
            )
            for member in fields(group)
        ),
        ("attempt", paths.attempt),
        ("public-summary", paths.public_summary),
    )


def _worker_command(binding, paths, transport):
    options = (
        ("repo-root", binding.root),
        ("expected-revision", binding.revision),
        ("expected-contract-sha256", binding.contract_sha256),
        *_worker_options(paths),
        ("internal-transport", transport.directory),
        ("expected-handoff-sha256", transport.expected_handoff_sha256),
    )
    return (
        sys.executable,
        str(binding.root / "scripts/run_external_evaluation.py"),
        *(
            argument
            for name, value in options
            for argument in (f"--{name}", str(value))
        ),
    )


def _run_observed_external(binding, paths, handoff):
    state = ExternalObservationState(binding, handoff)
    try:
        profile = resolve_external_source_profile(binding)
        external_identity(binding, profile, handoff)
        state.stage = "transport"
        with retain_internal_handoff(handoff) as transport, state.capture_body():
            state.command = _worker_command(binding, paths, transport)
            state.stage = "worker_observation"
            state.worker = observe_worker(state.command)
            source_runner._require_successful_worker(state.worker, state.command)
            state.stage = "completion_verification"
            state.snapshot = verify_external_completion_snapshot(
                binding,
                paths,
                expected_handoff=handoff,
                worker=state.worker,
                command=state.command,
            )
            state.stage = "transport_finalization"
        return ObservedExternalCompletion(state.worker, state.snapshot)
    except BaseException as error:
        state.retain_failure(error)
        raise


def _run_observed_sources(binding, internal_paths, external_paths):
    """Compose invented fixtures privately; this helper grants no access authority."""
    resolve_external_source_profile(binding)
    internal = source_runner._run_observed_internal(binding, internal_paths)
    try:
        handoff = build_internal_handoff(internal)
        external = _run_observed_external(binding, external_paths, handoff)
        return ObservedSourceCompletion(internal, external)
    except BaseException as error:
        try:
            error.source_internal = internal
        except BaseException:
            pass
        raise


def run_internal_external_process(
    root: Path,
    *,
    expected_revision: str,
    expected_contract_sha256: str,
    internal_paths: InternalRunPaths,
    external_paths: ExternalRunPaths,
) -> ObservedSourceCompletion:
    """Require a complete study freeze before either source pass can begin."""
    binding = bind_execution(
        root,
        expected_revision=expected_revision,
        expected_contract_sha256=expected_contract_sha256,
    )
    if not binding.protected_evaluation_ready:
        raise ExternalSourceExecutionError("pre_access_freeze_incomplete")
    profile = resolve_external_source_profile(binding)
    if not profile.protected_evaluation_ready:
        raise ExternalSourceExecutionError("pre_access_freeze_incomplete")
    return _run_observed_sources(binding, internal_paths, external_paths)
