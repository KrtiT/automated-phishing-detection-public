"""Same-parent source composition behind two independently closed access gates.

Only the parent running both workers supplies the retained internal expectations.
Caller-created records and historical handoff files cannot establish this lineage.
This pair supplies neither operational observations nor final study decisions.
"""

from dataclasses import dataclass
from pathlib import Path

from . import source_runner
from ._external_process_context import ExternalObservationState
from ._external_source_profile import resolve_external_source_profile
from ._external_source_records import (
    ExternalRunPaths,
    ExternalSourceExecutionError,
    external_identity,
)
from ._external_worker_commands import (
    _prepared_worker_command,
    _worker_command,
    _worker_options,
)
from ._prepared_external_io import io_scope
from .execution_preflight import bind_execution
from .external_source_completion import verify_external_completion_snapshot
from .external_source_handoff import ObservedExternalCompletion
from .internal_external_handoff import build_internal_handoff
from .internal_handoff_transport import retain_internal_handoff
from .internal_process_handoff import ObservedInternalCompletion
from .owned_worker import observe_worker
from .source_runner import InternalRunPaths

__all__ = [
    "ObservedSourceCompletion",
    "run_internal_external_process",
    "_worker_command",
    "_worker_options",
    "_prepared_worker_command",
]


@dataclass(frozen=True)
class ObservedSourceCompletion:
    internal: ObservedInternalCompletion
    external: ObservedExternalCompletion


def _run_observed_external(binding, paths, handoff, *, preparation=None):
    state = ExternalObservationState(binding, handoff, preparation=preparation)
    try:
        with io_scope(preparation):
            profile = resolve_external_source_profile(binding)
        extra = {} if preparation is None else {"preparation": preparation}
        external_identity(binding, profile, handoff, **extra)
        state.stage = "transport"
        with retain_internal_handoff(handoff) as transport, state.capture_body():
            _observe_and_verify(state, paths, transport)
            state.stage = "transport_finalization"
        return ObservedExternalCompletion(state.worker, state.snapshot)
    except BaseException as error:
        state.retain_failure(error)
        raise


def _observe_and_verify(state, paths, transport):
    preparation = state.preparation
    state.command = (
        _worker_command(state.binding, paths, transport)
        if preparation is None
        else _prepared_worker_command(state.binding, paths, transport, preparation)
    )
    state.stage = "worker_observation"
    state.worker = observe_worker(state.command)
    source_runner._require_successful_worker(state.worker, state.command)
    state.stage = "completion_verification"
    extra = {} if preparation is None else {"expected_preparation": preparation}
    state.snapshot = verify_external_completion_snapshot(
        state.binding,
        paths,
        expected_handoff=state.handoff,
        worker=state.worker,
        command=state.command,
        **extra,
    )


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
