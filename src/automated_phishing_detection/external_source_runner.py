"""Candidate external source worker; the public protected-access gate stays closed.

The private composition is for invented fixtures, never research authorization.
Only the observing parent can later accept a successful worker's retained output.
"""

from pathlib import Path

from . import _external_source_body as body
from ._external_primary_progress import attach_failure_progress
from ._external_source_records import (
    ExternalRunPaths,
    ExternalSourceExecutionError,
    build_external_public,
)
from ._prepared_external_io import io_scope
from ._prepared_external_runtime import run_held_preparation
from .bound_external_runtime import open_bound_external_session
from .execution_preflight import ExecutionBinding, bind_execution, recheck_binding
from .execution_receipt import publish_completion, record_failure
from .external_source_failure import retain_external_failure
from .internal_external_handoff import InternalHandoffPayloads
from .internal_failure import failure_kind
from .internal_handoff_transport import read_internal_handoff_transport


def _selected(first, later):
    return (
        later
        if isinstance(first, Exception) and not isinstance(later, Exception)
        else first
    )


def _failure_records(state, error, selected):
    progress, incomplete = None, False
    if state.attempt is not None and not state.publishing:
        try:
            progress = state.failures.snapshot(
                state.attempt, state.identity, state.stage, error
            )
            with io_scope(state.preparation):
                retain_external_failure(state.attempt, progress)
        except BaseException as persistence_error:
            incomplete = True
            selected = _selected(selected, persistence_error)
        try:
            with io_scope(state.preparation):
                record_failure(
                    state.attempt, stage=state.stage, error_type=failure_kind(selected)
                )
        except BaseException as persistence_error:
            incomplete = True
            selected = _selected(selected, persistence_error)
    return selected, progress, incomplete


def _failed(state, error):
    selected = state.failures.selected_error(error)
    selected, progress, incomplete = _failure_records(state, error, selected)
    if not isinstance(selected, Exception):
        try:
            attach_failure_progress(
                selected, progress, "external_execution_interrupted"
            )
        except BaseException:
            pass
        raise selected from None
    symbol = "failure_record_incomplete" if incomplete else "execution_failed"
    raise ExternalSourceExecutionError(f"{state.stage}: {symbol}") from None


def _complete(state):
    state.stage = "final_binding"
    with io_scope(state.preparation):
        recheck_binding(state.binding)
    state.stage = "summary"
    public = build_external_public(
        state.binding,
        state.profile,
        state.identity,
        state.attempt.reservation_sha256,
        state.outputs,
        state.failures.produced.public_summary,
        preparation=state.preparation,
    )
    state.stage = "publication"
    state.publishing = True
    with io_scope(state.preparation):
        return publish_completion(
            state.attempt,
            private_outputs=state.outputs,
            public_summary=public,
            public_path=state.paths.public_summary,
        )


def _run_bound_external(
    binding: ExecutionBinding,
    paths: ExternalRunPaths,
    *,
    handoff: InternalHandoffPayloads,
) -> Path:
    """Compose invented source fixtures without granting protected-data access."""
    state = body.ExternalRun(binding, paths, handoff)
    return _run(state)


def _run(state):
    try:
        body.preflight(state)
        state.stage = "model_loading"
        with (
            open_bound_external_session(
                state.binding,
                state.paths.artifacts,
                state.paths.secondary_artifacts,
                state.paths.drift_artifacts,
            ) as session,
            state.failures.capture_body(),
        ):
            body.produce(state, session)
        state.failures.session_closed = True
        if state.failures.original_error is not None:
            raise state.failures.original_error
        return _complete(state)
    except BaseException as error:
        _failed(state, error)


def run_external_evaluation(
    root: Path,
    *,
    expected_revision: str,
    expected_contract_sha256: str,
    paths: ExternalRunPaths,
    internal_transport: Path,
    expected_handoff_sha256: str,
) -> Path:
    """Require the complete frozen profile before inspecting supplied paths."""
    binding = bind_execution(
        root,
        expected_revision=expected_revision,
        expected_contract_sha256=expected_contract_sha256,
    )
    if not binding.protected_evaluation_ready:
        raise ExternalSourceExecutionError("pre_access_freeze_incomplete")
    profile = body.resolve_external_source_profile(binding)
    if not profile.protected_evaluation_ready:
        raise ExternalSourceExecutionError("external_profile_freeze_incomplete")
    handoff = read_internal_handoff_transport(
        internal_transport, expected_handoff_sha256=expected_handoff_sha256
    )
    return _run_bound_external(binding, paths, handoff=handoff)


def _run_bound_prepared_external(binding, paths, *, handoff, preparation):
    state = body.ExternalRun(binding, paths, handoff, preparation=preparation)
    return _run(state)


def run_prepared_external_evaluation(
    root,
    *,
    expected_revision,
    expected_contract_sha256,
    paths,
    internal_transport,
    expected_handoff_sha256,
    expected_preparation_reservation_sha256,
    expected_preparation_completion_sha256,
):
    binding = bind_execution(
        root,
        expected_revision=expected_revision,
        expected_contract_sha256=expected_contract_sha256,
    )
    if not binding.protected_evaluation_ready:
        raise ExternalSourceExecutionError("pre_access_freeze_incomplete")
    with io_scope(True):
        profile = body.resolve_external_source_profile(binding)
    if not profile.protected_evaluation_ready:
        raise ExternalSourceExecutionError("external_profile_freeze_incomplete")
    return run_held_preparation(
        binding,
        paths,
        internal_transport,
        expected_handoff_sha256,
        expected_preparation_reservation_sha256,
        expected_preparation_completion_sha256,
    )
