"""Owned prepared workers retain source lineage without authorizing continuation."""

from dataclasses import dataclass, field

from . import external_source_process as process
from ._external_process_context import ObservedExternalFailure
from ._external_source_records import (
    ExternalSourceExecutionError,
    PreparedExternalRunPaths,
)
from ._prepared_external_io import io_scope
from ._prepared_external_runtime import held_preparation
from ._prepared_internal_records import PreparedInternalRunPaths
from .execution_preflight import bind_execution


@dataclass(frozen=True)
class ObservedPreparedSourceCompletion:
    preparation: object = field(repr=False)
    internal: object
    external: object
    handoff: object = field(default=None, repr=False)


def _run_observed_prepared_internal(binding, paths, *, preparation):
    from .prepared_internal_runner import _run_observed_prepared_internal as observed

    return observed(binding, paths, preparation=preparation)


def _same_preparation(internal, external):
    if (
        type(internal) is not PreparedInternalRunPaths
        or type(external) is not PreparedExternalRunPaths
        or internal.preparation.absolute() != external.preparation.absolute()
    ):
        raise ExternalSourceExecutionError("invalid_preparation_paths")


def _retain_failure(error, preparation, internal):
    try:
        attributes = BaseException.__dict__["__dict__"].__get__(error)
        for name, value in (
            ("study_preparation", preparation),
            ("source_internal", internal),
        ):
            if value is not None and not dict.__contains__(attributes, name):
                dict.__setitem__(attributes, name, value)
    except BaseException:
        pass


def _run_observed_prepared_sources(
    binding, internal_paths, external_paths, preparation
):
    _same_preparation(internal_paths, external_paths)
    internal = None
    try:
        internal = _run_observed_prepared_internal(
            binding, internal_paths, preparation=preparation
        )
        handoff = process.build_internal_handoff(internal)
        external = process._run_observed_external(
            binding, external_paths, handoff, preparation=preparation
        )
        return ObservedPreparedSourceCompletion(
            preparation, internal, external, handoff
        )
    except BaseException as error:
        _retain_failure(error, preparation, internal)
        raise


def _retain_completed(error, binding, completed):
    try:
        _retain_failure(error, completed.preparation, completed.internal)
        attributes = BaseException.__dict__["__dict__"].__get__(error)
        if dict.__contains__(attributes, "external_failure"):
            return
        retained = ObservedExternalFailure(
            binding,
            completed.handoff,
            "preparation_finalization",
            completed.external.worker.command_sha256,
            completed.external.worker,
            completed.external.snapshot,
            None,
            completed.preparation,
        )
        dict.__setitem__(attributes, "external_failure", retained)
    except BaseException:
        pass


def _run_held_sources(binding, internal_paths, external_paths, reservation, completion):
    completed, preparation = None, None
    try:
        with held_preparation(
            binding, external_paths, reservation, completion
        ) as preparation:
            completed = _run_observed_prepared_sources(
                binding, internal_paths, external_paths, preparation
            )
        return completed
    except BaseException as error:
        if preparation is not None:
            _retain_failure(error, preparation, None)
        if completed is not None:
            _retain_completed(error, binding, completed)
        raise


def run_prepared_internal_external_process(
    root,
    *,
    expected_revision,
    expected_contract_sha256,
    internal_paths,
    external_paths,
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
        profile = process.resolve_external_source_profile(binding)
    if not profile.protected_evaluation_ready:
        raise ExternalSourceExecutionError("external_profile_freeze_incomplete")
    _same_preparation(internal_paths, external_paths)
    return _run_held_sources(
        binding,
        internal_paths,
        external_paths,
        expected_preparation_reservation_sha256,
        expected_preparation_completion_sha256,
    )
