"""Hold authenticated preparation across scoring without original source paths."""

from contextlib import contextmanager

from ._exception_cleanup import CleanupStack
from ._external_source_records import (
    ExternalSourceExecutionError,
    PreparedExternalRunPaths,
)
from ._prepared_failure_context import carry_failure_context
from .study_preparation_context import bound_preparation_context
from .study_preparation_transport import hold_study_preparation


def _reader(binding, paths, reservation_sha256, completion_sha256):
    if type(paths) is not PreparedExternalRunPaths:
        raise ExternalSourceExecutionError("invalid_external_run_paths")
    identity, unused, contents, unused_profile = bound_preparation_context(binding)
    return hold_study_preparation(
        paths.preparation,
        expected_identity=identity,
        expected_reservation_sha256=reservation_sha256,
        expected_completion_sha256=completion_sha256,
        source_spec_bytes=contents["data/sources.json"],
        preparation_summary_bytes=contents["reports/phiusiil-preparation-summary.json"],
    )


@contextmanager
def held_preparation(binding, paths, reservation_sha256, completion_sha256):
    context = _reader(binding, paths, reservation_sha256, completion_sha256)
    original = None
    try:
        with CleanupStack() as cleanup:
            cleanup.push(context.__exit__)
            preparation = context.__enter__()
            try:
                yield preparation
            except BaseException as error:
                original = error
                raise
    except BaseException as error:
        carry_failure_context(error, original)
        raise


def run_held_preparation(
    binding, paths, transport, handoff_sha256, reservation, completion
):
    from . import external_source_runner as runner

    with held_preparation(binding, paths, reservation, completion) as preparation:
        handoff = runner.read_internal_handoff_transport(
            transport, expected_handoff_sha256=handoff_sha256
        )
        return runner._run_bound_prepared_external(
            binding, paths, handoff=handoff, preparation=preparation
        )
