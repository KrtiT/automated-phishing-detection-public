"""Fixed commands for owned internal workers consuming retained preparation."""

import sys
from dataclasses import fields

from . import execution_receipt, source_runner
from ._prepared_internal_records import PreparedInternalRunPaths
from .bound_models import ArtifactPaths
from .bound_secondary import SecondaryArtifactPaths
from .internal_process_handoff import ObservedInternalCompletion, retain_worker_failure


def _worker_options(paths):
    return (
        ("preparation", paths.preparation),
        *(
            (member.name.replace("_", "-"), getattr(group, member.name))
            for group in (paths.artifacts, paths.secondary_artifacts)
            for member in fields(group)
        ),
        ("attempt", paths.attempt),
        ("public-summary", paths.public_summary),
    )


def _worker_command(binding, paths, *, reservation_sha256, completion_sha256):
    if (
        type(paths) is not PreparedInternalRunPaths
        or type(paths.artifacts) is not ArtifactPaths
        or type(paths.secondary_artifacts) is not SecondaryArtifactPaths
        or any(
            type(digest) is not str or not execution_receipt._SHA256.fullmatch(digest)
            for digest in (reservation_sha256, completion_sha256)
        )
    ):
        raise source_runner.SourceExecutionError("invalid_prepared_worker_inputs")
    options = (
        ("repo-root", binding.root),
        ("expected-revision", binding.revision),
        ("expected-contract-sha256", binding.contract_sha256),
        ("expected-preparation-reservation-sha256", reservation_sha256),
        ("expected-preparation-completion-sha256", completion_sha256),
        *_worker_options(paths),
    )
    return (
        sys.executable,
        str(binding.root / "scripts/run_prepared_internal_evaluation.py"),
        "--worker",
        *(
            argument
            for name, value in options
            for argument in (f"--{name}", str(value))
        ),
    )


def retain_held_worker(error, completed, binding):
    try:
        values = BaseException.__dict__["__dict__"].__get__(error)
        if type(completed) is ObservedInternalCompletion and not dict.__contains__(
            values, "worker_failure"
        ):
            retain_worker_failure(
                error, completed.worker, binding, "preparation_finalization"
            )
    except BaseException:
        pass
