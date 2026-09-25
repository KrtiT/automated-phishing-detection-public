"""A same-parent association, not serialized proof of an earlier process exit."""

from dataclasses import dataclass

from .execution_preflight import ExecutionBinding
from .internal_source_handoff import VerifiedInternalSnapshot
from .owned_worker import WorkerObservation


@dataclass(frozen=True)
class ObservedInternalCompletion:
    worker: WorkerObservation
    snapshot: VerifiedInternalSnapshot

    @property
    def public_summary(self) -> dict:
        return self.snapshot.public_summary


@dataclass(frozen=True)
class ObservedInternalFailure:
    worker: WorkerObservation
    binding: ExecutionBinding
    stage: str


def retain_worker_failure(error, worker, binding, stage) -> None:
    """Best-effort private retention must never replace the original exception."""
    try:
        error.worker_failure = ObservedInternalFailure(worker, binding, stage)
    except BaseException:
        pass
