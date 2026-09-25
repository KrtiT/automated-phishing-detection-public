"""Private association of actual observation and later rejection, never acceptance."""

from contextlib import contextmanager
from dataclasses import dataclass, field

from ._process_support import command_hash
from .execution_preflight import ExecutionBinding
from .external_source_handoff import VerifiedExternalSnapshot
from .internal_external_handoff import InternalHandoffPayloads
from .owned_worker import WorkerObservation


@dataclass(frozen=True)
class ObservedExternalFailure:
    binding: ExecutionBinding = field(repr=False)
    handoff: InternalHandoffPayloads = field(repr=False)
    stage: str
    command_sha256: str | None
    worker: WorkerObservation | None
    candidate_snapshot: VerifiedExternalSnapshot | None = field(repr=False)
    worker_progress: bytes | None = field(repr=False)
    preparation: object = field(default=None, repr=False)


@dataclass
class ExternalObservationState:
    binding: ExecutionBinding = field(repr=False)
    handoff: InternalHandoffPayloads = field(repr=False)
    stage: str = "external_preflight"
    command: tuple[str, ...] | None = field(default=None, repr=False)
    worker: WorkerObservation | None = None
    snapshot: VerifiedExternalSnapshot | None = field(default=None, repr=False)
    worker_progress: bytes | None = field(default=None, repr=False)
    preparation: object = field(default=None, repr=False)

    @contextmanager
    def capture_body(self):
        try:
            yield
        except BaseException as error:
            try:
                progress = vars(error).get("progress")
                if type(progress) is bytes:
                    self.worker_progress = progress
            except BaseException:
                pass
            raise

    def retain_failure(self, error: BaseException) -> None:
        try:
            progress = self.worker_progress
            if progress is None:
                progress = vars(error).get("progress")
            error.external_failure = ObservedExternalFailure(
                self.binding,
                self.handoff,
                self.stage,
                command_hash(self.command) if self.command is not None else None,
                self.worker,
                self.snapshot,
                progress if type(progress) is bytes else None,
                self.preparation,
            )
        except BaseException:
            pass
