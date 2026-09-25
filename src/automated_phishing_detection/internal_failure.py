"""Private parent-owned failure state survives numerical-session unwinding."""

import asyncio
import json
from contextlib import contextmanager

from ._external_primary_progress import attach_failure_progress
from .evaluation_producer import _json_bytes
from .internal_scientific_checkpoints import SCIENTIFIC_CHECKPOINT_PROTOCOL


def failure_kind(error: BaseException) -> str:
    if isinstance(error, asyncio.CancelledError):
        return "cancelled"
    if isinstance(error, KeyboardInterrupt):
        return "keyboard_interrupt"
    if isinstance(error, SystemExit):
        return "system_exit"
    return "execution_failed" if isinstance(error, Exception) else "interrupted"


def failure_exit_code(error: BaseException) -> int:
    if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)):
        return 130
    if type(error) is SystemExit and type(error.code) is int and 1 <= error.code <= 255:
        return error.code
    return 2


class InternalFailureState:
    def __init__(self, progress):
        self.progress = progress
        self.writer = None
        self.source_checkpoint_sha256 = None
        self.original_error = None
        self.body_exited = False
        self.session_closed = False

    @contextmanager
    def capture_body(self, scorer):
        try:
            yield
        except BaseException as error:
            self.original_error = error
            try:
                self.progress.observe_counts(
                    scorer, suppress_interruptions=not isinstance(error, Exception)
                )
            except BaseException as observation_error:
                self.original_error = observation_error
                raise
            raise
        finally:
            self.body_exited = True

    def selected_error(self, error: BaseException) -> BaseException:
        if self.original_error is not None and not isinstance(
            self.original_error, Exception
        ):
            return self.original_error
        return error

    def snapshot(
        self, attempt, identity: dict, stage: str, error: BaseException
    ) -> bytes:
        original = self.original_error if self.original_error is not None else error
        return _json_bytes(
            {
                "schema_version": 1,
                "protocol_id": SCIENTIFIC_CHECKPOINT_PROTOCOL,
                "status": "failed",
                "reservation_sha256": attempt.reservation_sha256,
                "execution": identity,
                "source_checkpoint_sha256": self.source_checkpoint_sha256,
                "stage": stage,
                "producer": json.loads(self.progress.snapshot()),
                "checkpoints": json.loads(self.writer.snapshot())
                if self.writer
                else None,
                "failure": failure_kind(original),
                "cleanup_failed": self.body_exited
                and not self.session_closed
                and error is not self.original_error,
            }
        )


def propagate_interruption(error: BaseException, progress: bytes | None) -> None:
    try:
        attach_failure_progress(error, progress, "internal_execution_interrupted")
    except BaseException:
        pass
    raise error from None
