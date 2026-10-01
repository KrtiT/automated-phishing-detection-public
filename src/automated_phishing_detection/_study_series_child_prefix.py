"""Hold four immutable root records while their parent directories can grow."""

import os
import stat
from contextlib import contextmanager

from . import _study_preparation_files as files
from . import _study_series_child_commands as commands
from ._exception_cleanup import CleanupStack
from ._prepared_failure_context import carry_failure_context
from ._study_child_root import _read
from ._study_series_admission import validate_series_frame


class SeriesChildPrefixError(ValueError):
    """The pinned live-parent prefix could not remain held consistently."""


def _context(binding, frame):
    validate_series_frame(frame)
    profile = commands._profile(binding)
    files.require(frame.profile_sha256 == binding.profile_sha256)
    files.require(frame.envelope_sha256 == binding.envelope_sha256)
    files.require(frame.history_index_sha256 == profile["history"]["index_sha256"])
    files.require(
        frame.origin_reservation_sha256 == profile["origin"]["root_reservation_sha256"]
    )
    files.require(frame.segment_ordinal == profile["segment"]["ordinal"])
    files.require(
        profile["segment"]["start_ordinal"]
        <= frame.cell_ordinal
        <= profile["segment"]["end_ordinal"]
    )
    return {
        name: commands.schema.lexical_path(profile["paths"][f"{name}_attempt"])
        for name in ("series", "segment")
    }


def _expected(frame):
    return {
        "series": {"reservation.json": frame.series_reservation_sha256},
        "segment": {
            "reservation.json": frame.segment_reservation_sha256,
            "segment-intent.json": frame.intent_sha256,
            "history-import.json": frame.predecessor_sha256,
        },
    }


def _check(held):
    for directory, states in held:
        directory.check()
        files.require(stat.S_IMODE(os.fstat(directory.descriptor).st_mode) == 0o700)
        for name, initial in states.items():
            files.require(files.capture(directory, name) == initial)
        directory.check()


def _acquire(cleanup, paths, frame):
    payloads, held = {}, []
    for kind, expectations in _expected(frame).items():
        directory = files.enter_directory(cleanup, paths[kind])
        states = {}
        files.require(stat.S_IMODE(os.fstat(directory.descriptor).st_mode) == 0o700)
        for name, expected in expectations.items():
            payloads[f"{kind}/{name}"], states[name] = _read(
                cleanup, directory, name, expected
            )
        held.append((directory, states))
    files.deferred(_check, held)
    cleanup.callback(files.deferred, _check, held)
    return payloads


def _validate(binding, frame, payloads):
    from .study_series_prefix import validate_series_child_prefix

    validate_series_child_prefix(binding, frame, tuple(sorted(payloads.items())))


@contextmanager
def hold_series_child_prefix(binding, frame):
    """Hold declarations only; live channel and full adoption are separate gates."""
    original = None
    try:
        paths = _context(binding, frame)
        with CleanupStack() as cleanup:
            payloads = _acquire(cleanup, paths, frame)
            _validate(binding, frame, payloads)
            try:
                yield tuple(sorted(payloads.items()))
            except BaseException as error:
                original = error
                raise
    except BaseException as error:
        carry_failure_context(error, original)
        if not isinstance(error, Exception) or error is original:
            raise error from None
        rejected = SeriesChildPrefixError("invalid_series_child_prefix")
        carry_failure_context(rejected, error)
        raise rejected from None
