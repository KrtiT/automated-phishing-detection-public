"""Hold fully verified retained history behind current explicit adoption."""

from contextlib import contextmanager
from dataclasses import dataclass, field

from ._exception_cleanup import CleanupStack
from ._study_series_history_cells import restore_cells
from ._study_series_history_context import archives
from ._study_series_history_files import HistoryFiles
from ._study_series_history_origin import restore_origin, verify_physical
from ._study_series_history_sources import restore_sources
from .study_series_execution import recheck_series_public_execution
from .study_series_runner import require_final_policy


class SeriesHistoryError(ValueError):
    """A symbolic retained-history rejection without private diagnostics."""


def _check(public, reader):
    try:
        recheck_series_public_execution(public)
        require_final_policy(public)
        reader.check()
    except Exception:
        raise SeriesHistoryError("invalid_series_history") from None


@dataclass(frozen=True)
class HeldSeriesHistory:
    index: object = field(repr=False)
    origin_metadata_bytes: bytes = field(repr=False)
    historical_prefix: tuple = field(repr=False)
    internal_snapshot: object = field(repr=False)
    external_snapshot: object = field(repr=False)
    _public: object = field(repr=False)
    _reader: HistoryFiles = field(repr=False)

    def check(self):
        _check(self._public, self._reader)


def _restore(public, reader):
    try:
        index, value, profile = archives(reader, public)
        origin = restore_origin(reader, value, profile, index)
        verify_physical(reader, value, origin)
        sources = restore_sources(reader, value, origin)
        prefix = restore_cells(reader, value, origin, sources)
        return HeldSeriesHistory(
            index, origin.metadata_bytes, prefix, *sources, public, reader
        )
    except Exception:
        raise SeriesHistoryError("invalid_series_history") from None


@contextmanager
def hold_series_history(public):
    reader = HistoryFiles()
    _check(public, reader)
    with CleanupStack() as cleanup:
        cleanup.callback(_check, public, reader)
        history = _restore(public, reader)
        history.check()
        yield history
