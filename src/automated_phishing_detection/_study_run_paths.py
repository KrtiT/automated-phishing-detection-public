"""Hold fresh output parents without reading or authorizing protected inputs."""

import os
import stat
from contextlib import contextmanager

from . import _study_preparation_files as files
from . import execution_receipt as receipt
from ._exception_cleanup import CleanupStack
from ._prepared_failure_context import carry_failure_context
from ._study_run_context import StudyRunError, output_paths, require
from .operational_cell_runner import OperationalCellPaths
from .operational_schedule import cell_for_ordinal

CELL_NAMES = tuple(
    f"cell-{ordinal:03d}-{suffix}"
    for ordinal in range(1, 126)
    for suffix in ("inputs", "attempt", "summary.json")
)


def cell_paths(paths, ordinal):
    cell_for_ordinal(ordinal)
    prefix = f"cell-{ordinal:03d}"
    return OperationalCellPaths(
        paths.accepted_inputs_directory,
        paths.cells_directory / f"{prefix}-inputs",
        paths.cells_directory / f"{prefix}-attempt",
        paths.cells_directory / f"{prefix}-summary.json",
    )


def _private(directory):
    require(stat.S_IMODE(os.fstat(directory.descriptor).st_mode) == 0o700)


class _OutputDirectories:
    def __init__(self, directories, paths):
        self.directories = directories
        self.cells = directories[paths.cells_directory]
        self.accepted_parent = directories[paths.accepted_inputs_directory.parent]

    def _check(self, expected_names):
        for directory in self.directories.values():
            directory.check()
        _private(self.cells)
        _private(self.accepted_parent)
        names = set(os.listdir(self.cells.descriptor))
        require(names <= set(CELL_NAMES))
        if expected_names is not None:
            require(type(expected_names) is tuple and names == set(expected_names))
        for name in names:
            metadata = receipt._entry(self.cells, name)
            require(metadata is not None)
            public = name.endswith("-summary.json")
            require((stat.S_ISREG if public else stat.S_ISDIR)(metadata.st_mode))
            require(stat.S_IMODE(metadata.st_mode) == (0o644 if public else 0o700))
            require(not public or metadata.st_nlink == 1)

    def check(self, expected_names=None):
        files.deferred(self._check, expected_names)


def _acquire(cleanup, paths):
    outputs = output_paths(paths)
    parents = {path.parent for path in outputs} | {paths.cells_directory}
    directories = {
        path: files.enter_directory(cleanup, path) for path in sorted(parents)
    }
    held = _OutputDirectories(directories, paths)
    held.check(())
    for output in outputs:
        if output != paths.cells_directory:
            files.deferred(
                receipt._require_absent, directories[output.parent], output.name
            )
    return held


@contextmanager
def hold_study_paths(paths):
    original = None
    try:
        with CleanupStack() as cleanup:
            try:
                held = _acquire(cleanup, paths)
                cleanup.callback(held.check)
                yield held
            except BaseException as error:
                original = error
                raise
    except BaseException as error:
        carry_failure_context(error, original)
        if not isinstance(error, Exception) or error is original:
            raise
        rejected = StudyRunError("invalid_study_output_paths")
        carry_failure_context(rejected, error)
        raise rejected from None
