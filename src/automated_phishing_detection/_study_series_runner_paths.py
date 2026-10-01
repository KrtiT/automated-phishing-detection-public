"""Pin empty fresh outputs and preserve every original history location."""

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from . import _operational_input_files as inputs
from . import _study_preparation_files as files
from . import execution_receipt as receipt
from ._exception_cleanup import CleanupStack
from ._study_run_paths import _OutputDirectories

OUTPUTS = (
    "series_attempt",
    "series_public_summary",
    "segment_attempt",
    "segment_public_summary",
    "historical_inputs_dir",
    "cells_dir",
)


def cell_names(profile):
    return tuple(
        f"cell-{ordinal:03d}-{suffix}"
        for ordinal in range(profile["segment"]["start_ordinal"], 126)
        for suffix in ("inputs", "attempt", "summary.json")
    )


def disjoint(profile, history):
    outputs = tuple(Path(profile["paths"][name]) for name in OUTPUTS)
    protected = (Path(profile["paths"]["repo_root"]),) + tuple(
        Path(path) for path, unused in history.index.file_refs
    )
    for index, first in enumerate(outputs):
        for second in outputs[index + 1 :] + protected:
            files.require(
                not first.is_relative_to(second) and not second.is_relative_to(first)
            )
    return outputs


def check_outputs(held, names, expected=None):
    held.check(expected)
    files.require(set(os.listdir(held.cells.descriptor)) <= set(names))


@contextmanager
def hold_outputs(profile, history):
    outputs = disjoint(profile, history)
    paths = SimpleNamespace(
        cells_directory=Path(profile["paths"]["cells_dir"]),
        accepted_inputs_directory=Path(profile["paths"]["historical_inputs_dir"]),
    )
    with CleanupStack() as cleanup:
        directories = {
            path: files.enter_directory(cleanup, path)
            for path in sorted({output.parent for output in outputs})
        }
        for output in outputs:
            files.deferred(
                receipt._require_absent, directories[output.parent], output.name
            )
        unused, cells = files.deferred(inputs._create, cleanup, paths.cells_directory)
        directories[paths.cells_directory] = cells
        held = _OutputDirectories(directories, paths)
        names = cell_names(profile)
        check_outputs(held, names, ())
        cleanup.callback(check_outputs, held, names)
        yield held
