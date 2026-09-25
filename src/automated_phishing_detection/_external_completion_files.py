"""Fixed once-read completion tree; filesystem consistency is not acceptance.

The observing parent checks actual exit and binding before entering this helper.
Directory descriptors stay open through reconstruction and final state checks.
No original publisher, internal source or model path is consumed here.
"""

import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from . import execution_receipt as receipt
from . import source_runner
from ._external_provenance_payloads import PROVENANCE_NAMES
from ._saved_external_bindings import PRIVATE_OUTPUTS

PAYLOAD_NAMES = PROVENANCE_NAMES | PRIVATE_OUTPUTS
_ROOT_FILES = frozenset({"reservation.json", "finalize.claim", "outcome.json"})


class ExternalCompletionFileError(ValueError):
    """Symbolic filesystem rejection without private paths or output contents."""


@dataclass(frozen=True)
class ExternalFileSnapshot:
    payloads: tuple[tuple[str, bytes], ...] = field(repr=False)


def _require(condition):
    if not condition:
        raise ExternalCompletionFileError("invalid_external_completion_files")


def _paths(attempt, public):
    attempt = receipt._absolute_path(attempt)
    public = receipt._absolute_path(public)
    _require(not public.is_relative_to(attempt))
    _require(len(PAYLOAD_NAMES) == 36 and not PROVENANCE_NAMES & PRIVATE_OUTPUTS)
    return attempt, public


@contextmanager
def _directories(attempt, public):
    with (
        receipt._directory(attempt) as root,
        receipt._directory(attempt / "checkpoints") as checkpoints,
        receipt._directory(attempt / "evidence") as evidence,
        receipt._directory(public.parent) as public_parent,
    ):
        yield (
            (root, "attempt", _ROOT_FILES),
            (checkpoints, "attempt/checkpoints", PAYLOAD_NAMES),
            (evidence, "attempt/evidence", PAYLOAD_NAMES),
            (public_parent, "", (public.name,)),
        )


def _check_directories(directories):
    for directory, prefix, names in directories:
        directory.check()
        if not prefix:
            continue
        expected = set(names)
        if prefix == "attempt":
            expected |= {"checkpoints", "evidence"}
        _require(stat.S_IMODE(os.fstat(directory.descriptor).st_mode) == 0o700)
        _require(set(os.listdir(directory.descriptor)) == expected)


def _capture_files(directories):
    captured = []
    for directory, prefix, names in directories:
        for name in sorted(names):
            state = receipt._entry(directory, name)
            _require(
                state is not None
                and stat.S_ISREG(state.st_mode)
                and state.st_nlink == 1
                and stat.S_IMODE(state.st_mode) == (0o600 if prefix else 0o644)
            )
            logical = f"{prefix}/{name}" if prefix else "public-summary.json"
            captured.append(
                (directory, name, logical, source_runner._file_state(state))
            )
    return tuple(captured)


def _read_files(captured):
    return ExternalFileSnapshot(
        tuple(
            (
                logical,
                source_runner._read_file_once(
                    directory.path / name, expected_state=initial
                ),
            )
            for directory, name, logical, initial in captured
        )
    )


def _recheck_files(directories, captured):
    _check_directories(directories)
    for directory, name, unused, initial in captured:
        current = receipt._entry(directory, name)
        _require(current is not None and source_runner._file_state(current) == initial)


@contextmanager
def snapshot_external_files(attempt: Path, public_summary: Path):
    """Yield immutable bytes, then reject any changed tree before returning."""
    body_error = None
    try:
        attempt, public = _paths(attempt, public_summary)
        with _directories(attempt, public) as directories:
            _check_directories(directories)
            captured = _capture_files(directories)
            snapshot = _read_files(captured)
            try:
                yield snapshot
            except BaseException as error:
                body_error = error
                raise
            _recheck_files(directories, captured)
    except Exception as error:
        if error is body_error:
            raise
        raise ExternalCompletionFileError("invalid_external_completion_files") from None
