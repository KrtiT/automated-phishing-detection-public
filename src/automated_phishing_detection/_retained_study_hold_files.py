"""Hold exact historical root files without accepting source or process authority."""

from contextlib import contextmanager
from functools import partial
from pathlib import Path

from . import _study_preparation_files as files
from . import _study_root_files as storage
from ._exception_cleanup import CleanupStack, preserve_cleanup
from ._study_root_records import HOLD_ORDER, StudyRootSnapshot, require

_ROOT_NAMES = ("reservation.json", *HOLD_ORDER, "finalize.claim", "outcome.json")


def _check(root, evidence, parent, states):
    storage.inventory(root, (*_ROOT_NAMES, "evidence"))
    storage.inventory(evidence, HOLD_ORDER)
    parent.check()
    for directory, name, unused, mode, initial in states:
        require(storage.capture(directory, name, mode) == initial)


def _capture(root, evidence, parent, public_name):
    members = (
        (root, _ROOT_NAMES, "attempt/", 0o600),
        (evidence, HOLD_ORDER, "attempt/evidence/", 0o600),
        (parent, (public_name,), "", 0o644),
    )
    return tuple(
        (
            directory,
            name,
            prefix + name if prefix else "public-summary.json",
            mode,
            storage.capture(directory, name, mode),
        )
        for directory, names, prefix, mode in members
        for name in names
    )


@contextmanager
def hold_snapshot(profile, reservation_sha256):
    root_path = Path(profile["paths"]["attempt"])
    public = Path(profile["paths"]["public-summary"])
    require(not public.is_relative_to(root_path))
    with CleanupStack() as cleanup:
        root = files.enter_directory(cleanup, root_path)
        evidence = files.enter_directory(cleanup, root_path / "evidence")
        parent = files.enter_directory(cleanup, public.parent)
        states = files.deferred(_capture, root, evidence, parent, public.name)
        check = partial(files.deferred, _check, root, evidence, parent, states)
        check()
        with preserve_cleanup(check):
            payloads = tuple(
                (logical, files.deferred(storage.read_file, directory, name, initial))
                for directory, name, logical, unused, initial in states
            )
            check()
            yield StudyRootSnapshot(reservation_sha256, payloads)
