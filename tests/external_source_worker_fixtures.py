"""Run the real external worker on invented sources with fixture model bindings.

Only source/producer/publication and owned-process composition are exercised.
The internal handoff and model/runtime binding remain explicitly invented seams.
"""

from contextlib import contextmanager
from pathlib import Path

import pytest
from test_external_source_runner import configured_case, session_owner

from automated_phishing_detection import external_source_runner
from automated_phishing_detection.internal_handoff_transport import (
    read_internal_handoff_transport,
)


@contextmanager
def teardown_failure(case, *arguments):
    with session_owner(case, *arguments) as session:
        yield session
    raise ValueError("invented teardown failure")


def run_child(root, transport, expected_handoff, count, mode):
    """Produce fresh external files in this child, not a preinstalled marker."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        case = configured_case(external_source_runner, Path(root), monkeypatch, count)
        handoff = read_internal_handoff_transport(
            Path(transport), expected_handoff_sha256=expected_handoff
        )
        if mode == "teardown":
            monkeypatch.setattr(
                external_source_runner,
                "open_bound_external_session",
                lambda *arguments: teardown_failure(case, *arguments),
            )
        external_source_runner._run_bound_external(
            case.binding, case.paths, handoff=handoff
        )
        if mode == "nonzero":
            raise SystemExit(17)
