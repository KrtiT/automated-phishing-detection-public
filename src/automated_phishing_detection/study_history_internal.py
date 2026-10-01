"""Restore internal science from pinned bytes without execution authority.

Original root authority, process custody, access rights and physical eligibility
remain separate. Retained replay artifacts feed unchanged scientific kernels;
no original source/model path, new fitting or transformer forward is required.
"""

from ._study_history_internal_context import authenticate
from ._study_history_internal_science import restore


class HistoricalInternalScienceError(ValueError):
    """A symbolic historical-science rejection without private diagnostics."""


def verify_historical_internal_science(
    payloads,
    *,
    expected_snapshot_sha256,
    expected_execution,
    expected_source_sha256,
    expected_attempt_directory,
):
    """Return immutable scientific data after exact original-context verification."""
    try:
        values, source, report = authenticate(
            payloads,
            expected_snapshot_sha256,
            expected_execution,
            expected_source_sha256,
            expected_attempt_directory,
        )
        return restore(values, expected_execution, source, report)
    except Exception:
        raise HistoricalInternalScienceError(
            "invalid_historical_internal_science"
        ) from None
