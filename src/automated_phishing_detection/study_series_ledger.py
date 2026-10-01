"""One parent's ordered fresh suffix; neither records nor construction grant access."""

from . import _study_series_ledger_acceptance as acceptance
from . import _study_series_ledger_context as context
from . import _study_series_ledger_issuance as issuance
from . import _study_series_ledger_records as records
from ._study_series_ledger_records import SeriesCellStop

__all__ = [
    "SeriesAdmissionLedger",
    "SeriesCellStop",
    "SeriesLedgerError",
    "snapshot_series_ledger",
]


class SeriesLedgerError(ValueError):
    """An ordered parent transition cannot be authenticated."""


def _transition(ledger, operation, *arguments):
    try:
        context.parent(ledger)
        return operation(ledger, *arguments)
    except BaseException as error:
        ledger._closed = True
        if not isinstance(error, Exception):
            raise
        raise SeriesLedgerError("invalid_series_ledger") from None


class SeriesAdmissionLedger:
    """Callbacks retain current owned facts; holder exits remain the caller's duty."""

    def __init__(
        self,
        public_binding,
        series_attempt,
        segment_attempt,
        intent_bytes,
        import_bytes,
        metadata_bytes,
    ):
        try:
            self.__dict__.update(
                context.initialize(
                    public_binding,
                    series_attempt,
                    segment_attempt,
                    intent_bytes,
                    import_bytes,
                    metadata_bytes,
                )
            )
        except Exception:
            raise SeriesLedgerError("invalid_series_ledger") from None

    def start_cell(self, cell, attempt, descriptor_bytes, binding_bytes):
        return _transition(
            self, context.start, cell, attempt, descriptor_bytes, binding_bytes
        )

    def issue(self, role, exact_command):
        return _transition(self, issuance.issue, role, exact_command)

    def accept_cell(self, candidate, observation, pair_intent_bytes):
        return _transition(
            self, acceptance.accept, candidate, observation, pair_intent_bytes
        )

    def stop_cell(self, actual_failure_state):
        return _transition(self, records.stop, actual_failure_state)


def snapshot_series_ledger(ledger):
    """Freeze factual suffix accounting after acceptance or an explicitly recorded stop."""
    try:
        context.schema.require(type(ledger) is SeriesAdmissionLedger)
        context.parent(ledger)
        return records.snapshot(ledger)
    except Exception:
        raise SeriesLedgerError("invalid_series_ledger") from None
