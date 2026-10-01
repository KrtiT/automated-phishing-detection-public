"""Reserve and retain the two roots before issuing any fresh cell admission."""

from pathlib import Path

from . import _study_preparation_files as files
from . import _study_series_runner_records as records
from . import study_series_prefix as prefix
from ._study_run_body import enter
from ._study_series_runner_storage import hold_root
from .operational_input_transport import retain_operational_root_inputs
from .study_series_ledger import SeriesAdmissionLedger


def reserve(state, cleanup):
    paths = state.profile["paths"]
    state.stage = "reservation"
    identity = prefix.series_identity(state.public)
    files.deferred(state.reserve_series, Path(paths["series_attempt"]), identity)
    state.series_writer = enter(
        cleanup,
        hold_root(
            state.series_attempt,
            Path(paths["series_public_summary"]),
            identity,
        ),
    )
    identity = prefix.segment_identity(state.public, state.series_attempt)
    files.deferred(state.reserve_segment, Path(paths["segment_attempt"]), identity)
    state.segment_writer = enter(
        cleanup,
        hold_root(
            state.segment_attempt,
            Path(paths["segment_public_summary"]),
            identity,
        ),
    )


def prepare(state, cleanup):
    state.stage = "input_retention"
    state.metadata_bytes = records.metadata(state)
    state.import_bytes = records.imported(state)
    state.intent_bytes = prefix.segment_intent_bytes(
        state.public,
        state.series_attempt,
        state.segment_attempt,
        state.import_bytes,
        state.metadata_bytes,
    )
    state.segment_writer.append("history-import.json", state.import_bytes)
    state.segment_writer.append("segment-intent.json", state.intent_bytes)
    enter(
        cleanup,
        retain_operational_root_inputs(
            Path(state.profile["paths"]["historical_inputs_dir"]),
            accepted_inputs=state.metadata_bytes,
        ),
    )
    state.ledger = SeriesAdmissionLedger(
        state.public,
        state.series_attempt,
        state.segment_attempt,
        state.intent_bytes,
        state.import_bytes,
        state.metadata_bytes,
    )
