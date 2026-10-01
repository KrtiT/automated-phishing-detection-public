"""Join imported inputs, full-matrix reductions and truthful root accounting."""

import json

from . import study_series_accounting as accounting
from . import study_series_prefix as prefix
from ._study_history_snapshot_records import digest
from .study_series_inputs import build_series_input_metadata
from .study_series_ledger import snapshot_series_ledger
from .study_series_reduction import reduce_series_science


def metadata(state):
    history = state.history
    return build_series_input_metadata(
        history.origin_metadata_bytes,
        state.public.profile_bytes,
        history.internal_snapshot,
        history.external_snapshot,
        expected_origin_sha256=digest(history.origin_metadata_bytes),
        expected_profile_sha256=state.public.profile_sha256,
        series_reservation_sha256=state.series_attempt.reservation_sha256,
        segment_reservation_sha256=state.segment_attempt.reservation_sha256,
    )


def imported(state):
    return prefix.history_import_bytes(
        state.public,
        state.series_attempt,
        state.segment_attempt,
        state.metadata_bytes,
        origin_metadata_bytes=state.history.origin_metadata_bytes,
        imported_prefix_length=len(state.history.historical_prefix),
        origin_accounting_sha256=state.profile["segment"][
            "predecessor_accounting_sha256"
        ],
    )


def reduce(state):
    history = state.history
    return reduce_series_science(
        history.historical_prefix,
        tuple(state.completed),
        selected_metadata_bytes=history.origin_metadata_bytes,
        current_context_bytes=state.metadata_bytes,
        profile_bytes=state.public.profile_bytes,
        expected_profile_sha256=state.public.profile_sha256,
        expected_selected_metadata_sha256=digest(history.origin_metadata_bytes),
        expected_current_context_sha256=digest(state.metadata_bytes),
        internal_snapshot=history.internal_snapshot,
        external_snapshot=history.external_snapshot,
    )


def account(state, status):
    if state.series_attempt is None:
        return
    buffers = dict(
        import_bytes=state.import_bytes,
        intent_bytes=state.intent_bytes,
        metadata_bytes=state.metadata_bytes,
    )
    selected = segment_account(state, status, buffers)
    state.series_accounting_bytes = accounting.series_accounting_bytes(
        state.public,
        state.series_attempt,
        state.history.index,
        status=status,
        **selected,
    )


def segment_account(state, status, buffers):
    if state.segment_attempt is None:
        return {}
    state.segment_accounting_bytes = accounting.segment_accounting_bytes(
        state.public,
        state.series_attempt,
        state.segment_attempt,
        status=status,
        stage=state.stage,
        ledger_bytes=None
        if state.ledger is None
        else snapshot_series_ledger(state.ledger),
        **buffers,
    )
    return dict(
        segment_attempt=state.segment_attempt,
        segment_accounting_bytes=state.segment_accounting_bytes,
        **buffers,
    )


def append_accounting(state, *, failure=False):
    for role in ("segment", "series"):
        writer, content = (
            getattr(state, f"{role}_writer"),
            getattr(state, f"{role}_accounting_bytes"),
        )
        if (
            writer is None
            or content is None
            or writer.closed
            or writer.failed
            or writer.held.publishing
        ):
            continue
        name = f"{role}-accounting.json"
        if name in writer.contents:
            if not failure or writer.contents[name] == content:
                continue
            name = "failure-accounting.json"
        if name not in writer.contents:
            writer.append(name, content)


def publication(state, role):
    writer = getattr(state, f"{role}_writer")
    content = getattr(state, f"{role}_accounting_bytes")
    outputs = dict(writer.contents) | {
        "operational-summary.json": state.reduced.operational_bytes,
        "study-evidence.json": state.reduced.study_bytes,
    }
    public = dict(
        schema_version=1,
        protocol=f"study-series-{role}-root-v1",
        status=f"{role}_evidence_published",
        profile_sha256=state.public.profile_sha256,
        envelope_sha256=state.public.envelope_sha256,
        reservation_sha256=writer.attempt.reservation_sha256,
        accounting_sha256=digest(content),
        private_sha256={name: digest(value) for name, value in outputs.items()},
        operational=json.loads(state.reduced.operational_bytes),
        study=json.loads(state.reduced.study_bytes),
        sessions=dict(
            historical_attempt=4,
            historical_prefix_length=len(state.history.historical_prefix),
            fresh_segment=2,
            fresh_start_ordinal=state.profile["segment"]["start_ordinal"],
            fresh_end_ordinal=125,
            single_session=False,
        ),
    )
    return writer.complete(outputs, public)
