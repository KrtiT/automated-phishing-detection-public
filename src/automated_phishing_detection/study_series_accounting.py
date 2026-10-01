"""Pure saved declarations, never live acceptance, custody or holder proof."""

from . import _study_series_accounting_records as records
from ._checkpoint_codec import canonical_bytes
from ._study_series_accounting_records import digest, schema


class SeriesAccountingError(ValueError):
    """Saved accounting contradicts its closed independently supplied context."""


def _segment_value(context, status, stage, ledger_bytes):
    imported = context["imported"]
    return dict(
        schema_version=1,
        protocol="study-series-segment-accounting-v1",
        series_reservation_sha256=context["pins"][0],
        segment_reservation_sha256=context["pins"][1],
        status=status,
        stage=stage,
        history_import_sha256=None if imported is None else digest(imported),
        ledger=None if ledger_bytes is None else schema.loads(ledger_bytes),
    )


def segment_accounting_bytes(
    public_binding,
    series_attempt,
    segment_attempt,
    *,
    status,
    stage,
    import_bytes=None,
    intent_bytes=None,
    metadata_bytes=None,
    ledger_bytes=None,
):
    try:
        context = records.context(
            public_binding,
            series_attempt,
            segment_attempt,
            (import_bytes, intent_bytes, metadata_bytes),
        )
        value = _segment_value(context, status, stage, ledger_bytes)
        return canonical_bytes(records.segment(value, context))
    except Exception:
        raise SeriesAccountingError("invalid_series_accounting") from None


def verify_segment_accounting(
    content,
    public_binding,
    series_attempt,
    segment_attempt,
    *,
    expected_sha256,
    import_bytes=None,
    intent_bytes=None,
    metadata_bytes=None,
):
    try:
        context = records.context(
            public_binding,
            series_attempt,
            segment_attempt,
            (import_bytes, intent_bytes, metadata_bytes),
        )
        return records.segment(schema.authenticated(content, expected_sha256), context)
    except Exception:
        raise SeriesAccountingError("invalid_series_accounting") from None


def _series(public, attempt, index, status, segment_attempt, content, buffers):
    selected = dict(
        ordinal=2, status="unattempted", reservation_sha256=None, accounting_sha256=None
    )
    if segment_attempt is not None:
        context = records.context(public, attempt, segment_attempt, buffers)
        value = records.segment(schema.loads(content), context)
        selected.update(
            status=value["status"],
            accounting_sha256=digest(content),
            reservation_sha256=segment_attempt.reservation_sha256,
        )
    else:
        schema.require(content is None and all(item is None for item in buffers))
    return records.series(public, attempt, index, status, selected)


def series_accounting_bytes(
    public_binding,
    series_attempt,
    history_index,
    *,
    status,
    segment_attempt=None,
    segment_accounting_bytes=None,
    import_bytes=None,
    intent_bytes=None,
    metadata_bytes=None,
):
    try:
        return canonical_bytes(
            _series(
                public_binding,
                series_attempt,
                history_index,
                status,
                segment_attempt,
                segment_accounting_bytes,
                (import_bytes, intent_bytes, metadata_bytes),
            )
        )
    except Exception:
        raise SeriesAccountingError("invalid_series_accounting") from None


def verify_series_accounting(
    content,
    public_binding,
    series_attempt,
    history_index,
    *,
    expected_sha256,
    segment_attempt=None,
    segment_accounting_bytes=None,
    import_bytes=None,
    intent_bytes=None,
    metadata_bytes=None,
):
    try:
        value = schema.authenticated(content, expected_sha256)
        expected = _series(
            public_binding,
            series_attempt,
            history_index,
            value["status"],
            segment_attempt,
            segment_accounting_bytes,
            (import_bytes, intent_bytes, metadata_bytes),
        )
        schema.same(value, expected)
        return value
    except Exception:
        raise SeriesAccountingError("invalid_series_accounting") from None
