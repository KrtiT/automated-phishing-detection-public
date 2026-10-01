"""Pure series root declarations; historical science and live authority are upstream.

Caller-constructed bindings, attempts and frames confer no provenance. The child
prefix authenticates declarations, not metadata content absent from these files.
"""

from . import _study_series_prefix_records as records
from . import _study_series_prefix_validation as validation
from ._checkpoint_codec import canonical_bytes


class SeriesPrefixError(ValueError):
    """Series prefix declarations differ from their exact supplied context."""


def series_identity(public_binding):
    try:
        return records.series(records.profile(public_binding), public_binding)
    except Exception:
        raise SeriesPrefixError("invalid_series_prefix") from None


def segment_identity(public_binding, series_attempt):
    try:
        value = records.profile(public_binding)
        pin = records.attempt(
            series_attempt,
            value["paths"]["series_attempt"],
            records.series(value, public_binding),
        )
        return records.segment(value, public_binding, pin)
    except Exception:
        raise SeriesPrefixError("invalid_series_prefix") from None


def history_import_bytes(
    public_binding,
    series_attempt,
    segment_attempt,
    metadata_bytes,
    *,
    origin_metadata_bytes,
    imported_prefix_length,
    origin_accounting_sha256,
):
    try:
        return validation.history(
            public_binding,
            series_attempt,
            segment_attempt,
            (
                metadata_bytes,
                origin_metadata_bytes,
                imported_prefix_length,
                origin_accounting_sha256,
            ),
        )
    except Exception:
        raise SeriesPrefixError("invalid_series_prefix") from None


def segment_intent_bytes(
    public_binding, series_attempt, segment_attempt, import_bytes, metadata_bytes
):
    try:
        value = records.profile(public_binding)
        pins = records.attempts(value, public_binding, series_attempt, segment_attempt)
        metadata = records.metadata(
            metadata_bytes, value, public_binding.profile_sha256, pins
        )
        metadata_pin = records.digest(metadata_bytes)
        records.import_record(
            import_bytes,
            value,
            public_binding,
            pins,
            metadata_pin,
            records.digest(canonical_bytes(metadata["origin"])),
        )
        return canonical_bytes(
            records.intent(
                value, public_binding, pins, records.digest(import_bytes), metadata_pin
            )
        )
    except Exception:
        raise SeriesPrefixError("invalid_series_prefix") from None


def validate_series_child_prefix(public_binding, frame, exact_four_payloads):
    try:
        validation.validate(public_binding, frame, exact_four_payloads)
    except Exception:
        raise SeriesPrefixError("invalid_series_prefix") from None
