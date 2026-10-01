"""Closed series prefix declarations, not historical verification or access."""

from pathlib import Path

from . import _operational_input_schema as schema
from . import _study_series_input_context as context
from . import execution_receipt as receipt
from ._checkpoint_codec import canonical_bytes
from ._study_history_snapshot_records import digest
from ._study_series_child_input_context import _origin
from .study_series_adoption import validate_series_adoption_header
from .study_series_execution import SeriesPublicBinding


def profile(binding):
    schema.require(type(binding) is SeriesPublicBinding)
    validate_series_adoption_header(
        binding.policy_bytes,
        binding.profile_bytes,
        binding.envelope_bytes,
        expected_profile_sha256=binding.profile_sha256,
        expected_envelope_sha256=binding.envelope_sha256,
    )
    return schema.loads(binding.profile_bytes)


def series(value, binding):
    return {
        "kind": "study_series",
        "protocol": "study-series-root-v1",
        "series_id": value["series_id"],
        "profile_sha256": binding.profile_sha256,
        "envelope_sha256": binding.envelope_sha256,
        "policy_sha256": value["policy_sha256"],
        "history_index_sha256": value["history"]["index_sha256"],
        "origin_reservation_sha256": value["origin"]["root_reservation_sha256"],
        "execution": context._execution(value),
    }


def segment(value, binding, series_pin):
    selected = value["segment"]
    return {
        "kind": "study_series_segment",
        "protocol": "study-series-segment-v1",
        "series_reservation_sha256": series_pin,
        "profile_sha256": binding.profile_sha256,
        "envelope_sha256": binding.envelope_sha256,
        "history_index_sha256": value["history"]["index_sha256"],
        "origin_reservation_sha256": value["origin"]["root_reservation_sha256"],
        "segment_ordinal": selected["ordinal"],
        "session_id": selected["session_id"],
        "start_ordinal": selected["start_ordinal"],
        "end_ordinal": selected["end_ordinal"],
    }


def reservation(directory, identity):
    return receipt._json_bytes(
        dict(
            schema_version=1, status="reserved", directory=directory, identity=identity
        ),
        "series_prefix_reservation",
    )


def attempt(value, directory, identity):
    schema.require(type(value) is receipt.Attempt)
    schema.require(
        isinstance(value.directory, Path) and value.directory == Path(directory)
    )
    schema.digest(value.reservation_sha256)
    schema.require(digest(reservation(directory, identity)) == value.reservation_sha256)
    return value.reservation_sha256


def attempts(value, binding, series_attempt, segment_attempt):
    series_pin = attempt(
        series_attempt, value["paths"]["series_attempt"], series(value, binding)
    )
    segment_pin = attempt(
        segment_attempt,
        value["paths"]["segment_attempt"],
        segment(value, binding, series_pin),
    )
    context._reservations(
        series_pin, segment_pin, value["origin"]["root_reservation_sha256"]
    )
    return series_pin, segment_pin


def metadata(content, value, profile_pin, pins):
    selected = schema.loads(content)
    schema.keys(selected, context.FIELDS)
    schema.require(
        type(selected["schema_version"]) is int and selected["schema_version"] == 1
    )
    schema.require(selected["kind"] == context.KIND)
    _origin(selected["origin"], value)
    context._current(selected, value, profile_pin)
    schema.require(selected["series_reservation_sha256"] == pins[0])
    schema.require(selected["root_reservation_sha256"] == pins[1])
    return selected


def history(value, binding, pins, metadata_pin, origin_pin):
    schema.digest(metadata_pin)
    schema.digest(origin_pin)
    return {
        "schema_version": 1,
        "protocol": "study-series-history-import-v1",
        "series_reservation_sha256": pins[0],
        "segment_reservation_sha256": pins[1],
        "profile_sha256": binding.profile_sha256,
        "history_index_sha256": value["history"]["index_sha256"],
        "origin_reservation_sha256": value["origin"]["root_reservation_sha256"],
        "origin_accounting_sha256": value["segment"]["predecessor_accounting_sha256"],
        "origin_metadata_sha256": origin_pin,
        "accepted_inputs_sha256": metadata_pin,
        "imported_prefix_length": value["segment"]["start_ordinal"] - 1,
        "eligible_prefix_review_sha256": value["history"][
            "eligible_prefix_review_sha256"
        ],
        "exposure_record_sha256": value["history"]["exposure_record_sha256"],
    }


def intent(value, binding, pins, import_pin, metadata_pin):
    from ._study_execution_policy import DEADLINES

    return {
        "schema_version": 1,
        "protocol": "study-series-segment-intent-v1",
        "series_reservation_sha256": pins[0],
        "segment_reservation_sha256": pins[1],
        "profile_sha256": binding.profile_sha256,
        "envelope_sha256": binding.envelope_sha256,
        "history_import_sha256": import_pin,
        "accepted_inputs_sha256": metadata_pin,
        "segment_ordinal": value["segment"]["ordinal"],
        "start_ordinal": value["segment"]["start_ordinal"],
        "end_ordinal": value["segment"]["end_ordinal"],
        "protective_deadlines_seconds": dict(DEADLINES),
    }


def import_record(content, value, binding, pins, metadata_pin, origin_pin=None):
    imported = schema.loads(content)
    expected = history(
        value,
        binding,
        pins,
        metadata_pin,
        imported["origin_metadata_sha256"] if origin_pin is None else origin_pin,
    )
    schema.require(canonical_bytes(expected) == content)
    return imported
