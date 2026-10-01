"""Authenticate four declared prefix records before any content interpretation."""

from . import _operational_input_schema as schema
from . import _study_series_prefix_records as records
from ._checkpoint_codec import canonical_bytes
from ._operational_cell_publication_records import inventory
from ._study_series_admission import validate_series_frame
from .study_series_execution import SeriesPublicBinding

NAMES = (
    "series/reservation.json",
    "segment/reservation.json",
    "segment/segment-intent.json",
    "segment/history-import.json",
)


def history(binding, series_attempt, segment_attempt, buffers):
    content, origin, prefix_length, accounting_pin = buffers
    value = records.profile(binding)
    pins = records.attempts(value, binding, series_attempt, segment_attempt)
    metadata = records.metadata(content, value, binding.profile_sha256, pins)
    schema.require(
        type(origin) is bytes and canonical_bytes(metadata["origin"]) == origin
    )
    schema.require(type(prefix_length) is int)
    schema.require(prefix_length == value["segment"]["start_ordinal"] - 1)
    schema.digest(accounting_pin)
    schema.require(accounting_pin == value["segment"]["predecessor_accounting_sha256"])
    expected = records.history(
        value, binding, pins, records.digest(content), records.digest(origin)
    )
    return canonical_bytes(expected)


def authenticate(binding, frame, payloads):
    validate_series_frame(frame)
    schema.require(type(binding) is SeriesPublicBinding)
    contents = inventory(payloads, NAMES)
    pins = (
        frame.series_reservation_sha256,
        frame.segment_reservation_sha256,
        frame.intent_sha256,
        frame.predecessor_sha256,
    )
    for name, pin in zip(NAMES, pins, strict=True):
        schema.require(records.digest(contents[name]) == pin)
    for content, pin in (
        (binding.profile_bytes, frame.profile_sha256),
        (binding.envelope_bytes, frame.envelope_sha256),
    ):
        schema.require(type(content) is bytes and records.digest(content) == pin)
    return contents, pins[:2]


def reservations(binding, value, pins, contents):
    expected = (
        records.series(value, binding),
        records.segment(value, binding, pins[0]),
    )
    for name, path, identity in zip(
        NAMES[:2], ("series_attempt", "segment_attempt"), expected, strict=True
    ):
        schema.require(
            contents[name] == records.reservation(value["paths"][path], identity)
        )
    records.context._reservations(
        pins[0], pins[1], value["origin"]["root_reservation_sha256"]
    )


def validate(binding, frame, payloads):
    contents, pins = authenticate(binding, frame, payloads)
    value = records.profile(binding)
    reservations(binding, value, pins, contents)
    schema.require(
        frame.origin_reservation_sha256 == value["origin"]["root_reservation_sha256"]
    )
    schema.require(frame.history_index_sha256 == value["history"]["index_sha256"])
    schema.require(frame.segment_ordinal == value["segment"]["ordinal"])
    schema.require(
        value["segment"]["start_ordinal"]
        <= frame.cell_ordinal
        <= value["segment"]["end_ordinal"]
    )
    records.import_record(
        contents[NAMES[3]], value, binding, pins, frame.accepted_inputs_sha256
    )
    expected = records.intent(
        value, binding, pins, frame.predecessor_sha256, frame.accepted_inputs_sha256
    )
    schema.require(contents[NAMES[2]] == canonical_bytes(expected))
