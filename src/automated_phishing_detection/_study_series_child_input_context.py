"""Scalar child-input joins; source science and live admission remain upstream."""

from hashlib import sha256

from . import _operational_input_schema as schema
from . import _study_series_input_context as context
from ._study_series_admission import validate_series_frame


def _origin(value, profile):
    schema.validate_metadata(value)
    schema.require(
        value["root_reservation_sha256"] == profile["origin"]["root_reservation_sha256"]
    )
    schema.same(
        value["execution"],
        context._execution(profile) | {"revision": profile["origin"]["revision"]},
    )
    schema.require(
        value["operational_profile_sha256"]
        == profile["components"]["original_operational"]
    )
    schema.require(
        value["external"]["execution"]["source_profile_sha256"]
        == profile["components"]["original_external"]
    )
    context._preparation(value, profile)
    context._source_pins(value, profile)


def _frame(value, profile, frame):
    schema.require(
        frame.series_reservation_sha256 == value["series_reservation_sha256"]
    )
    schema.require(frame.segment_reservation_sha256 == value["root_reservation_sha256"])
    schema.require(
        frame.origin_reservation_sha256 == value["origin"]["root_reservation_sha256"]
    )
    schema.require(frame.history_index_sha256 == value["history_index_sha256"])
    schema.require(frame.segment_ordinal == profile["segment"]["ordinal"])
    schema.require(
        profile["segment"]["start_ordinal"]
        <= frame.cell_ordinal
        <= profile["segment"]["end_ordinal"]
    )


def authenticate(metadata_bytes, profile_bytes, frame):
    validate_series_frame(frame)
    for content, expected in (
        (metadata_bytes, frame.accepted_inputs_sha256),
        (profile_bytes, frame.profile_sha256),
    ):
        schema.require(
            type(content) is bytes and sha256(content).hexdigest() == expected
        )
    profile = context._profile(profile_bytes, frame.profile_sha256)
    value = schema.authenticated(metadata_bytes, frame.accepted_inputs_sha256)
    schema.keys(value, context.FIELDS)
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(value["kind"] == context.KIND)
    _origin(value["origin"], profile)
    context._reservations(
        value["series_reservation_sha256"],
        value["root_reservation_sha256"],
        value["origin"]["root_reservation_sha256"],
    )
    context._current(value, profile, frame.profile_sha256)
    _frame(value, profile, frame)
    return value
