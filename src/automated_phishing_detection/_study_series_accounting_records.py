"""Closed segment context and exact sealed history references, without IO."""

import base64

from . import _operational_input_schema as schema
from . import _study_series_prefix_records as prefix
from ._checkpoint_codec import canonical_bytes
from ._study_history_snapshot_records import digest
from .study_series_history_index import (
    SeriesHistoryIndex,
    validate_series_history_index,
)

STAGES = (
    "validation",
    "output_preflight",
    "reservation",
    "input_retention",
    "cell_execution",
    "segment_publication",
    "segment_finalization",
)
SEGMENT_FIELDS = frozenset(
    (
        "schema_version",
        "protocol",
        "series_reservation_sha256",
        "segment_reservation_sha256",
        "status",
        "stage",
        "history_import_sha256",
        "ledger",
    )
)


def decoded(value):
    schema.require(type(value) is str)
    result = base64.b64decode(value, validate=True)
    schema.require(base64.b64encode(result).decode("ascii") == value)
    return result


def _intent(intent, imported, metadata, profile, public, pins):
    if intent is not None:
        schema.require(imported is not None and metadata is not None)
        expected = prefix.intent(
            profile, public, pins, digest(imported), digest(metadata)
        )
        schema.require(intent == canonical_bytes(expected))


def context(public, series, segment, buffers):
    profile = prefix.profile(public)
    pins = prefix.attempts(profile, public, series, segment)
    imported, intent, metadata = buffers
    if metadata is not None:
        selected = prefix.metadata(metadata, profile, public.profile_sha256, pins)
    if imported is not None:
        schema.require(metadata is not None)
        prefix.import_record(
            imported,
            profile,
            public,
            pins,
            digest(metadata),
            digest(canonical_bytes(selected["origin"])),
        )
    _intent(intent, imported, metadata, profile, public, pins)
    return dict(
        public=public,
        profile=profile,
        pins=pins,
        imported=imported,
        intent=intent,
        metadata=metadata,
    )


def _ledger(value, context):
    from ._study_series_accounting_ledger import validate

    if value["ledger"] is None:
        schema.require(value["status"] == "failed" and value["stage"] in STAGES[:4])
        return
    schema.require(
        all(context[name] is not None for name in ("imported", "intent", "metadata"))
    )
    validate(value["ledger"], context, complete=value["status"] == "complete")
    if value["stage"] in STAGES[:4]:
        schema.require(
            not value["ledger"]["admissions"]
            and all(
                slot["status"] == "unattempted" for slot in value["ledger"]["cells"]
            )
        )


def segment(value, context):
    schema.keys(value, SEGMENT_FIELDS)
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(value["protocol"] == "study-series-segment-accounting-v1")
    schema.require(value["status"] in ("complete", "failed"))
    schema.require(type(value["stage"]) is str and value["stage"] in STAGES)
    schema.same(
        (value["series_reservation_sha256"], value["segment_reservation_sha256"]),
        context["pins"],
    )
    imported = context["imported"]
    schema.require(
        value["history_import_sha256"]
        == (None if imported is None else digest(imported))
    )
    _ledger(value, context)
    if value["status"] == "complete":
        schema.require(value["stage"] == "segment_finalization")
    return value


def _history(index, public, profile):
    schema.require(type(index) is SeriesHistoryIndex)
    verified = validate_series_history_index(
        index.index_bytes,
        public.profile_bytes,
        expected_index_sha256=profile["history"]["index_sha256"],
        expected_profile_sha256=public.profile_sha256,
    )
    schema.require(index == verified)
    return schema.loads(verified.index_bytes)


def prior_attempts(history):
    return [
        dict(
            ordinal=member["ordinal"],
            reservation_sha256=member["root_reservation_sha256"],
            inventory_sha256=member["inventory"]["sha256"],
            disposition=member["disposition"],
        )
        for member in history["attempts"]
    ]


def series(public, attempt, index, status, selected):
    profile = prefix.profile(public)
    pin = prefix.attempt(
        attempt, profile["paths"]["series_attempt"], prefix.series(profile, public)
    )
    history = _history(index, public, profile)
    schema.require(status in ("complete", "failed"))
    schema.require(status != "complete" or selected["status"] == "complete")
    prior = prior_attempts(history)
    hold = history["original_hold"]
    return dict(
        schema_version=1,
        protocol="study-series-accounting-v1",
        series_reservation_sha256=pin,
        profile_sha256=public.profile_sha256,
        history_index_sha256=index.index_sha256,
        original_hold=dict(
            reservation_sha256=hold["reservation_sha256"],
            inventory_sha256=hold["inventory"]["sha256"],
        ),
        prior_attempts=prior,
        segment=selected,
        status=status,
    )
