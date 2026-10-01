"""Closed and fully repinned prefix records still require the declared context."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from study_series_adoption_fixtures import digest
from study_series_prefix_fixtures import (
    api,
    candidates,
    frame,
    imported,
    manifests,
    prefix_case,
    series_case,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["candidates", "manifests", "prefix_case", "series_case"]


class DigestText(str):
    pass


def test_import_accounting_pin_requires_exact_digest_type(prefix_case):
    pin = prefix_case.source.profile["segment"]["predecessor_accounting_sha256"]
    with pytest.raises(api().SeriesPrefixError):
        imported(prefix_case, origin_accounting_sha256=DigestText(pin))


def rejoin(case, name, content):
    values = dict(case.payloads) | {name: content}
    intent = json.loads(values["segment/segment-intent.json"])
    intent["history_import_sha256"] = digest(values["segment/history-import.json"])
    values["segment/segment-intent.json"] = canonical_bytes(intent)
    selected = frame(
        case,
        **{
            "series_reservation_sha256": digest(values["series/reservation.json"]),
            "segment_reservation_sha256": digest(values["segment/reservation.json"]),
            "intent_sha256": digest(values["segment/segment-intent.json"]),
            "predecessor_sha256": digest(values["segment/history-import.json"]),
        },
    )
    return selected, tuple(values.items())


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "protocol",
        "series_reservation_sha256",
        "segment_reservation_sha256",
        "profile_sha256",
        "history_index_sha256",
        "origin_reservation_sha256",
        "origin_accounting_sha256",
        "accepted_inputs_sha256",
        "imported_prefix_length",
        "eligible_prefix_review_sha256",
        "exposure_record_sha256",
        "extra",
    ),
)
def test_rehashed_import_fields_cannot_change_context(prefix_case, field):
    value = json.loads(prefix_case.imported)
    value[field] = "0" * 64 if field.endswith("sha256") else True
    selected, values = rejoin(
        prefix_case, "segment/history-import.json", canonical_bytes(value)
    )
    with pytest.raises(api().SeriesPrefixError):
        api().validate_series_child_prefix(prefix_case.binding, selected, values)


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "protocol",
        "series_reservation_sha256",
        "segment_reservation_sha256",
        "profile_sha256",
        "envelope_sha256",
        "accepted_inputs_sha256",
        "segment_ordinal",
        "start_ordinal",
        "end_ordinal",
        "protective_deadlines_seconds",
        "extra",
    ),
)
def test_rehashed_intent_fields_cannot_change_context(prefix_case, field):
    value = json.loads(prefix_case.intent)
    value[field] = "0" * 64 if field.endswith("sha256") else True
    selected, values = rejoin(
        prefix_case, "segment/segment-intent.json", canonical_bytes(value)
    )
    with pytest.raises(api().SeriesPrefixError):
        api().validate_series_child_prefix(prefix_case.binding, selected, values)


@pytest.mark.parametrize(
    "name", ("series/reservation.json", "segment/reservation.json")
)
@pytest.mark.parametrize(
    "field", ("identity", "directory", "status", "schema_version", "extra")
)
def test_rehashed_receipts_still_match_exact_identity_and_directory(
    prefix_case, name, field
):
    from automated_phishing_detection import execution_receipt as receipt

    value = json.loads(dict(prefix_case.payloads)[name])
    value[field] = True
    selected, values = rejoin(prefix_case, name, receipt._json_bytes(value, "invented"))
    with pytest.raises(api().SeriesPrefixError):
        api().validate_series_child_prefix(prefix_case.binding, selected, values)


@pytest.mark.parametrize(
    "field,value",
    (
        ("origin_metadata_bytes", b"{}"),
        ("origin_metadata_bytes", bytearray(b"{}")),
        ("imported_prefix_length", True),
        ("imported_prefix_length", 71),
        ("imported_prefix_length", 73),
        ("origin_accounting_sha256", "0" * 64),
    ),
)
def test_import_builder_checks_declared_origin_count_and_accounting(
    prefix_case, field, value
):
    with pytest.raises(api().SeriesPrefixError):
        imported(prefix_case, **{field: value})


def test_intent_builder_joins_import_origin_hash_to_actual_metadata(prefix_case):
    value = json.loads(prefix_case.imported) | {"origin_metadata_sha256": "0" * 64}
    with pytest.raises(api().SeriesPrefixError):
        api().segment_intent_bytes(
            prefix_case.binding,
            prefix_case.series,
            prefix_case.segment,
            canonical_bytes(value),
            prefix_case.metadata,
        )


def test_child_prefix_origin_pin_is_a_declaration_not_absent_metadata_proof(
    prefix_case,
):
    value = json.loads(prefix_case.imported) | {"origin_metadata_sha256": "0" * 64}
    selected, values = rejoin(
        prefix_case, "segment/history-import.json", canonical_bytes(value)
    )
    assert (
        api().validate_series_child_prefix(prefix_case.binding, selected, values)
        is None
    )


@pytest.mark.parametrize("kind", ("namespace", "directory", "string_path", "digest"))
@pytest.mark.parametrize("role", ("series", "segment"))
def test_only_exact_typed_actual_reservation_declarations_join(prefix_case, kind, role):
    current = getattr(prefix_case, role)
    variants = {
        "namespace": SimpleNamespace(**vars(current)),
        "directory": replace(current, directory=Path("/invented/wrong")),
        "string_path": replace(current, directory=str(current.directory)),
        "digest": replace(current, reservation_sha256="0" * 64),
    }
    case = SimpleNamespace(**(vars(prefix_case) | {role: variants[kind]}))
    with pytest.raises(api().SeriesPrefixError):
        imported(case)
