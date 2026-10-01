"""Independent frame pins precede parsing; no IO or scientific authority arises."""

import json
import os
from dataclasses import replace

import pytest
from study_series_prefix_fixtures import (
    api,
    candidates,
    frame,
    imported,
    manifests,
    prefix_case,
    series_case,
)
from test_study_history_external_boundaries import forbid_live

from automated_phishing_detection import _study_series_prefix_records as records
from automated_phishing_detection import execution_preflight, study_series_inputs
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["candidates", "manifests", "prefix_case", "series_case"]


def forbidden(*arguments, **keywords):
    pytest.fail("prefix reached IO, source science or parsing before authentication")


@pytest.mark.parametrize(
    "field",
    (
        "profile_sha256",
        "envelope_sha256",
        "series_reservation_sha256",
        "segment_reservation_sha256",
        "intent_sha256",
        "predecessor_sha256",
    ),
)
def test_independent_frame_pins_authenticate_before_header_parse(
    prefix_case, monkeypatch, field
):
    monkeypatch.setattr(records, "profile", forbidden)
    with pytest.raises(api().SeriesPrefixError, match="^invalid_series_prefix$"):
        api().validate_series_child_prefix(
            prefix_case.binding,
            frame(prefix_case, **{field: "0" * 64}),
            prefix_case.payloads,
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("origin_reservation_sha256", "0" * 64),
        ("history_index_sha256", "0" * 64),
        ("accepted_inputs_sha256", "0" * 64),
        ("cell_ordinal", 2),
    ),
)
def test_other_frame_scalar_joins_remain_required(prefix_case, field, value):
    with pytest.raises(api().SeriesPrefixError):
        api().validate_series_child_prefix(
            prefix_case.binding,
            frame(prefix_case, **{field: value}),
            prefix_case.payloads,
        )


@pytest.mark.parametrize(
    "kind", ("missing", "extra", "duplicate", "list", "member", "mutable")
)
def test_prefix_inventory_is_exact_four_immutable_unique_members(prefix_case, kind):
    values = prefix_case.payloads
    variants = {
        "missing": values[:-1],
        "extra": values + (("extra", b"x"),),
        "duplicate": values + (values[0],),
        "list": list(values),
        "member": (list(values[0]), *values[1:]),
        "mutable": ((values[0][0], bytearray(values[0][1])), *values[1:]),
    }
    with pytest.raises(api().SeriesPrefixError):
        api().validate_series_child_prefix(
            prefix_case.binding, frame(prefix_case), variants[kind]
        )


@pytest.mark.parametrize("name", ("policy_bytes", "profile_bytes", "envelope_bytes"))
def test_binding_header_cannot_be_replaced_by_constructed_carrier(prefix_case, name):
    changed = replace(
        prefix_case.binding, **{name: b"private malformed invented bytes"}
    )
    with pytest.raises(api().SeriesPrefixError, match="^invalid_series_prefix$"):
        api().series_identity(changed)


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "kind",
        "root_reservation_sha256",
        "series_reservation_sha256",
        "profile_sha256",
        "history_index_sha256",
        "execution",
        "primary",
        "origin",
        "extra",
    ),
)
def test_metadata_scalar_context_is_revalidated_without_sources(prefix_case, field):
    from types import SimpleNamespace

    value = json.loads(prefix_case.metadata)
    value[field] = "0" * 64 if field.endswith("sha256") else True
    case = SimpleNamespace(**(vars(prefix_case) | {"metadata": canonical_bytes(value)}))
    with pytest.raises(api().SeriesPrefixError):
        imported(case)


def test_pure_prefix_never_reads_or_recreates_live_or_scientific_state(
    prefix_case, monkeypatch
):
    forbid_live(monkeypatch)
    for target, name in (
        (os, "open"),
        (execution_preflight, "bind_execution"),
        (execution_preflight, "recheck_binding"),
        (study_series_inputs, "restore_series_cell_inputs"),
        (records.context, "_origin"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    assert imported(prefix_case) == prefix_case.imported
    assert (
        api().validate_series_child_prefix(
            prefix_case.binding, frame(prefix_case), prefix_case.payloads
        )
        is None
    )


@pytest.mark.parametrize(
    "name",
    (
        "series/reservation.json",
        "segment/reservation.json",
        "segment/segment-intent.json",
        "segment/history-import.json",
    ),
)
def test_stale_private_bytes_never_leak_in_error(prefix_case, name):
    values = dict(prefix_case.payloads) | {name: b"private invented secret"}
    with pytest.raises(api().SeriesPrefixError) as caught:
        api().validate_series_child_prefix(
            prefix_case.binding, frame(prefix_case), tuple(values.items())
        )
    assert str(caught.value) == "invalid_series_prefix"
    assert caught.value.__suppress_context__ is True
