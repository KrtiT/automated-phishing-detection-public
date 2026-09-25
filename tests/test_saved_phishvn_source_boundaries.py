"""Saved restoration preserves preparation policy without numerical access."""

from importlib import import_module

import pytest
from phishvn_source_fixtures import csv_bytes, decode, members, prepare, record
from test_saved_phishvn_source import encoded, module, restore, saved_sample

from automated_phishing_detection import phishvn_source

_FORBIDDEN_NUMERICAL = {
    "transformer_inference": ("load_transformer_cascade_bundle", "_load_model"),
    "length_inference": (
        "load_length_only_artifact",
        "score_length_only_authoritative",
    ),
    "secondary_tabular": ("load_secondary_model_bytes", "_fit", "_score_state"),
    "secondary_transformer": (
        "load_secondary_transformer_bytes",
        "fit_secondary_transformer",
        "score_secondary_transformer_urls",
    ),
    "external_primary": ("score_external_primary",),
    "external_producer": ("produce_external_evidence",),
}


def test_restoration_never_loads_fits_or_scores_models(monkeypatch):
    original, source, summary, pins = saved_sample()
    api = module()
    owners = {
        name: import_module(f"automated_phishing_detection.{name}")
        for name in _FORBIDDEN_NUMERICAL
    }

    def forbidden(*args, **kwargs):
        pytest.fail("saved restoration attempted model loading, fitting, or scoring")

    for name, functions in _FORBIDDEN_NUMERICAL.items():
        for function in functions:
            monkeypatch.setattr(owners[name], function, forbidden)
    assert (
        api.restore_phishvn_source(encoded(source), encoded(summary), pins=pins)
        == original
    )


def test_restoration_uses_existing_mapping_once_with_only_split_tables(monkeypatch):
    original, source, summary, pins = saved_sample()
    decode_rows = phishvn_source._decode_rows
    observed = []

    def observe(tables):
        observed.append(tables)
        return decode_rows(tables)

    monkeypatch.setattr(phishvn_source, "_decode_rows", observe)
    assert restore(source, summary, pins) == original
    assert len(observed) == 1
    assert set(observed[0]) == {
        "data/splits/url_train.csv",
        "data/splits/url_val.csv",
        "data/splits/url_test.csv",
    }


def test_cross_split_domain_quarantine_keeps_all_original_split_rows():
    records = [
        record(split, split, url="https://shared.example/path")
        for split in ("train", "val", "test")
    ]
    original, source, summary, pins = saved_sample(records)
    restored = restore(source, summary, pins)
    prepared = prepare(restored)
    assert prepared == prepare(original)
    assert len(restored.rows) == len(prepared.quarantine) == 3
    assert all(
        "domain_crosses_published_splits" in row.reason_codes
        for row in prepared.quarantine
    )


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"url": "bare.example"}, "invalid_or_missing_url"),
        ({"id": ""}, "missing_or_invalid_published_id"),
        ({"split": "unexpected"}, "invalid_published_split"),
    ],
)
def test_restoration_does_not_repair_invalid_publisher_values(changes, reason):
    contents = members([])
    original = record("private-canary", **changes)
    for name in ("data/dataset_url.csv", "data/splits/url_test.csv"):
        contents[name] = csv_bytes([original])
    decoded = decode(contents)
    pins = phishvn_source.PhishVNSourcePins(**decoded.public_summary["input_archive"])
    restored = module().restore_phishvn_source(
        decoded.private_outputs["publisher-source.json"],
        encoded(decoded.public_summary),
        pins=pins,
    )
    assert restored == decoded
    assert reason in prepare(restored).quarantine[0].reason_codes
