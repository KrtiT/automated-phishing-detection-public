"""Rehashed completion receipts still require the original closed projections."""

import json

import pytest
from study_history_cell_fixtures import (
    api,
    candidates,
    case,
    digest,
    hashes,
    history,
    manifests,
    restore,
)

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["candidates", "case", "history", "manifests"]


@pytest.mark.parametrize(
    "name,field,value",
    (
        ("attempt/finalize.claim", "operation", "failure"),
        ("attempt/finalize.claim", "reservation_sha256", "0" * 64),
        ("attempt/finalize.claim", "schema_version", True),
        ("attempt/outcome.json", "status", "failed"),
        ("attempt/outcome.json", "private_sha256", {}),
        ("attempt/outcome.json", "public_summary_sha256", "0" * 64),
        ("public-summary.json", "status", "complete"),
        ("public-summary.json", "protocol", "different"),
        ("public-summary.json", "schema_version", True),
        ("public-summary.json", "private_sha256", {}),
        ("public-summary.json", "execution", {}),
        ("public-summary.json", "cell", {}),
        ("public-summary.json", "summary", {}),
    ),
)
def test_rehashed_receipt_lies_are_rejected(history, name, field, value):
    values = dict(history.values)
    document = json.loads(values[name]) | {field: value}
    values[name] = receipt._json_bytes(document, "fixture")
    if name == "public-summary.json":
        outcome = json.loads(values["attempt/outcome.json"])
        outcome["public_summary_sha256"] = digest(values[name])
        values["attempt/outcome.json"] = receipt._json_bytes(outcome, "fixture")
    with pytest.raises(api().HistoricalCellScienceError):
        restore(
            history,
            payloads=tuple(values.items()),
            expected_snapshot_sha256=hashes(values),
        )


@pytest.mark.parametrize(
    "name",
    (
        "attempt/reservation.json",
        "attempt/finalize.claim",
        "attempt/outcome.json",
        "public-summary.json",
    ),
)
def test_receipts_require_original_codec_and_closed_fields(history, name):
    value = json.loads(history.values[name])
    for content in (
        canonical_bytes(value),
        receipt._json_bytes(value | {"extra": 1}, "fixture"),
    ):
        values = history.values | {name: content}
        with pytest.raises(api().HistoricalCellScienceError):
            restore(
                history,
                payloads=tuple(values.items()),
                expected_snapshot_sha256=hashes(values),
            )


@pytest.mark.parametrize(
    "directory",
    (
        "/invented/other",
        "relative",
        "/invented/../cell",
        "/invented/cell/",
        "/invented//cell",
        "/invented/cell\0",
        None,
    ),
)
def test_independent_original_directory_is_exact(history, directory):
    with pytest.raises(api().HistoricalCellScienceError):
        restore(history, expected_attempt_directory=directory)
