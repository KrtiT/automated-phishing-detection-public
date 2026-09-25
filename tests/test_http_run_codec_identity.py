"""Caller consistency is checked without claiming manifest/process authority."""

from dataclasses import replace

import pytest
from http_run_codec_fixtures import api, decode, document, requests

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.http_replay import ReplayRequest


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("expected_manifest_sha256", "b" * 64),
        ("expected_manifest_sha256", "A" * 64),
        ("expected_manifest_sha256", 1),
        ("expected_prevalence_basis_points", 10),
        ("expected_prevalence_basis_points", True),
        ("expected_concurrency", 1),
        ("expected_concurrency", 64.0),
        ("expected_run_index", 2),
        ("expected_run_index", True),
        ("expected_workload", "transformer_only"),
        ("expected_workload", None),
    ],
)
def test_expected_identity_is_independent_and_typed(field, invalid):
    with pytest.raises(api().HttpRunCodecError):
        decode(**{field: invalid})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[:-1],
        lambda rows: (*rows, rows[-1]),
        lambda rows: list(rows),
        lambda rows: (*rows[:10], rows[11], rows[10], *rows[12:]),
        lambda rows: (rows[1], *rows[1:]),
        lambda rows: (replace(rows[0], record_id="private-wrong"), *rows[1:]),
        lambda rows: (replace(rows[0], raw_url=""), *rows[1:]),
        lambda rows: (replace(rows[0], raw_url=1), *rows[1:]),
        lambda rows: (replace(rows[0], record_id="has space"), *rows[1:]),
        lambda rows: (
            {"record_id": "row-0", "raw_url": "https://example.test"},
            *rows[1:],
        ),
    ],
)
def test_full_expected_manifest_projection_is_checked(mutation):
    with pytest.raises(api().HttpRunCodecError):
        decode(expected_requests=mutation(requests()))


def test_raw_urls_are_validated_not_normalized_or_hashed_as_a_new_manifest():
    changed = (ReplayRequest("row-0", "not a normalized URL"), *requests()[1:])
    assert decode(expected_requests=changed).manifest_sha256 == "a" * 64


@pytest.mark.parametrize("phase", ["warmup", "measured"])
def test_serialized_order_is_bound_to_full_expected_order(phase):
    value = document()
    rows = value["run"][phase]
    rows[10]["record_id"], rows[11]["record_id"] = (
        rows[11]["record_id"],
        rows[10]["record_id"],
    )
    with pytest.raises(api().HttpRunCodecError):
        decode(canonical_bytes(value))


@pytest.mark.parametrize(
    "change",
    [
        lambda run: run["initial"].update(admitted_requests=1),
        lambda run: run["after_measured"].update(completed_requests=10999),
        lambda run: run["measured"][10].update(request_id="private-wrong"),
        lambda run: run["measured"][10]["response"].update(request_id="private-wrong"),
        lambda run: run["measured"][10]["response"].update(admission_sequence=1012),
        lambda run: run["measured"][10].update(elapsed_ms=2001),
        lambda run: run["measured"][0].update(error="unknown"),
        lambda run: run["measured"][0].update(response=run["measured"][10]["response"]),
        lambda run: run.update(measured_elapsed_ms=1),
    ],
)
def test_existing_counter_response_and_timing_gates_remain_in_force(change):
    value = document()
    change(value["run"])
    with pytest.raises(api().HttpRunCodecError, match="^invalid_http_run_record$"):
        decode(canonical_bytes(value))
