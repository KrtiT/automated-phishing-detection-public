"""Complete HTTP records retain every terminal outcome and both intervals."""

import json
from dataclasses import replace

import pytest
from http_run_codec_fixtures import api, complete_run, decode, document, wire_bytes

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.http_replay import summarize_run


@pytest.mark.parametrize("workload", ["fixed_cascade", "transformer_only"])
def test_complete_round_trip_preserves_every_typed_field(workload):
    run = complete_run(workload)
    encoded = api().encode_http_run(run)
    assert encoded == wire_bytes(workload)
    restored = decode(encoded, expected_workload=workload)
    assert restored == run
    assert api().encode_http_run(restored) == encoded
    assert len(restored.warmup) == 1000
    assert len(restored.measured) == 10000


def test_existing_reducer_includes_errors_in_latency_and_denominator():
    summary = summarize_run(decode())
    assert summary["request_count"] == 10000
    assert summary["request_errors"] == 6
    assert summary["request_error_rate"] == 0.0006
    assert summary["p50_ms"] == 51.0
    assert summary["p95_ms"] == 96.0
    assert summary["p99_ms"] == 100.0
    assert summary["measured_elapsed_ms"] == 10000.0
    assert summary["measured_drain_ms"] == 10001.0
    assert summary["client_attempts_per_second"] == 1000.0
    assert summary["successful_responses_per_second"] == 999.4


@pytest.mark.parametrize("field", ["measured_elapsed_ms", "measured_drain_ms"])
@pytest.mark.parametrize(
    "invalid", [None, True, 0, -1, "10000", float("inf"), float("nan")]
)
def test_encode_rejects_missing_or_invalid_complete_intervals(field, invalid):
    with pytest.raises(api().HttpRunCodecError, match="^invalid_http_run_record$"):
        api().encode_http_run(replace(complete_run(), **{field: invalid}))


@pytest.mark.parametrize("field", ["warmup", "measured"])
@pytest.mark.parametrize("size_change", [-1, 1])
def test_encode_rejects_shortened_or_extended_operational_phase(field, size_change):
    run = complete_run()
    rows = getattr(run, field)
    changed = rows[:-1] if size_change < 0 else (*rows, rows[-1])
    with pytest.raises(api().HttpRunCodecError):
        api().encode_http_run(replace(run, **{field: changed}))


@pytest.mark.parametrize("prevalence", [10, 500])
def test_transformer_only_rejects_sensitivity_prevalence(prevalence):
    with pytest.raises(api().HttpRunCodecError):
        api().encode_http_run(
            replace(
                complete_run("transformer_only"), prevalence_basis_points=prevalence
            )
        )


@pytest.mark.parametrize("prevalence", [10, 500])
def test_fixed_cascade_accepts_each_scheduled_prevalence(prevalence):
    run = replace(complete_run(), prevalence_basis_points=prevalence)
    assert (
        decode(api().encode_http_run(run), expected_prevalence_basis_points=prevalence)
        == run
    )


def test_concurrent_admission_order_is_not_dispatch_order():
    value = document()
    rows = value["run"]["measured"]
    (
        rows[10]["response"]["admission_sequence"],
        rows[11]["response"]["admission_sequence"],
    ) = 1012, 1011
    restored = decode(canonical_bytes(value))
    assert restored.measured[10].response.admission_sequence == 1012


def test_encoded_document_is_ascii_compact_sorted_and_newline_terminated():
    content = api().encode_http_run(complete_run())
    assert content.isascii() and content.endswith(b"\n")
    assert canonical_bytes(json.loads(content)) == content


def test_interruption_from_existing_validation_propagates_unchanged(monkeypatch):
    interruption = KeyboardInterrupt("invented")

    def interrupt(*args):
        raise interruption

    monkeypatch.setattr(api(), "summarize_run", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        api().encode_http_run(complete_run())
    assert caught.value is interruption
