"""Complete-record boundaries stay separate from reference and process gates."""

from dataclasses import replace

import pytest
from http_run_codec_fixtures import api, complete_run, decode, document, requests

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.http_replay import summarize_run


@pytest.mark.parametrize("invalid", ["", "has space", "control\x00", "\ud800"])
def test_encoder_rejects_source_ids_that_cannot_enter_replay(invalid):
    run = complete_run()
    changed = {
        phase: (
            replace(getattr(run, phase)[0], record_id=invalid),
            *getattr(run, phase)[1:],
        )
        for phase in ("warmup", "measured")
    }
    with pytest.raises(api().HttpRunCodecError):
        api().encode_http_run(replace(run, **changed))


def test_unicode_record_ids_are_preserved_in_canonical_ascii_bytes():
    run = complete_run()
    changed = {
        phase: (
            replace(getattr(run, phase)[0], record_id="é-例え"),
            *getattr(run, phase)[1:],
        )
        for phase in ("warmup", "measured")
    }
    encoded = api().encode_http_run(replace(run, **changed))
    expected = (replace(requests()[0], record_id="é-例え"), *requests()[1:])
    assert encoded.isascii() and b"\\u00e9" in encoded
    assert decode(encoded, expected_requests=expected).measured[0].record_id == "é-例え"


@pytest.mark.parametrize("workload", ["fixed_cascade", "transformer_only"])
def test_complete_all_error_run_is_evidence_not_success_or_missing_data(workload):
    run = complete_run(workload)
    measured = tuple(
        replace(row, error="transport", response=None, status_code=None)
        for row in run.measured
    )
    run = replace(run, measured=measured, after_measured=run.after_warmup)
    restored = decode(api().encode_http_run(run), expected_workload=workload)
    summary = summarize_run(restored)
    assert summary["request_errors"] == 10000
    assert summary["completed_requests"] == 0
    assert summary["client_attempts_per_second"] == 1000.0
    assert summary["successful_responses_per_second"] == 0.0
    assert summary["p99_ms"] == 100.0


@pytest.mark.parametrize("phase", ["warmup", "measured"])
@pytest.mark.parametrize("extra", [False, True])
def test_decoder_rejects_incomplete_and_oversized_phases(phase, extra):
    value = document()
    rows = value["run"][phase]
    value["run"][phase] = rows + [rows[-1]] if extra else rows[:-1]
    with pytest.raises(api().HttpRunCodecError):
        decode(canonical_bytes(value))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda run: replace(run, warmup=list(run.warmup)),
        lambda run: replace(run, measured=list(run.measured)),
        lambda run: replace(run, initial=run.initial.model_dump()),
        lambda run: replace(run, warmup=(None, *run.warmup[1:])),
        lambda run: replace(
            run, measured=(replace(run.measured[0], elapsed_ms=True), *run.measured[1:])
        ),
    ],
)
def test_encoder_requires_exact_typed_nested_records(mutation):
    with pytest.raises(api().HttpRunCodecError):
        api().encode_http_run(mutation(complete_run()))


def test_derived_throughput_overflow_does_not_escape_as_complete_record():
    run = complete_run()
    measured = tuple(replace(row, elapsed_ms=0.0) for row in run.measured)
    run = replace(run, measured=measured, measured_elapsed_ms=1e-320)
    with pytest.raises(api().HttpRunCodecError):
        api().encode_http_run(run)


@pytest.mark.parametrize("field", ["schema_version", "protocol", "run"])
def test_no_envelope_defaults(field):
    value = document()
    del value[field]
    with pytest.raises(api().HttpRunCodecError):
        decode(canonical_bytes(value))


def test_nested_duplicate_keys_are_rejected():
    value = canonical_bytes(document())
    value = value.replace(b'"elapsed_ms":1.0', b'"elapsed_ms":1.0,"elapsed_ms":1.0', 1)
    with pytest.raises(api().HttpRunCodecError):
        decode(value)
