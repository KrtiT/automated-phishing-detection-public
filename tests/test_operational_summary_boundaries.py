"""Summary completion cannot erase codec failures or interruption identity."""

from dataclasses import replace

import operational_summary_fixtures as fixtures
import pytest

matrix = fixtures.matrix


def test_nonfinite_derived_rate_is_rejected_by_real_complete_codec(matrix):
    run = matrix[0]
    measured = tuple(replace(row, elapsed_ms=0.0) for row in run.measured)
    run = replace(run, measured=measured, measured_elapsed_ms=1e-320)
    with pytest.raises(fixtures.api().OperationalSummaryError):
        fixtures.api().summarize_operational_runs((run, *matrix[1:]))


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_original_nonexception_from_codec_propagates(matrix, monkeypatch, error_type):
    module = fixtures.api()
    interruption = error_type("invented")

    def interrupt(run):
        raise interruption

    monkeypatch.setattr(module, "encode_http_run", interrupt)
    with pytest.raises(error_type) as caught:
        module.summarize_operational_runs(matrix)
    assert caught.value is interruption


def test_codec_exception_is_symbolic_without_private_diagnostic(matrix, monkeypatch):
    module = fixtures.api()

    def fail(run):
        raise RuntimeError("invented-private-url-and-path")

    monkeypatch.setattr(module, "encode_http_run", fail)
    with pytest.raises(module.OperationalSummaryError) as caught:
        module.summarize_operational_runs(matrix)
    assert str(caught.value) == "invalid_operational_summary"
    assert caught.value.__suppress_context__


def test_recorded_scorer_and_forward_failures_remain_descriptive_evidence(matrix):
    run = matrix[0]
    counts = run.after_measured.model_copy(
        update={
            "completed_requests": 8799,
            "failed_requests": 1,
            "transformer_forward_attempts": 4401,
        }
    )
    run = replace(run, after_measured=counts)
    group = fixtures.api().summarize_operational_runs((run, *matrix[1:]))["groups"][0]
    assert group["request_count"] == 50000
    assert group["admitted_requests"] == 40000
    assert group["completed_requests"] == 39999
    assert group["failed_requests"] == 1
    assert group["transformer_forward_attempts"] == 20001
    assert group["successful_transformer_scores"] == 20000
    assert group["physical_invocation_fraction"] == 20001 / 50000
