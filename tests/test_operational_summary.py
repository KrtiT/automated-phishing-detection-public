"""Every scheduled repeat contributes to the descriptive pooled result."""

import builtins
import socket
from pathlib import Path

import operational_summary_fixtures as fixtures
import pytest

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.http_replay import summarize_run
from automated_phishing_detection.operational_schedule import planned_cells
from automated_phishing_detection.shift_replay import summarize_shift_run

matrix = fixtures.matrix
report = fixtures.report
GROUP_FIELDS = {
    "workload",
    "prevalence_basis_points",
    "concurrency",
    "manifest_sha256",
    "run_indices",
    "run_request_counts",
    "request_count",
    "request_errors",
    "request_error_rate",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "admitted_requests",
    "completed_requests",
    "failed_requests",
    "transformer_forward_attempts",
    "successful_transformer_scores",
    "physical_invocation_fraction",
    "run_summaries",
}


def test_full_matrix_summary_api_exists(matrix):
    assert len(matrix) == 125
    assert callable(fixtures.api().summarize_operational_runs)


def test_full_matrix_produces_exact_closed_25_group_projection(report):
    assert set(report) == {"schema_version", "protocol", "groups"}
    assert report["schema_version"] == 1
    assert report["protocol"] == "operational-descriptive-summary-v1"
    assert len(report["groups"]) == 25
    assert canonical_bytes(report).isascii()
    for group, cell in zip(report["groups"], planned_cells()[::5], strict=True):
        assert set(group) == GROUP_FIELDS
        assert group["workload"] == cell.workload
        assert group["prevalence_basis_points"] == cell.prevalence_basis_points
        assert group["concurrency"] == cell.concurrency
        assert group["run_indices"] == [1, 2, 3, 4, 5]
        assert [run["cell_ordinal"] for run in group["run_summaries"]] == list(
            range(cell.ordinal, cell.ordinal + 5)
        )


def test_pooled_individual_quantiles_include_adverse_fifth_repeat(report):
    for group in report["groups"][:24]:
        assert group["run_request_counts"] == [10000] * 5
        assert group["request_count"] == 50000
        assert group["request_errors"] == 49996
        assert group["request_error_rate"] == 49996 / 50000
        assert (group["p50_ms"], group["p95_ms"], group["p99_ms"]) == (
            2100.0,
            2100.0,
            10000.0,
        )
        assert [run["p95_ms"] for run in group["run_summaries"]] == [2100.0] * 4 + [
            10000.0
        ]
        assert [run["request_errors"] for run in group["run_summaries"]] == [
            9999
        ] * 4 + [10000]


def test_physical_forward_fraction_uses_clients_not_responses_or_admissions(report):
    for group in report["groups"][:24]:
        fraction = 0.8 if group["workload"] == "transformer_only" else 0.4
        assert group["admitted_requests"] == group["completed_requests"] == 40000
        assert group["failed_requests"] == 0
        assert group["transformer_forward_attempts"] == fraction * 50000
        assert group["successful_transformer_scores"] == fraction * 50000
        assert group["physical_invocation_fraction"] == fraction


def test_shift_uses_full_five_n_including_errors_not_http_denominator(report):
    group = report["groups"][-1]
    assert group["prevalence_basis_points"] is None
    assert group["run_request_counts"] == [1001] * 5
    assert group["request_count"] == group["completed_requests"] == 5005
    assert group["request_errors"] == 10
    assert group["request_error_rate"] == 10 / 5005
    assert group["physical_invocation_fraction"] == 0


@pytest.mark.parametrize("index", [0, 4, 20, 89, 90, 119, 120, 124])
def test_run_summaries_are_unchanged_except_scheduled_ordinal(matrix, report, index):
    reducer = summarize_run if index < 120 else summarize_shift_run
    expected = reducer(matrix[index]) | {"cell_ordinal": index + 1}
    assert report["groups"][index // 5]["run_summaries"][index % 5] == expected


def test_full_projection_never_opens_files_or_sockets(matrix, report, monkeypatch):
    module = fixtures.api()

    def forbidden(*args, **kwargs):
        pytest.fail("pure summary attempted I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    assert module.summarize_operational_runs(matrix) == report
