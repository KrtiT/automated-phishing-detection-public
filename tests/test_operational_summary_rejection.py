"""No missing, relabeled, reordered, or differently sourced cell is pooled."""

from dataclasses import replace

import operational_summary_fixtures as fixtures
import pytest

from automated_phishing_detection.http_run_codec import encode_http_run
from automated_phishing_detection.shift_run_codec import encode_shift_run

matrix = fixtures.matrix


def rejected(runs):
    with pytest.raises(
        fixtures.api().OperationalSummaryError, match="^invalid_operational_summary$"
    ):
        fixtures.api().summarize_operational_runs(runs)


def changed(matrix, index, run):
    return (*matrix[:index], run, *matrix[index + 1 :])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda runs: runs[:-1],
        lambda runs: (*runs, runs[-1]),
        lambda runs: list(runs),
        lambda runs: (runs[1], runs[0], *runs[2:]),
        lambda runs: (runs[0], runs[0], *runs[2:]),
        lambda runs: (None, *runs[1:]),
    ],
)
def test_full_schedule_is_required_in_exact_order(matrix, mutation):
    rejected(mutation(matrix))


@pytest.mark.parametrize(
    "field,value",
    [
        ("workload", "transformer_only"),
        ("prevalence_basis_points", 10),
        ("concurrency", 8),
        ("concurrency", True),
        ("run_index", True),
    ],
)
def test_schedule_labels_are_not_caller_overrides(matrix, field, value):
    rejected(changed(matrix, 0, replace(matrix[0], **{field: value})))


@pytest.mark.parametrize("index", [4, 5, 29, 35, 90, 95, 119])
@pytest.mark.parametrize("change", ["hash", "order"])
def test_all_concurrencies_and_transformer_join_their_fixed_manifest(
    matrix, index, change
):
    run = fixtures.substitute_http(matrix[index], change)
    assert encode_http_run(run)
    rejected(changed(matrix, index, run))


@pytest.mark.parametrize("change", ["hash", "url", "order"])
def test_shift_repeats_join_full_ordered_raw_requests(matrix, change):
    run = matrix[-1]
    plan = run.plan
    if change == "hash":
        run = fixtures.substitute_shift_hash(run)
        assert encode_shift_run(run)
        rejected(changed(matrix, 124, run))
        return
    else:
        requests = plan.requests
        first = replace(requests[0], raw_url="https://different.test/")
        requests = (
            (first, *requests[1:])
            if change == "url"
            else (requests[1], requests[0], *requests[2:])
        )
        plan = replace(plan, requests=requests)
    if change == "url":
        assert encode_shift_run(replace(run, plan=plan))
    rejected(changed(matrix, 124, replace(run, plan=plan)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("measured_elapsed_ms", None),
        ("measured_drain_ms", None),
        ("measured_elapsed_ms", float("nan")),
        ("measured_elapsed_ms", True),
    ],
)
def test_existing_complete_timing_gates_are_not_relaxed(matrix, field, value):
    rejected(changed(matrix, 0, replace(matrix[0], **{field: value})))


def test_matching_shortened_warmup_still_fails_complete_codec(matrix):
    rejected(changed(matrix, 0, replace(matrix[0], warmup=matrix[0].warmup[:-1])))


def test_existing_drain_accounting_remains_mandatory(matrix):
    run = matrix[0]
    counts = run.after_measured.model_copy(update={"completed_requests": 8799})
    rejected(changed(matrix, 0, replace(run, after_measured=counts)))
