"""Original full-size125 workloads; constructed source context is not authority."""

from dataclasses import asdict, replace

import pytest
from operational_input_fixtures import candidates, case, manifests
from study_reduction_fixtures import full_case
from test_study_operational_records import api as records_api
from test_study_reduction import api

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.http_replay import (
    primary_http_summary,
    reference_invocations,
)
from automated_phishing_detection.operational_summary import summarize_operational_runs
from automated_phishing_detection.study_evidence import reduce_study_evidence

__all__ = ["candidates", "case", "manifests"]


@pytest.fixture(scope="module")
def complete(case):
    return full_case(records_api(), case)


def test_full125_reduction_matches_unchanged_kernels_and_family(complete):
    result = api().reduce_accepted_study(complete.accepted, complete.slots)
    expected = reduce_study_evidence(
        internal=complete.accepted.internal.snapshot.population,
        external=complete.accepted.external.snapshot.replay.evidence,
        reference=reference_invocations(complete.runs[0]),
        http=primary_http_summary(complete.runs[20:25]),
    )
    assert result.operational_bytes == canonical_bytes(
        summarize_operational_runs(complete.runs)
    )
    assert result.study_bytes == canonical_bytes(asdict(expected))
    assert len(result.operational["groups"]) == 25
    assert result.study["ablation_family"]["family_size"] == 4
    assert result.study["primary"]["hypotheses"]["H2"]["decision"] == "not_supported"
    result.operational["groups"].clear()
    result.study.clear()
    assert len(result.operational["groups"]) == 25 and result.study


def test_complete_collection_preserves_adverse_fifth_repeat_and_shift_errors(complete):
    result = api().reduce_accepted_study(complete.accepted, complete.slots).operational
    assert result["groups"][0]["request_count"] == 50000
    assert result["groups"][0]["request_errors"] == 49996
    assert result["groups"][0]["p99_ms"] == 10000
    assert result["groups"][-1]["request_count"] == 5005
    assert result["groups"][-1]["request_errors"] == 10


def test_changed_source_context_rejects_before_any_reducer(complete, monkeypatch):
    module = api()

    def forbidden(*args, **kwargs):
        pytest.fail("source mismatch reached reduction")

    monkeypatch.setattr(module, "reference_invocations", forbidden)
    changed = replace(
        complete.accepted, metadata_bytes=complete.accepted.metadata_bytes + b"\n"
    )
    with pytest.raises(module.StudyReductionError):
        module.reduce_accepted_study(changed, complete.slots)


def test_reference_extraction_failure_is_not_replaced_by_missing_evidence(
    complete, monkeypatch
):
    module = api()
    calls = []

    def fail(run):
        calls.append(run)
        raise ValueError("unadmitted reference attempt")

    monkeypatch.setattr(module, "reference_invocations", fail)
    with pytest.raises(module.StudyReductionError):
        module.reduce_accepted_study(complete.accepted, complete.slots)
    assert len(calls) == 1
    assert all(slot.status == "accepted" for slot in complete.slots)


def test_reduction_interruption_keeps_exact_original_object(complete, monkeypatch):
    interruption = KeyboardInterrupt("first")

    def fail(run):
        raise interruption

    monkeypatch.setattr(api(), "reference_invocations", fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        api().reduce_accepted_study(complete.accepted, complete.slots)
    assert caught.value is interruption


@pytest.mark.parametrize("count", [0, 1, 124])
def test_incomplete_prefix_never_reaches_run_restoration(complete, monkeypatch, count):
    module = api()

    def forbidden(*args, **kwargs):
        pytest.fail("incomplete accounting reached restoration")

    monkeypatch.setattr(module, "restore_runs", forbidden)
    prefix = tuple(slot.accepted for slot in complete.slots[:count])
    incomplete = records_api().freeze_cell_accounting(prefix)
    with pytest.raises(module.StudyReductionError):
        module.reduce_accepted_study(complete.accepted, incomplete)
    assert sum(slot.status == "accepted" for slot in incomplete) == count
