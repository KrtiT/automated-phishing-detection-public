"""Real retained-input producer children, held transport, and saved reconstruction."""

import json

import pytest
from prepared_internal_fixtures import restored_case
from prepared_internal_worker_fixtures import child_command
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)
from test_prepared_internal_runner import module

from automated_phishing_detection import evaluation_producer, saved_evidence
from automated_phishing_detection.internal_external_handoff import (
    build_internal_handoff,
)

__all__ = ["inputs", "preparation_api", "preparation_case", "runner"]


def expected_models(case, monkeypatch):
    content = evaluation_producer.binding_bytes(
        case.preparation.internal,
        case.session,
        evaluation_producer._thresholds(case.session.primary),
    )
    values = json.loads(content)
    monkeypatch.setattr(
        saved_evidence,
        "_EXPECTED_BINDING_CORE",
        {
            name: values[name]
            for name in ("artifact_hashes", "thresholds", "secondary", "gmm_audit")
        },
    )


@pytest.fixture
def child_case(preparation_api, preparation_case, inputs, monkeypatch):
    case = restored_case(preparation_api, preparation_case, inputs)
    expected_models(case, monkeypatch)
    for filename in (
        case.original.source_csv,
        case.original.suffix_rules,
        preparation_case.paths.archive,
    ):
        filename.unlink()
    return case


def run_owned(api, case, monkeypatch, mode):
    command = child_command(case, mode)
    monkeypatch.setattr(api, "_worker_command", lambda *args, **kwargs: command)
    return api._run_held(
        case.binding,
        case.paths,
        (case.identity, case.source, case.buffers, case.profile),
        case.preparation.reservation_sha256,
        case.preparation.completion_sha256,
        True,
    )


def test_actual_prepared_child_scientific_completion_is_accepted(
    child_case, monkeypatch
):
    api, case = module(), child_case
    result = run_owned(api, case, monkeypatch, "success")
    assert (
        result.worker.exit.exit_observed is True and result.worker.exit.exit_code == 0
    )
    assert result.snapshot.payload(
        "attempt/checkpoints/group_test.jsonl"
    ) == case.preparation.payload("group_test.jsonl")
    handoff = json.loads(build_internal_handoff(result).handoff_bytes)
    assert (
        handoff["execution"]["study_preparation_complete_sha256"]
        == case.preparation.completion_sha256
    )
    assert len(handoff["snapshot_sha256"]) == 35


def test_valid_child_publication_then_nonzero_exit_never_reaches_acceptance(
    child_case, monkeypatch
):
    api, case = module(), child_case
    monkeypatch.setattr(
        api,
        "verify_prepared_internal_completion_snapshot",
        lambda *args, **kwargs: pytest.fail("nonzero child reached acceptance"),
    )
    with pytest.raises(api.SourceExecutionError, match="worker_exit") as caught:
        run_owned(api, case, monkeypatch, "nonzero")
    assert case.paths.public_summary.is_file()
    assert caught.value.worker_failure.worker.exit.exit_code == 17


def test_child_session_teardown_retains_science_without_publication(
    child_case, monkeypatch
):
    api, case = module(), child_case
    with pytest.raises(api.SourceExecutionError, match="worker_exit"):
        run_owned(api, case, monkeypatch, "teardown")
    assert not case.paths.public_summary.exists()
    failure = json.loads((case.paths.attempt / "failure-progress.json").read_bytes())
    assert failure["cleanup_failed"] is True
    assert (case.paths.attempt / "scientific-checkpoints/completion.json").is_file()
