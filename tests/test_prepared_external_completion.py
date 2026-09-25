"""Saved consistency after real fixture scoring; no-op exit is unit lineage only."""

import sys
from dataclasses import replace

import pytest
from prepared_external_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    prepared_case,
    runner,
)

from automated_phishing_detection import external_source_completion as completion
from automated_phishing_detection import external_source_runner as worker
from automated_phishing_detection._external_completion_files import (
    snapshot_external_files,
)
from automated_phishing_detection._external_completion_records import (
    authenticate_external_records,
)
from automated_phishing_detection.owned_worker import observe_worker

__all__ = ["inputs", "preparation_api", "preparation_case", "prepared_case", "runner"]


def published(case):
    worker._run_bound_prepared_external(
        case.binding, case.paths, handoff=case.handoff, preparation=case.preparation
    )


def test_saved_prepared_completion_keeps_both_independent_replays(
    prepared_case, monkeypatch
):
    case, calls = prepared_case, []
    published(case)
    original_provenance = completion.verify_external_provenance
    original_science = completion.reconstruct_external_evidence

    def provenance(*args, **kwargs):
        calls.append("provenance")
        return original_provenance(*args, **kwargs)

    def science(*args, **kwargs):
        calls.append("science")
        return original_science(*args, **kwargs)

    monkeypatch.setattr(completion, "verify_external_provenance", provenance)
    monkeypatch.setattr(completion, "reconstruct_external_evidence", science)
    command = (sys.executable, "-c", "pass")
    snapshot = completion.verify_external_completion_snapshot(
        case.binding,
        case.paths,
        expected_handoff=case.handoff,
        worker=observe_worker(command),
        command=command,
        expected_preparation=case.preparation,
    )
    assert calls == ["provenance", "science"]
    assert len(snapshot.payloads) == 76


@pytest.mark.parametrize(
    "name",
    [
        "publisher-source.json",
        "publisher-summary.json",
        "suffix-rules.dat",
        "retained-test.jsonl",
        "quarantine.jsonl",
        "inventory.json",
        "preparation-summary.json",
    ],
)
def test_parent_bytes_reject_paired_checkpoint_evidence_substitution(
    prepared_case, name
):
    case = prepared_case
    published(case)
    with snapshot_external_files(
        case.paths.attempt, case.paths.public_summary
    ) as files:
        substituted = replace(
            files,
            payloads=tuple(
                (path, content + b" ")
                if path in (f"attempt/checkpoints/{name}", f"attempt/evidence/{name}")
                else (path, content)
                for path, content in files.payloads
            ),
        )
        with pytest.raises(completion.ExternalCompletionVerificationError):
            authenticate_external_records(
                substituted,
                case.paths.attempt,
                binding=case.binding,
                profile=case.profile,
                handoff=case.handoff,
                preparation=case.preparation,
            )


def test_prepared_completion_cannot_omit_parent_preparation(prepared_case):
    case = prepared_case
    published(case)
    command = (sys.executable, "-c", "pass")
    with pytest.raises(completion.ExternalCompletionVerificationError):
        completion.verify_external_completion_snapshot(
            case.binding,
            case.paths,
            expected_handoff=case.handoff,
            worker=observe_worker(command),
            command=command,
        )
