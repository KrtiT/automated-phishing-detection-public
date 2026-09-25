"""Saved science is joined to independent parent-held preparation bytes."""

from dataclasses import replace

import pytest
from prepared_internal_fixtures import (
    bind_saved_fixture,
    forbid_preparation,
    restored_case,
)
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)
from test_prepared_internal_runner import module

from automated_phishing_detection import source_completion, source_runner

__all__ = ["inputs", "preparation_api", "preparation_case", "runner"]


def verifier():
    name = "verify_prepared_internal_completion_snapshot"
    assert hasattr(source_completion, name), "missing prepared completion verifier"
    return getattr(source_completion, name)


@pytest.fixture
def prepared_published(preparation_api, preparation_case, inputs, monkeypatch):
    case = restored_case(preparation_api, preparation_case, inputs)
    module()._run_bound_prepared_internal(case.binding, case.paths, case.preparation)
    bind_saved_fixture(case, monkeypatch)
    return case


def test_saved_prepared_science_reuses_exact_parent_bytes_once(
    prepared_published, monkeypatch
):
    verify = verifier()
    case, reads = prepared_published, []
    original = source_runner._read_file_once

    def read(path, **kwargs):
        assert not path.is_relative_to(case.paths.preparation)
        reads.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(source_runner, "_read_file_once", read)
    forbid_preparation(monkeypatch)
    snapshot = verify(
        case.binding, case.paths, preparation=case.preparation, producer_exit_code=0
    )
    assert len(reads) == len(set(reads))
    assert len(snapshot.payloads) == 35
    assert snapshot.overlap_domains == case.preparation.overlap_domains
    assert snapshot.payload(
        "attempt/checkpoints/group_test.jsonl"
    ) == case.preparation.payload("group_test.jsonl")


@pytest.mark.parametrize("code", [None, False, 0.0, 17, -9])
def test_unaccepted_exit_never_reads_completion(prepared_published, monkeypatch, code):
    verify = verifier()
    case = prepared_published
    monkeypatch.setattr(
        source_runner,
        "_read_file_once",
        lambda *args, **kwargs: pytest.fail("read before exit acceptance"),
    )
    with pytest.raises(
        source_completion.CompletionVerificationError, match="producer_exit"
    ):
        verify(
            case.binding,
            case.paths,
            preparation=case.preparation,
            producer_exit_code=code,
        )


@pytest.mark.parametrize("field", ["reservation_sha256", "completion_sha256"])
def test_expected_preparation_digests_are_not_taken_from_child(
    prepared_published, field
):
    case = prepared_published
    changed = replace(case.preparation, **{field: "d" * 64})
    with pytest.raises(source_completion.CompletionVerificationError):
        verifier()(case.binding, case.paths, preparation=changed, producer_exit_code=0)


@pytest.mark.parametrize("name", ["group_test.jsonl", "source-overlap.json"])
def test_independent_expected_payload_join_rejects_substitution(
    prepared_published, name
):
    case = prepared_published
    changed = replace(
        case.preparation,
        payloads=tuple(
            (filename, content + b" ") if filename == name else (filename, content)
            for filename, content in case.preparation.payloads
        ),
    )
    with pytest.raises(
        source_completion.CompletionVerificationError, match="parent_preparation"
    ):
        verifier()(case.binding, case.paths, preparation=changed, producer_exit_code=0)


def test_original_verifier_does_not_accept_prepared_route(prepared_published):
    case = prepared_published
    with pytest.raises(
        source_completion.CompletionVerificationError, match="invalid_run_paths"
    ):
        source_completion.verify_internal_completion_snapshot(
            case.binding, case.paths, producer_exit_code=0
        )
    original_paths = replace(
        case.original,
        attempt=case.paths.attempt,
        public_summary=case.paths.public_summary,
    )
    with pytest.raises(
        source_completion.CompletionVerificationError, match="reservation_identity"
    ):
        source_completion.verify_internal_completion_snapshot(
            case.binding, original_paths, producer_exit_code=0
        )
