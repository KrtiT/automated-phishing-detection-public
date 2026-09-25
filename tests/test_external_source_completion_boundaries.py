"""Saved acceptance holds its snapshot through replay and binding rechecks."""

from pathlib import Path

import pytest
from external_completion_fixtures import external_completion_case
from test_external_source_completion import bind_fixture, module, verify

from automated_phishing_detection import (
    bound_models,
    bound_secondary,
    character_transformer,
    evaluation_producer,
    external_producer,
    secondary_transformer,
    source_runner,
    transformer_inference,
)


@pytest.mark.parametrize("location", ["provenance", "replay", "binding"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_failure_or_interruption_never_returns_completion(
    tmp_path, monkeypatch, location, error_type
):
    api = module()
    case = external_completion_case(tmp_path, monkeypatch)
    bind_fixture(api, case, monkeypatch)
    error = error_type("private-canary")

    def fail(*args, **kwargs):
        raise error

    operation = {
        "provenance": "verify_external_provenance",
        "replay": "reconstruct_external_evidence",
        "binding": "recheck_binding",
    }[location]
    monkeypatch.setattr(api, operation, fail)
    expected = (
        api.ExternalCompletionVerificationError
        if issubclass(error_type, Exception)
        else error_type
    )
    with pytest.raises(expected) as rejected:
        verify(api, case)
    if issubclass(error_type, Exception):
        assert str(rejected.value) == "invalid_external_completion"
    else:
        assert rejected.value is error


@pytest.mark.parametrize("directory", ["checkpoints", "evidence"])
def test_mutation_during_replay_is_rejected_before_return(
    tmp_path, monkeypatch, directory
):
    api = module()
    case = external_completion_case(tmp_path, monkeypatch)
    events = bind_fixture(api, case, monkeypatch)
    original = api.reconstruct_external_evidence

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        (case.paths.attempt / directory / "all-scores.jsonl").write_bytes(b"changed")
        return result

    monkeypatch.setattr(api, "reconstruct_external_evidence", changed)
    with pytest.raises(api.ExternalCompletionVerificationError):
        verify(api, case)
    assert events == ["recheck"]


def _forbid_models(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("saved acceptance attempted model access or new inference")

    for owner, name in (
        (Path, "read_bytes"),
        (Path, "read_text"),
        (bound_models, "load_bound_models"),
        (bound_secondary, "load_bound_secondary"),
        (bound_secondary, "score_bound_secondary"),
        (evaluation_producer, "score_primary_url"),
        (external_producer, "produce_external_evidence"),
        (secondary_transformer, "load_secondary_transformer_bytes"),
        (secondary_transformer, "score_secondary_transformer_urls"),
        (transformer_inference, "_load_transformer_cascade_bytes"),
        (character_transformer.CharacterTransformer, "forward"),
    ):
        monkeypatch.setattr(owner, name, forbidden)


def test_acceptance_never_opens_original_inputs_or_model_artifacts(
    tmp_path, monkeypatch
):
    api = module()
    case = external_completion_case(tmp_path, monkeypatch)
    bind_fixture(api, case, monkeypatch)
    original = source_runner._read_file_once
    reads = []

    def retained_only(filename, **options):
        assert filename.is_relative_to(case.paths.attempt) or (
            filename == case.paths.public_summary
        )
        reads.append(filename)
        return original(filename, **options)

    _forbid_models(monkeypatch)
    monkeypatch.setattr(source_runner, "_read_file_once", retained_only)
    snapshot = verify(api, case)
    assert len(reads) == len(set(reads)) == len(snapshot.payloads) == 76
