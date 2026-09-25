import importlib
import importlib.util
import json

import pytest
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)

__all__ = ["inputs", "preparation_api", "preparation_case", "runner"]


def module():
    name = "automated_phishing_detection.study_preparation_context"
    assert importlib.util.find_spec(name), "missing bound preparation context"
    return importlib.import_module(name)


def test_context_matches_actual_preparation_identity(preparation_case, monkeypatch):
    case = preparation_case
    api = preparation_api.__wrapped__()
    snapshot = api._run_bound_preparation(case.binding, case.paths)
    context = module()
    monkeypatch.setattr(
        context, "resolve_external_source_profile", lambda _: case.profile
    )
    identity, source, buffers, profile = context.bound_preparation_context(case.binding)
    assert (
        identity
        == json.loads(snapshot.payload("preparation-complete.json"))["execution"]
    )
    assert profile is case.profile
    assert source["expected_sha256"] == identity["partition_sha256"]
    assert set(buffers) == {context.SOURCE, context.PREPARATION}
    assert all(type(content) is bytes for content in buffers.values())


def test_context_rejects_inconsistent_suffix_pin(preparation_case, monkeypatch):
    case, context = preparation_case, module()
    monkeypatch.setattr(
        context, "resolve_external_source_profile", lambda _: case.profile
    )
    original = context.source_runner._public_sources

    def mismatched(binding):
        source, buffers = original(binding)
        return {**source, "suffix_rules_sha256": "f" * 64}, buffers

    monkeypatch.setattr(context.source_runner, "_public_sources", mismatched)
    with pytest.raises(
        context.StudyPreparationError, match="preparation_suffix_pin_mismatch"
    ):
        context.bound_preparation_context(case.binding)


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(7)])
def test_context_preserves_public_read_interruption(
    preparation_case, monkeypatch, interruption
):
    case, context = preparation_case, module()

    def interrupted(_):
        raise interruption

    monkeypatch.setattr(context, "resolve_external_source_profile", interrupted)
    with pytest.raises(type(interruption)) as captured:
        context.bound_preparation_context(case.binding)
    assert captured.value is interruption


def test_identity_projection_never_rereads_public_inputs(preparation_case, monkeypatch):
    case, context = preparation_case, module()
    monkeypatch.setattr(
        context, "resolve_external_source_profile", lambda _: case.profile
    )
    identity, source, unused, profile = context.bound_preparation_context(case.binding)
    project = getattr(context, "preparation_identity", None)
    assert callable(project), "missing pure preparation identity projection"
    monkeypatch.setattr(
        context.source_runner,
        "_public_sources",
        lambda *_: pytest.fail("pure identity read files"),
    )
    assert project(case.binding, profile, source) == identity
