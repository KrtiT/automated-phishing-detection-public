"""Prepared pair ordering retains actual internal scoring as a prerequisite."""

import importlib
import importlib.util
from dataclasses import replace

import pytest
from test_external_source_process_commands import command_case

from automated_phishing_detection._external_source_records import (
    PreparedExternalRunPaths,
)
from automated_phishing_detection._prepared_internal_records import (
    PreparedInternalRunPaths,
)


def api():
    name = "automated_phishing_detection.prepared_external_process"
    assert importlib.util.find_spec(name), "missing prepared observed pair"
    return importlib.import_module(name)


def paths():
    binding, old, unused = command_case()
    external = PreparedExternalRunPaths(
        old.archive,
        old.artifacts,
        old.secondary_artifacts,
        old.drift_artifacts,
        old.attempt,
        old.public_summary,
    )
    internal = PreparedInternalRunPaths(
        old.archive,
        old.artifacts,
        old.secondary_artifacts,
        old.attempt.with_name("internal"),
        old.public_summary.with_name("internal-public"),
    )
    return binding, internal, external


def configured_pair(module, monkeypatch):
    calls, internal, external, handoff, preparation = (
        [],
        object(),
        object(),
        object(),
        object(),
    )

    def observed(*args, **kwargs):
        calls.append("internal")
        assert kwargs["preparation"] is preparation
        return internal

    def handed(value):
        assert value is internal and calls == ["internal"]
        calls.append("handoff")
        return handoff

    def completed(*args, **kwargs):
        assert args[-1] is handoff and kwargs["preparation"] is preparation
        calls.append("external")
        return external

    monkeypatch.setattr(module, "_run_observed_prepared_internal", observed)
    monkeypatch.setattr(module.process, "build_internal_handoff", handed)
    monkeypatch.setattr(module.process, "_run_observed_external", completed)
    return calls, internal, external, preparation


def test_pair_keeps_observed_internal_before_handoff_and_external(monkeypatch):
    module = api()
    binding, internal_paths, external_paths = paths()
    calls, internal, external, preparation = configured_pair(module, monkeypatch)
    result = module._run_observed_prepared_sources(
        binding, internal_paths, external_paths, preparation
    )
    assert calls == ["internal", "handoff", "external"]
    assert (result.internal, result.external, result.preparation) == (
        internal,
        external,
        preparation,
    )


def test_mixed_preparation_directories_fail_before_internal(monkeypatch):
    module = api()
    binding, internal, external = paths()
    internal = replace(internal, preparation=internal.preparation.with_name("other"))
    monkeypatch.setattr(
        module,
        "_run_observed_prepared_internal",
        lambda *args, **kwargs: pytest.fail("launched"),
    )
    with pytest.raises(module.ExternalSourceExecutionError):
        module._run_observed_prepared_sources(binding, internal, external, object())
