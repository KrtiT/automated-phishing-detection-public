"""Both public readiness gates precede any transport or protected input access."""

from pathlib import Path

import pytest
import test_external_source_runner as fixtures

from automated_phishing_detection._external_source_profile import (
    CandidateExternalProfile,
)
from automated_phishing_detection.execution_preflight import ExecutionBinding

runner_api = fixtures.runner_api
runner_case = fixtures.runner_case


def gate_context(api, monkeypatch, binding_ready, profile_ready):
    binding = ExecutionBinding(Path("/invented"), "c" * 40, "d" * 64, (), "{}")
    profile, calls = CandidateExternalProfile(b"invented"), []
    monkeypatch.setattr(
        ExecutionBinding,
        "protected_evaluation_ready",
        property(lambda self: binding_ready),
    )
    monkeypatch.setattr(
        CandidateExternalProfile,
        "protected_evaluation_ready",
        property(lambda self: profile_ready),
    )
    monkeypatch.setattr(api, "bind_execution", lambda *args, **kwargs: binding)

    def resolved(actual):
        assert actual is binding
        calls.append("profile")
        return profile

    def forbidden(*args, **kwargs):
        pytest.fail("a false public gate allowed transport or execution")

    monkeypatch.setattr(api.body, "resolve_external_source_profile", resolved)
    monkeypatch.setattr(api, "read_internal_handoff_transport", forbidden)
    monkeypatch.setattr(api, "_run_bound_external", forbidden)
    return binding, calls


@pytest.mark.parametrize("binding_ready,profile_ready", [(False, True), (True, False)])
def test_each_false_gate_blocks_independently(
    runner_api, monkeypatch, binding_ready, profile_ready
):
    binding, calls = gate_context(runner_api, monkeypatch, binding_ready, profile_ready)
    with pytest.raises(runner_api.ExternalSourceExecutionError):
        runner_api.run_external_evaluation(
            binding.root,
            expected_revision=binding.revision,
            expected_contract_sha256=binding.contract_sha256,
            paths=object(),
            internal_transport=object(),
            expected_handoff_sha256=object(),
        )
    assert calls == (["profile"] if binding_ready else [])


def test_path_type_precedes_arbitrary_attribute_access(runner_api, runner_case):
    calls = []

    class UnknownPaths:
        @property
        def artifacts(self):
            calls.append("inspected")
            return None

    with pytest.raises(runner_api.ExternalSourceExecutionError):
        fixtures.run(runner_api, runner_case, paths=UnknownPaths())
    assert calls == []
