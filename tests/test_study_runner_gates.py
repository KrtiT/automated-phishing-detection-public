"""No candidate root entry can inspect private paths through a closed gate."""

import asyncio
import inspect
from types import SimpleNamespace

import pytest
import study_runner_fixtures as fixtures


async def forbidden(*arguments, **keywords):
    pytest.fail("closed profile entered private coordinator")


@pytest.mark.parametrize("gate", ["binding", "external", "operational"])
def test_each_closed_public_gate_precedes_all_private_context(
    tmp_path, monkeypatch, gate
):
    module = fixtures.api()
    binding = SimpleNamespace(protected_evaluation_ready=gate != "binding")
    external = SimpleNamespace(protected_evaluation_ready=gate != "external")
    operational = SimpleNamespace(
        protected_evaluation_ready=False, profile_sha256="c" * 64
    )
    monkeypatch.setattr(
        module, "bind_execution", lambda *arguments, **keywords: binding
    )
    monkeypatch.setattr(
        module, "resolve_external_source_profile", lambda binding: external
    )
    monkeypatch.setattr(
        module, "resolve_operational_profile", lambda binding: operational
    )

    monkeypatch.setattr(module, "_run_bound_study", forbidden)
    with pytest.raises(module.StudyRunError):
        asyncio.run(
            module.run_study(
                tmp_path,
                expected_revision="a" * 40,
                expected_contract_sha256="b" * 64,
                expected_operational_profile_sha256="c" * 64,
                paths=object(),
            )
        )


def test_public_entry_offers_no_resume_subset_command_or_deadline_override():
    names = set(inspect.signature(fixtures.api().run_study).parameters)
    assert names == {
        "root",
        "expected_revision",
        "expected_contract_sha256",
        "expected_operational_profile_sha256",
        "paths",
    }
