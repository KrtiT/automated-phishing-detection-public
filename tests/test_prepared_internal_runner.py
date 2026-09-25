"""A distinct retained-input route stays closed before inspecting private paths."""

import importlib
import importlib.util
from dataclasses import fields

import pytest
from test_source_runner import inputs, runner

__all__ = ["inputs", "runner"]


def module():
    name = "automated_phishing_detection.prepared_internal_runner"
    assert importlib.util.find_spec(name), "missing prepared internal runner"
    return importlib.import_module(name)


def test_prepared_paths_omit_every_original_source_path():
    api = module()
    assert [field.name for field in fields(api.PreparedInternalRunPaths)] == [
        "preparation",
        "artifacts",
        "secondary_artifacts",
        "attempt",
        "public_summary",
    ]


@pytest.mark.parametrize(
    "entry",
    ["run_prepared_internal_evaluation", "run_prepared_internal_process_with_evidence"],
)
def test_closed_prepared_entries_do_not_inspect_supplied_paths(
    inputs, monkeypatch, entry
):
    api = module()
    binding, unused_paths, unused_session, unused_events = inputs
    monkeypatch.setattr(api, "bind_execution", lambda *args, **kwargs: binding)

    def forbidden(*args, **kwargs):
        pytest.fail("closed prepared entry inspected supplied inputs")

    monkeypatch.setattr(api, "bound_preparation_context", forbidden)
    with pytest.raises(api.SourceExecutionError, match="pre_access_freeze_incomplete"):
        getattr(api, entry)(
            binding.root,
            expected_revision=binding.revision,
            expected_contract_sha256=binding.contract_sha256,
            paths=object(),
            expected_preparation_reservation_sha256="a" * 64,
            expected_preparation_completion_sha256="b" * 64,
        )
