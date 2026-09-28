"""A complete invented study envelope never opens any legacy standalone route."""

import asyncio
import importlib
import inspect

import pytest
from study_execution_fixtures import bind, execution_case

__all__ = ["execution_case"]

ROUTES = (
    ("source_runner", "run_internal_evaluation"),
    ("source_runner", "run_internal_process"),
    ("source_runner", "run_internal_process_with_evidence"),
    ("prepared_internal_runner", "run_prepared_internal_evaluation"),
    ("prepared_internal_runner", "run_prepared_internal_process_with_evidence"),
    ("external_source_runner", "run_external_evaluation"),
    ("external_source_runner", "run_prepared_external_evaluation"),
    ("external_source_process", "run_internal_external_process"),
    ("prepared_external_process", "run_prepared_internal_external_process"),
    ("study_preparation_runner", "run_study_preparation"),
    ("study_runner", "run_study"),
    ("operational_cell_service", "run_operational_service"),
    ("operational_cell_client", "run_operational_client"),
)


class UninspectablePaths:
    def __getattribute__(self, name):
        raise AssertionError("legacy gate inspected a protected path")


def arguments(function, authorization):
    values = {
        "root": authorization.base.root,
        "expected_revision": authorization.base.revision,
        "expected_contract_sha256": authorization.base.contract_sha256,
        "expected_operational_profile_sha256": authorization.operational.profile_sha256,
    }
    return {
        name: values.get(name, UninspectablePaths())
        for name in inspect.signature(function).parameters
    }


@pytest.mark.parametrize("module_name,function_name", ROUTES)
def test_study_envelope_does_not_admit_legacy_execution(
    execution_case, monkeypatch, module_name, function_name
):
    authorization = bind(execution_case)
    module = importlib.import_module(f"automated_phishing_detection.{module_name}")
    target = module.context if module_name.startswith("operational_cell_") else module
    monkeypatch.setattr(
        target, "bind_execution", lambda *args, **kwargs: authorization.base
    )
    function = getattr(module, function_name)
    assert authorization.base.protected_evaluation_ready is False
    assert authorization.external.protected_evaluation_ready is False
    assert authorization.operational.protected_evaluation_ready is False
    with pytest.raises(ValueError, match="pre_access_freeze_incomplete"):
        result = function(**arguments(function, authorization))
        if inspect.isawaitable(result):
            asyncio.run(result)
