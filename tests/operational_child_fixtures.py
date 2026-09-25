"""Invented child contexts never authorize protected execution."""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest


def child_module(role):
    name = f"automated_phishing_detection.operational_cell_{role}"
    assert importlib.util.find_spec(name), "missing fixed operational child boundary"
    return importlib.import_module(name)


@pytest.fixture
def child_context():
    name = "automated_phishing_detection._operational_child_context"
    assert importlib.util.find_spec(name), "missing operational child context"
    return importlib.import_module(name)


def options(role):
    result = {
        "expected_revision": "a" * 40,
        "expected_contract_sha256": "b" * 64,
        "expected_operational_profile_sha256": "c" * 64,
        "accepted_inputs_directory": object(),
        "cell_input_directory": object(),
        "expected_binding_sha256": "d" * 64,
    }
    if role == "service":
        result["artifacts"] = object()
    return result


def ready(value=True):
    return SimpleNamespace(protected_evaluation_ready=value, profile_sha256="c" * 64)
