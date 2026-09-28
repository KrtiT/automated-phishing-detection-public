"""One adopted cell reuses the original supervisor with exact role admissions."""

import importlib
import importlib.util
from hashlib import sha256
from types import SimpleNamespace

import pytest
from test_study_child_commands import authorization


def api():
    name = "automated_phishing_detection._study_authorized_cell"
    assert importlib.util.find_spec(name), "missing adopted cell adapter"
    return importlib.import_module(name)


def test_each_role_binds_actual_held_cell_inputs():
    module, calls = api(), []
    auth = authorization()
    inputs = SimpleNamespace(
        cell=SimpleNamespace(ordinal=3),
        binding_sha256="c" * 64,
        accepted_bytes=b"invented accepted",
    )
    ledger = SimpleNamespace(
        authorization=auth,
        source_results_sha256="d" * 64,
        issue=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    commands, admit = module.commands_and_admissions(ledger, inputs)
    for role, command in zip(("service", "client"), commands, strict=True):
        admit(role, command)
    assert [entry[0][0] for entry in calls] == ["service", "client"]
    for unused, keywords in calls:
        assert keywords == {
            "predecessor_sha256": "d" * 64,
            "accepted_inputs_sha256": sha256(inputs.accepted_bytes).hexdigest(),
            "cell_binding_sha256": "c" * 64,
        }


@pytest.mark.parametrize("role", ["external", "internal", "service"])
def test_no_different_role_or_command_can_obtain_cell_admission(role):
    module = api()
    inputs = SimpleNamespace(
        cell=SimpleNamespace(ordinal=3),
        binding_sha256="c" * 64,
        accepted_bytes=b"invented accepted",
    )
    ledger = SimpleNamespace(
        authorization=authorization(),
        source_results_sha256="d" * 64,
        issue=lambda *args, **kwargs: pytest.fail("wrong role/argv admitted"),
    )
    unused, admit = module.commands_and_admissions(ledger, inputs)
    with pytest.raises(ValueError):
        admit(role, ("invented", "other-command"))
