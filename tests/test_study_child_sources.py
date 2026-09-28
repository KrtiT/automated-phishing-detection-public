"""Study source roles reuse held preparation, never original raw input paths."""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest


def api():
    name = "automated_phishing_detection._study_child_sources"
    assert importlib.util.find_spec(name), "missing prepared child roles"
    return importlib.import_module(name)


def held():
    authorization = SimpleNamespace(
        base=object(), paths=SimpleNamespace(internal=object(), external=object())
    )
    frame = SimpleNamespace(
        preparation_reservation_sha256="a" * 64,
        preparation_completion_sha256="b" * 64,
        predecessor_sha256="c" * 64,
    )
    return SimpleNamespace(
        authorization=authorization, admission=SimpleNamespace(frame=frame)
    )


def test_internal_reuses_nonobserving_prepared_worker_body(monkeypatch):
    module, child, seen = api(), held(), []
    context = object()
    monkeypatch.setattr(module, "bound_preparation_context", lambda binding: context)
    checks = []
    monkeypatch.setattr(
        module, "recheck_held_child", lambda value: checks.append(value)
    )

    def run(*args, lifecycle_check):
        lifecycle_check()
        seen.append(args)

    monkeypatch.setattr(module, "_run_held", run)
    module.run_internal(child)
    assert checks == [child]
    assert seen == [
        (
            child.authorization.base,
            child.authorization.paths.internal,
            context,
            "a" * 64,
            "b" * 64,
            False,
        )
    ]


def test_external_reuses_exact_handoff_and_preparation(monkeypatch):
    module, child, seen = api(), held(), []
    args = SimpleNamespace(
        internal_transport=object(), expected_handoff_sha256="c" * 64
    )
    checks = []
    monkeypatch.setattr(
        module, "recheck_held_child", lambda value: checks.append(value)
    )

    def run(*arguments, lifecycle_check):
        lifecycle_check()
        seen.append(arguments)

    monkeypatch.setattr(module, "run_held_preparation", run)
    module.run_external(child, args)
    assert checks == [child]
    assert seen == [
        (
            child.authorization.base,
            child.authorization.paths.external,
            args.internal_transport,
            "c" * 64,
            "a" * 64,
            "b" * 64,
        )
    ]


def test_wrong_external_predecessor_rejects_before_input_read(monkeypatch):
    module, child = api(), held()
    args = SimpleNamespace(
        internal_transport=object(), expected_handoff_sha256="d" * 64
    )
    monkeypatch.setattr(
        module, "run_held_preparation", lambda *args: pytest.fail("input opened")
    )
    with pytest.raises(ValueError):
        module.run_external(child, args)
