"""Adopted source wiring preserves actual parent acceptance ordering."""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest


def api():
    name = "automated_phishing_detection._study_authorized_sources"
    assert importlib.util.find_spec(name), "missing adopted source composition"
    return importlib.import_module(name)


def external_stub(handoff, preparation, admissions, events, external):
    def run_external(*args, **kwargs):
        assert args[2] is handoff
        assert kwargs == {"preparation": preparation, "study_admissions": admissions}
        events.append("external")
        return external

    return run_external


def setup(monkeypatch):
    module = api()
    events = []
    authorization = SimpleNamespace(
        base=object(), paths=SimpleNamespace(internal=object(), external=object())
    )
    internal, external, handoff, preparation = (object() for unused in range(4))
    admissions = SimpleNamespace(
        authorization=authorization,
        internal_accepted=lambda value: events.append(("internal_accepted", value)),
        external_accepted=lambda value: events.append(("external_accepted", value)),
    )
    monkeypatch.setattr(module, "_validate", lambda *args: None)
    monkeypatch.setattr(
        module, "recheck_study_execution", lambda value: events.append("recheck")
    )
    monkeypatch.setattr(module, "build_internal_handoff", lambda value: handoff)

    def run_internal(*args, **kwargs):
        assert kwargs == {"preparation": preparation, "study_admissions": admissions}
        events.append("internal")
        return internal

    monkeypatch.setattr(module, "_run_observed_prepared_internal", run_internal)
    monkeypatch.setattr(
        module,
        "_run_observed_external",
        external_stub(handoff, preparation, admissions, events, external),
    )
    return SimpleNamespace(**locals())


def test_admission_follows_each_actual_saved_completion(monkeypatch):
    case = setup(monkeypatch)
    result = case.module.run_adopted_sources(
        case.authorization, case.preparation, case.admissions
    )
    assert result.preparation is case.preparation
    assert result.internal is case.internal and result.external is case.external
    assert case.events == [
        "recheck",
        "internal",
        ("internal_accepted", case.internal),
        "external",
        ("external_accepted", case.external),
        "recheck",
    ]


def test_failed_internal_never_launches_external(monkeypatch):
    case = setup(monkeypatch)

    def fail(*args, **kwargs):
        raise ValueError("invented_internal_failure")

    monkeypatch.setattr(case.module, "_run_observed_prepared_internal", fail)
    with pytest.raises(ValueError) as caught:
        case.module.run_adopted_sources(
            case.authorization, case.preparation, case.admissions
        )
    assert case.events == ["recheck"]
    assert caught.value.study_preparation is case.preparation


def test_external_failure_retains_actual_internal_and_preparation(monkeypatch):
    case = setup(monkeypatch)

    def fail(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(case.module, "_run_observed_external", fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        case.module.run_adopted_sources(
            case.authorization, case.preparation, case.admissions
        )
    assert caught.value.source_internal is case.internal
    assert caught.value.study_preparation is case.preparation
    assert "external_accepted" not in [
        entry[0] for entry in case.events if isinstance(entry, tuple)
    ]
