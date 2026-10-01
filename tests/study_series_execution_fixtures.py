"""Invented public binding; science and Git are isolated at this boundary."""

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest
from study_series_adoption_fixtures import digest, make_case, refresh

from automated_phishing_detection import _study_series_policy as policy
from automated_phishing_detection import execution_preflight as preflight
from automated_phishing_detection._checkpoint_codec import canonical_bytes


def api():
    name = "automated_phishing_detection.study_series_execution"
    assert find_spec(name), "missing shared public-only series binding"
    return import_module(name)


def seal(case):
    refresh(case)
    case.envelope_path.write_bytes(canonical_bytes(case.envelope))
    return dict(
        expected_revision=case.base.revision,
        expected_profile_sha256=digest(case.envelope["profile"]),
        envelope_path=case.envelope_path,
        expected_envelope_sha256=digest(case.envelope),
    )


@pytest.fixture
def public_case(tmp_path, monkeypatch):
    module = api()
    case = make_case()
    value = case.envelope["profile"]
    case.envelope_path = tmp_path / "invented-envelope.json"
    case.base = preflight.ExecutionBinding(
        Path(value["paths"]["repo_root"]),
        value["execution"]["revision"],
        value["execution"]["contract_sha256"],
        tuple(sorted(value["source_artifact_scope"].items())),
        "{}",
    )
    case.events = []
    case.components = SimpleNamespace(authorizes_execution=False)
    case.external, case.operational = object(), object()
    _patch(case, module, monkeypatch)
    seal(case)
    return case


def _patch(case, module, monkeypatch):

    def bound(*args, **kwargs):
        case.events.append("preflight")
        return case.base

    monkeypatch.setattr(preflight, "bind_execution", bound)
    monkeypatch.setattr(
        preflight, "recheck_binding", lambda base: case.events.append("recheck")
    )
    monkeypatch.setattr(
        module,
        "_components",
        lambda *args: (case.external, case.operational, case.components),
    )
    monkeypatch.setattr(
        module.admission_io, "committed_policy", lambda *args: policy.policy_bytes()
    )


def bind(case, **overrides):
    return api().bind_series_public_execution(
        case.base.root, **(seal(case) | overrides)
    )
