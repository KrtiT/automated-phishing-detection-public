"""Invented decisions and paths, never evidence of real approval or access."""

from hashlib import sha256
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest
from study_execution_profile_fixtures import profile

from automated_phishing_detection import _external_source_profile as external
from automated_phishing_detection import _operational_profile as operational
from automated_phishing_detection import execution_preflight as preflight
from automated_phishing_detection._checkpoint_codec import canonical_bytes


def api():
    name = "automated_phishing_detection.study_execution"
    assert find_spec(name), "missing study-only execution authorization binder"
    return import_module(name)


def decision(kind, digest):
    return {
        "status": "approved",
        "approver": "Invented fixture reviewer",
        "recorded_at": "2026-09-28T00:00:00Z",
        "subject_kind": kind,
        "subject_sha256": digest,
        "decision_text": "Invented test decision only; no real study is authorized.",
    }


def seal(case, changed=None):
    value = profile(case) if changed is None else changed
    digest = sha256(canonical_bytes(value)).hexdigest()
    envelope = {
        "schema_version": 1,
        "envelope_id": "study-adoption-envelope-v1",
        "profile": value,
        "decisions": {
            "method": decision("method", value["method_sha256"]),
            "advisor": decision("policy", value["policy_sha256"]),
            "profile": decision("study-profile", digest),
            "access": decision("study-profile", digest),
        },
        "revoked": False,
    }
    return envelope


def write_envelope(case, envelope):
    content = canonical_bytes(envelope)
    case.envelope_path.write_bytes(content)
    return sha256(content).hexdigest()


def bind(case, envelope=None):
    digest = write_envelope(case, seal(case) if envelope is None else envelope)
    return api().bind_study_execution(
        case.root,
        expected_revision=case.base.revision,
        envelope_path=case.envelope_path,
        expected_envelope_sha256=digest,
    )


def _base(tmp_path, source):
    required = {
        *external._IMPLEMENTATIONS,
        *operational._REQUIRED,
        "reports/phiusiil-preparation-summary.json",
        "scripts/run_adopted_study.py",
        "scripts/run_study_child.py",
    }
    hashes = {name: sha256(name.encode()).hexdigest() for name in required}
    hashes["data/sources.json"] = sha256(source).hexdigest()
    return preflight.ExecutionBinding(
        tmp_path / "public",
        "a" * 40,
        "b3847581d73cba2a01d9f25d39548bb6cd96903129b0e382359faa6df97ab7c4",
        tuple(sorted(hashes.items())),
        '{"invented_runtime":true}',
    )


def _patch_public_reads(case, source, monkeypatch):
    original_read = preflight._read_regular

    def read_public(root, relative):
        if root != case.root:
            return original_read(root, relative)
        case.events.append(("public", relative))
        return {
            "data/sources.json": source,
            "data/study-execution-policy-v1.json": case.policy,
        }[relative]

    monkeypatch.setattr(preflight, "_read_regular", read_public)
    monkeypatch.setattr(preflight, "recheck_binding", lambda value: None)
    monkeypatch.setattr(preflight, "bind_execution", lambda *args, **kwargs: case.base)

    def git(root, *arguments):
        if arguments[0] == "ls-tree":
            return (
                b"100644 blob " + b"c" * 40 + b"\tdata/study-execution-policy-v1.json\0"
            )
        assert arguments == ("cat-file", "blob", "c" * 40)
        return case.policy

    monkeypatch.setattr(preflight, "_git", git)


@pytest.fixture
def execution_case(tmp_path, monkeypatch):
    api()
    repository = Path(__file__).parents[1]
    source = (repository / "data/sources.json").read_bytes()
    policy = (repository / "data/study-execution-policy-v1.json").read_bytes()
    base = _base(tmp_path, source)
    case = SimpleNamespace(
        root=base.root,
        base=base,
        policy=policy,
        events=[],
        envelope_path=tmp_path / "invented-envelope.json",
    )
    _patch_public_reads(case, source, monkeypatch)
    case.external = external.resolve_external_source_profile(case.base)
    case.operational = operational.resolve_operational_profile(case.base)
    case.events.clear()
    return case
