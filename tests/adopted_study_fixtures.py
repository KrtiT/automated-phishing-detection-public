"""Invented authorization records around existing capacity and retained fixtures."""

import asyncio
import importlib
import importlib.util
from hashlib import sha256
from types import SimpleNamespace

import study_runner_fixtures as original

from automated_phishing_detection._checkpoint_codec import canonical_bytes


def api(name="adopted_study_runner"):
    qualified = f"automated_phishing_detection.{name}"
    assert importlib.util.find_spec(qualified), "missing adopted study coordinator"
    return importlib.import_module(qualified)


def authorization(case):
    policy = canonical_bytes({"invented": "policy"})
    profile = canonical_bytes({"invented": "profile"})
    envelope = canonical_bytes({"invented": "envelope"})
    return SimpleNamespace(
        base=case.binding,
        operational=case.profile,
        paths=case.paths,
        deadlines=case.deadlines,
        policy_bytes=policy,
        profile_bytes=profile,
        envelope_bytes=envelope,
        policy_sha256=sha256(policy).hexdigest(),
        profile_sha256=sha256(profile).hexdigest(),
        envelope_sha256=sha256(envelope).hexdigest(),
    )


def setup(tmp_path, prepared, monkeypatch, preparation=None):
    case = original.setup(tmp_path, prepared, monkeypatch, preparation)
    case.module = api()
    case.authorization = authorization(case)
    monkeypatch.setattr(case.module, "recheck_study_execution", lambda auth: None)
    return case


def execute(case):
    return asyncio.run(case.module._run_adopted_bound(case.authorization))
