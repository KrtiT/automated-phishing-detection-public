"""Invented external publications and independent byte expectations only."""

import importlib
import importlib.util
import json
from hashlib import sha256
from types import SimpleNamespace

from external_completion_fixtures import external_completion_case

from automated_phishing_detection import execution_receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_completion_records import _LOGICAL_NAMES
from automated_phishing_detection._external_preparation_outputs import _PUBLIC_INPUTS


def api():
    name = "automated_phishing_detection.study_history_external"
    assert importlib.util.find_spec(name), "missing historical external science"
    return importlib.import_module(name)


def digest(content):
    return sha256(content).hexdigest()


def arguments(case, payloads):
    public = json.loads(payloads["public-summary.json"])
    return dict(
        expected_snapshot_sha256={
            name: digest(value) for name, value in payloads.items()
        },
        expected_execution=public["execution"],
        expected_source_sha256={
            name: dict(case.binding.source_hashes)[name]
            for name, unused in _PUBLIC_INPUTS
        },
        expected_profile_bytes=case.profile.canonical_bytes,
        expected_handoff_bytes=case.handoff.handoff_bytes,
        expected_overlap_bytes=case.handoff.overlap_bytes,
        expected_attempt_directory=str(case.paths.attempt),
    )


def make_case(tmp_path, monkeypatch, count=5):
    case = external_completion_case(tmp_path, monkeypatch, count)
    payloads = {
        name: (
            case.paths.public_summary
            if name == "public-summary.json"
            else case.paths.attempt / name.removeprefix("attempt/")
        ).read_bytes()
        for name in _LOGICAL_NAMES
    }
    return SimpleNamespace(
        case=case, payloads=payloads, arguments=arguments(case, payloads)
    )


def verify(history, **changes):
    return api().verify_historical_external_science(
        history.payloads, **(history.arguments | changes)
    )


def repin(history):
    history.arguments["expected_snapshot_sha256"] = {
        name: digest(content) for name, content in history.payloads.items()
    }


def rewrite_public(history, public):
    history.payloads["public-summary.json"] = execution_receipt._json_bytes(
        public, "unit"
    )
    outcome = json.loads(history.payloads["attempt/outcome.json"])
    outcome["public_summary_sha256"] = digest(history.payloads["public-summary.json"])
    history.payloads["attempt/outcome.json"] = execution_receipt._json_bytes(
        outcome, "unit"
    )
    repin(history)


def replace_output(history, name, content):
    for directory in ("checkpoints", "evidence"):
        history.payloads[f"attempt/{directory}/{name}"] = content
    public = json.loads(history.payloads["public-summary.json"])
    for field in ("private_sha256", "checkpoint_sha256"):
        public[field][name] = digest(content)
    if name in public["composition"]["private_sha256"]:
        public["composition"]["private_sha256"][name] = digest(content)
    outcome = json.loads(history.payloads["attempt/outcome.json"])
    outcome["private_sha256"][name] = digest(content)
    history.payloads["attempt/outcome.json"] = execution_receipt._json_bytes(
        outcome, "unit"
    )
    rewrite_public(history, public)


def mutate_output(history, name, change):
    value = json.loads(history.payloads[f"attempt/evidence/{name}"])
    change(value)
    replace_output(history, name, canonical_bytes(value))
