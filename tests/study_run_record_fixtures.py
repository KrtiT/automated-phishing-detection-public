"""Invented retained preparation and caller-only branch fixtures, never authority."""

import importlib
import importlib.util
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from operational_profile_fixtures import REQUIRED
from retained_study_preparation_fixtures import (
    completion,
    make_case,
    restore,
    retained_outputs,
)
from study_feasibility_fixtures import assess, external_rows, internal_rows

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection import retained_study_preparation
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_profile import (
    CandidateOperationalProfile,
    _projection,
)
from automated_phishing_detection.execution_preflight import ExecutionBinding

EXECUTION_FIELDS = (
    "revision",
    "execution_contract_sha256",
    "runtime_sha256",
    "source_spec_sha256",
)
REQUIREMENTS = (
    "internal_negative_rows",
    "internal_positive_domains",
    "certified_rows",
    "gold_positive_domains",
    "tranco_rows",
    "external_complete_windows",
    "http_100_negative_rows",
    "http_100_positive_rows",
    "http_10_negative_rows",
    "http_10_positive_rows",
    "http_500_negative_rows",
    "http_500_positive_rows",
    "shift_warmup_rows",
)


def api():
    name = "automated_phishing_detection._study_run_records"
    assert importlib.util.find_spec(name), "missing pure whole-study records"
    return importlib.import_module(name)


def attempt(identity):
    directory = Path("/invented/study")
    content = receipt._json_bytes(
        {
            "schema_version": 1,
            "status": "reserved",
            "directory": str(directory),
            "identity": identity,
        },
        "fixture",
    )
    return receipt.Attempt(directory, sha256(content).hexdigest())


@pytest.fixture(scope="module")
def prepared():
    source = make_case()
    source.identity["runtime_sha256"] = sha256(b"{}").hexdigest()
    outputs = retained_outputs(source)
    outputs["preparation-complete.json"] = completion(
        outputs, source.identity, source.reservation
    )
    source.snapshot = replace(source.snapshot, payloads=tuple(outputs.items()))
    source.completion = sha256(outputs["preparation-complete.json"]).hexdigest()
    return bound_case(source, restore(retained_study_preparation, source))


def bound_case(source, restored):
    hashes = {name: sha256(name.encode()).hexdigest() for name in REQUIRED}
    hashes["data/sources.json"] = restored.execution["source_spec_sha256"]
    binding = ExecutionBinding(
        Path("/invented/checkout"),
        source.identity["revision"],
        source.identity["execution_contract_sha256"],
        tuple(sorted(hashes.items())),
        "{}",
    )
    profile = CandidateOperationalProfile(canonical_bytes(_projection(binding)))
    identity = {
        "kind": "whole_study",
        "protocol": "study-root-v1",
        **{name: restored.execution[name] for name in EXECUTION_FIELDS},
        "operational_profile_sha256": profile.profile_sha256,
    }
    reserved = attempt(identity)
    execution = identity | {"reservation_sha256": reserved.reservation_sha256}
    return SimpleNamespace(
        preparation=restored,
        execution=execution,
        binding=binding,
        profile=profile,
        attempt=reserved,
    )


def branch(prepared, feasibility):
    payloads = dict(prepared.preparation.payloads)
    payloads["feasibility.json"] = canonical_bytes(feasibility)
    completion = json.loads(payloads["preparation-complete.json"])
    completion["input_sha256"]["feasibility.json"] = sha256(
        payloads["feasibility.json"]
    ).hexdigest()
    payloads["preparation-complete.json"] = canonical_bytes(completion)
    return replace(
        prepared.preparation,
        payloads=tuple(payloads.items()),
        completion_sha256=sha256(payloads["preparation-complete.json"]).hexdigest(),
    )


def isolated_shortage(prepared, name):
    feasibility = assess((), external_rows(()))
    feasibility["shortages"] = [
        item for item in feasibility["shortages"] if item["requirement"] == name
    ]
    return branch(prepared, feasibility)


def capacity(prepared):
    roles = ("gold", "gold", "certified", "tranco") + ("silver",) * 996
    return branch(prepared, assess(internal_rows(9990, 500, 2), external_rows(roles)))
