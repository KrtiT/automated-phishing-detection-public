"""Invented public acceptance chains around invented retained drift snapshots."""

import copy
import importlib
import json
from dataclasses import asdict, replace
from hashlib import sha256
from types import ModuleType, SimpleNamespace

import pytest
from test_retained_drift import _public_bytes, _rebind, _snapshots

from automated_phishing_detection import (
    bound_drift,
    bound_secondary,
    development_execution,
    phiusiil,
    seed_probe_execution,
)


def snapshot_chain() -> tuple[dict, bytes]:
    data = _snapshots()
    preparation = json.loads(data["preparation_summary"])
    source = copy.deepcopy(preparation["declared_sources"])
    source["phiusiil"].update(phiusiil._OFFICIAL_PHIUSIIL_METADATA)
    source["public_suffix_list"].update(phiusiil._OFFICIAL_PSL_METADATA)
    source_bytes = _public_bytes(source)
    preparation["declared_sources"] = source
    preparation["source_spec_sha256"] = sha256(source_bytes).hexdigest()
    preparation_bytes = _public_bytes(preparation)
    pins = replace(
        data["pins"], preparation_summary_sha256=sha256(preparation_bytes).hexdigest()
    )
    reference = json.loads(data["training_reference"])
    audit = json.loads(data["validation_audit"])
    public = data["expected_drift_summary"]
    for document in (reference, audit, public):
        document["input_hashes"] = asdict(pins)
    return _rebind(reference, audit, public, preparation_bytes, pins), source_bytes


def _drift_member(data: dict) -> dict:
    summary = {
        "member": "drift",
        "status": "development_member_completed",
        "result": data["expected_drift_summary"],
        "private_sha256": data["expected_drift_summary"]["private_sha256"],
    }
    return {
        "member": "drift",
        "summary": summary,
        "public_summary_sha256": seed_probe_execution._digest(summary),
        "checks": {
            "authenticated_training_and_validation_membership": True,
            "independent_drift_score_recomputation": False,
            "retained_drift_arithmetic": True,
        },
    }


def _retained_audit(data: dict) -> dict:
    return {
        "stage": "retained_audit",
        "status": "development_correction_stage_completed",
        "result": {
            "status": "retained_development_members_audited",
            "protected_evaluation_authorized": False,
            "original_aggregate_accepted": False,
            "analysis_stage": "development_validation_only",
            "fits": 0,
            "input_hashes": asdict(data["pins"]),
            "members": [
                _drift_member(data),
                *({"member": name} for name in development_execution.STEPS[1:-1]),
            ],
        },
    }


def accepted_report(data: dict) -> bytes:
    completion = {
        "status": "completed_secondary_development_correction",
        "worker_exit_codes": {"random_forest": 0, "retained_audit": 0},
        "protected_evaluation_authorized": False,
        "original_aggregate_accepted": False,
        "analysis_stage": "development_validation_only",
        "execution": {"pins": asdict(data["pins"])},
        "retained_audit": _retained_audit(data),
    }
    return _public_bytes(
        {
            "status": "accepted_development_evidence",
            "completion": completion,
            "completion_summary_sha256": seed_probe_execution._digest(completion),
            "execution_observation": {
                "parent_exit_code": 0,
                "worker_exit_codes": completion["worker_exit_codes"],
            },
        }
    )


@pytest.fixture
def restorer() -> ModuleType:
    name = "automated_phishing_detection.retained_external_drift"
    assert importlib.util.find_spec(name), "missing byte-only accepted drift restorer"
    return importlib.import_module(name)


@pytest.fixture
def chain(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    data, source = snapshot_chain()
    report = accepted_report(data)
    public = (
        (bound_drift._REPORT, report),
        (bound_drift._PREPARATION, data["preparation_summary"]),
        (bound_drift._SOURCE, source),
    )
    monkeypatch.setitem(
        bound_secondary.PUBLIC_REPORTS,
        "tabular",
        (bound_drift._REPORT, sha256(report).hexdigest()),
    )
    return SimpleNamespace(
        data=data,
        public=public,
        arguments=(data["training_reference"], data["validation_audit"], public),
    )
