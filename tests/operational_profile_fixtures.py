"""Invented public identities; no research source or artifact is opened."""

from dataclasses import replace
from hashlib import sha256
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest

from automated_phishing_detection import execution_preflight as preflight

COMMON = [
    "--repo-root",
    "--expected-revision",
    "--expected-contract-sha256",
    "--expected-operational-profile-sha256",
    "--accepted-inputs-dir",
    "--cell-input-dir",
    "--expected-binding-sha256",
]
WORKING = [
    "reservation.json",
    "process-pair-intent.json",
    "service-intent.json",
    "service-started.json",
    "service-process.json",
    "client-intent.json",
    "client-started.json",
    "client-process.json",
    "service-role.json",
    "client-role.json",
    "service-ready.json",
    "service-stop.json",
    "service-cleanup.json",
    "warmup.json",
    "measured.json",
    "run.json",
    "process-pair.json",
]
STUDY_REQUIRED = (
    "scripts/run_study.py",
    "scripts/run_prepared_internal_evaluation.py",
    "scripts/run_prepared_external_evaluation.py",
    *(
        f"src/automated_phishing_detection/{name}.py"
        for name in (
            "_study_cli_protocol",
            "_study_profile",
            "study_runner",
            "_study_run_body",
            "_study_run_schema",
            "_study_root_records",
            "study_preparation_retention",
            "_internal_handoff_validation",
            "_external_completion_files",
            "_external_completion_records",
        )
    ),
)
REQUIRED = (
    "data/sources.json",
    "reports/phiusiil-preparation-summary.json",
    "data/operational-workloads-v1.json",
    "data/http-replay-contract-v1.json",
    "data/shift-execution-contract-v1.json",
    "data/evaluation-manifest-contract-v1.json",
    "src/automated_phishing_detection/_operational_profile.py",
    "src/automated_phishing_detection/_operational_cell_protocol.py",
    "src/automated_phishing_detection/operational_schedule.py",
    "scripts/run_operational_service.py",
    "scripts/run_operational_client.py",
    "pyproject.toml",
    "uv.lock",
    *STUDY_REQUIRED,
)


def api():
    name = "automated_phishing_detection._operational_profile"
    assert find_spec(name) is not None, "missing closed operational profile"
    return import_module(name)


@pytest.fixture
def profile_case(monkeypatch):
    hashes = {name: sha256(name.encode()).hexdigest() for name in REQUIRED}
    hashes["src/automated_phishing_detection/other_bound_module.py"] = "f" * 64
    binding = preflight.ExecutionBinding(
        Path("/invented/public-checkout"),
        "a" * 40,
        "b" * 64,
        tuple(sorted(hashes.items())),
        '{"invented_runtime":true}',
    )
    case = SimpleNamespace(binding=binding, hashes=hashes, checks=[])
    monkeypatch.setattr(preflight, "recheck_binding", case.checks.append)
    return case


def resolve(case, **changes):
    return api().resolve_operational_profile(replace(case.binding, **changes))
