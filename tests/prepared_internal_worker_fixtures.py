"""Actual fixture producer children; numerical bindings are explicitly synthetic."""

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from test_evaluation_producer import synthetic_session

from automated_phishing_detection import evaluation_producer, phiusiil, source_runner
from automated_phishing_detection._prepared_internal_records import (
    PreparedInternalRunPaths,
)
from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.bound_secondary import SecondaryArtifactPaths
from automated_phishing_detection.execution_preflight import ExecutionBinding


def child_command(case, mode):
    binding, paths = case.binding, case.paths
    payload = {
        "root": str(binding.root),
        "revision": binding.revision,
        "contract": binding.contract_sha256,
        "pins": binding.source_hashes,
        "runtime": binding.runtime_json,
        "identity": case.identity,
        "reservation": case.preparation.reservation_sha256,
        "completion": case.preparation.completion_sha256,
        "mode": mode,
        "preparation": str(paths.preparation),
        "attempt": str(paths.attempt),
        "public_summary": str(paths.public_summary),
        "artifacts": [str(value) for value in vars(paths.artifacts).values()],
        "secondary": [str(value) for value in vars(paths.secondary_artifacts).values()],
    }
    program = (
        f"import sys;sys.path.insert(0,{str(Path(__file__).parent)!r});"
        "from prepared_internal_worker_fixtures import run_child;run_child(sys.argv[1])"
    )
    return sys.executable, "-c", program, json.dumps(payload)


@contextmanager
def session_owner(session, mode):
    yield session
    if mode == "teardown":
        raise ValueError("invented session teardown failed")


def forbidden(*args, **kwargs):
    raise AssertionError("prepared child attempted original source reconstruction")


def child_inputs(value):
    binding = ExecutionBinding(
        Path(value["root"]),
        value["revision"],
        value["contract"],
        tuple(tuple(member) for member in value["pins"]),
        value["runtime"],
    )
    paths = PreparedInternalRunPaths(
        Path(value["preparation"]),
        ArtifactPaths(*map(Path, value["artifacts"])),
        SecondaryArtifactPaths(*map(Path, value["secondary"])),
        Path(value["attempt"]),
        Path(value["public_summary"]),
    )
    return binding, paths


def run_child(content):
    from automated_phishing_detection import prepared_internal_runner

    value = json.loads(content)
    binding, paths = child_inputs(value)
    with pytest.MonkeyPatch.context() as monkeypatch:
        session, *_ = synthetic_session(evaluation_producer, monkeypatch)
        monkeypatch.setattr(source_runner, "recheck_binding", lambda _: None)
        monkeypatch.setattr(
            source_runner,
            "open_bound_evaluation_session",
            lambda *args: session_owner(session, value["mode"]),
        )
        for name in ("_parse_csv_rows", "resolve_rows", "assign_splits"):
            monkeypatch.setattr(phiusiil, name, forbidden)
        source, buffers = source_runner._public_sources(binding)
        prepared_internal_runner._run_held(
            binding,
            paths,
            (value["identity"], source, buffers, None),
            value["reservation"],
            value["completion"],
            False,
        )
        if value["mode"] == "nonzero":
            raise SystemExit(17)
