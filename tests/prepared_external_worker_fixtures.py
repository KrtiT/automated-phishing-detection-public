"""Real owned children with invented retained bytes and fixture numerical bindings."""

import json
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from prepared_external_fixtures import _configure, _scientific

from automated_phishing_detection import external_source_runner as worker
from automated_phishing_detection import study_preparation_context as context
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_source_profile import (
    CandidateExternalProfile,
)
from automated_phishing_detection._prepared_external_records import (
    PreparedExternalRunPaths,
)
from automated_phishing_detection._prepared_external_runtime import held_preparation
from automated_phishing_detection.bound_drift import DriftArtifactPaths
from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.bound_secondary import SecondaryArtifactPaths
from automated_phishing_detection.execution_preflight import ExecutionBinding


def child_command(case, transport, mode):
    value = {
        "binding": asdict(case.binding),
        "paths": asdict(case.paths),
        "profile": case.profile.projection(),
        "mode": mode,
        "reservation": case.preparation.reservation_sha256,
        "completion": case.preparation.completion_sha256,
        "transport": str(transport.directory),
        "handoff": transport.expected_handoff_sha256,
    }
    program = (
        f"import sys;sys.path.insert(0,{str(Path(__file__).parent)!r});"
        "from prepared_external_worker_fixtures import run_child;run_child(sys.argv[1])"
    )
    return sys.executable, "-c", program, json.dumps(value, default=str)


def child_case(value):
    binding, paths = value["binding"], value["paths"]
    binding["root"] = Path(binding["root"])
    binding["source_hashes"] = tuple(map(tuple, binding["source_hashes"]))
    paths["artifacts"] = ArtifactPaths(
        **{name: Path(path) for name, path in paths["artifacts"].items()}
    )
    paths["secondary_artifacts"] = SecondaryArtifactPaths(
        **{name: Path(path) for name, path in paths["secondary_artifacts"].items()}
    )
    paths["drift_artifacts"] = DriftArtifactPaths(
        **{name: Path(path) for name, path in paths["drift_artifacts"].items()}
    )
    for name in ("preparation", "attempt", "public_summary"):
        paths[name] = Path(paths[name])
    return SimpleNamespace(
        binding=ExecutionBinding(**binding),
        paths=PreparedExternalRunPaths(**paths),
        profile=CandidateExternalProfile(canonical_bytes(value["profile"])),
        events=[],
        active=False,
    )


@contextmanager
def session_owner(case, mode):
    yield case.session
    if mode == "teardown":
        raise ValueError("invented external session teardown failed")


def forbidden(*args, **kwargs):
    raise AssertionError("prepared child reacquired original archive")


def run_child(content):
    value = json.loads(content)
    case = child_case(value)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            context, "resolve_external_source_profile", lambda _: case.profile
        )
        with held_preparation(
            case.binding, case.paths, value["reservation"], value["completion"]
        ) as preparation:
            case.preparation = preparation
            _scientific(case, monkeypatch)
            _configure(case, monkeypatch)
            monkeypatch.setattr(worker.body, "_source", forbidden)
            monkeypatch.setattr(
                worker,
                "open_bound_external_session",
                lambda *args: session_owner(case, value["mode"]),
            )
            handoff = worker.read_internal_handoff_transport(
                Path(value["transport"]), expected_handoff_sha256=value["handoff"]
            )
            worker._run_bound_prepared_external(
                case.binding, case.paths, handoff=handoff, preparation=preparation
            )
        if value["mode"] == "nonzero":
            raise SystemExit(17)
