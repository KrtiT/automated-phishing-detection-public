"""Install only public metadata and numerical fixtures in actual study children."""

import atexit
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from study_authorized_process_metadata_fixtures import patch_public
from study_authorized_process_science_fixtures import install_science

from automated_phishing_detection.execution_preflight import ExecutionBinding

PATCH = None


def bootstrap():
    global PATCH
    value = json.loads(Path(os.environ["STUDY_PROCESS_FIXTURE"]).read_bytes())
    base = value["base"]
    base["root"] = Path(base["root"])
    base["source_hashes"] = tuple(map(tuple, base["source_hashes"]))
    case = SimpleNamespace(
        base=ExecutionBinding(**base),
        policy=value["policy"].encode(),
        archive_pins=value["archive_pins"],
    )
    PATCH = pytest.MonkeyPatch()
    install_science(case, PATCH, write_public=False)
    patch_public(case, PATCH)
    _operational(PATCH)
    if value["mode"] == "nonzero":
        atexit.register(lambda: os._exit(17))


def _operational(monkeypatch):
    if "--role" not in sys.argv:
        return
    role = sys.argv[sys.argv.index("--role") + 1]
    if role not in ("service", "client"):
        return
    from operational_cell_runner_children import install_synthetic_owner

    from automated_phishing_detection import _operational_child_context as context

    monkeypatch.setattr(context, "recheck_binding", lambda binding: None)
    if role == "service":
        install_synthetic_owner()
