import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from automated_phishing_detection import execution_receipt as receipt

NAMES = (
    "repo-root",
    "expected-revision",
    "expected-contract-sha256",
    "expected-operational-profile-sha256",
    "source-csv",
    "suffix-rules",
    "archive",
    "preparation-attempt",
    "internal-attempt",
    "internal-public-summary",
    "external-attempt",
    "external-public-summary",
    "attempt",
    "public-summary",
    "accepted-inputs-dir",
    "cells-dir",
    "length-only",
    "logistic-l1",
    "transformer-bundle",
    "gmm",
    "formatting",
    "permutation-42",
    "permutation-43",
    "permutation-44",
    "permutation-45",
    "permutation-46",
    "random-forest",
    "seed-43-weights",
    "seed-44-weights",
    "seed-45-weights",
    "seed-46-weights",
    "training-reference",
    "validation-audit",
)
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_study.py"


@pytest.fixture
def cli():
    assert SCRIPT.is_file(), "missing whole-study CLI"
    specification = importlib.util.spec_from_file_location("study_cli_fixture", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def arguments():
    return [value for name in NAMES for value in (f"--{name}", f"invented/{name}")]


def result(status):
    content = receipt._json_bytes({"status": status}, "fixture")
    return SimpleNamespace(snapshot=SimpleNamespace(payload=lambda name: content))
