import importlib
import importlib.util
import json
from hashlib import sha256
from types import SimpleNamespace

import pytest
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)

__all__ = ["inputs", "preparation_api", "preparation_case", "runner", "retained_case"]


def module():
    name = "automated_phishing_detection.study_preparation_transport"
    assert importlib.util.find_spec(name), "missing held preparation reader"
    return importlib.import_module(name)


@pytest.fixture
def retained_case(preparation_case, preparation_api):
    case = preparation_case
    snapshot = preparation_api._run_bound_preparation(case.binding, case.paths)
    completion = snapshot.payload("preparation-complete.json")
    expected = {
        "expected_identity": json.loads(completion)["execution"],
        "expected_reservation_sha256": snapshot.reservation_sha256,
        "expected_completion_sha256": sha256(completion).hexdigest(),
        "source_spec_bytes": (case.binding.root / "data/sources.json").read_bytes(),
        "preparation_summary_bytes": (
            case.binding.root / "reports/phiusiil-preparation-summary.json"
        ).read_bytes(),
    }
    return SimpleNamespace(
        case=case, directory=case.paths.attempt, snapshot=snapshot, expected=expected
    )
