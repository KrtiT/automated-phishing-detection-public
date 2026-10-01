"""Synthetic owned-observation arguments, never evidence of actual study exits."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._operational_cell_protocol import WORKING_NAMES
from automated_phishing_detection._operational_process_records import ProcessObservation


def api():
    name = "automated_phishing_detection.study_series_cell_acceptance"
    assert importlib.util.find_spec(name), "missing series working-cell acceptance"
    return importlib.import_module(name)


def arguments(case):
    original = case.arguments
    return {
        "attempt": receipt.Attempt(
            Path(original["expected_attempt_directory"]),
            original["expected_cell_reservation_sha256"],
        ),
        "expected_identity": json.loads(case.values["attempt/reservation.json"])[
            "identity"
        ],
        "inputs": case.inputs,
        **{
            name: original[name]
            for name in (
                "profile_bytes",
                "expected_profile_sha256",
                "expected_metadata_sha256",
                "internal_snapshot",
                "external_snapshot",
            )
        },
        "observation": ProcessObservation(case.values["attempt/process-pair.json"]),
        "service_command": ("/invented/python", "service"),
        "client_command": ("/invented/python", "client"),
        "expected_deadlines": dict(startup=300, shutdown=180, terminate=10, kill=10),
    }


def payloads(case):
    return tuple((name, case.values[f"attempt/{name}"]) for name in WORKING_NAMES)


def working(case, *, values=None, **changes):
    return api().verify_series_working_cell(
        payloads(case) if values is None else values, **(arguments(case) | changes)
    )


@pytest.fixture(scope="module")
def accepted_working(fresh_cell):
    return working(fresh_cell)


def publish(case, verified, *, values=None, content=None):
    return api().verify_series_published_cell(
        tuple(case.values.items()) if values is None else values,
        working=verified,
        expected_public_bytes=case.values["public-summary.json"]
        if content is None
        else content,
    )


def forbid(*arguments, **keywords):
    pytest.fail("cell acceptance fabricated ownership or reached forbidden computation")


def forbid_constructors(monkeypatch):
    names = {
        "ProcessObservation",
        "AcceptedOperationalInputs",
        "VerifiedOperationalCell",
        "VerifiedOperationalWorking",
        "ExecutionBinding",
        "WorkerObservation",
    }
    guarded = set()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("automated_phishing_detection."):
            continue
        for value in tuple(vars(module).values()):
            if isinstance(value, type) and (
                value.__name__ in names or value.__name__.startswith("Observed")
            ):
                if value not in guarded:
                    monkeypatch.setattr(value, "__init__", forbid)
                    guarded.add(value)
    assert len(guarded) >= len(names)
