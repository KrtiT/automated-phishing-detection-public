"""Invented working bytes for filesystem tests, not scientific process proof."""

import importlib
from dataclasses import dataclass
from types import SimpleNamespace

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._operational_cell_protocol import (
    PRIVATE_NAMES,
    WORKING_NAMES,
)
from automated_phishing_detection._operational_process_records import ProcessObservation

IDENTITY = {"kind": "invented_cell"}


def module():
    return importlib.import_module(
        "automated_phishing_detection.operational_cell_completion"
    )


@dataclass(frozen=True)
class Working:
    payloads: tuple

    @property
    def private_outputs(self):
        return {
            name: content for name, content in self.payloads if name in PRIVATE_NAMES
        }


@dataclass(frozen=True)
class Snapshot:
    payloads: tuple


def setup(tmp_path, monkeypatch):
    attempt = receipt.reserve_attempt(tmp_path.resolve() / "attempt", identity=IDENTITY)
    case = SimpleNamespace(
        attempt=attempt, public=tmp_path.resolve() / "public.json", calls=[]
    )
    case.options = {
        "inputs": object(),
        "accepted": object(),
        "observation": ProcessObservation(b"invented actual-observer shape"),
        "service_command": ("invented", "service"),
        "client_command": ("invented", "client"),
        "expected_deadlines": dict.fromkeys(
            ("startup", "shutdown", "terminate", "kill"), 1.0
        ),
    }

    def verify(payloads, **expected):
        case.calls.append(("working", payloads, expected))
        return Working(payloads)

    def publish(payloads, **expected):
        case.calls.append(("published", payloads, expected))
        return Snapshot(payloads)

    monkeypatch.setattr(module(), "_verify_working", verify)
    monkeypatch.setattr(module(), "_verify_published", publish)
    monkeypatch.setattr(
        module(), "_build_public", lambda *args, **kwargs: {"status": "invented_only"}
    )
    return case


def write_working(case):
    for name in WORKING_NAMES:
        if name != "reservation.json":
            path = case.attempt.directory / name
            path.write_bytes(name.encode())
            path.chmod(0o600)


def holder(case):
    return module().hold_operational_cell(
        case.attempt, case.public, expected_identity=IDENTITY
    )
