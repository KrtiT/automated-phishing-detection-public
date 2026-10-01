"""Temporary invented file lifecycle, with science stubbed only for IO isolation."""

import importlib
import importlib.util
from types import SimpleNamespace

from operational_cell_io_fixtures import IDENTITY, Snapshot, Working, write_working

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._operational_process_records import ProcessObservation

__all__ = ["write_working"]


def module():
    name = "automated_phishing_detection.study_series_cell_completion"
    assert importlib.util.find_spec(name), "missing held series cell completion"
    return importlib.import_module(name)


def _options():
    return dict(
        inputs=object(),
        profile_bytes=b"invented",
        expected_profile_sha256="1" * 64,
        expected_metadata_sha256="2" * 64,
        internal_snapshot=object(),
        external_snapshot=object(),
        observation=ProcessObservation(b"invented observer shape"),
        service_command=("invented", "service"),
        client_command=("invented", "client"),
        expected_deadlines=dict(startup=300, shutdown=180, terminate=10, kill=10),
    )


def setup(tmp_path, monkeypatch):
    module()
    attempt = receipt.reserve_attempt(tmp_path.resolve() / "attempt", identity=IDENTITY)
    case = SimpleNamespace(
        attempt=attempt,
        public=tmp_path.resolve() / "public.json",
        calls=[],
        options=_options(),
    )

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


def holder(case):
    return module().hold_series_cell(
        case.attempt, case.public, expected_identity=IDENTITY
    )
