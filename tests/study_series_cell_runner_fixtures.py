"""Real temporary retention and science, with synthetic process observations."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

from operational_cell_process_fixtures import process_records
from study_reduction_fixtures import _http
from study_series_input_fixtures import candidates, manifests, series_case
from study_series_ledger_fixtures import ledger, prepared
from study_series_prefix_fixtures import make_prefix_case
from test_study_series_cell_runner import api

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection import operational_cell_runner as original
from automated_phishing_detection.http_run_checkpoints import _checkpoint
from automated_phishing_detection.http_run_codec import encode_http_run
from automated_phishing_detection.operational_input_transport import (
    retain_operational_root_inputs,
)

__all__ = ["candidates", "manifests", "series_case"]


def setup(tmp_path, series_case, monkeypatch, *, allow_candidate=True):
    root = tmp_path.resolve()
    cells = root / "cells"
    cells.mkdir(mode=0o700)
    prefix = make_prefix_case(
        series_case,
        paths=dict(
            series_attempt=str(root / "series"),
            segment_attempt=str(root / "segment"),
            historical_inputs_dir=str(root / "accepted"),
            cells_dir=str(cells),
        ),
    )
    selected = prepared(prefix)
    case = SimpleNamespace(
        prefix=prefix, selected=selected, ledger=ledger(prefix), events=[]
    )
    monkeypatch.setattr(api(), "recheck_series_public_execution", lambda _: None)
    if allow_candidate:
        monkeypatch.setattr(api().context, "require_final_policy", lambda _: None)
    return case


@contextmanager
def inputs(case):
    from pathlib import Path

    path = Path(case.prefix.source.profile["paths"]["historical_inputs_dir"])
    with retain_operational_root_inputs(path, accepted_inputs=case.prefix.metadata):
        yield


def execute(case, **changes):
    arguments = dict(
        public=case.prefix.binding,
        metadata_bytes=case.prefix.metadata,
        internal_snapshot=case.prefix.source.internal,
        external_snapshot=case.prefix.source.external,
        cell=case.selected.cell,
        admissions=case.ledger,
    )
    return asyncio.run(api()._run_series_cell(**(arguments | changes)))


def install_observer(case, monkeypatch):
    async def observe(attempt, **options):
        assert attempt == case.selected.attempt
        case.events.append("observe")
        commands = (options["service_command"], options["client_command"])
        values, case.observation = process_records(
            case.selected.inputs.computational,
            attempt.reservation_sha256,
            commands,
            options["deadlines"],
        )
        for role, command, pid in zip(
            ("service", "client"), commands, (321, 654), strict=True
        ):
            admission = options["series_admissions"](role, command)
            admission.on_launched(pid)
            admission.on_observed(True, 0)
        _retain(attempt, values, options["writer"], case.selected.inputs.computational)
        return case.observation

    monkeypatch.setattr(api(), "observe_series_operational_children", observe)


def _retain(attempt, values, writer, computational):
    run = _http(computational.cell, computational)
    values.update(
        {
            "run.json": encode_http_run(run),
            "warmup.json": _checkpoint(run, measured=False),
            "measured.json": _checkpoint(run, measured=True),
        }
    )
    with receipt._directory(attempt.directory) as directory:
        for name in ("process-pair-intent.json", *sorted(values)):
            if name not in values:
                continue
            content = values.pop(name)
            if name in original._PARENT_NAMES:
                writer(attempt, name, content)
            else:
                receipt._install_record(directory, name, content)
