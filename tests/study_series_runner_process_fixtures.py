"""Synthetic observed process records exercise real suffix cells and publication."""

from types import SimpleNamespace

from operational_cell_process_fixtures import process_records
from shift_run_codec_fixtures import checkpoints as shift_checkpoints
from study_reduction_fixtures import _http, _shift
from study_series_ledger_fixtures import prepared

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection import operational_cell_runner as original
from automated_phishing_detection import study_series_cell_runner as cell_runner
from automated_phishing_detection.http_run_checkpoints import _checkpoint
from automated_phishing_detection.http_run_codec import encode_http_run
from automated_phishing_detection.shift_run_codec import encode_shift_run


def install(case, monkeypatch, *, stop_ordinal=None):
    monkeypatch.setattr(
        cell_runner.context, "require_final_policy", lambda public: None
    )
    monkeypatch.setattr(
        cell_runner, "recheck_series_public_execution", lambda public: None
    )

    async def observe(attempt, **options):
        return observed(case, stop_ordinal, attempt, options)

    monkeypatch.setattr(cell_runner, "observe_series_operational_children", observe)


def observed(case, stop_ordinal, attempt, options):
    ledger = options["series_admissions"].__self__
    ordinal = ledger._current["cell"].ordinal
    case.events.append(("cell", ordinal))
    if ordinal == stop_ordinal:
        raise ValueError("invented stopped cell")
    prefix = SimpleNamespace(
        source=case.source, binding=ledger._public, metadata=ledger._metadata
    )
    selected = prepared(prefix, ordinal)
    commands = options["service_command"], options["client_command"]
    values, observation = process_records(
        selected.inputs.computational,
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
    retain(attempt, values, options["writer"], selected.inputs.computational)
    return observation


def run_records(inputs):
    if inputs.cell.workload == "shift_period":
        run = _shift(inputs.cell, inputs)
        warmup, measured = shift_checkpoints(run)
        encoded = encode_shift_run(run)
    else:
        run = _http(inputs.cell, inputs)
        warmup, measured = (
            _checkpoint(run, measured=False),
            _checkpoint(run, measured=True),
        )
        encoded = encode_http_run(run)
    return {"run.json": encoded, "warmup.json": warmup, "measured.json": measured}


def retain(attempt, values, writer, inputs):
    values.update(run_records(inputs))
    with receipt._directory(attempt.directory) as directory:
        for name in ("process-pair-intent.json", *sorted(values)):
            if name not in values:
                continue
            content = values.pop(name)
            if name in original._PARENT_NAMES:
                writer(attempt, name, content)
            else:
                receipt._install_record(directory, name, content)
