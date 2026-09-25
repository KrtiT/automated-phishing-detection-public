"""Invented full records; constructed observations are not actual exit evidence."""

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from http_run_codec_fixtures import complete_run
from operational_cell_process_fixtures import process_records
from operational_input_fixtures import build

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection import operational_inputs
from automated_phishing_detection._operational_process_records import _bytes
from automated_phishing_detection.http_run_checkpoints import _checkpoint
from automated_phishing_detection.http_run_codec import encode_http_run
from automated_phishing_detection.operational_cell_inputs import (
    bind_cell_descriptor,
    build_cell_descriptor,
    restore_cell_inputs,
)
from automated_phishing_detection.operational_schedule import planned_cells


def digest(content):
    return sha256(content).hexdigest()


def http_run(inputs):
    original = complete_run(inputs.cell.workload)
    phases = {}
    for phase in ("warmup", "measured"):
        rows = []
        for position, row in enumerate(getattr(original, phase)):
            request_id = f"{inputs.manifest_sha256}.64.1.{phase}.{position}"
            response = row.response
            if response is not None:
                response = response.model_copy(update={"request_id": request_id})
            rows.append(
                replace(
                    row,
                    record_id=inputs.requests[position].record_id,
                    request_id=request_id,
                    response=response,
                )
            )
        phases[phase] = tuple(rows)
    return replace(original, manifest_sha256=inputs.manifest_sha256, **phases)


def _reservation(identity):
    reservation = receipt._json_bytes(
        {
            "schema_version": 1,
            "status": "reserved",
            "directory": "/invented/cell",
            "identity": identity,
        },
        "fixture",
    )
    return receipt.Attempt(Path("/invented/cell"), digest(reservation)), reservation


def _restore(accepted, selected, attempt):
    binding = bind_cell_descriptor(
        selected.descriptor_bytes, cell_reservation_sha256=attempt.reservation_sha256
    )
    return restore_cell_inputs(
        accepted.metadata_bytes,
        selected.descriptor_bytes,
        binding,
        selected.manifest_bytes,
        expected_binding_sha256=digest(binding),
        expected_cell_reservation_sha256=attempt.reservation_sha256,
    )


def cell_case(api, source, *, ordinal=21):
    accepted = build(operational_inputs, source)
    selected = build_cell_descriptor(accepted, planned_cells()[ordinal - 1])
    identity = api.cell_identity(accepted, selected.descriptor_bytes)
    attempt, reservation = _reservation(identity)
    inputs = _restore(accepted, selected, attempt)
    commands = (("/invented/python", "service.py"), ("/invented/python", "client.py"))
    deadlines = {"startup": 1.0, "shutdown": 2.0, "terminate": 3.0, "kill": 4.0}
    payloads, observation = process_records(
        inputs, attempt.reservation_sha256, commands, deadlines
    )
    payloads["reservation.json"] = reservation
    arguments = {
        "attempt": attempt,
        "expected_identity": identity,
        "inputs": inputs,
        "accepted": accepted,
        "observation": observation,
        "service_command": commands[0],
        "client_command": commands[1],
        "expected_deadlines": deadlines,
    }
    return SimpleNamespace(payloads=payloads, arguments=arguments, inputs=inputs)


def http_case(api, source):
    case = cell_case(api, source)
    run = http_run(case.inputs)
    case.payloads.update(
        {
            "warmup.json": _checkpoint(run, measured=False),
            "measured.json": _checkpoint(run, measured=True),
            "run.json": encode_http_run(run),
        }
    )
    return case


def verify(api, case, **changes):
    return api.verify_working_cell(
        tuple(case.payloads.items()), **(case.arguments | changes)
    )


def changed_payload(case, name, mutate):
    payloads = dict(case.payloads)
    value = json.loads(payloads[name])
    mutate(value)
    payloads[name] = _bytes(value)
    return replace_case(case, payloads)


def replace_case(case, payloads):
    return SimpleNamespace(
        payloads=payloads, arguments=case.arguments, inputs=case.inputs
    )
