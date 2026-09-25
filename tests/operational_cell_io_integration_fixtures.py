"""Real pure validators over invented runs; no process-proof claim is made."""

from hashlib import sha256
from types import SimpleNamespace

from operational_cell_acceptance_fixtures import http_case, process_records
from operational_cell_shift_fixtures import shift_case
from operational_input_fixtures import candidates, case, manifests

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection import operational_cell_acceptance as acceptance
from automated_phishing_detection.operational_cell_inputs import (
    bind_cell_descriptor,
    restore_cell_inputs,
)

__all__ = ["candidates", "case", "manifests"]


def rebind(inputs, reservation):
    binding = bind_cell_descriptor(
        inputs.descriptor_bytes, cell_reservation_sha256=reservation
    )
    return restore_cell_inputs(
        inputs.accepted_bytes,
        inputs.descriptor_bytes,
        binding,
        inputs.manifest_bytes,
        expected_binding_sha256=sha256(binding).hexdigest(),
        expected_cell_reservation_sha256=reservation,
    )


def disk_case(tmp_path, source, workload):
    original = (
        http_case(acceptance, source)
        if workload == "http"
        else shift_case(acceptance, source)
    )
    identity = original.arguments["expected_identity"]
    attempt = receipt.reserve_attempt(tmp_path.resolve() / "cell", identity=identity)
    inputs = rebind(original.inputs, attempt.reservation_sha256)
    commands = tuple(
        original.arguments[name] for name in ("service_command", "client_command")
    )
    payloads, observation = process_records(
        inputs,
        attempt.reservation_sha256,
        commands,
        original.arguments["expected_deadlines"],
    )
    payloads.update(
        {
            name: original.payloads[name]
            for name in ("warmup.json", "measured.json", "run.json")
        }
    )
    return ready_case(tmp_path, original, attempt, payloads, inputs, observation)


def ready_case(tmp_path, original, attempt, payloads, inputs, observation):
    return SimpleNamespace(
        attempt=attempt,
        identity=original.arguments["expected_identity"],
        payloads=payloads,
        public=tmp_path.resolve() / "public.json",
        arguments={
            name: value
            for name, value in original.arguments.items()
            if name not in ("attempt", "expected_identity")
        }
        | {"inputs": inputs, "observation": observation},
    )
