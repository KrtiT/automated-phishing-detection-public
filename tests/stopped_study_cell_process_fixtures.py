"""Direct invented cancellation bytes, without synthetic live observation objects."""

from hashlib import sha256

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_process_records import _bytes
from automated_phishing_detection._study_execution_policy import DEADLINES


def role(pid):
    return dict(
        pid=pid,
        exit_code=0,
        exit_observed=True,
        forced=False,
        signals=[],
        stdout_sha256=sha256(b"").hexdigest(),
        stderr_sha256=sha256(b"").hexdigest(),
    )


def never_launched():
    return dict(
        pid=None,
        exit_code=None,
        exit_observed=False,
        forced=False,
        signals=[],
        stdout_sha256=None,
        stderr_sha256=None,
    )


def lifecycle(cell, pid):
    common = dict(schema_version=1, pid=pid, workload=cell.workload)
    return {
        "service-ready.json": _bytes(
            common | dict(status="ready", host="127.0.0.1", port=54321)
        ),
        "service-cleanup.json": _bytes(common | dict(status="clean")),
        "service-stop.json": _bytes(dict(status="requested")),
    }


def observation(payloads, reservation, pid, readiness):
    return dict(
        schema_version=1,
        reservation_sha256=reservation,
        status="failed",
        research_accepted=False,
        failure="parent_cancelled",
        record_failures=[],
        readiness_sha256=sha256(payloads["service-ready.json"]).hexdigest()
        if readiness
        else None,
        cleanup_sha256=sha256(payloads["service-cleanup.json"]).hexdigest(),
        stop_sent=True,
        service=role(pid),
        client=never_launched(),
    )


def process_records(cell, reservation, binding, metadata, *, pid, readiness):
    payloads = lifecycle(cell, pid)
    payloads.update(
        {
            "process-pair-intent.json": _bytes(
                dict(
                    schema_version=1,
                    reservation_sha256=reservation,
                    service_command_sha256="a" * 64,
                    client_command_sha256="b" * 64,
                    deadlines=dict(DEADLINES),
                )
            ),
            "process-pair.json": _bytes(
                observation(payloads, reservation, pid, readiness)
            ),
        }
    )
    return payloads | _service(cell, binding, metadata, pid)


def _service(cell, binding, metadata, pid):
    return {
        "service-intent.json": _bytes(dict(command_sha256="a" * 64)),
        "service-started.json": _bytes(dict(pid=pid)),
        "service-process.json": _bytes(role(pid)),
        "service-role.json": canonical_bytes(
            dict(
                schema_version=1,
                protocol="operational-role-v1",
                role="service",
                binding_sha256=binding,
                pid=pid,
                command_sha256="a" * 64,
                base_url="http://127.0.0.1:54321",
                workload=cell.workload,
            )
            | metadata["primary"]
        ),
    }


def finalization(reservation):
    common = dict(schema_version=1, reservation_sha256=reservation)
    return {
        "finalize.claim": receipt._json_bytes(
            common | dict(operation="failure"), "fixture"
        ),
        "outcome.json": receipt._json_bytes(
            common | dict(status="failed", stage="observation", error_type="cancelled"),
            "fixture",
        ),
    }
