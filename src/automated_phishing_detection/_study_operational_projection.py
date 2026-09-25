"""Nine closed administrative fields; no private paths, PIDs, or raw bytes."""

import json
from dataclasses import asdict

from ._study_operational_validation import digest


def _stop(value):
    result = {
        "stage": value.stage,
        "progress_sha256": None if value.progress is None else digest(value.progress),
    }
    failure = value.failure
    if failure is not None:
        result["publishing"] = failure.publishing
        if failure.attempt is not None:
            result["reservation_sha256"] = failure.attempt.reservation_sha256
        if failure.observation is not None:
            result["observation_sha256"] = digest(failure.observation.record)
    return result


def project(slot):
    result = {
        "cell": asdict(slot.cell),
        "status": slot.status,
        "retention": None,
        "snapshot_sha256": None,
        "observation_sha256": None,
        "stage": None,
        "reservation_sha256": None,
        "progress_sha256": None,
        "publishing": None,
    }
    if slot.accepted is not None:
        record = slot.accepted
        result.update(
            retention="compact",
            snapshot_sha256=dict(record.snapshot_sha256),
            observation_sha256=digest(record.observation.record),
            reservation_sha256=json.loads(record.binding_bytes)[
                "cell_reservation_sha256"
            ],
        )
    elif slot.returned is not None:
        result.update(
            retention="unpacked",
            observation_sha256=digest(slot.returned.observation.record),
        )
    elif slot.stopped is not None:
        result.update(_stop(slot.stopped))
    return result
