"""Constructed record shapes for accounting, explicitly not process authority."""

from dataclasses import asdict
from hashlib import sha256

from operational_cell_process_fixtures import role_value

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_cell_protocol import (
    PRIVATE_NAMES,
    PROTOCOL,
    SNAPSHOT_NAMES,
)
from automated_phishing_detection._operational_process_records import (
    ProcessObservation,
    _bytes,
)
from automated_phishing_detection.operational_schedule import cell_for_ordinal


def digest(content):
    return sha256(content).hexdigest()


def observation(reservation):
    return ProcessObservation(
        _bytes(
            {
                "schema_version": 1,
                "reservation_sha256": reservation,
                "status": "observed",
                "research_accepted": False,
                "failure": None,
                "record_failures": [],
                "readiness_sha256": "a" * 64,
                "cleanup_sha256": "b" * 64,
                "stop_sent": True,
                "service": role_value(321),
                "client": role_value(654),
            }
        )
    )


def described(cell):
    return canonical_bytes(
        {
            "schema_version": 1,
            "kind": "operational-cell-descriptor-v1",
            "root_reservation_sha256": "e" * 64,
            "cell": asdict(cell),
            "manifest_sha256": "f" * 64,
            "accepted_inputs_sha256": "d" * 64,
        }
    )


def bound(descriptor, reservation):
    return canonical_bytes(
        {
            "schema_version": 1,
            "kind": "operational-cell-binding-v1",
            "descriptor_sha256": digest(descriptor),
            "cell_reservation_sha256": reservation,
        }
    )


def public_bytes(cell, descriptor, reservation, hashes):
    execution = {
        "kind": "operational_cell",
        "protocol": PROTOCOL,
        "revision": "1" * 40,
        "execution_contract_sha256": "2" * 64,
        "runtime_sha256": "3" * 64,
        "source_spec_sha256": "4" * 64,
        "operational_profile_sha256": "5" * 64,
        "root_reservation_sha256": "e" * 64,
        "descriptor_sha256": digest(descriptor),
        "reservation_sha256": reservation,
    }
    return receipt._json_bytes(
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "operational_evidence_published",
            "execution": execution,
            "cell": asdict(cell),
            "summary": {},
            "private_sha256": {
                name: hashes[f"attempt/{name}"] for name in PRIVATE_NAMES
            },
        },
        "fixture",
    )


def compact(api, ordinal):
    cell = cell_for_ordinal(ordinal)
    reservation = f"{ordinal:064x}"
    observed, descriptor = observation(reservation), described(cell)
    hashes = dict.fromkeys(SNAPSHOT_NAMES, "a" * 64)
    hashes["attempt/reservation.json"] = reservation
    for prefix in ("attempt", "attempt/evidence"):
        hashes[f"{prefix}/run.json"] = digest(b"{}\n")
        hashes[f"{prefix}/process-pair.json"] = digest(observed.record)
    public = public_bytes(cell, descriptor, reservation, hashes)
    hashes["public-summary.json"] = digest(public)
    return api.AcceptedOperationalRun(
        cell,
        observed,
        b"{}\n",
        descriptor,
        bound(descriptor, reservation),
        public,
        tuple(sorted(hashes.items())),
    )
