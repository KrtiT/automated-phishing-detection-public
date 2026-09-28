"""Invented complete authorization ledgers, never proof of historical processes."""

import json
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

from study_admission_fixtures import frame
from study_operational_fixtures import bound, compact
from study_run_public_fixtures import success_case

from automated_phishing_detection import _adopted_study_records as records
from automated_phishing_detection import study_operational_records as operational
from automated_phishing_detection._adopted_study_cell_evidence import evidence
from automated_phishing_detection._adopted_study_issuance import _entry
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_operational_projection import project


def cell_evidence(ordinal, execution, accepted):
    retained = compact(operational, ordinal)
    described = json.loads(retained.descriptor_bytes)
    described.update(
        root_reservation_sha256=execution["reservation_sha256"],
        accepted_inputs_sha256=accepted,
    )
    descriptor = canonical_bytes(described)
    reservation = json.loads(retained.binding_bytes)["cell_reservation_sha256"]
    intent = pair_intent(reservation)
    hashes = dict(retained.snapshot_sha256)
    for prefix in ("attempt", "attempt/evidence"):
        hashes[f"{prefix}/process-pair-intent.json"] = sha256(intent).hexdigest()
    retained = replace(
        retained,
        descriptor_bytes=descriptor,
        binding_bytes=bound(descriptor, reservation),
        snapshot_sha256=tuple(sorted(hashes.items())),
    )
    projection = project(
        operational.OperationalCellSlot(retained.cell, accepted=retained)
    )
    return retained, intent, projection


def pair_intent(reservation):
    return canonical_bytes(
        {
            "schema_version": 1,
            "reservation_sha256": reservation,
            "service_command_sha256": "a" * 64,
            "client_command_sha256": "b" * 64,
            "deadlines": dict(startup=300, shutdown=180, terminate=10, kill=10),
        }
    )


def entry_for(
    role,
    context,
    *,
    cell=None,
    predecessor=None,
    accepted=None,
    binding=None,
    command="a" * 64,
    pid=123,
):
    admission = frame(
        role=role,
        **context,
        predecessor_sha256=predecessor,
        accepted_inputs_sha256=accepted,
        cell_binding_sha256=binding,
        command_sha256=command,
    )
    entry = _entry(admission, cell)
    entry.update(launched_pid=pid, exit_observed=True, exit_code=0, accepted=True)
    return entry


def complete_ledger(prepared, manifests):
    execution, checkpoints, unused, attempt = success_case(prepared, manifests)
    execution = execution | {
        "protocol": records.PROTOCOL,
        "study_profile_sha256": "1" * 64,
        "adoption_envelope_sha256": "2" * 64,
        "study_policy_sha256": "3" * 64,
    }
    contents = dict(checkpoints)
    accepted = json.loads(contents["source-results.json"])
    metadata = accepted["accepted_inputs"]
    context = _context(execution, contents)
    ledger = dict(
        schema_version=1,
        protocol="study-authorization-ledger-v1",
        intent_sha256=context["intent_sha256"],
        barrier_sha256=context["barrier_sha256"],
        handoff_sha256=metadata["external"]["execution"]["internal_handoff_sha256"],
        source_results_sha256=sha256(contents["source-results.json"]).hexdigest(),
        accepted_inputs_sha256=accepted["accepted_inputs_sha256"],
        admissions=[],
        cell_acceptances=[],
    )
    _sources(ledger, context, metadata)
    projections = _cells(ledger, context, execution)
    accounting = json.loads(contents["study-accounting.json"]) | {"cells": projections}
    contents["study-accounting.json"] = canonical_bytes(
        {"scientific_accounting_bytes": records.encoded(canonical_bytes(accounting))}
    )
    return SimpleNamespace(ledger=ledger, execution=execution, contents=contents)


def _context(execution, contents):
    barrier = json.loads(contents["prediction-barrier.json"])
    return dict(
        root_reservation_sha256=execution["reservation_sha256"],
        intent_sha256=sha256(contents["study-intent.json"]).hexdigest(),
        barrier_sha256=sha256(contents["prediction-barrier.json"]).hexdigest(),
        preparation_reservation_sha256=barrier["study_preparation_reservation_sha256"],
        preparation_completion_sha256=barrier["study_preparation_complete_sha256"],
    )


def _sources(ledger, context, metadata):
    for role in ("internal", "external"):
        worker = metadata[role]["worker"]
        ledger["admissions"].append(
            entry_for(
                role,
                context,
                predecessor=None if role == "internal" else ledger["handoff_sha256"],
                command=worker["command_sha256"],
                pid=worker["exit"]["pid"],
            )
        )


def _cells(ledger, context, execution):
    projections = []
    for ordinal in range(1, 126):
        retained, intent, projection = cell_evidence(
            ordinal, execution, ledger["accepted_inputs_sha256"]
        )
        ledger["cell_acceptances"].append(evidence(retained, intent))
        for role, command, pid in (
            ("service", "a" * 64, 321),
            ("client", "b" * 64, 654),
        ):
            ledger["admissions"].append(
                entry_for(
                    role,
                    context,
                    cell=retained.cell,
                    predecessor=ledger["source_results_sha256"],
                    accepted=ledger["accepted_inputs_sha256"],
                    binding=sha256(retained.binding_bytes).hexdigest(),
                    command=command,
                    pid=pid,
                )
            )
        projections.append(projection)
    return projections
