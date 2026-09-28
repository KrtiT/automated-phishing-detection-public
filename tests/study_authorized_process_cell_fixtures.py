"""Genuine first scheduled HTTP pair after actual accepted source child exits."""

import asyncio

from automated_phishing_detection import _study_preparation_files as files
from automated_phishing_detection import execution_receipt, operational_cell_runner
from automated_phishing_detection._adopted_study_records import (
    scientific_execution,
    study_identity,
)
from automated_phishing_detection._study_authorized_cell import run_adopted_cell
from automated_phishing_detection._study_run_paths import cell_paths
from automated_phishing_detection._study_run_records import source_results
from automated_phishing_detection.operational_input_transport import (
    retain_operational_root_inputs,
)
from automated_phishing_detection.operational_inputs import build_accepted_inputs
from automated_phishing_detection.operational_schedule import cell_for_ordinal
from automated_phishing_detection.study_operational_records import retain_accepted_cell


def run_first_cell(case, sources, monkeypatch):
    auth, ledger = case.authorization, case.ledger
    accepted = _accepted(case, sources)
    content = _retain_source_results(case, accepted)
    auth.paths.cells_directory.mkdir(mode=0o700)
    auth.paths.accepted_inputs_directory.parent.chmod(0o700)
    monkeypatch.setattr(
        operational_cell_runner, "recheck_binding", lambda binding: None
    )
    with retain_operational_root_inputs(
        auth.paths.accepted_inputs_directory, accepted_inputs=accepted.metadata_bytes
    ):
        ledger.inputs_retained(accepted, content)
        cell = cell_for_ordinal(1)
        ledger.cell_started(cell)
        completion = asyncio.run(
            run_adopted_cell(
                auth, accepted, cell, paths=cell_paths(auth.paths, 1), admissions=ledger
            )
        )
        retained = retain_accepted_cell(completion, accepted=accepted)
        ledger.cell_accepted(
            cell,
            retained,
            pair_intent_bytes=dict(completion.snapshot.payloads)[
                "attempt/process-pair-intent.json"
            ],
        )
    return completion


def _accepted(case, sources):
    auth = case.authorization
    return build_accepted_inputs(
        sources.internal,
        sources.external,
        binding=auth.base,
        root_reservation_sha256=case.ledger.attempt.reservation_sha256,
        operational_profile_sha256=auth.operational.profile_sha256,
    )


def _retain_source_results(case, accepted):
    execution = study_identity(case.authorization) | {
        "reservation_sha256": case.ledger.attempt.reservation_sha256
    }
    content = source_results(accepted, execution=scientific_execution(execution))
    with execution_receipt._directory(case.authorization.paths.attempt) as directory:
        states = {
            name: files.capture(directory, name)
            for name in (
                "reservation.json",
                "study-intent.json",
                "prediction-barrier.json",
            )
        }
        files.append(directory, states, "source-results.json", content)
    return content
