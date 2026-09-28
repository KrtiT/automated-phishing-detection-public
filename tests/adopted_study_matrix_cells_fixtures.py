"""Invented cell observations for composed coordinator tests, never process proof."""

import json
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

from adopted_study_ledger_fixtures import cell_evidence

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._owned_process_exit import OwnedProcessExit
from automated_phishing_detection._process_support import command_hash
from automated_phishing_detection.owned_worker import WorkerObservation


def observed(ledger, role, command, pid, **keywords):
    with ledger.issue(role, command, **keywords) as admission:
        admission.launched(pid)
        admission.observed(True, 0)
    return WorkerObservation(
        command_hash(command), OwnedProcessExit(pid, True, 0), "a" * 64, "b" * 64
    )


def repin_public(retained, hashes):
    public = json.loads(retained.public_bytes)
    public["execution"].update(
        root_reservation_sha256=json.loads(retained.descriptor_bytes)[
            "root_reservation_sha256"
        ],
        descriptor_sha256=sha256(retained.descriptor_bytes).hexdigest(),
    )
    public["private_sha256"] = {
        name: hashes[f"attempt/{name}"] for name in public["private_sha256"]
    }
    content = receipt._json_bytes(public, "invented")
    hashes["public-summary.json"] = sha256(content).hexdigest()
    return replace(
        retained, public_bytes=content, snapshot_sha256=tuple(sorted(hashes.items()))
    )


def retained_cell(ledger, ordinal):
    retained, content, unused = cell_evidence(
        ordinal,
        {"reservation_sha256": ledger.attempt.reservation_sha256},
        ledger.accepted_inputs_sha256,
    )
    intent = json.loads(content)
    commands = {
        role: ("invented", role, str(ordinal)) for role in ("service", "client")
    }
    for role, command in commands.items():
        intent[f"{role}_command_sha256"] = command_hash(command)
    content = canonical_bytes(intent)
    hashes = dict(retained.snapshot_sha256)
    for prefix in ("attempt", "attempt/evidence"):
        hashes[f"{prefix}/process-pair-intent.json"] = sha256(content).hexdigest()
    return repin_public(retained, hashes), content, commands


def observe_pair(admissions, retained, commands):
    for role, pid in (("service", 321), ("client", 654)):
        observed(
            admissions,
            role,
            commands[role],
            pid,
            predecessor_sha256=admissions.source_results_sha256,
            accepted_inputs_sha256=admissions.accepted_inputs_sha256,
            cell_binding_sha256=sha256(retained.binding_bytes).hexdigest(),
        )


def cell_result(paths, retained, intent):
    paths.cell_input_directory.mkdir(mode=0o700)
    paths.attempt.mkdir(mode=0o700)
    paths.public_summary.write_bytes(b"{}")
    paths.public_summary.chmod(0o644)
    return SimpleNamespace(
        retained=retained,
        snapshot=SimpleNamespace(
            payloads=(("attempt/process-pair-intent.json", intent),)
        ),
    )


def cell_worker(case, fail_ordinal):
    async def cell(authorization, accepted, selected, *, paths, admissions):
        assert authorization is case.authorization and accepted is case.accepted
        assert admissions.source_results_sha256 is not None
        assert (case.paths.attempt / "source-results.json").is_file()
        case.events.append(selected.ordinal)
        retained, intent, commands = retained_cell(admissions, selected.ordinal)
        observe_pair(admissions, retained, commands)
        if selected.ordinal == fail_ordinal:
            raise ValueError("invented cell failure")
        result = cell_result(paths, retained, intent)
        case.returns.append(result)
        return result

    return cell
