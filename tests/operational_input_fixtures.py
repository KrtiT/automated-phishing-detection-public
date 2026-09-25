"""Invented retained object shapes, explicitly not proof of source execution."""

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from external_completion_lineage_fixtures import OVERLAP_DOMAINS, _overlap
from operational_input_external_fixtures import external_case
from test_evaluation_manifest import candidates, manifests

from automated_phishing_detection import _internal_handoff_validation as internal
from automated_phishing_detection import execution_receipt, saved_evidence
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._owned_process_exit import OwnedProcessExit
from automated_phishing_detection.evaluation_producer import ManifestOutcome
from automated_phishing_detection.execution_preflight import ExecutionBinding
from automated_phishing_detection.internal_process_handoff import (
    ObservedInternalCompletion,
)
from automated_phishing_detection.internal_source_handoff import (
    VerifiedInternalSnapshot,
)
from automated_phishing_detection.owned_worker import WorkerObservation

__all__ = ["candidates", "manifests", "source_case", "case", "digest", "build"]


def digest(content):
    return sha256(content).hexdigest()


def worker(pid):
    return WorkerObservation(
        "1" * 64, OwnedProcessExit(pid, True, 0), "2" * 64, "3" * 64
    )


def _internal_execution(payloads):
    execution = {
        **internal.EXECUTION_CONSTANTS,
        **{name: "a" * 64 for name in internal.EXECUTION_HASHES},
        "revision": "c" * 40,
        "runtime_sha256": digest(b"{}"),
        "source_interface": "retained_study_preparation_v1",
        "study_preparation_reservation_sha256": "d" * 64,
        "study_preparation_complete_sha256": "e" * 64,
    }
    execution.update(
        {key: digest(payloads[path]) for key, path in internal.SOURCE_LINKS.items()}
    )
    return execution


def _internal_case(manifests):
    payloads = {name: name.encode() for name in internal.SNAPSHOT_NAMES}
    execution = _internal_execution(payloads)
    payloads[internal.OVERLAP_NAME] = _overlap(execution)
    payloads["public-summary.json"] = execution_receipt._json_bytes(
        {"execution": execution}, "fixture"
    )
    payloads["attempt/evidence/bindings.json"] = canonical_bytes(
        saved_evidence._EXPECTED_BINDING_CORE
    )
    snapshot = VerifiedInternalSnapshot(
        tuple(sorted(payloads.items())),
        (),
        (),
        OVERLAP_DOMAINS,
        tuple(
            (key, ManifestOutcome("prepared", value))
            for key, value in sorted(manifests.items())
        ),
    )
    return ObservedInternalCompletion(worker(123), snapshot)


def source_case(manifests, count=1001):
    observed = _internal_case(manifests)
    execution = observed.public_summary["execution"]
    binding = ExecutionBinding(
        Path("/invented/checkout"),
        execution["revision"],
        execution["execution_contract_sha256"],
        tuple(
            (path.removeprefix("source/"), execution[name])
            for name, path in internal.SOURCE_LINKS.items()
            if path.startswith("source/")
        ),
        "{}",
    )
    return SimpleNamespace(
        internal=observed,
        external=external_case(observed, count, worker(456)),
        binding=binding,
        reservation="f" * 64,
        profile="9" * 64,
    )


@pytest.fixture(scope="module")
def case(manifests):
    return source_case(manifests)


def build(api, case, **changes):
    return api.build_accepted_inputs(
        case.internal,
        case.external,
        **(
            {
                "binding": case.binding,
                "root_reservation_sha256": case.reservation,
                "operational_profile_sha256": case.profile,
            }
            | changes
        ),
    )


def change_payload(observation, name, mutate):
    payloads = dict(observation.snapshot.payloads)
    payloads[name] = canonical_bytes(mutate(json.loads(payloads[name])))
    return replace(
        observation,
        snapshot=replace(
            observation.snapshot, payloads=tuple(sorted(payloads.items()))
        ),
    )
