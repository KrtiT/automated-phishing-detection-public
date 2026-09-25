"""Real fixture production and saved reconstruction; no-op exits are unit lineage."""

import json
import sys
from contextlib import nullcontext
from dataclasses import replace

import pytest
from prepared_external_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    prepared_case,
    runner,
)
from prepared_external_pair_fixtures import internal_paths
from saved_external_scorer_fixtures import FixturePrimaryScorer
from test_operational_inputs import module

from automated_phishing_detection import (
    bound_secondary,
    evaluation_producer,
    external_source_completion,
    external_source_runner,
    saved_evidence,
    source_completion,
    source_runner,
)
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.external_source_handoff import (
    ObservedExternalCompletion,
)
from automated_phishing_detection.internal_external_handoff import (
    build_internal_handoff,
)
from automated_phishing_detection.internal_process_handoff import (
    ObservedInternalCompletion,
)
from automated_phishing_detection.owned_worker import observe_worker

__all__ = ["inputs", "preparation_api", "preparation_case", "prepared_case", "runner"]


def shared_models(case):
    primary = case.session.evaluation.primary
    hashes = dict(primary.models.artifact_hashes) | {
        "cascade.json": "6" * 64,
        "transformer.json": "7" * 64,
    }
    models = replace(primary.models, artifact_hashes=tuple(sorted(hashes.items())))
    evaluation = replace(
        case.session.evaluation, primary=replace(primary, models=models)
    )
    case.session = replace(case.session, evaluation=evaluation)


def internal_completion(case, paths, monkeypatch, worker):
    monkeypatch.setattr(source_runner, "recheck_binding", lambda _: None)
    monkeypatch.setattr(
        source_runner,
        "open_bound_evaluation_session",
        lambda *args: nullcontext(case.session.evaluation),
    )
    monkeypatch.setattr(
        evaluation_producer,
        "score_bound_secondary",
        bound_secondary.score_bound_secondary,
    )
    source_runner._run_bound_internal(case.binding, paths, preparation=case.preparation)
    binding = json.loads((paths.attempt / "evidence/bindings.json").read_bytes())
    monkeypatch.setattr(
        saved_evidence,
        "_EXPECTED_BINDING_CORE",
        {
            name: binding[name]
            for name in ("artifact_hashes", "thresholds", "secondary", "gmm_audit")
        },
    )
    snapshot = source_completion.verify_prepared_internal_completion_snapshot(
        case.binding, paths, preparation=case.preparation, producer_exit_code=0
    )
    return ObservedInternalCompletion(worker, snapshot)


def external_completion(case, internal, command, worker):
    case.handoff = build_internal_handoff(internal)
    primary = case.session.evaluation.primary
    primary = replace(primary, scorer=FixturePrimaryScorer(primary.models.cascade))
    case.session = replace(
        case.session, evaluation=replace(case.session.evaluation, primary=primary)
    )
    external_source_runner._run_bound_prepared_external(
        case.binding, case.paths, handoff=case.handoff, preparation=case.preparation
    )
    snapshot = external_source_completion.verify_external_completion_snapshot(
        case.binding,
        case.paths,
        expected_handoff=case.handoff,
        worker=worker,
        command=command,
        expected_preparation=case.preparation,
    )
    return ObservedExternalCompletion(worker, snapshot)


def forbid_reconstruction(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("accepted projection repeated reconstruction or model loading")

    monkeypatch.setattr(source_runner, "_read_file_once", forbidden)
    monkeypatch.setattr(
        source_completion, "reconstruct_internal_evidence_and_population", forbidden
    )
    monkeypatch.setattr(
        external_source_completion, "reconstruct_external_evidence", forbidden
    )


def test_projects_genuinely_reconstructed_source_pair_without_repeating_work(
    prepared_case, inputs, monkeypatch
):
    case = prepared_case
    shared_models(case)
    command = (sys.executable, "-c", "pass")
    worker = observe_worker(command)
    internal = internal_completion(
        case, internal_paths(case, inputs), monkeypatch, worker
    )
    external = external_completion(case, internal, command, worker)
    forbid_reconstruction(monkeypatch)
    accepted = module().build_accepted_inputs(
        internal,
        external,
        binding=case.binding,
        root_reservation_sha256="f" * 64,
        operational_profile_sha256="9" * 64,
    )
    value = json.loads(accepted.metadata_bytes)
    assert accepted.internal is internal and accepted.external is external
    assert len(value["primary"]["artifact_hashes"]) == 7
    assert (
        len(internal.snapshot.payloads) == 35 and len(external.snapshot.payloads) == 76
    )
    assert canonical_bytes(value["internal"]) == case.handoff.handoff_bytes
