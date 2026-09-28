"""Constructed observations test coordinator composition, not historical exits."""

import json
from dataclasses import asdict
from hashlib import sha256
from types import SimpleNamespace

from adopted_study_matrix_cells_fixtures import cell_worker, observed
from study_runner_matrix_fixtures import public_summary, reduction

from automated_phishing_detection import _adopted_study_ledger as ledger_module
from automated_phishing_detection import _study_authorized_cell as cell_module
from automated_phishing_detection import _study_authorized_sources as source_module
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.external_source_handoff import (
    ObservedExternalCompletion,
)


def source_worker(case):
    def sources(authorization, preparation, ledger):
        assert authorization is case.authorization and preparation is case.retained
        assert ledger.barrier_sha256 is not None
        assert (case.paths.attempt / "prediction-barrier.json").is_file()
        case.events.append("sources")
        internal = SimpleNamespace(
            worker=observed(ledger, "internal", ("invented", "internal"), 111)
        )
        ledger.internal_accepted(internal)
        external = ObservedExternalCompletion(
            observed(
                ledger,
                "external",
                ("invented", "external"),
                222,
                predecessor_sha256=ledger.handoff_sha256,
            ),
            None,
        )
        ledger.external_accepted(external)
        case.sources = SimpleNamespace(internal=internal, external=external)
        case.accepted = accepted_inputs(case.sources, ledger.handoff_sha256)
        return case.sources

    return sources


def accepted_inputs(sources, handoff):
    metadata = {
        role: {"worker": asdict(getattr(sources, role).worker), "execution": {}}
        for role in ("internal", "external")
    }
    metadata["external"]["execution"]["internal_handoff_sha256"] = handoff
    return SimpleNamespace(metadata_bytes=canonical_bytes(metadata))


def source_results(accepted, *, execution):
    return canonical_bytes(
        {
            "execution": execution,
            "accepted_inputs": json.loads(accepted.metadata_bytes),
            "accepted_inputs_sha256": sha256(accepted.metadata_bytes).hexdigest(),
        }
    )


def install(case, monkeypatch, *, fail_ordinal=None, reduction_error=None):
    case.returns = []
    monkeypatch.setattr(
        ledger_module,
        "build_internal_handoff",
        lambda observed: SimpleNamespace(handoff_bytes=b"invented handoff"),
    )
    monkeypatch.setattr(source_module, "run_adopted_sources", source_worker(case))
    monkeypatch.setattr(
        cell_module, "run_adopted_cell", cell_worker(case, fail_ordinal)
    )
    monkeypatch.setattr(
        case.body, "build_accepted_inputs", lambda *args, **kwargs: case.accepted
    )
    monkeypatch.setattr(
        case.body, "retain_accepted_cell", lambda result, **kwargs: result.retained
    )
    monkeypatch.setattr(
        case.body, "reduce_accepted_study", reduction(case, reduction_error)
    )
    monkeypatch.setattr(case.body.records, "source_results", source_results)
    monkeypatch.setattr(case.body.records, "root_public_summary", public_summary)
