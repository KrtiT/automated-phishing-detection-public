"""Mocked scoring/cell boundaries test order, not actual source or exit proof."""

import json
from hashlib import sha256
from types import SimpleNamespace

from study_operational_fixtures import compact

from automated_phishing_detection import study_operational_records
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.study_reduction import ReducedStudyBytes


def public_summary(*, execution, checkpoints, reduced=None):
    contents = dict(checkpoints)
    extras = (
        {}
        if reduced is None
        else {
            "operational-summary.json": reduced.operational_bytes,
            "study-evidence.json": reduced.study_bytes,
        }
    )
    result = {
        "schema_version": 1,
        "protocol": "study-root-v1",
        "status": "whole_study_hold" if reduced is None else "study_evidence_published",
        "execution": execution,
        "accounting_sha256": sha256(contents["study-accounting.json"]).hexdigest(),
        "private_sha256": {
            name: sha256(content).hexdigest()
            for name, content in (contents | extras).items()
        },
    }
    if reduced is None:
        result["feasibility"] = json.loads(contents["prediction-barrier.json"])[
            "feasibility"
        ]
    else:
        result.update(operational=reduced.operational, study=reduced.study)
    return result


def source_worker(case):
    def sources(binding, internal, external, preparation):
        assert preparation is case.retained
        barrier = json.loads(
            (case.paths.attempt / "prediction-barrier.json").read_bytes()
        )
        assert barrier["status"] == "necessary_capacity_present"
        assert barrier["predictions_started"] is False
        case.events.append("sources")
        return case.sources

    return sources


def cell_worker(case, fail_ordinal):
    async def cell(
        binding, profile, accepted, selected, *, paths, artifacts, deadlines
    ):
        assert (
            binding is case.binding
            and profile is case.profile
            and accepted is case.accepted
        )
        assert (
            artifacts is case.paths.internal.artifacts and deadlines == case.deadlines
        )
        case.events.append(selected.ordinal)
        if selected.ordinal == fail_ordinal:
            raise ValueError("invented cell failure")
        paths.cell_input_directory.mkdir(mode=0o700)
        paths.attempt.mkdir(mode=0o700)
        paths.public_summary.write_bytes(b"{}")
        paths.public_summary.chmod(0o644)
        result = SimpleNamespace(ordinal=selected.ordinal)
        case.returns.append(result)
        return result

    return cell


def reduction(case, reduction_error):
    def reduce(accepted, slots):
        assert accepted is case.accepted and len(slots) == 125
        assert all(slot.status == "accepted" for slot in slots)
        case.events.append("reduce")
        if reduction_error is not None:
            raise reduction_error
        return ReducedStudyBytes(
            b'{"invented_operational":true}\n', b'{"invented_study":true}\n'
        )

    return reduce


def install(case, monkeypatch, *, fail_ordinal=None, reduction_error=None):
    case.sources = SimpleNamespace(internal=object(), external=object())
    case.accepted = SimpleNamespace(metadata_bytes=b'{"invented":true}\n')
    case.returns = []
    monkeypatch.setattr(
        case.body, "_run_observed_prepared_sources", source_worker(case)
    )
    monkeypatch.setattr(
        case.body, "build_accepted_inputs", lambda *arguments, **keywords: case.accepted
    )
    monkeypatch.setattr(case.body, "_run_bound_cell", cell_worker(case, fail_ordinal))
    monkeypatch.setattr(
        case.body,
        "retain_accepted_cell",
        lambda result, **keywords: compact(study_operational_records, result.ordinal),
    )
    monkeypatch.setattr(
        case.body, "reduce_accepted_study", reduction(case, reduction_error)
    )
    monkeypatch.setattr(
        case.body.records,
        "source_results",
        lambda accepted, **keywords: canonical_bytes({"invented": True}),
    )
    monkeypatch.setattr(case.body.records, "root_public_summary", public_summary)
