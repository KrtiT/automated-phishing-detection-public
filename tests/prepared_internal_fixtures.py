"""Real byte restoration and fixture scoring, never research inputs or models."""

import json
from hashlib import sha256
from types import SimpleNamespace

from automated_phishing_detection import source_runner
from automated_phishing_detection._prepared_internal_records import (
    PreparedInternalRunPaths,
)


def _restore(snapshot, binding):
    from automated_phishing_detection.retained_study_preparation import (
        restore_study_preparation,
    )

    content = snapshot.payload("preparation-complete.json")
    identity = json.loads(content)["execution"]
    source, buffers = source_runner._public_sources(binding)
    prepared = restore_study_preparation(
        snapshot,
        expected_identity=identity,
        expected_reservation_sha256=snapshot.reservation_sha256,
        expected_completion_sha256=sha256(content).hexdigest(),
        source_spec_bytes=buffers[source_runner._SOURCE],
        preparation_summary_bytes=buffers[source_runner._PREPARATION],
    )
    return prepared, identity, source, buffers


def restored_case(preparation_api, preparation_case, inputs):
    case = preparation_case
    snapshot = preparation_api._run_bound_preparation(case.binding, case.paths)
    prepared, identity, source, buffers = _restore(snapshot, case.binding)
    original = inputs[1]
    paths = PreparedInternalRunPaths(
        case.paths.attempt,
        original.artifacts,
        original.secondary_artifacts,
        original.attempt.parent / "scoring",
        original.attempt.parent / "scoring.json",
    )
    return SimpleNamespace(
        binding=case.binding,
        original=original,
        paths=paths,
        preparation=prepared,
        snapshot=snapshot,
        identity=identity,
        source=source,
        buffers=buffers,
        session=case.session,
        events=case.events,
        profile=case.profile,
    )


def forbid_preparation(monkeypatch):
    from automated_phishing_detection import (
        evaluation_producer,
        phiusiil,
        source_overlap,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("retained scoring repeated source preparation")

    monkeypatch.setattr(phiusiil, "_parse_csv_rows", forbidden)
    monkeypatch.setattr(phiusiil, "resolve_rows", forbidden)
    monkeypatch.setattr(phiusiil, "assign_splits", forbidden)
    monkeypatch.setattr(source_overlap, "reconstruct_source_overlap", forbidden)
    monkeypatch.setattr(evaluation_producer, "parse_internal_partition", forbidden)


def bind_saved_fixture(case, monkeypatch):
    from automated_phishing_detection import saved_evidence

    bindings = json.loads((case.paths.attempt / "evidence/bindings.json").read_bytes())
    monkeypatch.setattr(
        saved_evidence,
        "_EXPECTED_BINDING_CORE",
        {
            name: bindings[name]
            for name in ("artifact_hashes", "thresholds", "secondary", "gmm_audit")
        },
    )
