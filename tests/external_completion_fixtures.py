"""Invented completion trees with real byte reconstruction and retained arithmetic.

The observed no-op worker and declared internal handoff are unit lineage only;
they never establish that a worker produced or accepted this evidence.
"""

import sys
from dataclasses import fields
from types import SimpleNamespace

import phishvn_source_fixtures as publisher
import saved_external_artifact_fixtures as artifacts
import saved_external_fixtures as scientific
from external_completion_lineage_fixtures import (
    OVERLAP_DOMAINS,
    fixture_binding,
    fixture_profile,
    rebound_snapshot_chain,
    unit_handoff,
)

from automated_phishing_detection import execution_receipt, phishvn, protocol_preflight
from automated_phishing_detection._external_source_checkpoints import (
    SCIENTIFIC_ORDER,
    ExternalCheckpointWriter,
)
from automated_phishing_detection._external_source_records import (
    ExternalRunPaths,
    build_external_public,
    external_identity,
)
from automated_phishing_detection.bound_drift import DriftArtifactPaths
from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.bound_secondary import SecondaryArtifactPaths
from automated_phishing_detection.external_source_provenance import (
    build_external_provenance,
)
from automated_phishing_detection.owned_worker import observe_worker
from automated_phishing_detection.phishvn_source import (
    PhishVNSourcePins,
    decode_phishvn_archive,
)


def _archive(count):
    mappings = (
        ("tinnhiemmang", "gold", "phishing"),
        ("tinnhiem_web", "gold", "benign"),
        ("tinnhiemmang", "silver", "phishing"),
        ("chongluadao", "bronze", "phishing"),
        ("tranco", "silver", "benign"),
    )
    rows = []
    for index in range(count):
        source, tier, label = mappings[index % len(mappings)]
        rows.append(
            publisher.record(
                f"external-{index}",
                url=f"https://host{index}.invented{index}.com/path",
                source=source,
                tier=tier,
                label=label,
            )
        )
    return publisher.bundle(publisher.members(rows))


def _paths(root):
    def inventory(record_type, prefix):
        return record_type(
            *(root / f"{prefix}-{field.name}" for field in fields(record_type))
        )

    return ExternalRunPaths(
        root / "invented-publisher.zip",
        root / "invented-suffix.dat",
        inventory(ArtifactPaths, "primary"),
        inventory(SecondaryArtifactPaths, "secondary"),
        inventory(DriftArtifactPaths, "drift"),
        root / "attempt",
        root / "public.json",
    )


def _science(monkeypatch, prepared, suffix, count):
    monkeypatch.setattr(
        artifacts, "snapshot_chain", lambda: rebound_snapshot_chain(suffix)
    )
    monkeypatch.setattr(scientific, "prepared_external", lambda unused: prepared)
    return scientific.saved_external_bundle(monkeypatch, count)


def _retain(case):
    case.provenance = build_external_provenance(
        case.decoded,
        case.prepared,
        suffix_rules=case.suffix,
        internal_handoff=case.handoff.handoff_bytes,
        internal_overlap=case.handoff.overlap_bytes,
        execution=case.identity,
        reservation_sha256=case.attempt.reservation_sha256,
    )
    writer = ExternalCheckpointWriter(case.attempt, identity=case.identity)
    writer.begin(case.provenance)
    for name in SCIENTIFIC_ORDER:
        writer(name, case.produced.private_outputs[name])
    return writer.complete(case.produced.private_outputs)


def _publish(case):
    case.identity = external_identity(case.binding, case.profile, case.handoff)
    case.attempt = execution_receipt.reserve_attempt(
        case.paths.attempt, identity=case.identity
    )
    case.filesinputs = _retain(case)
    case.expectedpublic = build_external_public(
        case.binding,
        case.profile,
        case.identity,
        case.attempt.reservation_sha256,
        case.filesinputs,
        case.produced.public_summary,
    )
    execution_receipt.publish_completion(
        case.attempt,
        private_outputs=case.filesinputs,
        public_summary=case.expectedpublic,
        public_path=case.paths.public_summary,
    )


def _prepare(case):
    case.archivebytes = case.archive.content
    case.decoded = decode_phishvn_archive(
        case.archivebytes, pins=PhishVNSourcePins(**case.archive.pins)
    )
    case.prepared = phishvn.prepare_external_rows(
        case.decoded.rows,
        published_split_counts=case.decoded.published_split_counts,
        test_split="test",
        suffix_rules=protocol_preflight.parse_suffix_rules(case.suffix.decode()),
        phiusiil_domains=OVERLAP_DOMAINS,
    )


def _context(case, tmp_path, monkeypatch, count):
    case.bundle = _science(monkeypatch, case.prepared, case.suffix, count)
    case.produced = case.bundle.produced
    case.binding = fixture_binding(
        tmp_path / "invented-checkout", case.bundle.session.drift.public_inputs
    )
    case.handoff = unit_handoff(
        case.binding,
        case.suffix,
        case.bundle.session.drift.reference.pins.source_csv_sha256,
    )
    case.profile = fixture_profile(case.binding, case.archive.pins, case.suffix)
    case.command = (sys.executable, "-c", "pass")
    case.worker = observe_worker(case.command)


def external_completion_case(tmp_path, monkeypatch, count=5):
    """Construct unit evidence without claiming actual producer supervision."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    case = SimpleNamespace(
        paths=_paths(tmp_path),
        archive=_archive(count),
        suffix=b"com\n",
    )
    _prepare(case)
    _context(case, tmp_path, monkeypatch, count)
    _publish(case)
    return case
