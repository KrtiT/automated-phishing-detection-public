"""Restore same-parent retained consistency, not source or scoring authority.

Expected identities must come from the observing parent, not the candidate bytes.
Dictionary-bearing views are fresh; obtain each once per consuming adapter.
"""

import json
from dataclasses import dataclass, field

from . import _retained_preparation_external as external
from . import _retained_preparation_records as records
from . import evaluation_producer, phishvn, protocol_preflight
from ._phishvn_archive import PhishVNSourcePins
from ._retained_preparation_records import StudyPreparationRestoreError
from ._source_checkpoint_structure import verify_internal_preparation_structure
from ._study_preparation_records import PreparedStudySnapshot
from .evaluation_producer import PreparedInternal
from .phishvn import PreparedExternal
from .phishvn_source import DecodedPhishVNSource
from .saved_phishvn_source import restore_phishvn_source
from .source_checkpoints import CHECKPOINT_NAMES
from .source_overlap import ReconstructedSource
from .study_feasibility import assess_preparation_feasibility


@dataclass(frozen=True)
class RestoredStudyPreparation:
    reservation_sha256: str
    completion_sha256: str
    payloads: tuple[tuple[str, bytes], ...] = field(repr=False)
    internal: PreparedInternal = field(repr=False)
    overlap_domains: frozenset[str] = field(repr=False)
    _archive_pins: PhishVNSourcePins = field(repr=False)

    def payload(self, name: str) -> bytes:
        for retained_name, content in self.payloads:
            if retained_name == name:
                return content
        raise KeyError(name)

    @property
    def external(self) -> PreparedExternal:
        return external.restore_external(dict(self.payloads))

    @property
    def publisher(self) -> DecodedPhishVNSource:
        return restore_phishvn_source(
            self.payload("publisher-source.json"),
            self.payload("publisher-summary.json"),
            pins=self._archive_pins,
        )

    @property
    def reconstructed_internal(self) -> ReconstructedSource:
        return ReconstructedSource(
            self.payload("group_test.jsonl"),
            self.overlap_domains,
            {"source-overlap.json": self.payload("source-overlap.json")},
            json.loads(self.payload("source-reconstruction.json"))["reconstruction"],
        )

    @property
    def execution(self) -> dict:
        return json.loads(self.payload("preparation-complete.json"))["execution"]

    @property
    def feasibility(self) -> dict:
        return json.loads(self.payload("feasibility.json"))

    @property
    def scoring_source(self) -> dict:
        return {
            "source_interface": "retained_study_preparation_v1",
            "study_preparation_reservation_sha256": self.reservation_sha256,
            "study_preparation_complete_sha256": self.completion_sha256,
        }


def _internal(outputs, execution, reservation, source_bytes, report_bytes):
    source, report = records.public_inputs(
        outputs, execution, source_bytes, report_bytes
    )
    rules = protocol_preflight.parse_suffix_rules(
        outputs["suffix-rules.dat"].decode("utf-8")
    )
    prepared = evaluation_producer.parse_internal_partition(
        outputs["group_test.jsonl"], suffix_rules=rules, **source
    )
    domains = verify_internal_preparation_structure(
        {name: outputs[name] for name in CHECKPOINT_NAMES},
        internal=prepared,
        preparation_report=report,
        execution=execution,
        reservation_sha256=reservation,
    )
    phishvn._validate_domains(domains, rules)
    return prepared, domains, rules


def _restore(snapshot, execution, reservation, completion, source_bytes, report_bytes):
    outputs, execution = records.authenticate(
        snapshot, execution, reservation, completion
    )
    internal, domains, rules = _internal(
        outputs, execution, reservation, source_bytes, report_bytes
    )
    pins = PhishVNSourcePins(
        execution["archive_sha256"], execution["archive_size_bytes"]
    )
    publisher = restore_phishvn_source(
        outputs["publisher-source.json"], outputs["publisher-summary.json"], pins=pins
    )
    prepared = external.validate_external(outputs, publisher, rules, domains)
    records.require(
        outputs["feasibility.json"]
        == assess_preparation_feasibility(internal.records, prepared)
    )
    return RestoredStudyPreparation(
        reservation, completion, snapshot.payloads, internal, domains, pins
    )


def restore_study_preparation(
    snapshot: PreparedStudySnapshot,
    *,
    expected_identity: dict,
    expected_reservation_sha256: str,
    expected_completion_sha256: str,
    source_spec_bytes: bytes,
    preparation_summary_bytes: bytes,
) -> RestoredStudyPreparation:
    """Validate retained records without raw reads, preparation, sampling or scoring."""
    try:
        return _restore(
            snapshot,
            expected_identity,
            expected_reservation_sha256,
            expected_completion_sha256,
            source_spec_bytes,
            preparation_summary_bytes,
        )
    except Exception:
        raise StudyPreparationRestoreError(
            "invalid_retained_study_preparation"
        ) from None
