"""Immutable external evidence kept by the same observing parent.

Caller-constructed values and serialized exit claims establish no process proof
or access authorization. Fresh views do not repeat source reads or reconstruction.
"""

import json
from dataclasses import dataclass, field

from ._external_completion_files import ExternalFileSnapshot
from .evaluation_stream import ExternalEvidence
from .external_evidence_types import ScoredExternalRow
from .external_replay import ReplayedExternal
from .hypothesis_evaluation import SavedControls, SavedPopulation, WindowCounts
from .owned_worker import WorkerObservation
from .paired_evaluation import BinaryPrediction, EvaluationRecord
from .policy_replay import PolicyReplay
from .probe_replay import MonitorReplay

_Columns = tuple[tuple[str, tuple[BinaryPrediction, ...]], ...]


@dataclass(frozen=True)
class _Population:
    records: tuple[EvaluationRecord, ...] = field(repr=False)
    columns: _Columns = field(repr=False)

    def view(self) -> SavedPopulation:
        return SavedPopulation(self.records, dict(self.columns))


@dataclass(frozen=True)
class VerifiedExternalSnapshot:
    payloads: tuple[tuple[str, bytes], ...] = field(repr=False)
    profile_bytes: bytes = field(repr=False)
    rows: tuple[ScoredExternalRow, ...] = field(repr=False)
    populations: tuple[tuple[str, _Population], ...] = field(repr=False)
    control_ids: tuple[str, ...] = field(repr=False)
    control_columns: _Columns = field(repr=False)
    external_windows: WindowCounts = field(repr=False)
    policy: PolicyReplay = field(repr=False)
    role_counts: tuple[tuple[str, int], ...] = field(repr=False)
    monitors: tuple[MonitorReplay, ...] = field(repr=False)

    def payload(self, name: str) -> bytes:
        return dict(self.payloads)[name]

    @property
    def public_summary(self) -> dict:
        return json.loads(self.payload("public-summary.json"))

    @property
    def replay(self) -> ReplayedExternal:
        evidence = ExternalEvidence(
            {name: population.view() for name, population in self.populations},
            SavedControls(self.control_ids, dict(self.control_columns)),
            self.external_windows,
            self.policy,
            dict(self.role_counts),
        )
        return ReplayedExternal(self.rows, evidence, self.monitors)


def _columns(predictions) -> _Columns:
    return tuple(sorted((name, tuple(values)) for name, values in predictions.items()))


def freeze_external_snapshot(
    files: ExternalFileSnapshot, profile_bytes: bytes, replay: ReplayedExternal
) -> VerifiedExternalSnapshot:
    """Retain already-verified immutable components, without asserting authority."""
    evidence = replay.evidence
    populations = tuple(
        sorted(
            (
                name,
                _Population(
                    tuple(population.records), _columns(population.predictions)
                ),
            )
            for name, population in evidence.populations.items()
        )
    )
    return VerifiedExternalSnapshot(
        tuple(files.payloads),
        profile_bytes,
        tuple(replay.rows),
        populations,
        tuple(evidence.controls.record_ids),
        _columns(evidence.controls.predictions),
        evidence.external_windows,
        evidence.replay,
        tuple(sorted(evidence.role_counts.items())),
        tuple(replay.monitors),
    )


@dataclass(frozen=True)
class ObservedExternalCompletion:
    worker: WorkerObservation
    snapshot: VerifiedExternalSnapshot = field(repr=False)

    @property
    def public_summary(self) -> dict:
        return self.snapshot.public_summary
