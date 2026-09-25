"""Immutable verified bytes and population for a same-parent internal handoff.

This value records completion consistency. It does not prove that its caller
supervised a process or authorize a later protected evaluation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .hypothesis_evaluation import SavedPopulation
from .paired_evaluation import BinaryPrediction, EvaluationRecord

if TYPE_CHECKING:
    from .evaluation_producer import ManifestOutcome


@dataclass(frozen=True)
class VerifiedInternalSnapshot:
    payloads: tuple[tuple[str, bytes], ...]
    records: tuple[EvaluationRecord, ...]
    prediction_columns: tuple[tuple[str, tuple[BinaryPrediction, ...]], ...]
    overlap_domains: frozenset[str]
    manifest_outcomes: tuple[tuple[int, ManifestOutcome], ...] = field(repr=False)

    def payload(self, name: str) -> bytes:
        for retained_name, content in self.payloads:
            if retained_name == name:
                return content
        raise KeyError(name)

    @property
    def public_summary(self) -> dict:
        return json.loads(self.payload("public-summary.json"))

    @property
    def population(self) -> SavedPopulation:
        return SavedPopulation(self.records, dict(self.prediction_columns))

    @property
    def manifests(self) -> dict[int, ManifestOutcome]:
        return dict(self.manifest_outcomes)


def _payload_name(path: Path, attempt: Path, public_summary: Path) -> str:
    if path == public_summary:
        return "public-summary.json"
    return f"attempt/{path.relative_to(attempt).as_posix()}"


def freeze_internal_snapshot(
    contents: Mapping[Path, bytes],
    source_buffers: Mapping[str, bytes],
    *,
    attempt: Path,
    public_summary: Path,
    population: SavedPopulation,
    overlap_domains: frozenset[str],
    manifests: Mapping[int, ManifestOutcome],
) -> VerifiedInternalSnapshot:
    payloads = [
        (_payload_name(path, attempt, public_summary), content)
        for path, content in contents.items()
    ]
    payloads.extend(
        (f"source/{name}", content) for name, content in source_buffers.items()
    )
    columns = ((name, tuple(column)) for name, column in population.predictions.items())
    return VerifiedInternalSnapshot(
        tuple(sorted(payloads)),
        tuple(population.records),
        tuple(sorted(columns)),
        frozenset(overlap_domains),
        tuple(sorted(manifests.items())),
    )
