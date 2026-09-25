"""Rich external score records preserve nullable outcomes and source provenance."""

from dataclasses import dataclass

from .bound_secondary import SecondarySeedScore, SecondaryTabularScore
from .phishvn import PreparedExternalRow
from .primary_scores import PrimaryURLScores


@dataclass(frozen=True)
class ScoredExternalRow:
    record: PreparedExternalRow
    primary: PrimaryURLScores
    secondary_tabular: tuple[SecondaryTabularScore, ...]
    secondary_seeds: tuple[SecondarySeedScore, ...]
    standardized_monitor_features: tuple[float, ...]
    policy_probability: float
    policy_decision: int
    drift_override: bool
    logical_stage2_mask: bool
