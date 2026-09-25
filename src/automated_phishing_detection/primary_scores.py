"""Label-independent primary singleton evidence shared by evaluation streams."""

from dataclasses import dataclass

from .selective_inference import InferenceCounts


@dataclass(frozen=True)
class PrimaryURLScores:
    features: tuple[float, ...]
    length_probability: float
    stage1_probability: float
    transformer_probability: float
    cascade_probability: float
    length_decision: int
    stage1_decision: int
    transformer_decision: int
    cascade_decision: int
    band_selected: bool
    monitor_probability: float
    negative_log_likelihood: float
    length_scoring_audit_json: str
    stage1_scoring_audit_json: str
    inference_counts: InferenceCounts
