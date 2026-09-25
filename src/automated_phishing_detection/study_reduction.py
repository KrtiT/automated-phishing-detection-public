"""Reduce only a complete accepted matrix with the existing scientific kernels."""

import json
from dataclasses import asdict, dataclass, field

from . import _operational_input_schema as schema
from . import _study_operational_validation as validation
from ._checkpoint_codec import canonical_bytes
from ._study_reduction_runs import match_summaries, restore_runs
from .http_replay import primary_http_summary, reference_invocations
from .operational_inputs import AcceptedOperationalInputs
from .operational_summary import summarize_operational_runs
from .study_evidence import reduce_study_evidence


class StudyReductionError(ValueError):
    """A supplied study cannot be reduced; absence is not invented in its place."""


@dataclass(frozen=True)
class ReducedStudyBytes:
    operational_bytes: bytes = field(repr=False)
    study_bytes: bytes = field(repr=False)

    @property
    def operational(self):
        return json.loads(self.operational_bytes)

    @property
    def study(self):
        return json.loads(self.study_bytes)


def reduce_accepted_study(accepted, slots) -> ReducedStudyBytes:
    try:
        schema.require(type(accepted) is AcceptedOperationalInputs)
        validation.slots(slots)
        schema.require(all(slot.accepted is not None for slot in slots))
        runs = restore_runs(accepted, slots)
        reference = reference_invocations(runs[0])
        http = primary_http_summary(runs[20:25])
        operational = summarize_operational_runs(runs)
        match_summaries(operational, slots)
        study = reduce_study_evidence(
            internal=accepted.internal.snapshot.population,
            external=accepted.external.snapshot.replay.evidence,
            reference=reference,
            http=http,
        )
        return ReducedStudyBytes(
            canonical_bytes(operational), canonical_bytes(asdict(study))
        )
    except Exception:
        raise StudyReductionError("invalid_study_reduction") from None
