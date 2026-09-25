"""Preparation-only values confer neither source authority nor scoring permission."""

from dataclasses import dataclass, field
from pathlib import Path

from ._external_source_profile import CandidateExternalProfile
from .execution_preflight import ExecutionBinding
from .execution_receipt import Attempt


class StudyPreparationError(ValueError):
    """A symbolic preparation rejection without private parser diagnostics."""


@dataclass(frozen=True)
class StudyPreparationPaths:
    source_csv: Path
    suffix_rules: Path
    archive: Path
    attempt: Path


@dataclass(frozen=True)
class PreparedStudySnapshot:
    reservation_sha256: str
    payloads: tuple[tuple[str, bytes], ...] = field(repr=False)

    def payload(self, name: str) -> bytes:
        for retained_name, content in self.payloads:
            if retained_name == name:
                return content
        raise KeyError(name)


@dataclass(repr=False)
class PreparationState:
    binding: ExecutionBinding
    paths: StudyPreparationPaths
    stage: str = "public_preflight"
    attempt: Attempt | None = None
    profile: CandidateExternalProfile | None = None
    identity: dict | None = None
    source: dict | None = None
    source_buffers: dict | None = None
    writer: object = None
    outputs: dict[str, bytes] = field(default_factory=dict)
