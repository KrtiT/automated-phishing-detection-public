"""Retained-input paths omit every original-source acquisition capability."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .bound_models import ArtifactPaths
from .bound_secondary import SecondaryArtifactPaths


@dataclass(frozen=True)
class PreparedInternalRunPaths:
    preparation: Path
    artifacts: ArtifactPaths
    secondary_artifacts: SecondaryArtifactPaths
    attempt: Path
    public_summary: Path


def require_paths(paths):
    from .source_runner import SourceExecutionError

    if (
        type(paths) is not PreparedInternalRunPaths
        or type(paths.artifacts) is not ArtifactPaths
        or type(paths.secondary_artifacts) is not SecondaryArtifactPaths
    ):
        raise SourceExecutionError("invalid_prepared_run_paths")


def _expected_preparation(binding, source):
    pins = dict(binding.source_hashes)
    return {
        "kind": "study_preparation_only",
        "protocol": "study-preparation-v1",
        "revision": binding.revision,
        "execution_contract_sha256": binding.contract_sha256,
        "runtime_sha256": sha256(binding.runtime_json.encode()).hexdigest(),
        "source_spec_sha256": pins["data/sources.json"],
        "preparation_summary_sha256": pins["reports/phiusiil-preparation-summary.json"],
        "partition_sha256": source["expected_sha256"],
        "source_csv_sha256": source["source_csv_sha256"],
        "suffix_rules_sha256": source["suffix_rules_sha256"],
    }


def prepared_source(binding, source, preparation):
    from .execution_receipt import _SHA256
    from .retained_study_preparation import RestoredStudyPreparation
    from .source_runner import SourceExecutionError

    if type(preparation) is not RestoredStudyPreparation:
        raise SourceExecutionError("invalid_retained_preparation")
    expected = _expected_preparation(binding, source)
    execution = preparation.execution
    if any(execution.get(name) != value for name, value in expected.items()):
        raise SourceExecutionError("preparation_binding_mismatch")
    result = preparation.scoring_source
    if any(
        type(value) is not str or not _SHA256.fullmatch(value)
        for value in (preparation.reservation_sha256, preparation.completion_sha256)
    ):
        raise SourceExecutionError("invalid_preparation_digest")
    return result
