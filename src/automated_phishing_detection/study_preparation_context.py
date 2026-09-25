"""Public-only preparation identity shared by acquisition and retained adapters."""

from . import _study_preparation_files as files
from . import source_runner
from ._external_records_validation import context
from ._external_source_profile import resolve_external_source_profile
from ._study_preparation_records import StudyPreparationError

SOURCE = "data/sources.json"
PREPARATION = "reports/phiusiil-preparation-summary.json"


def preparation_identity(binding, profile, source):
    """Project existing authenticated source claims without filesystem access."""
    projection, pins = context(binding, profile)
    if source["suffix_rules_sha256"] != profile.suffix_rules_sha256:
        raise StudyPreparationError("preparation_suffix_pin_mismatch")
    return {
        "kind": "study_preparation_only",
        "protocol": "study-preparation-v1",
        **projection["execution"],
        "source_profile_sha256": profile.profile_sha256,
        "preparation_summary_sha256": pins[PREPARATION],
        "source_csv_sha256": source["source_csv_sha256"],
        "suffix_rules_sha256": source["suffix_rules_sha256"],
        "partition_sha256": source["expected_sha256"],
        "archive_sha256": profile.archive_pins.archive_sha256,
        "archive_size_bytes": profile.archive_pins.archive_size_bytes,
    }


def context_for_profile(binding, profile):
    context(binding, profile)
    source, buffers = source_runner._public_sources(binding)
    return preparation_identity(binding, profile, source), source, buffers, profile


def _bound_context(binding):
    return context_for_profile(binding, resolve_external_source_profile(binding))


def bound_preparation_context(binding):
    """Authenticate public metadata only; neither inspect private paths nor authorize."""
    return files.deferred(_bound_context, binding)
