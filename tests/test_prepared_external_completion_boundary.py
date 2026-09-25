"""Preparation expectations cannot silently downgrade to the original mode."""

import pytest
from external_completion_fixtures import external_completion_case

from automated_phishing_detection import external_source_completion as completion
from automated_phishing_detection._external_completion_files import (
    snapshot_external_files,
)
from automated_phishing_detection._external_completion_records import (
    ExternalCompletionVerificationError,
    authenticate_external_records,
)


def test_original_files_reject_a_preparation_expectation(tmp_path, monkeypatch):
    case = external_completion_case(tmp_path, monkeypatch)
    with snapshot_external_files(
        case.paths.attempt, case.paths.public_summary
    ) as files:
        with pytest.raises(ExternalCompletionVerificationError):
            authenticate_external_records(
                files,
                case.paths.attempt,
                binding=case.binding,
                profile=case.profile,
                handoff=case.handoff,
                preparation=object(),
            )


def test_original_completion_rejects_prepared_mode_before_file_reads(
    tmp_path, monkeypatch
):
    case = external_completion_case(tmp_path, monkeypatch)
    monkeypatch.setattr(
        completion, "resolve_external_source_profile", lambda _: case.profile
    )

    def forbidden(*args, **kwargs):
        pytest.fail("mismatched input mode read completion files")

    monkeypatch.setattr(completion, "snapshot_external_files", forbidden)
    with pytest.raises(ExternalCompletionVerificationError):
        completion.verify_external_completion_snapshot(
            case.binding,
            case.paths,
            expected_handoff=case.handoff,
            worker=case.worker,
            command=case.command,
            expected_preparation=object(),
        )
