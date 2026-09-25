import asyncio
import base64
import json

import pytest
from study_root_retention_fixtures import api, append_all, complete, manager, root_case

from automated_phishing_detection import _study_preparation_files as files
from automated_phishing_detection import execution_receipt as receipt


@pytest.mark.parametrize(
    "original", [KeyboardInterrupt(), SystemExit(0), asyncio.CancelledError()]
)
@pytest.mark.parametrize("later", [OSError("cleanup"), KeyboardInterrupt("later")])
def test_original_interruption_survives_final_check(
    tmp_path, monkeypatch, original, later
):
    module, case = api(), root_case(tmp_path)
    with pytest.raises(type(original)) as caught:
        with manager(module, case) as writer:
            writer.append("study-intent.json", case.contents["study-intent.json"])
            monkeypatch.setattr(writer._held, "check", lambda: throw(later))
            raise original
    assert caught.value is original
    assert not writer.publishing


def throw(error):
    raise error


@pytest.mark.parametrize("after", [False, True])
def test_publish_error_is_permanent_even_without_installed_claim(
    tmp_path, monkeypatch, after
):
    module, case = api(), root_case(tmp_path)
    original = receipt.publish_completion

    def fail(*arguments, **keywords):
        if after:
            original(*arguments, **keywords)
        raise OSError("invented publication failure")

    monkeypatch.setattr(receipt, "publish_completion", fail)
    with pytest.raises(module.StudyRootRetentionError):
        with manager(module, case) as writer:
            append_all(writer, case)
            complete(writer, case)
    assert writer.publishing
    assert case.public_path.exists() is after
    assert writer.candidate is None
    with pytest.raises(module.StudyRootRetentionError):
        complete(writer, case)


def test_failed_partial_write_preserves_pending_bytes_without_retry(
    tmp_path, monkeypatch
):
    module, case = api(), root_case(tmp_path)
    content = case.contents["study-intent.json"]
    original = files.write_file

    def fail(directory, name, value, mode):
        original(directory, name, value[:3], mode)
        raise OSError("invented partial write")

    monkeypatch.setattr(files, "write_file", fail)
    with pytest.raises(module.StudyRootRetentionError):
        with manager(module, case) as writer:
            writer.append("study-intent.json", content)
    assert (case.attempt.directory / "study-intent.json").read_bytes() == content[:3]
    progress = json.loads(writer.snapshot())
    assert progress["confirmed_sha256"] == {}
    assert (
        base64.b64decode(progress["pending_checkpoint_bytes"]["study-intent.json"])
        == content
    )
    assert writer.payloads == (("study-intent.json", content),)


def test_later_interruption_keeps_actual_missing_only_associations(
    tmp_path, monkeypatch
):
    module, case = api(), root_case(tmp_path)
    original, later = RuntimeError("body"), KeyboardInterrupt("cleanup")
    original.source_internal = object()
    later.external_failure = object()
    with pytest.raises(KeyboardInterrupt) as caught:
        with manager(module, case) as writer:
            monkeypatch.setattr(writer._held, "check", lambda: throw(later))
            raise original
    assert caught.value is later
    assert later.source_internal is original.source_internal
    assert later.external_failure is not None


def test_late_holder_failure_retains_candidate_without_promotion(tmp_path, monkeypatch):
    module, case = api(), root_case(tmp_path)
    with pytest.raises(module.StudyRootRetentionError):
        with manager(module, case) as writer:
            append_all(writer, case)
            candidate = complete(writer, case)
            monkeypatch.setattr(writer._held, "check", lambda: throw(OSError("late")))
    assert writer.candidate is candidate
    assert writer.publishing
    assert case.public_path.exists()
