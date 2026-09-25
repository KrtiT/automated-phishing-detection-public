"""Original producer interruptions and completed evidence survive cleanup failures."""

import asyncio
import json
from contextlib import contextmanager

import pytest
import test_external_source_runner as fixtures

runner_api = fixtures.runner_api
runner_case = fixtures.runner_case


def failed_session(case, cleanup_error):
    @contextmanager
    def opened(*args):
        try:
            with fixtures.session_owner(case, *args) as session:
                yield session
        finally:
            raise cleanup_error

    return opened


@pytest.mark.parametrize(
    "kind", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
@pytest.mark.parametrize("cleanup_kind", [OSError, KeyboardInterrupt])
def test_real_producer_first_interruption_survives_session_cleanup(
    runner_api, runner_case, monkeypatch, kind, cleanup_kind
):
    case, first = runner_case, kind("private-canary")
    scorer = case.session.evaluation.primary.scorer
    original = scorer.score_all

    def interrupted(url):
        if scorer.urls:
            raise first
        return original(url)

    monkeypatch.setattr(scorer, "score_all", interrupted)
    monkeypatch.setattr(
        runner_api,
        "open_bound_external_session",
        failed_session(case, cleanup_kind("cleanup")),
    )
    with pytest.raises(kind) as caught:
        fixtures.run(runner_api, case)
    assert caught.value is first and len(scorer.urls) == 1
    private = (case.paths.attempt / "external-failure.json").read_bytes()
    assert caught.value.progress == private
    assert json.loads(private)["cleanup_failed"] is True
    assert "private-canary" not in str(caught.value)
    assert not case.paths.public_summary.exists()


def later_failure(api, case, monkeypatch, boundary):
    def broken(*args):
        raise OSError("private-canary")

    if boundary == "session":
        monkeypatch.setattr(
            api,
            "open_bound_external_session",
            failed_session(case, OSError("cleanup")),
        )
    else:
        attribute = (
            "recheck_binding" if boundary == "binding" else "build_external_public"
        )
        monkeypatch.setattr(api, attribute, broken)


@pytest.mark.parametrize("boundary", ["session", "binding", "summary"])
def test_completed_producer_is_retained_through_later_failure(
    runner_api, runner_case, monkeypatch, boundary
):
    case, observations = runner_case, []
    owner = runner_api.body.ExternalSourceFailureState
    original = owner.snapshot

    def snapshot(state, *args):
        observations.append((state.produced, state.session_closed))
        return original(state, *args)

    monkeypatch.setattr(owner, "snapshot", snapshot)
    later_failure(runner_api, case, monkeypatch, boundary)
    with pytest.raises(runner_api.ExternalSourceExecutionError):
        fixtures.run(runner_api, case)
    produced, closed = observations[0]
    assert produced.public_summary == case.produced.public_summary
    assert produced.private_outputs == case.produced.private_outputs
    assert closed is (boundary != "session")
    assert (case.paths.attempt / "external-failure.json").is_file()


@pytest.mark.parametrize("after", [False, True])
def test_publication_failure_never_retries_or_replaces_outcome(
    runner_api, runner_case, monkeypatch, after
):
    original, calls = runner_api.publish_completion, []

    def broken(*args, **kwargs):
        calls.append(True)
        if after:
            original(*args, **kwargs)
        raise OSError("publication failed")

    def forbidden(*args, **kwargs):
        pytest.fail("publication failure attempted replacement failure record")

    monkeypatch.setattr(runner_api, "publish_completion", broken)
    monkeypatch.setattr(runner_api, "record_failure", forbidden)
    monkeypatch.setattr(runner_api, "retain_external_failure", forbidden)
    with pytest.raises(runner_api.ExternalSourceExecutionError, match="publication"):
        fixtures.run(runner_api, runner_case)
    assert calls == [True]
    assert runner_case.paths.public_summary.exists() is after


def test_private_failure_precedes_failure_outcome(runner_api, runner_case, monkeypatch):
    original, calls = runner_api.record_failure, []

    def checked(attempt, **kwargs):
        assert (attempt.directory / "external-failure.json").is_file()
        calls.append(True)
        return original(attempt, **kwargs)

    runner_case.paths.suffix_rules.write_bytes(b"invalid")
    monkeypatch.setattr(runner_api, "record_failure", checked)
    with pytest.raises(runner_api.ExternalSourceExecutionError):
        fixtures.run(runner_api, runner_case)
    assert calls == [True]


@pytest.mark.parametrize("kind", [OSError, KeyboardInterrupt])
def test_failure_persistence_keeps_first_interruption_and_attempts_each_once(
    runner_api, runner_case, monkeypatch, kind
):
    first, calls = kind("original"), []

    def interrupted(*args, **kwargs):
        raise first

    def sidecar(*args):
        calls.append("sidecar")
        raise KeyboardInterrupt("first persistence")

    def outcome(*args, **kwargs):
        calls.append("outcome")
        raise SystemExit("later persistence")

    monkeypatch.setattr(runner_api.body, "produce_external_evidence", interrupted)
    monkeypatch.setattr(runner_api, "retain_external_failure", sidecar)
    monkeypatch.setattr(runner_api, "record_failure", outcome)
    with pytest.raises(KeyboardInterrupt) as caught:
        fixtures.run(runner_api, runner_case)
    assert (caught.value is first) is (kind is KeyboardInterrupt)
    assert calls == ["sidecar", "outcome"]
    assert caught.value.progress is not None
