"""Exact review scope rejects metadata substitutions without protected reads."""

import json
import os
from hashlib import sha256
from pathlib import Path

import pytest
import study_execution_fixtures as fixtures

from automated_phishing_detection._checkpoint_codec import canonical_bytes

execution_case = fixtures.execution_case


@pytest.mark.parametrize(
    "change",
    [
        "revision",
        "contract",
        "external",
        "operational",
        "policy",
        "method",
        "scope",
        "extra",
        "session",
        "commitment",
        "invocation",
        "root",
        "paths",
        "relative",
        "traversal",
        "unnormalized",
        "leading_double_slash",
        "model_overlap",
        "output_overlap",
    ],
)
def test_substituted_profile_scope_is_rejected(execution_case, change):
    case = execution_case
    value = fixtures.profile(case)
    replacements = {
        "revision": (value["execution"], "revision", "d" * 40),
        "contract": (value["execution"], "contract_sha256", "d" * 64),
        "external": (value["components"], "external", "d" * 64),
        "operational": (value["components"], "operational", "d" * 64),
        "policy": (value, "policy_sha256", "d" * 64),
        "method": (value, "method_sha256", "d" * 64),
        "scope": (value["source_artifact_scope"], "extra.py", "d" * 64),
        "extra": (value, "ready", True),
        "session": (value["session"]["requirements"], "power", "battery"),
        "commitment": (value["session"], "operator_commitment", "already_observed"),
        "invocation": (value["invocation"]["arguments"], "resume", True),
        "root": (value["paths"], "repo-root", "/substituted"),
        "paths": (value["paths"], "extra", "/extra"),
        "relative": (value["paths"], "archive", "relative.zip"),
        "traversal": (value["paths"], "archive", "/invented/../archive.zip"),
        "unnormalized": (value["paths"], "archive", "/invented//archive.zip"),
        "leading_double_slash": (value["paths"], "archive", "//invented/archive.zip"),
        "model_overlap": (value["paths"], "gmm", str(case.root / "private.json")),
        "output_overlap": (value["paths"], "attempt", value["paths"]["archive"]),
    }
    owner, name, replacement = replacements[change]
    owner[name] = replacement
    with pytest.raises(fixtures.api().StudyExecutionError):
        fixtures.bind(case, fixtures.seal(case, value))


@pytest.mark.parametrize("change", ["noncanonical", "duplicate", "extra", "revoked"])
def test_bad_envelope_is_rejected_without_public_metadata_reads(execution_case, change):
    case = execution_case
    value = fixtures.seal(case)
    if change == "extra":
        value["approved"] = True
    if change == "revoked":
        value["revoked"] = True
    content = canonical_bytes(value)
    if change == "noncanonical":
        content += b"\n"
    if change == "duplicate":
        content = b'{"revoked":true,' + content[1:]
    case.envelope_path.write_bytes(content)
    with pytest.raises(fixtures.api().StudyExecutionError):
        fixtures.api().bind_study_execution(
            case.root,
            expected_revision=case.base.revision,
            envelope_path=case.envelope_path,
            expected_envelope_sha256=sha256(content).hexdigest(),
        )
    assert case.events == []


@pytest.mark.parametrize("change", ["missing", "symlink", "changed", "wrong_name"])
def test_policy_must_be_exact_committed_regular_bytes(
    execution_case, monkeypatch, change
):
    case = execution_case
    previous = fixtures.preflight._git

    def altered(root, *arguments):
        if change == "changed" and arguments[0] == "cat-file":
            return case.policy + b" "
        if arguments[0] != "ls-tree":
            return previous(root, *arguments)
        if change == "missing":
            return b""
        result = previous(root, *arguments)
        return (
            result.replace(b"100644", b"120000")
            if change == "symlink"
            else result.replace(b"policy-v1", b"policy-v2")
            if change == "wrong_name"
            else result
        )

    monkeypatch.setattr(fixtures.preflight, "_git", altered)
    with pytest.raises(fixtures.api().StudyExecutionError):
        fixtures.bind(case)


def test_lexical_profile_binding_never_stats_or_resolves_protected_paths(
    execution_case, monkeypatch
):
    case = execution_case
    digest = fixtures.write_envelope(case, fixtures.seal(case))

    def forbidden(*args, **kwargs):
        pytest.fail("profile construction inspected a filesystem path")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "stat", forbidden)
        scoped.setattr(Path, "resolve", forbidden)
        result = fixtures.api().bind_study_execution(
            case.root,
            expected_revision=case.base.revision,
            envelope_path=case.envelope_path,
            expected_envelope_sha256=digest,
        )
    assert len(json.loads(result.profile_bytes)["paths"]) == 30


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_envelope_alias_cannot_authorize(execution_case, alias):
    case = execution_case
    digest = fixtures.write_envelope(case, fixtures.seal(case))
    original = case.envelope_path.with_name("original.json")
    case.envelope_path.rename(original)
    if alias == "symlink":
        case.envelope_path.symlink_to(original)
    else:
        os.link(original, case.envelope_path)
    with pytest.raises(fixtures.api().StudyExecutionError):
        fixtures.api().bind_study_execution(
            case.root,
            expected_revision=case.base.revision,
            envelope_path=case.envelope_path,
            expected_envelope_sha256=digest,
        )


@pytest.mark.parametrize("slot", ["profile", "access"])
def test_final_decisions_cannot_approve_only_method_or_policy(execution_case, slot):
    value = fixtures.seal(execution_case)
    value["decisions"][slot] = fixtures.decision(
        "policy", value["profile"]["policy_sha256"]
    )
    with pytest.raises(fixtures.api().StudyExecutionError):
        fixtures.bind(execution_case, value)
