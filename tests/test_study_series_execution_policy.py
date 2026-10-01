from types import SimpleNamespace

import pytest
from study_series_adoption_fixtures import digest

from automated_phishing_detection import _study_series_execution_io as module
from automated_phishing_detection import _study_series_policy as policy
from automated_phishing_detection import execution_preflight as preflight


def setup(tmp_path, monkeypatch):
    base = SimpleNamespace(root=tmp_path, revision="a" * 40)
    content = policy.policy_bytes()
    target = tmp_path / policy.POLICY_PATH
    target.parent.mkdir()
    target.write_bytes(content)
    case = SimpleNamespace(
        base=base,
        content=content,
        target=target,
        tree=b"100644 blob " + b"b" * 40 + b"\t" + policy.POLICY_PATH.encode() + b"\0",
        blob=content,
        calls=[],
    )

    def git(root, *arguments):
        assert root == base.root
        case.calls.append(arguments)
        return case.tree if arguments[0] == "ls-tree" else case.blob

    monkeypatch.setattr(preflight, "_git", git)
    return case


def test_regular_policy_matches_exact_selected_commit_blob(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    assert module.committed_policy(case.base, digest(case.content)) == case.content
    assert case.calls == [
        ("ls-tree", "-z", case.base.revision, "--", policy.POLICY_PATH),
        ("cat-file", "blob", "b" * 40),
    ]


@pytest.mark.parametrize("operation", ("symlink", "hardlink", "changed", "wrong_pin"))
def test_unsafe_or_unpinned_policy_never_reaches_git(tmp_path, monkeypatch, operation):
    case = setup(tmp_path, monkeypatch)
    if operation in ("symlink", "hardlink"):
        original = case.target.with_name("original.json")
        case.target.rename(original)
        if operation == "symlink":
            case.target.symlink_to(original)
        else:
            case.target.hardlink_to(original)
    elif operation == "changed":
        case.target.write_bytes(b"changed")
    expected = "0" * 64 if operation == "wrong_pin" else digest(case.content)
    with pytest.raises(ValueError):
        module.committed_policy(case.base, expected)
    assert case.calls == []


@pytest.mark.parametrize(
    "tree",
    (
        b"",
        b"\0",
        b"bad\0",
        b"100644 blob " + b"b" * 40 + b"\twrong\0",
        b"120000 blob " + b"b" * 40 + b"\tdata/study-series-policy-v1.json\0",
        b"100644 tree " + b"b" * 40 + b"\tdata/study-series-policy-v1.json\0",
    ),
)
def test_wrong_git_tree_record_rejects(tmp_path, monkeypatch, tree):
    case = setup(tmp_path, monkeypatch)
    case.tree = tree
    with pytest.raises(ValueError):
        module.committed_policy(case.base, digest(case.content))


def test_even_correct_worktree_policy_must_equal_committed_blob(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    case.blob = b"different committed bytes"
    with pytest.raises(ValueError):
        module.committed_policy(case.base, digest(case.content))
