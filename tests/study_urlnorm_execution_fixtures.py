"""Metadata-only amendment fixtures, never real scientific execution authority."""

import adopted_study_profile_fixtures as success
import study_execution_fixtures as original
from study_urlnorm_fixtures import policy_bytes, profile, seal

POLICY_PATH = "data/study-execution-policy-v3.json"


def install_policy(case, monkeypatch):
    read = original.preflight._read_regular
    case.policy = policy_bytes()

    def selected(root, relative):
        if root == case.root and relative == POLICY_PATH:
            case.events.append(("public", relative))
            return case.policy
        return read(root, relative)

    def git(root, *arguments):
        if arguments[0] == "ls-tree":
            assert arguments[-1] == POLICY_PATH
            return b"100644 blob " + b"c" * 40 + b"\t" + POLICY_PATH.encode() + b"\0"
        assert arguments == ("cat-file", "blob", "c" * 40)
        return case.policy

    monkeypatch.setattr(original.preflight, "_read_regular", selected)
    monkeypatch.setattr(original.preflight, "_git", git)


def bound(case, monkeypatch):
    value = seal(profile(case))
    install_policy(case, monkeypatch)
    return original.bind(case, value)


def successful_case(prepared, manifests, monkeypatch):
    case = success.make_case(prepared, manifests)
    case.profile = profile(case)
    case.policy = policy_bytes()
    monkeypatch.setattr(success, "seal", lambda unused, current: seal(current))
    return case
