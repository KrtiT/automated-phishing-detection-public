"""Public metadata substitution only; binder and all admission gates stay real."""

import json
from dataclasses import asdict, fields, replace
from hashlib import sha256
from pathlib import Path

from study_execution_fixtures import seal, write_envelope
from study_execution_profile_fixtures import profile

from automated_phishing_detection import _external_source_profile as external
from automated_phishing_detection import _operational_profile as operational
from automated_phishing_detection import execution_preflight as preflight
from automated_phishing_detection._study_execution_policy import CONTRACT_SHA256


def patch_public(case, monkeypatch):
    policy_path = {
        "study-execution-policy-v1": "data/study-execution-policy-v1.json",
        "study-execution-policy-v2": "data/study-execution-policy-v2.json",
    }[json.loads(case.policy)["policy_id"]]
    monkeypatch.setattr(preflight, "bind_execution", lambda *args, **kwargs: case.base)
    monkeypatch.setattr(preflight, "recheck_binding", lambda binding: None)
    monkeypatch.setattr(external, "EXPECTED_FORMAT", case.archive_pins)

    def git(root, *arguments):
        if arguments[0] == "ls-tree":
            assert arguments[-1] == policy_path
            return b"100644 blob " + b"c" * 40 + b"\t" + policy_path.encode() + b"\0"
        assert arguments == ("cat-file", "blob", "c" * 40)
        return case.policy

    monkeypatch.setattr(preflight, "_git", git)


def bind_metadata(case, monkeypatch):
    repository = Path(__file__).parents[1]
    case.policy = (repository / "data/study-execution-policy-v1.json").read_bytes()
    (case.base.root / "data/study-execution-policy-v1.json").write_bytes(case.policy)
    required = {
        *external._IMPLEMENTATIONS,
        *operational._REQUIRED,
        "scripts/run_study_child.py",
        "scripts/run_adopted_study.py",
    }
    pins = {name: sha256(name.encode()).hexdigest() for name in required}
    pins.update(dict(case.base.source_hashes))
    case.base = replace(
        case.base,
        contract_sha256=CONTRACT_SHA256,
        source_hashes=tuple(sorted(pins.items())),
    )
    case.root = case.base.root
    patch_public(case, monkeypatch)
    case.external = external.resolve_external_source_profile(case.base)
    case.operational = operational.resolve_operational_profile(case.base)


def authorization(case):
    from automated_phishing_detection.study_execution import bind_study_execution

    value = profile(case)
    mapping = value["paths"]
    for name in ("source_csv", "suffix_rules", "archive"):
        mapping[name.replace("_", "-")] = str(getattr(case.preparation_paths, name))
    mapping["preparation-attempt"] = str(case.preparation_paths.attempt)
    for group in (case.original.artifacts, case.original.secondary_artifacts):
        for member in fields(group):
            mapping[member.name.replace("_", "-")] = str(getattr(group, member.name))
    digest = write_envelope(case, seal(case, value))
    return bind_study_execution(
        case.root,
        expected_revision=case.base.revision,
        envelope_path=case.envelope_path,
        expected_envelope_sha256=digest,
    )


def write_child_bootstrap(case, monkeypatch, mode):
    directory = case.root.parent / "bootstrap"
    directory.mkdir()
    fixture = directory / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "base": asdict(case.base),
                "archive_pins": case.archive_pins,
                "policy": case.policy.decode(),
                "mode": mode,
            },
            default=str,
        )
    )
    tests = Path(__file__).parent
    (directory / "sitecustomize.py").write_text(
        "from study_authorized_process_child_fixtures import bootstrap\nbootstrap()\n"
    )
    monkeypatch.setenv("STUDY_PROCESS_FIXTURE", str(fixture))
    monkeypatch.setenv(
        "PYTHONPATH", ":".join(map(str, (directory, tests, tests.parent / "src")))
    )
    script = case.root / "scripts/run_study_child.py"
    script.parent.mkdir()
    script.write_bytes((tests.parent / "scripts/run_study_child.py").read_bytes())
