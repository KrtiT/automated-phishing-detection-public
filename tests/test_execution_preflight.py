"""Checkout/runtime authentication using only synthetic temporary repositories."""

import copy
import json
import os
import subprocess
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from automated_phishing_detection import execution_preflight as preflight

CONTRACT_PATH = "data/execution-binding-contract-v3.json"
PACKAGE = "src/automated_phishing_detection"
TEMPLATE = Path(__file__).resolve().parents[1] / CONTRACT_PATH
HISTORICAL_CONTRACT_PATH = "data/execution-binding-contract-v2.json"
HISTORICAL_CONTRACT_SHA256 = (
    "887f771381927dfe1b9268a45f4e605baf3e9a7caee2b7005cdfe68b1be516e1"
)
NEW_PUBLIC_PATHS = {
    "data/secondary-development-contract-v1.json",
    "data/secondary-seed-probe-contract-v1.json",
    "reports/secondary-development-correction-v2-summary.json",
    "reports/secondary-seed-probe-correction-v1-summary.json",
}


def test_historical_v2_contract_bytes_are_preserved():
    historical = TEMPLATE.parents[1] / HISTORICAL_CONTRACT_PATH
    assert sha256(historical.read_bytes()).hexdigest() == HISTORICAL_CONTRACT_SHA256


def test_binding_uses_only_the_fixed_v3_contract_path():
    assert preflight._CONTRACT_PATH == CONTRACT_PATH


def test_new_code_rejects_historical_v2_contract_schema():
    historical = TEMPLATE.parents[1] / HISTORICAL_CONTRACT_PATH
    with pytest.raises(preflight.ExecutionPreflightError, match="contract schema"):
        preflight._contract(historical.read_bytes())


def test_v3_profile_preserves_historical_runtime_and_all_public_pins():
    assert TEMPLATE.is_file(), "the prospective v3 profile must exist"
    historical = json.loads(
        (TEMPLATE.parents[1] / HISTORICAL_CONTRACT_PATH).read_bytes()
    )
    current = json.loads(TEMPLATE.read_bytes())
    assert current["schema_version"] == 3
    assert current["contract_id"] == "execution-binding-v3"
    assert current["protected_evaluation_ready"] is False
    assert current["runtime"] == historical["runtime"]
    previous_pins = historical["public_file_sha256"]
    current_pins = current["public_file_sha256"]
    assert len(previous_pins) == 26
    assert current_pins == {
        **previous_pins,
        HISTORICAL_CONTRACT_PATH: HISTORICAL_CONTRACT_SHA256,
        **{
            name: sha256((TEMPLATE.parents[1] / name).read_bytes()).hexdigest()
            for name in NEW_PUBLIC_PATHS
        },
    }
    assert preflight._contract(TEMPLATE.read_bytes()) == current


def test_v3_secondary_report_pins_match_the_bound_loader():
    from automated_phishing_detection.bound_secondary import PUBLIC_REPORTS

    assert TEMPLATE.is_file(), "the prospective v3 profile must exist"
    pins = json.loads(TEMPLATE.read_bytes())["public_file_sha256"]
    assert {relative: pins[relative] for relative, _ in PUBLIC_REPORTS.values()} == {
        relative: digest for relative, digest in PUBLIC_REPORTS.values()
    }


def test_historical_callers_use_fixed_v2_interfaces():
    from automated_phishing_detection import (
        development_execution,
        seed_probe_execution,
    )

    for module in (development_execution, seed_probe_execution):
        assert module.bind_execution is preflight._bind_historical_v2_execution
        assert module.recheck_binding is preflight._recheck_historical_v2_binding


def test_historical_binder_rejects_other_pins_before_reading(monkeypatch):
    monkeypatch.setattr(
        preflight,
        "_read_regular",
        lambda *args: pytest.fail("read preceded fixed historical pin validation"),
    )
    with pytest.raises(preflight.ExecutionPreflightError, match="historical"):
        preflight._bind_historical_v2_execution(
            Path("unused"),
            expected_revision="a" * 40,
            expected_contract_sha256="b" * 64,
        )


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def git(root, *args):
    return (
        subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            env={
                key: value
                for key, value in os.environ.items()
                if not key.startswith("GIT_")
            },
        )
        .stdout.decode()
        .strip()
    )


def write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def commit(root):
    git(root, "add", "--all")
    git(
        root,
        "-c",
        "user.name=Synthetic",
        "-c",
        "user.email=synthetic@example.test",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "--no-verify",
        "-m",
        "Synthetic fixture",
    )
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "--quiet")
    contract = json.loads(TEMPLATE.read_bytes())
    for name in contract["public_file_sha256"]:
        content = canonical({"synthetic_public_path": name}).encode()
        write(root, name, content)
        contract["public_file_sha256"][name] = sha256(content).hexdigest()
    write(root, CONTRACT_PATH, canonical(contract).encode())
    write(root, f"{PACKAGE}/__init__.py", b'"""Synthetic package."""\n')
    write(
        root, f"{PACKAGE}/execution_preflight.py", b'"""Synthetic binding module."""\n'
    )
    write(root, f"{PACKAGE}/model.py", b"VALUE = 1\n")
    write(root, "scripts/measurement.py", b"SYNTHETIC = True\n")
    write(root, "pyproject.toml", b'[project]\nname = "synthetic"\n')
    write(root, "uv.lock", b"version = 1\n")
    write(root, ".gitignore", b"private/\n__pycache__/\n")
    revision = commit(root)
    runtime = copy.deepcopy(contract["runtime"])
    imports = {
        "automated_phishing_detection": (
            str(root / PACKAGE / "__init__.py"),
            str(root / PACKAGE / "__init__.py"),
            (str(root / PACKAGE),),
        ),
        "automated_phishing_detection.execution_preflight": (
            str(root / PACKAGE / "execution_preflight.py"),
            str(root / PACKAGE / "execution_preflight.py"),
            (),
        ),
    }
    monkeypatch.setattr(preflight, "_probe_runtime", lambda: copy.deepcopy(runtime))
    monkeypatch.setattr(preflight, "_imported_project_modules", lambda: imports.copy())
    for name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
        monkeypatch.delenv(name, raising=False)
    return SimpleNamespace(
        root=root,
        contract=contract,
        runtime=runtime,
        imports=imports,
        revision=revision,
    )


def bind(repository, **overrides):
    arguments = {
        "expected_revision": repository.revision,
        "expected_contract_sha256": sha256(
            (repository.root / CONTRACT_PATH).read_bytes()
        ).hexdigest(),
    }
    return preflight.bind_execution(repository.root, **{**arguments, **overrides})


@pytest.fixture
def historical_repository(repository):
    root = TEMPLATE.parents[1]
    content = (root / HISTORICAL_CONTRACT_PATH).read_bytes()
    contract = json.loads(content)
    for name in contract["public_file_sha256"]:
        write(repository.root, name, (root / name).read_bytes())
    write(repository.root, HISTORICAL_CONTRACT_PATH, content)
    (repository.root / CONTRACT_PATH).unlink()
    repository.revision = commit(repository.root)
    repository.contract = contract
    return repository


def test_historical_binding_and_recheck_do_not_require_v3(historical_repository):
    repository = historical_repository
    result = preflight._bind_historical_v2_execution(
        repository.root,
        expected_revision=repository.revision,
        expected_contract_sha256=HISTORICAL_CONTRACT_SHA256,
    )
    assert result.protected_evaluation_ready is False
    assert CONTRACT_PATH not in dict(result.source_hashes)
    assert dict(result.source_hashes)[HISTORICAL_CONTRACT_PATH] == (
        HISTORICAL_CONTRACT_SHA256
    )
    preflight._recheck_historical_v2_binding(result)
    with pytest.raises(preflight.ExecutionPreflightError, match="missing"):
        preflight.recheck_binding(result)


def test_historical_contract_cannot_be_replaced_with_relinked_bytes(
    historical_repository, monkeypatch
):
    repository = historical_repository
    changed = copy.deepcopy(repository.contract)
    changed["scope"] = "changed historical contract"
    write(repository.root, HISTORICAL_CONTRACT_PATH, canonical(changed).encode())
    repository.revision = commit(repository.root)
    monkeypatch.setattr(
        preflight,
        "_probe_runtime",
        lambda: pytest.fail("runtime preceded historical pin authentication"),
    )
    with pytest.raises(preflight.ExecutionPreflightError, match="SHA-256"):
        preflight._bind_historical_v2_execution(
            repository.root,
            expected_revision=repository.revision,
            expected_contract_sha256=HISTORICAL_CONTRACT_SHA256,
        )


def test_historical_recheck_rejects_active_v3_binding(repository):
    result = bind(repository)
    with pytest.raises(preflight.ExecutionPreflightError, match="historical"):
        preflight._recheck_historical_v2_binding(result)


def test_historical_recheck_detects_committed_source_change(historical_repository):
    repository = historical_repository
    result = preflight._bind_historical_v2_execution(
        repository.root,
        expected_revision=repository.revision,
        expected_contract_sha256=HISTORICAL_CONTRACT_SHA256,
    )
    write(repository.root, f"{PACKAGE}/model.py", b"VALUE = 2\n")
    repository.revision = commit(repository.root)
    with pytest.raises(preflight.ExecutionPreflightError, match="revision"):
        preflight._recheck_historical_v2_binding(result)


@pytest.mark.parametrize("override_pin", [False, True])
def test_historical_committed_files_always_enforce_the_fixed_v2_bytes(
    historical_repository, override_pin
):
    repository = historical_repository
    changed = canonical({"replacement": "historical profile"}).encode()
    write(repository.root, HISTORICAL_CONTRACT_PATH, changed)
    repository.revision = commit(repository.root)
    public = (
        {HISTORICAL_CONTRACT_PATH: sha256(changed).hexdigest()} if override_pin else {}
    )
    with pytest.raises(preflight.ExecutionPreflightError, match="SHA-256"):
        preflight._historical_v2_committed_files(
            repository.root, repository.revision, public
        )


def test_binding_authenticates_selected_committed_bytes_and_is_frozen(repository):
    result = bind(repository)
    expected = set(repository.contract["public_file_sha256"]) | {
        CONTRACT_PATH,
        "pyproject.toml",
        "uv.lock",
        "scripts/measurement.py",
        f"{PACKAGE}/__init__.py",
        f"{PACKAGE}/execution_preflight.py",
        f"{PACKAGE}/model.py",
    }
    assert type(result) is preflight.ExecutionBinding
    assert result.root == repository.root.resolve()
    assert result.revision == repository.revision
    assert (
        result.contract_sha256
        == sha256((repository.root / CONTRACT_PATH).read_bytes()).hexdigest()
    )
    assert result.source_hashes == tuple(
        sorted(
            (name, sha256((repository.root / name).read_bytes()).hexdigest())
            for name in expected
        )
    )
    assert result.runtime_json == canonical(repository.runtime)
    assert result.protected_evaluation_ready is False
    with pytest.raises(FrozenInstanceError):
        result.revision = "0" * 40
    preflight.recheck_binding(result)


@pytest.mark.parametrize(
    "name,value",
    [
        ("expected_revision", "a" * 39),
        ("expected_revision", "A" * 40),
        ("expected_revision", None),
        ("expected_contract_sha256", "a" * 63),
        ("expected_contract_sha256", "A" * 64),
        ("expected_contract_sha256", True),
    ],
)
def test_reviewed_pins_are_exact_lowercase_hex(repository, name, value):
    with pytest.raises(preflight.ExecutionPreflightError, match="revision|contract"):
        bind(repository, **{name: value})


def test_wrong_reviewed_revision_is_rejected(repository):
    with pytest.raises(preflight.ExecutionPreflightError, match="revision"):
        bind(repository, expected_revision="0" * 40)


def test_wrong_contract_hash_is_rejected_before_runtime_probe(repository, monkeypatch):
    monkeypatch.setattr(
        preflight,
        "_probe_runtime",
        lambda: pytest.fail("runtime preceded authentication"),
    )
    with pytest.raises(preflight.ExecutionPreflightError, match="contract"):
        bind(repository, expected_contract_sha256="0" * 64)


def test_historical_expected_hash_is_rejected_before_runtime_or_models(
    repository, monkeypatch
):
    monkeypatch.setattr(
        preflight,
        "_probe_runtime",
        lambda: pytest.fail("runtime preceded v2 authentication"),
    )
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda name: pytest.fail("module import preceded v2 authentication"),
    )
    with pytest.raises(preflight.ExecutionPreflightError, match="contract"):
        bind(repository, expected_contract_sha256=HISTORICAL_CONTRACT_SHA256)


@pytest.mark.parametrize("kind", ["untracked", "modified", "staged", "deleted"])
def test_dirty_checkouts_are_rejected(repository, kind):
    if kind == "untracked":
        write(repository.root, "notes.txt", b"untracked\n")
    elif kind == "deleted":
        (repository.root / f"{PACKAGE}/model.py").unlink()
    else:
        write(repository.root, f"{PACKAGE}/model.py", b"VALUE = 2\n")
        if kind == "staged":
            git(repository.root, "add", f"{PACKAGE}/model.py")
    with pytest.raises(preflight.ExecutionPreflightError, match="clean"):
        bind(repository)


def test_assume_unchanged_cannot_hide_modified_authenticated_source(repository):
    name = f"{PACKAGE}/model.py"
    git(repository.root, "update-index", "--assume-unchanged", name)
    write(repository.root, name, b"VALUE = 2\n")
    assert git(repository.root, "status", "--porcelain") == ""
    with pytest.raises(preflight.ExecutionPreflightError, match="committed|bytes"):
        bind(repository)


def test_commit_replacement_cannot_redirect_reviewed_source(repository):
    original = repository.revision
    write(repository.root, f"{PACKAGE}/model.py", b"VALUE = 2\n")
    replacement = commit(repository.root)
    git(repository.root, "replace", original, replacement)
    git(repository.root, "update-ref", "HEAD", original)
    assert git(repository.root, "status", "--porcelain") == ""
    with pytest.raises(preflight.ExecutionPreflightError, match="clean|committed"):
        bind(repository)


def test_blob_replacement_cannot_redirect_reviewed_source(repository):
    name = f"{PACKAGE}/model.py"
    original = repository.revision
    original_blob = git(repository.root, "rev-parse", f"{original}:{name}")
    write(repository.root, name, b"VALUE = 2\n")
    replacement = commit(repository.root)
    replacement_blob = git(repository.root, "rev-parse", f"{replacement}:{name}")
    git(repository.root, "replace", original_blob, replacement_blob)
    git(repository.root, "update-ref", "HEAD", original)
    git(repository.root, "read-tree", original)
    git(repository.root, "update-index", "--assume-unchanged", name)
    assert git(repository.root, "status", "--porcelain") == ""
    with pytest.raises(preflight.ExecutionPreflightError, match="committed|bytes"):
        bind(repository)


def test_subdirectory_cannot_bind_parent_repository(repository):
    child = repository.root / "private" / "not-a-repository"
    child.mkdir(parents=True)
    with pytest.raises(preflight.ExecutionPreflightError, match="top-level"):
        preflight.bind_execution(
            child,
            expected_revision=repository.revision,
            expected_contract_sha256="a" * 64,
        )


def test_nested_repository_uses_its_own_commit_not_parent(repository, tmp_path):
    git(tmp_path, "init", "--quiet")
    result = bind(repository)
    assert result.revision == repository.revision


def test_git_environment_cannot_redirect_repository(repository, monkeypatch, tmp_path):
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "not-git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "wrong-tree"))
    result = bind(repository)
    assert result.root == repository.root


def test_ignored_private_files_are_not_opened_or_traversed(repository, monkeypatch):
    private = repository.root / "private"
    private.mkdir()
    os.mkfifo(private / "do-not-open")
    opened = []
    original = preflight._read_regular

    def observe(root, relative):
        opened.append(relative)
        assert not relative.startswith("private/")
        return original(root, relative)

    monkeypatch.setattr(preflight, "_read_regular", observe)
    result = bind(repository)
    assert len(opened) >= len(result.source_hashes)
    assert not any(name.startswith("private/") for name in opened)


def test_source_symlink_is_rejected_even_if_committed(repository):
    name = f"{PACKAGE}/model.py"
    (repository.root / name).unlink()
    (repository.root / name).symlink_to("__init__.py")
    repository.revision = commit(repository.root)
    with pytest.raises(preflight.ExecutionPreflightError, match="regular|mode|symlink"):
        bind(repository)


def test_ancestor_symlink_cannot_redirect_regular_reader(repository, tmp_path):
    (repository.root / "private").mkdir()
    write(repository.root, "private/target.py", b"never authenticated\n")
    (repository.root / "redirect").symlink_to(
        repository.root / "private", target_is_directory=True
    )
    with pytest.raises(preflight.ExecutionPreflightError, match="regular|symlink"):
        preflight._read_regular(repository.root, "redirect/target.py")


@pytest.mark.parametrize(
    "name", ["../outside", "/etc/passwd", "data/raw/source.json", "private/model.json"]
)
def test_contract_cannot_expand_the_public_read_allowlist(repository, name):
    repository.contract["public_file_sha256"][name] = "a" * 64
    write(repository.root, CONTRACT_PATH, canonical(repository.contract).encode())
    repository.revision = commit(repository.root)
    with pytest.raises(preflight.ExecutionPreflightError, match="public.*allowlist"):
        bind(repository)


def test_committed_public_summary_still_must_match_contract_digest(repository):
    name = next(iter(repository.contract["public_file_sha256"]))
    write(repository.root, name, b'{"different":"committed public evidence"}')
    repository.revision = commit(repository.root)
    with pytest.raises(preflight.ExecutionPreflightError, match="public.*SHA-256"):
        bind(repository)


@pytest.mark.parametrize("name", sorted(NEW_PUBLIC_PATHS | {HISTORICAL_CONTRACT_PATH}))
def test_changed_v3_public_pin_is_rejected_before_runtime_or_models(
    repository, monkeypatch, name
):
    write(repository.root, name, b'{"changed":"public metadata"}')
    repository.revision = commit(repository.root)
    monkeypatch.setattr(
        preflight,
        "_probe_runtime",
        lambda: pytest.fail("runtime preceded public pin authentication"),
    )
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda module: pytest.fail("module import preceded public pin authentication"),
    )
    with pytest.raises(preflight.ExecutionPreflightError, match="public.*SHA-256"):
        bind(repository)


@pytest.mark.parametrize(
    "change",
    [
        "ready",
        "schema",
        "legacy_schema",
        "identity",
        "legacy_identity",
        "duplicate",
        "nonfinite",
    ],
)
def test_contract_schema_is_strict(repository, change):
    payload = copy.deepcopy(repository.contract)
    if change == "ready":
        payload["protected_evaluation_ready"] = True
    elif change == "schema":
        payload["schema_version"] = True
    elif change == "legacy_schema":
        payload["schema_version"] = 1
    elif change == "identity":
        payload["contract_id"] = "unknown"
    elif change == "legacy_identity":
        payload["contract_id"] = "execution-binding-v1"
    raw = canonical(payload)
    if change == "duplicate":
        raw = '{"schema_version":1,' + raw[1:]
    elif change == "nonfinite":
        raw = raw[:-1] + ',"invalid":NaN}'
    write(repository.root, CONTRACT_PATH, raw.encode())
    repository.revision = commit(repository.root)
    with pytest.raises(preflight.ExecutionPreflightError, match="contract"):
        bind(repository)


@pytest.mark.parametrize(
    "change",
    [
        "foreign_file",
        "foreign_origin",
        "missing_file",
        "foreign_package_path",
        "unbound_file",
    ],
)
def test_current_project_imports_must_come_from_bound_sources(
    repository, change, tmp_path
):
    name = "automated_phishing_detection.execution_preflight"
    filename, origin, paths = repository.imports[name]
    if change == "foreign_file":
        filename = str(tmp_path / "installed.py")
    elif change == "foreign_origin":
        origin = str(tmp_path / "installed.py")
    elif change == "missing_file":
        filename = None
    elif change == "foreign_package_path":
        paths = (str(tmp_path),)
    else:
        write(repository.root, "private/unbound.py", b"VALUE=1\n")
        filename = origin = str(repository.root / "private/unbound.py")
    repository.imports[name] = filename, origin, paths
    with pytest.raises(preflight.ExecutionPreflightError, match="import"):
        bind(repository)


def test_valid_late_import_is_allowed_but_foreign_late_import_is_not(
    repository, tmp_path
):
    binding = bind(repository)
    path = str(repository.root / PACKAGE / "model.py")
    repository.imports["automated_phishing_detection.model"] = path, path, ()
    preflight.recheck_binding(binding)
    path = str(tmp_path / "foreign.py")
    repository.imports["automated_phishing_detection.foreign"] = path, path, ()
    with pytest.raises(preflight.ExecutionPreflightError, match="import"):
        preflight.recheck_binding(binding)


def test_import_name_must_match_its_authenticated_module_path(repository):
    path = str(repository.root / PACKAGE / "model.py")
    repository.imports["automated_phishing_detection.execution_preflight"] = (
        path,
        path,
        (),
    )
    with pytest.raises(preflight.ExecutionPreflightError, match="import"):
        bind(repository)


@pytest.mark.parametrize(
    "key,value",
    [
        ("python", None),
        ("platform", []),
        ("hardware", {}),
        ("blas", None),
        ("mps", {"built": 1, "available": True}),
        ("environment", {}),
        ("numerical_threads", True),
    ],
)
def test_malformed_nested_runtime_contract_is_rejected(repository, key, value):
    repository.contract["runtime"][key] = value
    write(repository.root, CONTRACT_PATH, canonical(repository.contract).encode())
    repository.revision = commit(repository.root)
    with pytest.raises(preflight.ExecutionPreflightError, match="contract runtime"):
        bind(repository)


def test_source_change_during_probe_cannot_hide_behind_index_flags(
    repository, monkeypatch
):
    name = f"{PACKAGE}/model.py"
    git(repository.root, "update-index", "--assume-unchanged", name)

    def mutate():
        write(repository.root, name, b"VALUE = 8\n")
        return repository.runtime

    monkeypatch.setattr(preflight, "_probe_runtime", mutate)
    with pytest.raises(preflight.ExecutionPreflightError, match="bytes changed"):
        bind(repository)


@pytest.mark.parametrize(
    "path,value",
    [
        (("python",), "3.10.20"),
        (("package_versions", "numpy"), "2.3.0"),
        (("package_versions", "httpcore"), "0.0.0"),
        (("platform", "system"), "Linux"),
        (("platform", "machine"), "x86_64"),
        (("hardware", "hw.model"), "Mac00,0"),
        (("hardware", "hw.memsize"), "1"),
        (("hardware", "macos_build"), "wrong"),
        (("blas", "name"), "accelerate"),
        (("blas", "version"), "0.3.30"),
        (("mps", "available"), False),
        (("mps", "built"), False),
        (("numerical_threads",), 2),
    ],
)
def test_runtime_mismatches_fail_closed(repository, path, value):
    target = repository.runtime
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(preflight.ExecutionPreflightError, match="runtime"):
        bind(repository)


@pytest.mark.parametrize(
    "name", ["PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"]
)
@pytest.mark.parametrize("value", ["1", "true", "", " 0", "false"])
def test_unsafe_mps_environment_rejected_before_probe(
    repository, monkeypatch, name, value
):
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        preflight,
        "_probe_runtime",
        lambda: pytest.fail("unsafe environment reached probe"),
    )
    with pytest.raises(preflight.ExecutionPreflightError, match=name):
        bind(repository)


def test_unset_and_zero_mps_flags_have_the_same_identity(repository, monkeypatch):
    initial = bind(repository)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "0")
    assert bind(repository) == initial


def test_recheck_rejects_source_runtime_or_binding_drift(repository):
    binding = bind(repository)
    repository.runtime["hardware"]["hw.logicalcpu"] = "8"
    with pytest.raises(preflight.ExecutionPreflightError, match="runtime"):
        preflight.recheck_binding(binding)
    repository.runtime["hardware"]["hw.logicalcpu"] = "16"
    with pytest.raises(preflight.ExecutionPreflightError, match="binding"):
        preflight.recheck_binding(replace(binding, runtime_json="{}"))
    write(repository.root, f"{PACKAGE}/model.py", b"VALUE = 3\n")
    with pytest.raises(preflight.ExecutionPreflightError, match="clean"):
        preflight.recheck_binding(binding)


def test_public_runtime_pins_cover_actual_locked_runtime_dependencies():
    contract = json.loads(TEMPLATE.read_bytes())
    assert {
        "starlette",
        "pydantic-core",
        "httpcore",
        "anyio",
        "torch",
        "numpy",
        "scipy",
        "scikit-learn",
        "threadpoolctl",
    } <= set(contract["runtime"]["package_versions"])
    lock = (TEMPLATE.parents[1] / "uv.lock").read_text()
    for name, version in contract["runtime"]["package_versions"].items():
        assert f'[[package]]\nname = "{name}"\nversion = "{version}"' in lock
    assert contract["protected_evaluation_ready"] is False


@pytest.mark.parametrize("failure", [None, "ignored_limit", "broken_restore"])
def test_numerical_probe_initializes_owner_state_and_restores_it(failure):
    state = {"threads": 4}
    events = []

    def threads():
        events.append("torch_query")
        return state["threads"]

    class Limit:
        def __init__(self, *, limits):
            events.append("limit")
            assert limits == 1
            state["threads"] = 4 if failure == "ignored_limit" else 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("restore")
            state["threads"] = 2 if failure == "broken_restore" else 4

    torch = SimpleNamespace(get_num_threads=threads)
    pools = SimpleNamespace(
        threadpool_limits=Limit,
        threadpool_info=lambda: [
            {
                "filepath": "synthetic-openblas",
                "internal_api": "openblas",
                "num_threads": state["threads"],
            }
        ],
    )
    if failure:
        with pytest.raises(preflight.ExecutionPreflightError, match="runtime|thread"):
            preflight._verify_thread_limit(torch, pools)
    else:
        assert preflight._verify_thread_limit(torch, pools) == 1
    assert events[0] == "torch_query"
    assert events.index("restore") > events.index("limit")
    if failure != "broken_restore":
        assert state["threads"] == 4


def test_hardware_probe_uses_only_fixed_nonidentifying_commands(monkeypatch):
    expected = json.loads(TEMPLATE.read_bytes())["runtime"]["hardware"]
    observed = []

    def execute(command, **kwargs):
        observed.append(tuple(command))
        if command[0] == "/usr/sbin/sysctl":
            assert command[1] == "-n"
            value = expected[command[2]]
        else:
            assert command[0] == "/usr/bin/sw_vers"
            value = expected[
                {"-productVersion": "macos_version", "-buildVersion": "macos_build"}[
                    command[1]
                ]
            ]
        assert kwargs["timeout"] > 0
        return SimpleNamespace(returncode=0, stdout=f"{value}\n".encode(), stderr=b"")

    monkeypatch.setattr(preflight.subprocess, "run", execute)
    assert preflight._hardware_metadata() == expected
    assert len(observed) == 7
    assert all(
        "serial" not in " ".join(row).lower() and "hostname" not in " ".join(row)
        for row in observed
    )


def test_runtime_probe_does_not_execute_a_model(monkeypatch):
    expected = json.loads(TEMPLATE.read_bytes())["runtime"]
    monkeypatch.setattr(
        preflight, "_runtime_versions", lambda: expected["package_versions"]
    )
    monkeypatch.setattr(preflight, "_hardware_metadata", lambda: expected["hardware"])
    monkeypatch.setattr(
        preflight.platform, "python_version", lambda: expected["python"]
    )
    monkeypatch.setattr(preflight.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(preflight.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(preflight, "_verify_thread_limit", lambda torch, pools: 1)
    for key in expected["environment"]:
        monkeypatch.delenv(key, raising=False)
    modules = {
        "numpy": SimpleNamespace(
            __config__=SimpleNamespace(
                CONFIG={"Build Dependencies": {"blas": expected["blas"]}}
            )
        ),
        "torch": SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_built=lambda: True, is_available=lambda: True)
            )
        ),
        "threadpoolctl": SimpleNamespace(),
    }
    monkeypatch.setattr(
        preflight.importlib, "import_module", lambda name: modules[name]
    )
    assert preflight._probe_runtime() == expected
