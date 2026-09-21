"""Bind reviewed public code/runtime identity without opening research inputs.

Call only in an idle isolated process, before starting a numerical owner session.
The binding never grants protected-data access or authenticates private artifacts.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath

_CONTRACT_PATH = "data/execution-binding-contract-v2.json"
_PACKAGE_ROOT = "src/automated_phishing_detection"
_PUBLIC_PATHS = frozenset(
    {
        "data/evaluation-contract-v1.json",
        "data/evaluation-manifest-contract-v1.json",
        "data/execution-binding-contract-v1.json",
        "data/http-replay-contract-v1.json",
        "data/operational-workloads-v1.json",
        "data/rq1-baseline-contract-v2.json",
        "data/rq1-baseline-contract.json",
        "data/rq1-transformer-cascade-contract-v1.json",
        "data/rq1-transformer-cascade-contract-v2.json",
        "data/rq2-gmm-development-contract-v1.json",
        "data/secondary-analysis-contract-v1.json",
        "data/shift-execution-contract-v1.json",
        "data/singleton-inference-amendment-v1.json",
        "data/sources.json",
        "docs/advisor-approval/2026-08-16-realignment-matrix.md",
        "reports/inference-compatibility-v1-preflight-correction.json",
        "reports/inference-compatibility-v1.json",
        "reports/phiusiil-preparation-summary.json",
        "reports/rq1-baseline-v2-summary.json",
        "reports/rq1-saga-convergence-v1-execution.json",
        "reports/rq1-saga-convergence-v2-execution.json",
        "reports/rq1-transformer-cascade-v2-execution.json",
        "reports/rq1-transformer-cascade-v2-retry-execution.json",
        "reports/rq1-transformer-cascade-v2-summary.json",
        "reports/rq2-gmm-development-v1-description.json",
        "reports/rq2-gmm-development-v1-summary.json",
    }
)
_RUNTIME_PACKAGES = (
    "annotated-doc",
    "annotated-types",
    "anyio",
    "certifi",
    "click",
    "cloudpickle",
    "exceptiongroup",
    "fastapi",
    "filelock",
    "fsspec",
    "h11",
    "httpcore",
    "httpx",
    "idna",
    "jinja2",
    "joblib",
    "markupsafe",
    "mpmath",
    "networkx",
    "numpy",
    "pydantic",
    "pydantic-core",
    "scikit-learn",
    "scipy",
    "starlette",
    "sympy",
    "threadpoolctl",
    "torch",
    "typing-extensions",
    "typing-inspection",
    "uvicorn",
)
_VERSIONED_IMPORTS = {
    "fastapi": "fastapi",
    "httpcore": "httpcore",
    "httpx": "httpx",
    "numpy": "numpy",
    "pydantic": "pydantic",
    "pydantic-core": "pydantic_core",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "starlette": "starlette",
    "threadpoolctl": "threadpoolctl",
    "torch": "torch",
    "uvicorn": "uvicorn",
}
_ENVIRONMENT_FLAGS = ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH")
_HARDWARE_COMMANDS = {
    "hw.model": ("/usr/sbin/sysctl", "-n", "hw.model"),
    "machdep.cpu.brand_string": ("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"),
    "hw.memsize": ("/usr/sbin/sysctl", "-n", "hw.memsize"),
    "hw.physicalcpu": ("/usr/sbin/sysctl", "-n", "hw.physicalcpu"),
    "hw.logicalcpu": ("/usr/sbin/sysctl", "-n", "hw.logicalcpu"),
    "macos_version": ("/usr/bin/sw_vers", "-productVersion"),
    "macos_build": ("/usr/bin/sw_vers", "-buildVersion"),
}


class ExecutionPreflightError(ValueError):
    """The reviewed checkout or declared runtime cannot be authenticated."""


@dataclass(frozen=True)
class ExecutionBinding:
    root: Path
    revision: str
    contract_sha256: str
    source_hashes: tuple[tuple[str, str], ...]
    runtime_json: str

    @property
    def protected_evaluation_ready(self) -> bool:
        return False


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _exact_hex(value: object, length: int, name: str) -> None:
    if type(value) is not str or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ExecutionPreflightError(
            f"{name} must be {length} lowercase hex characters"
        )


def _git(root: Path, *args: str) -> bytes:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        result = subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "-C",
                str(root),
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                *args,
            ],
            capture_output=True,
            check=False,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExecutionPreflightError("Git checkout authentication failed") from exc
    if result.returncode:
        raise ExecutionPreflightError("Git checkout authentication failed")
    return result.stdout


def _check_checkout(root: Path, revision: str) -> None:
    try:
        top = Path(
            _git(root, "rev-parse", "--show-toplevel").decode().strip()
        ).resolve()
    except UnicodeError as exc:
        raise ExecutionPreflightError("Git top-level path is invalid") from exc
    if top != root:
        raise ExecutionPreflightError("root must itself be the Git top-level directory")
    if _git(root, "rev-parse", "HEAD").decode().strip() != revision:
        raise ExecutionPreflightError(
            "checkout revision differs from reviewed revision"
        )
    if _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ):
        raise ExecutionPreflightError(
            "checkout must be clean, including untracked files"
        )


def _read_regular(root: Path, relative: str) -> bytes:
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in (".", "..") for part in path.parts)
    ):
        raise ExecutionPreflightError(
            "authenticated file must be a relative regular path"
        )
    directory = descriptor = None
    try:
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for part in path.parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = child
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ExecutionPreflightError(
                "authenticated file must be regular, not an alias"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read()
        after = os.fstat(descriptor)

        def identity(value):
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if identity(before) != identity(after) or len(content) != before.st_size:
            raise ExecutionPreflightError("authenticated file changed during reading")
        return content
    except OSError as exc:
        raise ExecutionPreflightError(
            "authenticated path is missing, symlinked or not regular"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory is not None:
            os.close(directory)


def _contract(content: bytes) -> dict:
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate key")
            result[name] = value
        return result

    def reject_constant(value):
        raise ValueError("nonfinite constant")

    try:
        value = json.loads(
            content, object_pairs_hook=unique, parse_constant=reject_constant
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ExecutionPreflightError("execution contract JSON is invalid") from exc
    required = {
        "schema_version",
        "contract_id",
        "status",
        "protected_evaluation_ready",
        "scope",
        "source_binding",
        "import_binding",
        "public_file_sha256",
        "runtime",
        "runtime_checks",
        "recheck",
        "remaining_bindings",
    }
    if (
        type(value) is not dict
        or set(value) != required
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 2
        or value["contract_id"] != "execution-binding-v2"
        or value["status"] != "specified_synthetic_preflight"
        or value["protected_evaluation_ready"] is not False
    ):
        raise ExecutionPreflightError(
            "execution contract schema or access status is invalid"
        )
    public = value["public_file_sha256"]
    if type(public) is not dict or set(public) != _PUBLIC_PATHS:
        raise ExecutionPreflightError(
            "contract public file allowlist must match exactly"
        )
    for digest in public.values():
        _exact_hex(digest, 64, "contract public SHA-256")
    runtime = value["runtime"]
    if (
        type(runtime) is not dict
        or set(runtime)
        != {
            "python",
            "package_versions",
            "platform",
            "hardware",
            "blas",
            "mps",
            "environment",
            "numerical_threads",
        }
        or type(runtime["package_versions"]) is not dict
        or set(runtime["package_versions"]) != set(_RUNTIME_PACKAGES)
        or any(
            type(version) is not str or not version
            for version in runtime["package_versions"].values()
        )
    ):
        raise ExecutionPreflightError("execution contract runtime schema is invalid")
    string_maps = {
        "platform": {"system", "machine"},
        "hardware": set(_HARDWARE_COMMANDS),
        "blas": {"name", "version"},
        "environment": set(_ENVIRONMENT_FLAGS),
    }
    for field, keys in string_maps.items():
        mapping = runtime[field]
        if (
            type(mapping) is not dict
            or set(mapping) != keys
            or any(type(item) is not str or not item for item in mapping.values())
        ):
            raise ExecutionPreflightError(
                "execution contract runtime metadata is invalid"
            )
    if (
        type(runtime["python"]) is not str
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", runtime["python"]) is None
        or type(runtime["mps"]) is not dict
        or set(runtime["mps"]) != {"built", "available"}
        or any(type(flag) is not bool for flag in runtime["mps"].values())
        or type(runtime["numerical_threads"]) is not int
        or runtime["numerical_threads"] != 1
    ):
        raise ExecutionPreflightError(
            "execution contract runtime capabilities are invalid"
        )
    return value


def _committed_files(root: Path, revision: str, public: dict) -> dict[str, str]:
    required = set(public) | {_CONTRACT_PATH, "pyproject.toml", "uv.lock"}
    selected = {}
    tree = _git(root, "ls-tree", "-r", "-z", "--full-tree", revision)
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, filename = entry.split(b"\t", 1)
        name = filename.decode("utf-8")
        source = name.endswith(".py") and name.startswith(
            (f"{_PACKAGE_ROOT}/", "scripts/")
        )
        if not source and name not in required:
            continue
        mode, kind, object_id = metadata.decode("ascii").split()
        if kind != "blob" or mode not in ("100644", "100755"):
            raise ExecutionPreflightError(
                "authenticated files must have regular Git modes"
            )
        content = _read_regular(root, name)
        committed = _git(root, "cat-file", "blob", object_id)
        if content != committed:
            raise ExecutionPreflightError(
                "authenticated bytes differ from committed blob"
            )
        digest = sha256(content).hexdigest()
        if name in public and digest != public[name]:
            raise ExecutionPreflightError(
                "public file SHA-256 differs from the contract"
            )
        selected[name] = digest
    required |= {
        f"{_PACKAGE_ROOT}/__init__.py",
        f"{_PACKAGE_ROOT}/execution_preflight.py",
    }
    if not required <= selected.keys():
        raise ExecutionPreflightError("required authenticated file is not committed")
    return selected


def _imported_project_modules() -> dict[str, tuple[object, object, tuple]]:
    result = {}
    for name, module in sys.modules.copy().items():
        if name == "automated_phishing_detection" or name.startswith(
            "automated_phishing_detection."
        ):
            result[name] = (
                getattr(module, "__file__", None),
                getattr(getattr(module, "__spec__", None), "origin", None),
                tuple(getattr(module, "__path__", ())),
            )
    return result


def _check_imports(root: Path, hashes: dict[str, str]) -> None:
    package = root / _PACKAGE_ROOT
    modules = _imported_project_modules()
    if not modules or "automated_phishing_detection.execution_preflight" not in modules:
        raise ExecutionPreflightError("execution preflight import identity is missing")
    for name, (filename, origin, search_paths) in modules.items():
        expected_path = "src/" + name.replace(".", "/")
        for raw_path in (filename, origin):
            if type(raw_path) is not str:
                raise ExecutionPreflightError(
                    "project import must identify a source file"
                )
            path = Path(raw_path)
            try:
                relative = path.relative_to(root).as_posix()
                path.relative_to(package)
            except ValueError as exc:
                raise ExecutionPreflightError(
                    "project import is outside the authenticated root"
                ) from exc
            if (
                relative not in hashes
                or path.resolve() != path
                or relative
                not in (expected_path + ".py", expected_path + "/__init__.py")
            ):
                raise ExecutionPreflightError(
                    "project import is not an authenticated source file"
                )
        if filename != origin:
            raise ExecutionPreflightError("project import file and spec origin differ")
        for raw_path in search_paths:
            if type(raw_path) is not str:
                raise ExecutionPreflightError("project import search path is invalid")
            path = Path(raw_path)
            if not path.is_relative_to(package) or path.resolve() != path:
                raise ExecutionPreflightError(
                    "project import search path leaves authenticated package"
                )


def _environment() -> dict[str, str]:
    for name in _ENVIRONMENT_FLAGS:
        if os.environ.get(name) not in (None, "0"):
            raise ExecutionPreflightError(f"{name} must be unset or exactly '0'")
    return {name: "disabled" for name in _ENVIRONMENT_FLAGS}


def _hardware_metadata() -> dict[str, str]:
    values = {}
    for key, command in _HARDWARE_COMMANDS.items():
        try:
            result = subprocess.run(
                command, capture_output=True, check=False, timeout=5
            )
            text = result.stdout.decode("utf-8").strip()
        except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
            raise ExecutionPreflightError("runtime hardware probe failed") from exc
        if result.returncode or not text or "\n" in text:
            raise ExecutionPreflightError(
                "runtime hardware probe returned invalid metadata"
            )
        values[key] = text
    return values


def _runtime_versions() -> dict[str, str]:
    try:
        versions = {
            name: importlib.metadata.version(name) for name in _RUNTIME_PACKAGES
        }
        for distribution, module_name in _VERSIONED_IMPORTS.items():
            if (
                importlib.import_module(module_name).__version__
                != versions[distribution]
            ):
                raise ExecutionPreflightError(
                    "runtime imported module/distribution versions disagree"
                )
    except (ImportError, AttributeError) as exc:
        raise ExecutionPreflightError(
            "runtime package metadata is unavailable"
        ) from exc
    return versions


def _verify_thread_limit(torch, threadpoolctl) -> int:
    before_torch = torch.get_num_threads()

    def state():
        return sorted(
            (
                str(pool.get("filepath")),
                str(pool.get("internal_api")),
                pool.get("num_threads"),
            )
            for pool in threadpoolctl.threadpool_info()
        )

    before = state()
    try:
        with threadpoolctl.threadpool_limits(limits=1):
            limited = state()
            if (
                not limited
                or any(row[2] != 1 for row in limited)
                or torch.get_num_threads() != 1
            ):
                raise ExecutionPreflightError(
                    "runtime did not honor one numerical thread"
                )
    finally:
        if torch.get_num_threads() != before_torch or state() != before:
            raise ExecutionPreflightError(
                "runtime thread probe did not restore entry state"
            )
    return 1


def _probe_runtime() -> dict:
    environment = _environment()
    versions = _runtime_versions()
    numpy = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    threadpoolctl = importlib.import_module("threadpoolctl")
    blas = numpy.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {})
    return {
        "python": platform.python_version(),
        "package_versions": versions,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "hardware": _hardware_metadata(),
        "blas": {
            "name": str(blas.get("name", "unknown")),
            "version": str(blas.get("version", "unknown")),
        },
        "mps": {
            "built": torch.backends.mps.is_built(),
            "available": torch.backends.mps.is_available(),
        },
        "environment": environment,
        "numerical_threads": _verify_thread_limit(torch, threadpoolctl),
    }


def _check_runtime(observed: dict, expected: dict) -> None:
    # Vendor build suffixes do not change the frozen OpenBLAS release.
    comparable = dict(observed)
    blas = observed.get("blas")
    if type(blas) is dict and type(blas.get("version")) is str:
        version = blas["version"]
        required = expected["blas"]["version"]
        if version == required or version.startswith(
            tuple(required + suffix for suffix in (".", "-", "+"))
        ):
            comparable["blas"] = {**blas, "version": required}
    if _canonical_json(comparable) != _canonical_json(expected):
        raise ExecutionPreflightError(
            "runtime identity differs from the execution contract"
        )


def bind_execution(
    root: Path, *, expected_revision: str, expected_contract_sha256: str
) -> ExecutionBinding:
    """Authenticate public execution identity only, never protected-data readiness."""
    _exact_hex(expected_revision, 40, "expected_revision")
    _exact_hex(expected_contract_sha256, 64, "expected_contract_sha256")
    _environment()
    try:
        root = Path(root).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise ExecutionPreflightError("checkout root is invalid") from exc
    _check_checkout(root, expected_revision)
    content = _read_regular(root, _CONTRACT_PATH)
    if sha256(content).hexdigest() != expected_contract_sha256:
        raise ExecutionPreflightError(
            "execution contract SHA-256 differs from reviewed pin"
        )
    contract = _contract(content)
    hashes = _committed_files(root, expected_revision, contract["public_file_sha256"])
    _check_imports(root, hashes)
    runtime = _probe_runtime()
    _check_runtime(runtime, contract["runtime"])
    _check_imports(root, hashes)
    _check_checkout(root, expected_revision)
    if any(
        sha256(_read_regular(root, name)).hexdigest() != digest
        for name, digest in hashes.items()
    ):
        raise ExecutionPreflightError("authenticated bytes changed during preflight")
    return ExecutionBinding(
        root,
        expected_revision,
        expected_contract_sha256,
        tuple(sorted(hashes.items())),
        _canonical_json(runtime),
    )


def recheck_binding(binding: ExecutionBinding) -> None:
    """Fail if code, imported paths, public evidence or runtime has drifted."""
    if type(binding) is not ExecutionBinding:
        raise ExecutionPreflightError("use a typed execution binding")
    current = bind_execution(
        binding.root,
        expected_revision=binding.revision,
        expected_contract_sha256=binding.contract_sha256,
    )
    if current != binding:
        raise ExecutionPreflightError(
            "execution identity differs from the prior binding"
        )
