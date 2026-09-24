"""The probe-correction CLI exposes only its prospective narrow authority."""

import ast
import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_secondary_probe_correction.py"
PATH_NAMES = (
    "validation",
    "suffix_rules",
    "length_only",
    "logistic_l1",
    "transformer_bundle",
    "gmm",
    "drift_reference",
    "drift_audit",
    "original_attempt",
    "attempt",
    "public_summary",
)
FAILURE_MESSAGE = "Seed/probe correction stopped; consult its retained evidence.\n"
RAW_STDOUT_SENTINEL = b"private-raw-stdout-sentinel\n"
RAW_STDERR_SENTINEL = b"private-raw-stderr-sentinel\n"
CHILD_STDOUT_SENTINEL = b"private-child-stdout-sentinel\n"
CHILD_STDERR_SENTINEL = b"private-child-stderr-sentinel\n"
NATIVE_STDOUT_SENTINEL = b"private-native-buffered-stdout-sentinel"
NATIVE_STDERR_SENTINEL = b"private-native-stderr-sentinel"
NATIVE_ENTRY_STDOUT_SENTINEL = b"public-native-entry-stdout|"
PROJECT_IMPORT_MARKER_ENV = "PROBE_CORRECTION_PROJECT_IMPORT_MARKER"
DISPATCH_MARKER_ENV = "PROBE_CORRECTION_DISPATCH_MARKER"
EXPECTED_PROJECT_IMPORTS = (
    (
        0,
        "automated_phishing_detection.seed_probe_correction",
        (("bind_seed_probe_correction", None),),
    ),
    (
        0,
        "automated_phishing_detection.seed_probe_correction_runner",
        (("STAGES", "runner_stages"),),
    ),
    (
        0,
        "automated_phishing_detection.seed_probe_correction_runner",
        (
            ("ProbeCorrectionPaths", None),
            ("run_probe_correction", None),
            ("run_probe_correction_worker", None),
            ("verify_probe_correction", None),
        ),
    ),
)


@pytest.fixture
def cli():
    assert SCRIPT.exists(), "missing probe-correction CLI"
    spec = importlib.util.spec_from_file_location("probe_correction_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with module._discard_boundary_streams():
        module._load_boundaries()
    return module


def binding_args(*, omit=None):
    values = (
        ("repo-root", "/invented/repository"),
        ("expected-revision", "a" * 40),
        ("expected-profile-sha256", "b" * 64),
    )
    return [
        item for name, value in values if name != omit for item in ("--" + name, value)
    ]


def path_args(*, omit=None):
    return [
        item
        for name in PATH_NAMES
        if name != omit
        for item in ("--" + name.replace("_", "-"), f"/invented/{name}")
    ]


def expected_paths(cli):
    return cli.ProbeCorrectionPaths(
        **{name: Path(f"/invented/{name}") for name in PATH_NAMES}
    )


def forbid_dispatch(cli, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid CLI arguments reached an authority boundary")

    for name in (
        "bind_seed_probe_correction",
        "run_probe_correction",
        "run_probe_correction_worker",
        "verify_probe_correction",
    ):
        monkeypatch.setattr(cli, name, forbidden)


def write_raw_and_inherited_private_streams():
    os.write(1, RAW_STDOUT_SENTINEL)
    os.write(2, RAW_STDERR_SENTINEL)
    child_code = (
        "import os;"
        f"os.write(1, {CHILD_STDOUT_SENTINEL!r});"
        f"os.write(2, {CHILD_STDERR_SENTINEL!r})"
    )
    subprocess.run([sys.executable, "-c", child_code], check=True)


def project_imports(source):
    tree = ast.parse(source)
    return tuple(
        (
            node.level,
            node.module,
            tuple((alias.name, alias.asname) for alias in node.names),
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.level > 0 or node.module.startswith("automated_phishing_detection"))
    )


def write_fake_project_package(
    tmp_path,
    *,
    import_source="",
    stages=("retained_seed_audit", "probes"),
    path_names=PATH_NAMES,
):
    package = tmp_path / "automated_phishing_detection"
    package.mkdir()
    package.joinpath("__init__.py").write_text(
        f"""import os
from pathlib import Path

marker = os.environ.get({PROJECT_IMPORT_MARKER_ENV!r})
if marker is not None:
    Path(marker).write_text("imported", encoding="utf-8")

{import_source}
""",
        encoding="utf-8",
    )
    package.joinpath("seed_probe_correction.py").write_text(
        f"""import os
from pathlib import Path

def bind_seed_probe_correction(*args, **kwargs):
    marker = os.environ.get({DISPATCH_MARKER_ENV!r})
    if marker is not None:
        Path(marker).write_text("dispatched", encoding="utf-8")
""",
        encoding="utf-8",
    )
    fields_source = "\n".join(f"    {name}: object" for name in path_names)
    package.joinpath("seed_probe_correction_runner.py").write_text(
        f"""from dataclasses import dataclass

STAGES = {stages!r}

@dataclass(frozen=True)
class ProbeCorrectionPaths:
{fields_source}

def run_probe_correction(*args, **kwargs): pass
def run_probe_correction_worker(*args, **kwargs): pass
def verify_probe_correction(*args, **kwargs): pass
""",
        encoding="utf-8",
    )


def fake_project_environment(tmp_path, import_marker, dispatch_marker=None):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)
    environment[PROJECT_IMPORT_MARKER_ENV] = str(import_marker)
    if dispatch_marker is not None:
        environment[DISPATCH_MARKER_ENV] = str(dispatch_marker)
    return environment


def noisy_project_import_source(*, fail=False):
    child_code = (
        "import os;"
        f"os.write(1, {CHILD_STDOUT_SENTINEL!r});"
        f"os.write(2, {CHILD_STDERR_SENTINEL!r})"
    )
    failure = (
        '\nraise RuntimeError("/private/project-import-failure")\n' if fail else ""
    )
    return f"""import ctypes
import subprocess
import sys

print("private-python-import-stdout")
print("private-python-import-stderr", file=sys.stderr)
os.write(1, {RAW_STDOUT_SENTINEL!r})
os.write(2, {RAW_STDERR_SENTINEL!r})

runtime = ctypes.CDLL(None)
printf = runtime.printf
printf.argtypes = (ctypes.c_char_p,)
printf.restype = ctypes.c_int
fputs = runtime.fputs
fputs.argtypes = (ctypes.c_char_p, ctypes.c_void_p)
fputs.restype = ctypes.c_int

def native_stream_pointer(name):
    for symbol in (name, f"__{{name}}p"):
        try:
            return ctypes.c_void_p.in_dll(runtime, symbol)
        except ValueError:
            pass
    raise RuntimeError("native stream unavailable")

printf({NATIVE_STDOUT_SENTINEL!r})
fputs({NATIVE_STDERR_SENTINEL!r}, native_stream_pointer("stderr"))
subprocess.run([sys.executable, "-c", {child_code!r}], check=True)
{failure}"""


def test_imports_only_probe_correction_binding_and_runner_boundary():
    assert SCRIPT.exists(), "missing probe-correction CLI"
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    plain_project_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.startswith("automated_phishing_detection")
    }
    assert plain_project_imports == set()
    assert project_imports(source) == EXPECTED_PROJECT_IMPORTS


def test_import_boundary_rejects_extra_name_from_approved_runner_module():
    source = SCRIPT.read_text(encoding="utf-8")
    approved_import = (
        "    from automated_phishing_detection.seed_probe_correction_runner import ("
    )
    mutant = source.replace(
        approved_import,
        "    from automated_phishing_detection.seed_probe_correction_runner "
        "import _read_probe_inputs\n" + approved_import,
        1,
    )
    assert mutant != source
    assert project_imports(mutant) != EXPECTED_PROJECT_IMPORTS


def test_parser_error_redacts_unrecognized_option_and_value(cli, monkeypatch, capsys):
    forbid_dispatch(cli, monkeypatch)
    secret = "/invented/private-research-secret.csv"
    with pytest.raises(SystemExit) as error:
        cli.main(binding_args() + path_args() + ["--train", secret])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == FAILURE_MESSAGE
    assert secret not in output.out + output.err
    assert "unrecognized" not in output.err
    assert "usage:" not in output.err


@pytest.mark.parametrize(
    "missing",
    ["repo-root", "expected-revision", "expected-profile-sha256"],
)
def test_required_identity_pin_omission_stops_before_dispatch(
    cli, monkeypatch, missing
):
    observed = []

    def observe(*args, **kwargs):
        observed.append((args, kwargs))

    for name in (
        "bind_seed_probe_correction",
        "run_probe_correction",
        "run_probe_correction_worker",
        "verify_probe_correction",
    ):
        monkeypatch.setattr(cli, name, observe)
    with pytest.raises(SystemExit) as error:
        cli.main(binding_args(omit=missing) + ["--check"])
    assert error.value.code == 2
    assert observed == []


@pytest.mark.parametrize(
    "missing",
    ["repo-root", "expected-revision", "expected-profile-sha256"],
)
def test_required_identity_pin_error_is_sanitized(cli, monkeypatch, capsys, missing):
    forbid_dispatch(cli, monkeypatch)
    with pytest.raises(SystemExit) as error:
        cli.main(binding_args(omit=missing) + ["--check"])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == FAILURE_MESSAGE


def abbreviated_profile_args():
    return [
        "--repo-root",
        "/invented/repository",
        "--expected-revision",
        "a" * 40,
        "--expected-profile",
        "b" * 64,
        "--check",
    ]


def test_abbreviated_option_is_rejected_before_dispatch(cli, monkeypatch):
    observed = []
    monkeypatch.setattr(
        cli,
        "bind_seed_probe_correction",
        lambda *args, **kwargs: observed.append((args, kwargs)),
    )
    with pytest.raises(SystemExit) as error:
        cli.main(abbreviated_profile_args())
    assert error.value.code == 2
    assert observed == []


def test_abbreviated_option_error_is_sanitized(cli, monkeypatch, capsys):
    forbid_dispatch(cli, monkeypatch)
    with pytest.raises(SystemExit) as error:
        cli.main(abbreviated_profile_args())
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == FAILURE_MESSAGE


# SP-CORR-10: metadata-only check accepts no research path or unauthorized mode.
def test_check_binds_only_correction_metadata(cli, monkeypatch, capsys):
    observed = []

    def bind(root, **kwargs):
        observed.append((root, kwargs))

    monkeypatch.setattr(cli, "bind_seed_probe_correction", bind)
    monkeypatch.setattr(
        cli,
        "run_probe_correction",
        lambda *args, **kwargs: pytest.fail("metadata check ran parent"),
    )
    monkeypatch.setattr(
        cli,
        "run_probe_correction_worker",
        lambda *args, **kwargs: pytest.fail("metadata check ran worker"),
    )
    monkeypatch.setattr(
        cli,
        "verify_probe_correction",
        lambda *args, **kwargs: pytest.fail("metadata check ran verification"),
    )

    assert cli.main(binding_args() + ["--check"]) == 0
    assert observed == [
        (
            Path("/invented/repository"),
            {
                "expected_revision": "a" * 40,
                "expected_profile_sha256": "b" * 64,
            },
        )
    ]
    output = capsys.readouterr()
    assert output.out == (
        "Seed/probe correction metadata binding verified. "
        "Research inputs were not read.\n"
    )
    assert output.err == ""


@pytest.mark.parametrize("name", PATH_NAMES)
def test_check_rejects_every_research_or_output_path_before_binding(
    cli, monkeypatch, name
):
    forbid_dispatch(cli, monkeypatch)
    option = "--" + name.replace("_", "-")
    with pytest.raises(SystemExit) as error:
        cli.main(binding_args() + ["--check", option, "/must-not-open"])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "extra",
    [
        ["--producer-exit-code", "0"],
        ["--verify"],
        ["--worker", "probes"],
    ],
)
def test_check_rejects_noncheck_mode_arguments(cli, monkeypatch, extra):
    forbid_dispatch(cli, monkeypatch)
    with pytest.raises(SystemExit) as error:
        cli.main(binding_args() + ["--check"] + extra)
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["parent", "worker", "verify"])
@pytest.mark.parametrize("missing", PATH_NAMES)
def test_every_execution_or_verification_mode_requires_every_path(
    cli, monkeypatch, mode, missing
):
    forbid_dispatch(cli, monkeypatch)
    arguments = binding_args() + path_args(omit=missing)
    if mode == "worker":
        arguments.extend(("--worker", "retained_seed_audit"))
    elif mode == "verify":
        arguments.extend(("--verify", "--producer-exit-code", "0"))
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2


def test_parent_dispatches_exact_binding_and_paths(cli, monkeypatch, capsys):
    observed = []
    monkeypatch.setattr(
        cli,
        "run_probe_correction",
        lambda root, **kwargs: observed.append((root, kwargs)),
    )
    monkeypatch.setattr(
        cli,
        "bind_seed_probe_correction",
        lambda *args, **kwargs: pytest.fail("parent dispatched metadata check"),
    )
    monkeypatch.setattr(
        cli,
        "run_probe_correction_worker",
        lambda *args, **kwargs: pytest.fail("parent dispatched worker"),
    )
    monkeypatch.setattr(
        cli,
        "verify_probe_correction",
        lambda *args, **kwargs: pytest.fail("parent dispatched verifier"),
    )

    assert cli.main(binding_args() + path_args()) == 0
    assert observed == [
        (
            Path("/invented/repository"),
            {
                "expected_revision": "a" * 40,
                "expected_profile_sha256": "b" * 64,
                "paths": expected_paths(cli),
            },
        )
    ]
    output = capsys.readouterr()
    assert output.out == (
        "Seed/probe correction sequence completed; verify the observed exit "
        "and saved evidence.\n"
    )
    assert output.err == ""


def test_parent_hands_off_relative_lexical_paths_without_normalizing(cli, monkeypatch):
    observed = []
    root_text = "relative/root/../repository"
    path_values = {name: f"relative/{name}/../{name}-sentinel" for name in PATH_NAMES}
    arguments = [
        "--repo-root",
        root_text,
        "--expected-revision",
        "a" * 40,
        "--expected-profile-sha256",
        "b" * 64,
    ] + [
        item
        for name in PATH_NAMES
        for item in ("--" + name.replace("_", "-"), path_values[name])
    ]
    monkeypatch.setattr(
        cli,
        "run_probe_correction",
        lambda root, **kwargs: observed.append((root, kwargs)),
    )
    monkeypatch.setattr(
        cli,
        "bind_seed_probe_correction",
        lambda *args, **kwargs: pytest.fail("parent dispatched metadata check"),
    )
    monkeypatch.setattr(
        cli,
        "run_probe_correction_worker",
        lambda *args, **kwargs: pytest.fail("parent dispatched worker"),
    )
    monkeypatch.setattr(
        cli,
        "verify_probe_correction",
        lambda *args, **kwargs: pytest.fail("parent dispatched verifier"),
    )

    assert cli.main(arguments) == 0
    root, options = observed[0]
    assert root == Path(root_text)
    assert str(root) == root_text
    assert options["paths"] == cli.ProbeCorrectionPaths(
        **{name: Path(value) for name, value in path_values.items()}
    )
    assert {
        name: str(getattr(options["paths"], name)) for name in PATH_NAMES
    } == path_values


@pytest.mark.parametrize("stage", ["retained_seed_audit", "probes"])
def test_hidden_worker_dispatches_exact_stage(cli, monkeypatch, capsys, stage):
    observed = []
    monkeypatch.setattr(
        cli,
        "run_probe_correction_worker",
        lambda root, **kwargs: observed.append((root, kwargs)),
    )
    monkeypatch.setattr(
        cli,
        "bind_seed_probe_correction",
        lambda *args, **kwargs: pytest.fail("worker dispatched metadata check"),
    )
    monkeypatch.setattr(
        cli,
        "run_probe_correction",
        lambda *args, **kwargs: pytest.fail("worker dispatched parent"),
    )
    monkeypatch.setattr(
        cli,
        "verify_probe_correction",
        lambda *args, **kwargs: pytest.fail("worker dispatched verifier"),
    )

    arguments = binding_args() + path_args() + ["--worker", stage]
    assert cli.main(arguments) == 0
    assert observed == [
        (
            Path("/invented/repository"),
            {
                "expected_revision": "a" * 40,
                "expected_profile_sha256": "b" * 64,
                "paths": expected_paths(cli),
                "stage": stage,
            },
        )
    ]
    output = capsys.readouterr()
    assert output.out == "Seed/probe correction stage evidence published.\n"
    assert output.err == ""


@pytest.mark.parametrize("producer_exit_code", [0, 2, -9])
def test_verify_dispatches_exact_producer_exit_code(
    cli, monkeypatch, capsys, producer_exit_code
):
    observed = []
    monkeypatch.setattr(
        cli,
        "verify_probe_correction",
        lambda root, **kwargs: observed.append((root, kwargs)),
    )
    monkeypatch.setattr(
        cli,
        "bind_seed_probe_correction",
        lambda *args, **kwargs: pytest.fail("verifier dispatched metadata check"),
    )
    monkeypatch.setattr(
        cli,
        "run_probe_correction",
        lambda *args, **kwargs: pytest.fail("verifier dispatched parent"),
    )
    monkeypatch.setattr(
        cli,
        "run_probe_correction_worker",
        lambda *args, **kwargs: pytest.fail("verifier dispatched worker"),
    )

    arguments = (
        binding_args()
        + path_args()
        + ["--verify", "--producer-exit-code", str(producer_exit_code)]
    )
    assert cli.main(arguments) == 0
    assert observed == [
        (
            Path("/invented/repository"),
            {
                "expected_revision": "a" * 40,
                "expected_profile_sha256": "b" * 64,
                "paths": expected_paths(cli),
                "producer_exit_code": producer_exit_code,
            },
        )
    ]
    output = capsys.readouterr()
    assert output.out == "Saved seed/probe correction evidence verified.\n"
    assert output.err == ""


@pytest.mark.parametrize(
    "mode_arguments",
    [
        ["--verify"],
        ["--producer-exit-code", "0"],
        ["--worker", "probes", "--producer-exit-code", "0"],
    ],
)
def test_producer_exit_code_and_verify_are_valid_only_together(
    cli, monkeypatch, mode_arguments
):
    forbid_dispatch(cli, monkeypatch)
    with pytest.raises(SystemExit) as error:
        cli.main(binding_args() + path_args() + mode_arguments)
    assert error.value.code == 2


@pytest.mark.parametrize(
    "forbidden",
    [
        ["--train", "/invented/train"],
        ["--retry"],
        ["--resume"],
    ],
)
def test_train_retry_and_resume_are_rejected(cli, monkeypatch, forbidden):
    forbid_dispatch(cli, monkeypatch)
    with pytest.raises(SystemExit) as error:
        cli.main(binding_args() + path_args() + forbidden)
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("mode", "mode_arguments"),
    [
        ("check", ["--check"]),
        ("parent", []),
        ("worker", ["--worker", "probes"]),
        ("verify", ["--verify", "--producer-exit-code", "0"]),
    ],
)
def test_all_caught_failures_use_one_fixed_public_message(
    cli, monkeypatch, capfd, mode, mode_arguments
):
    private_failure = type("PrivateFailureType", (Exception,), {})
    stdout_sentinel = "private-stdout-sentinel"
    stderr_sentinel = "private-stderr-sentinel"
    cause_sentinel = "/private/chained-cause"

    def fail(*args, **kwargs):
        print(stdout_sentinel)
        print(stderr_sentinel, file=sys.stderr)
        write_raw_and_inherited_private_streams()
        try:
            raise ValueError(cause_sentinel)
        except ValueError as cause:
            raise private_failure(
                "/private/input: https://private.example/record"
            ) from cause

    for name in (
        "bind_seed_probe_correction",
        "run_probe_correction",
        "run_probe_correction_worker",
        "verify_probe_correction",
    ):
        monkeypatch.setattr(cli, name, fail)
    arguments = binding_args() + mode_arguments
    if mode != "check":
        arguments += path_args()

    assert cli.main(arguments) == 2
    output = capfd.readouterr()
    combined = output.out + output.err
    assert output.out == ""
    assert output.err == FAILURE_MESSAGE
    for secret in (
        stdout_sentinel,
        stderr_sentinel,
        cause_sentinel,
        "PrivateFailureType",
        "https://private.example/record",
        "Traceback",
        RAW_STDOUT_SENTINEL.decode("ascii").strip(),
        RAW_STDERR_SENTINEL.decode("ascii").strip(),
        CHILD_STDOUT_SENTINEL.decode("ascii").strip(),
        CHILD_STDERR_SENTINEL.decode("ascii").strip(),
    ):
        assert secret not in combined


@pytest.mark.parametrize("chained", [False, True])
def test_base_exceptions_fail_closed_without_private_detail(
    cli, monkeypatch, capfd, chained
):
    private_detail = "/private/base-exception-detail"
    private_cause = "/private/base-exception-cause"
    private_failure = type("PrivateBaseFailure", (BaseException,), {})

    def fail(*args, **kwargs):
        if not chained:
            raise SystemExit(private_detail)
        try:
            raise ValueError(private_cause)
        except ValueError as cause:
            raise private_failure(private_detail) from cause

    monkeypatch.setattr(cli, "bind_seed_probe_correction", fail)

    assert cli.main(binding_args() + ["--check"]) == 2
    output = capfd.readouterr()
    assert output.out == ""
    assert output.err == FAILURE_MESSAGE
    combined = output.out + output.err
    for secret in (private_detail, private_cause, "PrivateBaseFailure", "Traceback"):
        assert secret not in combined


@pytest.mark.parametrize(
    ("mode_arguments", "selected", "success"),
    [
        (
            ["--check"],
            "bind_seed_probe_correction",
            "Seed/probe correction metadata binding verified. "
            "Research inputs were not read.\n",
        ),
        (
            [],
            "run_probe_correction",
            "Seed/probe correction sequence completed; verify the observed exit "
            "and saved evidence.\n",
        ),
        (
            ["--worker", "probes"],
            "run_probe_correction_worker",
            "Seed/probe correction stage evidence published.\n",
        ),
        (
            ["--verify", "--producer-exit-code", "0"],
            "verify_probe_correction",
            "Saved seed/probe correction evidence verified.\n",
        ),
    ],
)
def test_successful_boundaries_also_discard_underlying_streams(
    cli, monkeypatch, capfd, mode_arguments, selected, success
):
    stdout_sentinel = "successful-private-stdout"
    stderr_sentinel = "successful-private-stderr"

    def noisy(*args, **kwargs):
        print(stdout_sentinel)
        print(stderr_sentinel, file=sys.stderr)
        write_raw_and_inherited_private_streams()

    for name in (
        "bind_seed_probe_correction",
        "run_probe_correction",
        "run_probe_correction_worker",
        "verify_probe_correction",
    ):
        monkeypatch.setattr(
            cli,
            name,
            noisy
            if name == selected
            else lambda *args, **kwargs: pytest.fail("wrong authority boundary"),
        )
    arguments = binding_args() + mode_arguments
    if "--check" not in mode_arguments:
        arguments += path_args()

    assert cli.main(arguments) == 0
    output = capfd.readouterr()
    combined = output.out + output.err
    assert output.out == success
    assert output.err == ""
    assert stdout_sentinel not in combined
    assert stderr_sentinel not in combined
    for secret in (
        RAW_STDOUT_SENTINEL,
        RAW_STDERR_SENTINEL,
        CHILD_STDOUT_SENTINEL,
        CHILD_STDERR_SENTINEL,
    ):
        assert secret.decode("ascii").strip() not in combined


@pytest.mark.parametrize("should_fail", [False, True])
@pytest.mark.parametrize("destructive", [False, True])
def test_cached_non_fd_stream_writes_are_rolled_back(
    cli, monkeypatch, should_fail, destructive
):
    stdout_before = "preexisting-public-stdout|"
    stderr_before = "preexisting-public-stderr|"
    public_stdout = io.StringIO(stdout_before)
    public_stderr = io.StringIO(stderr_before)
    stdout_position = len(stdout_before) if not should_fail else 3
    stderr_position = len(stderr_before) if should_fail else 5
    public_stdout.seek(stdout_position)
    public_stderr.seek(stderr_position)
    cached_stdout = public_stdout
    cached_stderr = public_stderr

    def boundary(*args, **kwargs):
        if destructive:
            cached_stdout.seek(0)
            cached_stdout.truncate()
            cached_stderr.seek(0)
            cached_stderr.truncate()
        cached_stdout.write("private-cached-stringio-stdout")
        cached_stderr.write("private-cached-stringio-stderr")
        if should_fail:
            try:
                raise ValueError("private-stringio-cause")
            except ValueError as cause:
                raise RuntimeError("private-stringio-failure") from cause

    monkeypatch.setattr(cli.sys, "stdout", public_stdout)
    monkeypatch.setattr(cli.sys, "stderr", public_stderr)
    monkeypatch.setattr(cli, "bind_seed_probe_correction", boundary)

    assert cli.main(binding_args() + ["--check"]) == (2 if should_fail else 0)
    success = (
        "Seed/probe correction metadata binding verified. "
        "Research inputs were not read.\n"
    )
    assert public_stdout.getvalue() == stdout_before + ("" if should_fail else success)
    assert public_stderr.getvalue() == stderr_before + (
        FAILURE_MESSAGE if should_fail else ""
    )
    assert public_stdout.tell() == (
        stdout_position if should_fail else len(stdout_before + success)
    )
    assert public_stderr.tell() == (
        len(stderr_before + FAILURE_MESSAGE) if should_fail else stderr_position
    )


@pytest.mark.parametrize("fails", [False, True])
def test_boundary_guard_closes_every_owned_descriptor(cli, monkeypatch, capsys, fails):
    opened = []
    closed = []
    real_dup = cli.os.dup
    real_open = cli.os.open
    real_close = cli.os.close

    def tracked_dup(descriptor):
        duplicate = real_dup(descriptor)
        opened.append(duplicate)
        return duplicate

    def tracked_open(path, flags):
        descriptor = real_open(path, flags)
        opened.append(descriptor)
        return descriptor

    def tracked_close(descriptor):
        closed.append(descriptor)
        real_close(descriptor)

    def boundary(*args, **kwargs):
        if fails:
            raise ValueError("private-boundary-failure")

    monkeypatch.setattr(cli.os, "dup", tracked_dup)
    monkeypatch.setattr(cli.os, "open", tracked_open)
    monkeypatch.setattr(cli.os, "close", tracked_close)
    monkeypatch.setattr(cli, "bind_seed_probe_correction", boundary)

    assert cli.main(binding_args() + ["--check"]) == (2 if fails else 0)
    capsys.readouterr()
    assert len(opened) == 3
    assert sorted(closed) == sorted(opened)


@pytest.mark.parametrize("inheritable_states", [(False, True), (True, False)])
@pytest.mark.parametrize("fails", [False, True])
def test_boundary_guard_preserves_standard_descriptor_inheritable_state(
    cli, monkeypatch, capsys, inheritable_states, fails
):
    original = {descriptor: os.get_inheritable(descriptor) for descriptor in (1, 2)}
    observed = []

    def boundary(*args, **kwargs):
        observed.append(tuple(os.get_inheritable(descriptor) for descriptor in (1, 2)))
        if fails:
            raise RuntimeError("private-inheritable-failure")

    monkeypatch.setattr(cli, "bind_seed_probe_correction", boundary)
    try:
        for descriptor, state in zip((1, 2), inheritable_states, strict=True):
            os.set_inheritable(descriptor, state)
        assert cli.main(binding_args() + ["--check"]) == (2 if fails else 0)
        assert observed == [(True, True)]
        assert (
            tuple(os.get_inheritable(descriptor) for descriptor in (1, 2))
            == inheritable_states
        )
    finally:
        for descriptor, state in original.items():
            os.set_inheritable(descriptor, state)
    capsys.readouterr()


def test_real_cli_help_hides_worker_and_unauthorized_modes():
    assert SCRIPT.exists(), "missing probe-correction CLI"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, check=False
    )
    assert result.returncode == 0
    help_text = result.stdout.decode("utf-8")
    assert "--check" in help_text
    assert "--verify" in help_text
    assert "--producer-exit-code" in help_text
    for hidden in ("--worker", "--train", "--retry", "--resume"):
        assert hidden not in help_text
    assert result.stderr == b""


@pytest.mark.parametrize(
    ("arguments", "returncode", "stderr"),
    [
        (["--help"], 0, b""),
        (
            ["--unrecognized-private-path", "/invented/private-import-path"],
            2,
            FAILURE_MESSAGE.encode("ascii"),
        ),
    ],
)
def test_parser_only_paths_do_not_load_or_leak_project_imports(
    tmp_path, arguments, returncode, stderr
):
    write_fake_project_package(
        tmp_path,
        import_source="os.write(2, b'private-import-stderr-sentinel\\n')",
    )
    import_marker = tmp_path / "project-imported"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        check=False,
        env=fake_project_environment(tmp_path, import_marker),
    )
    assert result.returncode == returncode
    assert result.stderr == stderr
    assert not import_marker.exists()
    assert b"private-import-stderr-sentinel" not in result.stdout + result.stderr


def test_valid_check_suppresses_all_project_import_streams(tmp_path):
    write_fake_project_package(
        tmp_path,
        import_source=noisy_project_import_source(),
    )
    import_marker = tmp_path / "project-imported"
    dispatch_marker = tmp_path / "binding-dispatched"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), *binding_args(), "--check"],
        capture_output=True,
        check=False,
        env=fake_project_environment(tmp_path, import_marker, dispatch_marker),
    )

    assert result.returncode == 0
    assert result.stdout == (
        b"Seed/probe correction metadata binding verified. "
        b"Research inputs were not read.\n"
    )
    assert result.stderr == b""
    assert import_marker.read_text(encoding="utf-8") == "imported"
    assert dispatch_marker.read_text(encoding="utf-8") == "dispatched"


def test_project_import_failure_discards_noise_and_uses_fixed_failure(tmp_path):
    write_fake_project_package(
        tmp_path,
        import_source=noisy_project_import_source(fail=True),
    )
    import_marker = tmp_path / "project-imported"
    dispatch_marker = tmp_path / "binding-dispatched"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), *binding_args(), "--check"],
        capture_output=True,
        check=False,
        env=fake_project_environment(tmp_path, import_marker, dispatch_marker),
    )

    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == FAILURE_MESSAGE.encode("ascii")
    assert import_marker.read_text(encoding="utf-8") == "imported"
    assert not dispatch_marker.exists()


@pytest.mark.parametrize(
    ("stages", "path_names"),
    [
        (("retained_seed_audit", "probes", "unauthorized"), PATH_NAMES),
        (("retained_seed_audit", "probes"), PATH_NAMES[:-1]),
    ],
    ids=("stage-mismatch", "path-schema-mismatch"),
)
def test_boundary_schema_mismatch_stops_before_dispatch(tmp_path, stages, path_names):
    write_fake_project_package(
        tmp_path,
        stages=stages,
        path_names=path_names,
    )
    import_marker = tmp_path / "project-imported"
    dispatch_marker = tmp_path / "binding-dispatched"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), *binding_args(), "--check"],
        capture_output=True,
        check=False,
        env=fake_project_environment(tmp_path, import_marker, dispatch_marker),
    )

    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == FAILURE_MESSAGE.encode("ascii")
    assert import_marker.read_text(encoding="utf-8") == "imported"
    assert not dispatch_marker.exists()


def test_real_process_propagates_boundary_failure_exit_code():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *binding_args(), "--check"],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == FAILURE_MESSAGE.encode("ascii")


def test_real_parser_failure_redacts_argument_value():
    secret = "/invented/private-research-secret.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *binding_args(),
            *path_args(),
            "--train",
            secret,
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == FAILURE_MESSAGE.encode("ascii")
    assert secret.encode("ascii") not in result.stdout + result.stderr


def test_cleanup_base_exception_restores_descriptors_before_fixed_failure():
    private_args = binding_args() + ["--check"]
    harness = f"""
import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location("cleanup_failure_cli", {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with module._discard_boundary_streams():
    module._load_boundaries()

class PrivateCleanupFailure(BaseException):
    pass

class FlushBomb:
    def __init__(self, stream, fail_on=None):
        self.stream = stream
        self.fail_on = fail_on
        self.flushes = 0

    def write(self, value):
        return self.stream.write(value)

    def flush(self):
        self.flushes += 1
        if self.flushes == self.fail_on:
            raise PrivateCleanupFailure("/private/cleanup-flush")
        return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)

original_stdout = sys.stdout
original_stderr = sys.stderr
before = tuple(
    (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
    for descriptor in (1, 2)
)
sys.stdout = FlushBomb(original_stdout, fail_on=2)
sys.stderr = FlushBomb(original_stderr)
module.bind_seed_probe_correction = lambda *args, **kwargs: None
try:
    exit_code = module.main({private_args!r})
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
after = tuple(
    (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
    for descriptor in (1, 2)
)
if after != before:
    exit_code = 97
raise SystemExit(exit_code)
"""
    result = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == FAILURE_MESSAGE.encode("ascii")
    combined = result.stdout + result.stderr
    for secret in (b"PrivateCleanupFailure", b"/private/cleanup-flush", b"Traceback"):
        assert secret not in combined


@pytest.mark.parametrize("failure_mode", ["native", "restore", "close"])
def test_fd_cleanup_base_exception_does_not_skip_remaining_cleanup(failure_mode):
    private_args = binding_args() + ["--check"]
    harness = f"""
import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location("fd_cleanup_cli", {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with module._discard_boundary_streams():
    module._load_boundaries()

class PrivateCleanupFailure(BaseException):
    pass

real_dup2 = os.dup2
real_close = os.close
dup2_calls = 0
close_calls = 0
native_flush_calls = 0

def guarded_native_flush():
    global native_flush_calls
    native_flush_calls += 1
    if {failure_mode!r} == "native" and native_flush_calls == 2:
        raise PrivateCleanupFailure("/private/native-flush-failure")

def guarded_dup2(source, target, *, inheritable=True):
    global dup2_calls
    dup2_calls += 1
    result = real_dup2(source, target, inheritable=inheritable)
    if {failure_mode!r} == "restore" and dup2_calls == 3:
        raise PrivateCleanupFailure("/private/restore-failure")
    return result

def guarded_close(descriptor):
    global close_calls
    close_calls += 1
    result = real_close(descriptor)
    if {failure_mode!r} == "close" and close_calls == 1:
        raise PrivateCleanupFailure("/private/close-failure")
    return result

os.set_inheritable(1, True)
os.set_inheritable(2, False)
module.os.dup2 = guarded_dup2
module.os.close = guarded_close
if {failure_mode!r} == "native":
    module._flush_native_streams = guarded_native_flush
module.bind_seed_probe_correction = lambda *args, **kwargs: None
exit_code = module.main({private_args!r})
if dup2_calls != 4 or close_calls != 3:
    exit_code = 96
if (os.get_inheritable(1), os.get_inheritable(2)) != (True, False):
    exit_code = 97
raise SystemExit(exit_code)
"""
    result = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == FAILURE_MESSAGE.encode("ascii")
    combined = result.stdout + result.stderr
    for secret in (
        b"PrivateCleanupFailure",
        b"/private/restore-failure",
        b"/private/close-failure",
        b"/private/native-flush-failure",
        b"Traceback",
    ):
        assert secret not in combined


def test_nonzero_final_native_flush_restores_descriptors_before_failure(
    cli, monkeypatch, capfd
):
    class FakeFlush:
        argtypes = None
        restype = None

        def __init__(self):
            self.calls = 0

        def __call__(self, stream):
            assert stream is None
            self.calls += 1
            return 0 if self.calls == 1 else -1

    class FakeRuntime:
        def __init__(self, flush):
            self.fflush = flush

    flush = FakeFlush()
    runtime = FakeRuntime(flush)
    before = tuple(
        (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
        for descriptor in (1, 2)
    )
    monkeypatch.setattr(cli.ctypes, "CDLL", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(cli, "bind_seed_probe_correction", lambda *args, **kwargs: None)

    assert cli.main(binding_args() + ["--check"]) == 2
    assert flush.calls == 2
    assert flush.argtypes == (cli.ctypes.c_void_p,)
    assert flush.restype is cli.ctypes.c_int
    assert (
        tuple(
            (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
            for descriptor in (1, 2)
        )
        == before
    )
    output = capfd.readouterr()
    assert output.out == ""
    assert output.err == FAILURE_MESSAGE


@pytest.mark.parametrize(
    ("should_fail", "returncode", "stdout", "stderr"),
    [
        (
            False,
            0,
            NATIVE_ENTRY_STDOUT_SENTINEL
            + b"Seed/probe correction metadata binding verified. "
            b"Research inputs were not read.\n",
            b"",
        ),
        (
            True,
            2,
            NATIVE_ENTRY_STDOUT_SENTINEL,
            FAILURE_MESSAGE.encode("ascii"),
        ),
    ],
)
def test_real_process_suppresses_cached_raw_and_inherited_boundary_streams(
    should_fail, returncode, stdout, stderr
):
    private_args = binding_args() + ["--check"]
    child_code = (
        "import os;"
        f"os.write(1, {CHILD_STDOUT_SENTINEL!r});"
        f"os.write(2, {CHILD_STDERR_SENTINEL!r})"
    )
    harness = f"""
import importlib.util
import ctypes
import os
import subprocess
import sys

spec = importlib.util.spec_from_file_location("raw_fd_cli", {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with module._discard_boundary_streams():
    module._load_boundaries()

runtime = ctypes.CDLL(None)
printf = runtime.printf
printf.argtypes = (ctypes.c_char_p,)
printf.restype = ctypes.c_int
fputs = runtime.fputs
fputs.argtypes = (ctypes.c_char_p, ctypes.c_void_p)
fputs.restype = ctypes.c_int

def native_stream_pointer(name):
    for symbol in (name, f"__{{name}}p"):
        try:
            return ctypes.c_void_p.in_dll(runtime, symbol)
        except ValueError:
            pass
    raise RuntimeError("native stream unavailable")

native_stderr = native_stream_pointer("stderr")
printf({NATIVE_ENTRY_STDOUT_SENTINEL!r})
cached_stdout = sys.stdout
cached_stderr = sys.stderr

class PrivateFailure(Exception):
    pass

def fail(*args, **kwargs):
    cached_stdout.write("cached-private-stdout-sentinel")
    cached_stderr.write("cached-private-stderr-sentinel")
    printf({NATIVE_STDOUT_SENTINEL!r})
    fputs({NATIVE_STDERR_SENTINEL!r}, native_stderr)
    os.write(1, {RAW_STDOUT_SENTINEL!r})
    os.write(2, {RAW_STDERR_SENTINEL!r})
    subprocess.run([sys.executable, "-c", {child_code!r}], check=True)
    if {should_fail!r}:
        try:
            raise ValueError("/private/chained-cause")
        except ValueError as cause:
            raise PrivateFailure("/private/boundary") from cause

module.bind_seed_probe_correction = fail
raise SystemExit(module.main({private_args!r}))
"""
    result = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True,
        check=False,
    )
    assert result.returncode == returncode
    assert result.stdout == stdout
    assert result.stderr == stderr
    combined = result.stdout + result.stderr
    for secret in (
        RAW_STDOUT_SENTINEL,
        RAW_STDERR_SENTINEL,
        CHILD_STDOUT_SENTINEL,
        CHILD_STDERR_SENTINEL,
        NATIVE_STDOUT_SENTINEL,
        NATIVE_STDERR_SENTINEL,
        b"/private/chained-cause",
        b"/private/boundary",
        b"cached-private-stdout-sentinel",
        b"cached-private-stderr-sentinel",
        b"PrivateFailure",
        b"Traceback",
    ):
        assert secret not in combined
