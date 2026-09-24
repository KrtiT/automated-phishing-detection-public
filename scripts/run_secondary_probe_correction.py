"""Audit retained seed evidence, run one zero-fit probe, or verify saved evidence."""

import argparse
import ctypes
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import fields
from pathlib import Path

STAGES = ("retained_seed_audit", "probes")
PATH_ARGUMENTS = (
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
FAILURE_MESSAGE = "Seed/probe correction stopped; consult its retained evidence."
_BOUNDARIES_LOADED = False


class PublicArgumentParser(argparse.ArgumentParser):
    def error(self, _message):
        self.exit(2, FAILURE_MESSAGE + "\n")


def _load_boundaries():
    global _BOUNDARIES_LOADED
    global ProbeCorrectionPaths
    global bind_seed_probe_correction
    global run_probe_correction
    global run_probe_correction_worker
    global verify_probe_correction

    if _BOUNDARIES_LOADED:
        return
    from automated_phishing_detection.seed_probe_correction import (
        bind_seed_probe_correction,
    )
    from automated_phishing_detection.seed_probe_correction_runner import (
        STAGES as runner_stages,
    )
    from automated_phishing_detection.seed_probe_correction_runner import (
        ProbeCorrectionPaths,
        run_probe_correction,
        run_probe_correction_worker,
        verify_probe_correction,
    )

    if (
        tuple(runner_stages) != STAGES
        or tuple(field.name for field in fields(ProbeCorrectionPaths)) != PATH_ARGUMENTS
    ):
        raise RuntimeError("correction CLI boundary mismatch")
    _BOUNDARIES_LOADED = True


def _flush_native_streams():
    runtime = ctypes.CDLL(None, use_errno=True)
    flush = runtime.fflush
    flush.argtypes = (ctypes.c_void_p,)
    flush.restype = ctypes.c_int
    if flush(None) != 0:
        raise OSError(ctypes.get_errno(), "native stream flush failed")


def _snapshot_public_streams(streams):
    snapshots = []
    seen = set()
    for stream in streams:
        identity = id(stream)
        if identity in seen:
            continue
        seen.add(identity)
        methods = ("getvalue", "tell", "seek", "truncate", "write")
        if all(callable(getattr(stream, name, None)) for name in methods):
            snapshots.append((stream, stream.getvalue(), stream.tell()))
    return snapshots


def _restore_public_stream(snapshot):
    stream, content, position = snapshot
    stream.seek(0)
    stream.write(content)
    stream.truncate()
    stream.seek(position)


@contextmanager
def _discard_boundary_streams():
    public_streams = (sys.stdout, sys.stderr)
    for stream in public_streams:
        stream.flush()
    _flush_native_streams()
    stream_snapshots = _snapshot_public_streams(public_streams)
    saved = {}
    inheritable = {}
    sink = None
    restore_error = None
    try:
        for descriptor in (1, 2):
            inheritable[descriptor] = os.get_inheritable(descriptor)
            saved[descriptor] = os.dup(descriptor)
        sink = os.open(os.devnull, os.O_WRONLY)
        for descriptor in (1, 2):
            os.dup2(sink, descriptor, inheritable=True)
        with open(os.devnull, "w", encoding="utf-8") as discard:
            with redirect_stdout(discard), redirect_stderr(discard):
                yield
    finally:
        for snapshot in stream_snapshots:
            try:
                _restore_public_stream(snapshot)
            except BaseException as error:
                if restore_error is None:
                    restore_error = error
        for stream in public_streams:
            try:
                stream.flush()
            except BaseException as error:
                if restore_error is None:
                    restore_error = error
        try:
            _flush_native_streams()
        except BaseException as error:
            if restore_error is None:
                restore_error = error
        for descriptor in (1, 2):
            duplicate = saved.get(descriptor)
            if duplicate is None:
                continue
            try:
                os.dup2(
                    duplicate,
                    descriptor,
                    inheritable=inheritable[descriptor],
                )
            except BaseException as error:
                if restore_error is None:
                    restore_error = error
            try:
                os.close(duplicate)
            except BaseException as error:
                if restore_error is None:
                    restore_error = error
        if sink is not None:
            try:
                os.close(sink)
            except BaseException as error:
                if restore_error is None:
                    restore_error = error
        if restore_error is not None:
            raise restore_error


def parser():
    command = PublicArgumentParser(description=__doc__, allow_abbrev=False)
    command.add_argument("--repo-root", type=Path, required=True)
    command.add_argument("--expected-revision", required=True)
    command.add_argument("--expected-profile-sha256", required=True)
    mode = command.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help=(
            "Check committed correction metadata and runtime without reading "
            "research inputs."
        ),
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="Verify saved correction evidence without research-source reads.",
    )
    mode.add_argument("--worker", choices=STAGES, help=argparse.SUPPRESS)
    command.add_argument("--producer-exit-code", type=int)
    for name in PATH_ARGUMENTS:
        command.add_argument("--" + name.replace("_", "-"), type=Path)
    return command


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    supplied = {name: getattr(args, name) for name in PATH_ARGUMENTS}
    if args.check and (
        any(value is not None for value in supplied.values())
        or args.producer_exit_code is not None
    ):
        command.error("--check accepts no research-input, output or exit arguments")
    if not args.check and any(value is None for value in supplied.values()):
        command.error(
            "execution or verification requires every named input/output path"
        )
    if args.verify != (args.producer_exit_code is not None):
        command.error(
            "--verify requires --producer-exit-code, which is valid only for verification"
        )
    binding = {
        "expected_revision": args.expected_revision,
        "expected_profile_sha256": args.expected_profile_sha256,
    }
    try:
        with _discard_boundary_streams():
            _load_boundaries()
            if args.check:
                bind_seed_probe_correction(args.repo_root, **binding)
            else:
                paths = ProbeCorrectionPaths(**supplied)
                if args.verify:
                    verify_probe_correction(
                        args.repo_root,
                        **binding,
                        paths=paths,
                        producer_exit_code=args.producer_exit_code,
                    )
                elif args.worker is not None:
                    run_probe_correction_worker(
                        args.repo_root,
                        **binding,
                        paths=paths,
                        stage=args.worker,
                    )
                else:
                    run_probe_correction(args.repo_root, **binding, paths=paths)
    except BaseException:
        print(FAILURE_MESSAGE, file=sys.stderr)
        return 2
    if args.check:
        print(
            "Seed/probe correction metadata binding verified. "
            "Research inputs were not read."
        )
    elif args.verify:
        print("Saved seed/probe correction evidence verified.")
    elif args.worker is not None:
        print("Seed/probe correction stage evidence published.")
    else:
        print(
            "Seed/probe correction sequence completed; verify the observed exit "
            "and saved evidence."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
