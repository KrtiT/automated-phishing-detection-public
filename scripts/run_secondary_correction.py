"""Audit retained development evidence, then run one separately identified RF fit."""

import argparse
import sys
from pathlib import Path

from automated_phishing_detection.development_correction import bind_correction
from automated_phishing_detection.development_correction_runner import (
    STAGES,
    CorrectionPaths,
    run_correction,
    run_correction_worker,
)

_PATH_ARGUMENTS = (
    "train",
    "validation",
    "suffix_rules",
    "original_attempt",
    "attempt",
    "public_summary",
)


def parser():
    command = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    command.add_argument("--repo-root", type=Path, required=True)
    command.add_argument("--expected-revision", required=True)
    command.add_argument("--expected-profile-sha256", required=True)
    mode = command.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify committed code, runtime and public bindings without research inputs.",
    )
    mode.add_argument("--worker", choices=STAGES, help=argparse.SUPPRESS)
    for name in _PATH_ARGUMENTS:
        command.add_argument(f"--{name.replace('_', '-')}", type=Path)
    return command


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    supplied = {name: getattr(args, name) for name in _PATH_ARGUMENTS}
    if args.check and any(value is not None for value in supplied.values()):
        command.error("--check accepts no research-input or output paths")
    if not args.check and any(value is None for value in supplied.values()):
        command.error(
            "execution requires "
            + ", ".join(f"--{name.replace('_', '-')}" for name in _PATH_ARGUMENTS)
        )
    binding = {
        "expected_revision": args.expected_revision,
        "expected_profile_sha256": args.expected_profile_sha256,
    }
    try:
        if args.check:
            bind_correction(args.repo_root, **binding)
        else:
            paths = CorrectionPaths(**supplied)
            if args.worker is None:
                run_correction(args.repo_root, **binding, paths=paths)
            else:
                run_correction_worker(
                    args.repo_root, **binding, paths=paths, stage=args.worker
                )
    except Exception:
        # Detailed, allowlisted failures belong to durable receipts, not the terminal.
        print("Development correction stopped.", file=sys.stderr)
        return 2
    if args.check:
        print("Correction metadata binding verified. Research inputs were not read.")
    elif args.worker is None:
        print("Development correction verified.")
    else:
        print("Correction stage evidence published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
