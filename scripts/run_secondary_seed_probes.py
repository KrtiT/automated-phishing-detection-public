"""Run the fixed development seed/probe sequence, or verify its saved evidence."""

import argparse
import sys
from dataclasses import fields
from pathlib import Path

from automated_phishing_detection.seed_probe_execution import bind_seed_probe_execution
from automated_phishing_detection.seed_probe_runner import (
    STAGES,
    SeedProbePaths,
    run_seed_probe_worker,
    run_seed_probes,
    verify_seed_probe_run,
)

PATH_ARGUMENTS = tuple(field.name for field in fields(SeedProbePaths))


def parser():
    command = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    command.add_argument("--repo-root", type=Path, required=True)
    command.add_argument("--expected-revision", required=True)
    command.add_argument("--expected-profile-sha256", required=True)
    mode = command.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Check committed code, public bindings and runtime without reading research inputs.",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="Verify saved outputs without fitting or rereading research sources.",
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
    if args.check and any(value is not None for value in supplied.values()):
        command.error("--check accepts no research-input or output paths")
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
        if args.check:
            bind_seed_probe_execution(args.repo_root, **binding)
        else:
            paths = SeedProbePaths(**supplied)
            if args.verify:
                verify_seed_probe_run(
                    args.repo_root,
                    **binding,
                    paths=paths,
                    producer_exit_code=args.producer_exit_code,
                )
            elif args.worker is not None:
                run_seed_probe_worker(
                    args.repo_root, **binding, paths=paths, stage=args.worker
                )
            else:
                run_seed_probes(args.repo_root, **binding, paths=paths)
    except Exception:
        print(
            "Seed/probe execution or verification stopped; consult its retained evidence.",
            file=sys.stderr,
        )
        return 2
    if args.check:
        print("Seed/probe metadata binding verified. Research inputs were not read.")
    elif args.verify:
        print("Saved seed/probe evidence verified.")
    elif args.worker is not None:
        print("Seed/probe stage evidence published.")
    else:
        print(
            "Seed/probe sequence completed; verify the observed exit and saved evidence."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
