"""Run internal evidence production only after a complete execution-profile freeze.

The current profile is closed. No override permits protected-data access.
"""

import argparse
import sys
from pathlib import Path

from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.bound_secondary import SecondaryArtifactPaths
from automated_phishing_detection.source_runner import (
    InternalRunPaths,
    run_internal_evaluation,
    run_internal_process,
)


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--repo-root", type=Path, required=True)
    command.add_argument("--expected-revision", required=True)
    command.add_argument("--expected-contract-sha256", required=True)
    command.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    for name in (
        "source-csv",
        "suffix-rules",
        "length-only",
        "logistic-l1",
        "transformer-bundle",
        "gmm",
        "formatting",
        "permutation-42",
        "permutation-43",
        "permutation-44",
        "permutation-45",
        "permutation-46",
        "random-forest",
        "seed-43-weights",
        "seed-44-weights",
        "seed-45-weights",
        "seed-46-weights",
        "attempt",
        "public-summary",
    ):
        command.add_argument(f"--{name}", type=Path, required=True)
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    paths = InternalRunPaths(
        args.source_csv,
        args.suffix_rules,
        ArtifactPaths(
            args.length_only, args.logistic_l1, args.transformer_bundle, args.gmm
        ),
        SecondaryArtifactPaths(
            args.formatting,
            args.permutation_42,
            args.permutation_43,
            args.permutation_44,
            args.permutation_45,
            args.permutation_46,
            args.random_forest,
            args.seed_43_weights,
            args.seed_44_weights,
            args.seed_45_weights,
            args.seed_46_weights,
        ),
        args.attempt,
        args.public_summary,
    )
    try:
        run = run_internal_evaluation if args.worker else run_internal_process
        run(
            args.repo_root,
            expected_revision=args.expected_revision,
            expected_contract_sha256=args.expected_contract_sha256,
            paths=paths,
        )
    except Exception as exc:
        print(f"Internal execution stopped: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(
        "Internal evidence published."
        if args.worker
        else "Internal completion verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
