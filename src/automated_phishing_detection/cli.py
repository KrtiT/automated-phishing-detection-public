"""Command-line entry point for research protocol utilities."""

import argparse
import csv
import json
import sys
from pathlib import Path

from . import phiusiil, protocol_preflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phishing-research")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate-manifest", description="Validate a synthetic split manifest."
    )
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--suffix-rules", required=True, type=Path)

    prepare = commands.add_parser(
        "prepare-phiusiil",
        description="Prepare deterministic domain-grouped PhiUSIIL splits.",
    )
    prepare.add_argument("--csv", required=True, type=Path)
    prepare.add_argument("--suffix-rules", required=True, type=Path)
    prepare.add_argument("--source-spec", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--summary", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-manifest":
        return protocol_preflight.main(
            [
                "--manifest",
                str(args.manifest),
                "--suffix-rules",
                str(args.suffix_rules),
            ]
        )
    if args.command == "prepare-phiusiil":
        try:
            summary = phiusiil.prepare_phiusiil(
                csv_path=args.csv,
                suffix_rules_path=args.suffix_rules,
                source_spec_path=args.source_spec,
                output_dir=args.output_dir,
                summary_path=args.summary,
            )
        except (
            OSError,
            UnicodeError,
            csv.Error,
            json.JSONDecodeError,
            phiusiil.PreparationError,
            protocol_preflight.PreflightError,
        ) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(summary, sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
