"""Command-line entry point for research protocol utilities."""

import argparse
import csv
import json
import sys
from pathlib import Path

from . import baselines, phiusiil, protocol_preflight, transformer_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phishing-research", allow_abbrev=False)
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

    fit_baselines = commands.add_parser(
        "fit-baselines",
        description="Fit frozen RQ1 baselines and select validation thresholds.",
    )
    fit_baselines.add_argument("--train", required=True, type=Path)
    fit_baselines.add_argument("--validation", required=True, type=Path)
    fit_baselines.add_argument("--preparation-summary", required=True, type=Path)
    fit_baselines.add_argument("--contract", required=True, type=Path)
    fit_baselines.add_argument("--output-dir", required=True, type=Path)
    fit_baselines.add_argument("--summary", required=True, type=Path)

    fit_transformer = commands.add_parser(
        "fit-transformer-cascade",
        description="Fit the frozen RQ1 character transformer and fixed cascade.",
        allow_abbrev=False,
    )
    fit_transformer.add_argument("--train", required=True, type=Path)
    fit_transformer.add_argument("--validation", required=True, type=Path)
    fit_transformer.add_argument("--preparation-summary", required=True, type=Path)
    fit_transformer.add_argument("--baseline-contract", required=True, type=Path)
    fit_transformer.add_argument("--logistic-l1-artifact", required=True, type=Path)
    fit_transformer.add_argument("--transformer-contract", required=True, type=Path)
    fit_transformer.add_argument("--output-dir", required=True, type=Path)
    fit_transformer.add_argument("--summary", required=True, type=Path)
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
    if args.command == "fit-baselines":
        try:
            summary = baselines.fit_baselines(
                train_path=args.train,
                validation_path=args.validation,
                preparation_summary_path=args.preparation_summary,
                contract_path=args.contract,
                output_dir=args.output_dir,
                summary_path=args.summary,
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            baselines.BaselineError,
        ) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.command == "fit-transformer-cascade":
        try:
            summary = transformer_pipeline.fit_transformer_cascade(
                train_path=args.train,
                validation_path=args.validation,
                preparation_summary_path=args.preparation_summary,
                baseline_contract_path=args.baseline_contract,
                logistic_l1_artifact_path=args.logistic_l1_artifact,
                transformer_contract_path=args.transformer_contract,
                output_dir=args.output_dir,
                summary_path=args.summary,
            )
        except (OSError, transformer_pipeline.TransformerPipelineError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(summary, sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
