"""Verify public checkout/runtime identity without opening research inputs."""

import argparse
import json
import sys
from pathlib import Path

from automated_phishing_detection.execution_preflight import (
    ExecutionPreflightError,
    bind_execution,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--repo-root", type=Path, required=True)
    command.add_argument("--expected-revision", required=True)
    command.add_argument("--expected-contract-sha256", required=True)
    return command


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        binding = bind_execution(
            args.repo_root,
            expected_revision=args.expected_revision,
            expected_contract_sha256=args.expected_contract_sha256,
        )
    except ExecutionPreflightError as exc:
        print(f"Execution preflight failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": "verified_preflight",
                "protected_evaluation_ready": False,
                "research_measurements_run": False,
                "revision": binding.revision,
                "contract_sha256": binding.contract_sha256,
                "source_file_count": len(binding.source_hashes),
                "runtime": json.loads(binding.runtime_json),
            },
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
