"""Import-light fixed study identity options and symbolic exit handling."""

import argparse
import sys
from pathlib import Path


def parser(description):
    command = argparse.ArgumentParser(description=description, allow_abbrev=False)
    command.add_argument("--repo-root", type=Path, required=True)
    command.add_argument("--expected-revision", required=True)
    command.add_argument("--envelope", type=Path, required=True)
    command.add_argument("--expected-envelope-sha256", required=True)
    return command


def keywords(arguments):
    return {
        "expected_revision": arguments.expected_revision,
        "envelope_path": arguments.envelope,
        "expected_envelope_sha256": arguments.expected_envelope_sha256,
    }


def failed(prefix, error):
    from .internal_failure import failure_exit_code, failure_kind

    print(f"{prefix}: {failure_kind(error)}", file=sys.stderr)
    return failure_exit_code(error)
