"""Run exactly one live parent-owned role of an approved fresh study."""

import asyncio
from pathlib import Path

from automated_phishing_detection._study_authorized_cli import failed, parser


def _parser(role=None):
    command = parser(__doc__)
    command.add_argument(
        "--role", choices=("internal", "external", "service", "client"), required=True
    )
    if role == "external":
        command.add_argument("--internal-transport", type=Path, required=True)
        command.add_argument("--expected-handoff-sha256", required=True)
    if role in ("service", "client"):
        command.add_argument("--cell-ordinal", type=int, required=True)
        command.add_argument("--expected-binding-sha256", required=True)
    return command


def main(argv=None):
    preliminary, unused = _parser().parse_known_args(argv)
    arguments = _parser(preliminary.role).parse_args(argv)
    try:
        from automated_phishing_detection._study_child_runtime import run_child

        asyncio.run(run_child(arguments))
    except BaseException as error:
        return failed("Study child stopped", error)
    print("Study child completed; parent verification required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
