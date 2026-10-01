"""Run one fixed service or client beneath its live adopted series parent."""

import asyncio

from automated_phishing_detection._study_authorized_cli import failed, parser


def _parser():
    command = parser(__doc__)
    command.add_argument("--role", choices=("service", "client"), required=True)
    command.add_argument("--expected-profile-sha256", required=True)
    command.add_argument("--cell-ordinal", type=int, required=True)
    command.add_argument("--expected-binding-sha256", required=True)
    return command


def main(argv=None):
    arguments = _parser().parse_args(argv)
    try:
        from automated_phishing_detection._study_series_child_runtime import run_child

        asyncio.run(run_child(arguments))
    except BaseException as error:
        return failed("Study series child stopped", error)
    print("Study series child completed; parent verification required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
