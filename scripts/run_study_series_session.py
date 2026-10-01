"""Supervise one adopted series continuation; no retry or execution overrides."""

from automated_phishing_detection._study_authorized_cli import failed, parser
from automated_phishing_detection.study_series_session import run_series_session


def _parser():
    command = parser(__doc__)
    command.add_argument("--expected-profile-sha256", required=True)
    return command


def main(argv=None):
    arguments = _parser().parse_args(argv)
    try:
        result = run_series_session(arguments)
    except BaseException as error:
        return failed("Study series physical session stopped", error)
    if result == 0:
        print("Study series session ended; independent final verification required.")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
