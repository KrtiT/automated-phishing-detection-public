"""Run the fixed continuation once beneath a live physical supervisor."""

import asyncio

from automated_phishing_detection._study_authorized_cli import failed, keywords, parser
from automated_phishing_detection.study_series_execution import (
    bind_series_public_execution,
)
from automated_phishing_detection.study_series_history import hold_series_history
from automated_phishing_detection.study_series_runner import _run_series_bound
from automated_phishing_detection.study_series_supervision import (
    consume_series_supervision,
)


def _parser():
    command = parser(__doc__)
    command.add_argument("--expected-profile-sha256", required=True)
    return command


async def _run(arguments):
    with consume_series_supervision(arguments) as supervisor:
        public = bind_series_public_execution(
            arguments.repo_root,
            **keywords(arguments),
            expected_profile_sha256=arguments.expected_profile_sha256,
        )
        return await _run_series_bound(
            public,
            hold_history=hold_series_history,
            lifecycle_check=supervisor.check,
        )


def main(argv=None):
    arguments = _parser().parse_args(argv)
    try:
        asyncio.run(_run(arguments))
    except BaseException as error:
        return failed("Study series stopped", error)
    print("Study series published; independent final verification required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
