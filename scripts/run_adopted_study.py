"""Run one fresh study only under an explicitly approved complete profile."""

import asyncio
import json

from automated_phishing_detection._study_authorized_cli import failed, keywords, parser


async def _run(arguments):
    from automated_phishing_detection.adopted_study_runner import run_adopted_study

    return await run_adopted_study(arguments.repo_root, **keywords(arguments))


def main(argv=None):
    arguments = parser(__doc__).parse_args(argv)
    try:
        result = asyncio.run(_run(arguments))
        status = json.loads(result.snapshot.payload("public-summary.json"))["status"]
        message = {
            "whole_study_hold": "Study held: insufficient population capacity.",
            "study_evidence_published": "Study evidence published.",
        }[status]
    except BaseException as error:
        return failed("Study execution stopped", error)
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
