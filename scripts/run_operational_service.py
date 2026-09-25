"""Run the fixed operational service child; protected access remains closed."""

from automated_phishing_detection import _operational_child_cli as cli
from automated_phishing_detection.operational_cell_service import (
    run_operational_service,
)


def parser():
    return cli.parser("service")


def main(argv=None):
    return cli.main("service", run_operational_service, argv)


if __name__ == "__main__":
    raise SystemExit(main())
