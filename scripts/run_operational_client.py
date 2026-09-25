"""Run the fixed operational client child; protected access remains closed."""

from automated_phishing_detection import _operational_child_cli as cli
from automated_phishing_detection.operational_cell_client import run_operational_client


def parser():
    return cli.parser("client")


def main(argv=None):
    return cli.main("client", run_operational_client, argv)


if __name__ == "__main__":
    raise SystemExit(main())
