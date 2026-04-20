"""Minimal module entrypoint for the STREAM package."""

from textwrap import dedent


def main() -> int:
    message = dedent(
        """
        STREAM package is importable.

        Suggested next steps:
          1. export PYTHONPATH=$PWD/src
          2. inspect configs/manifests under configs/
          3. run a stage script with --dry-run
        """
    ).strip()
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
