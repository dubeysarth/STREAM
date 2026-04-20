from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_us.human_use import extract_us_human_use
from devp_us.paths import DevpUSPaths
from devp_us.runtime import RunContext, build_parser


def main() -> None:
    parser = build_parser("Extract CAMELS-US human-use features from LISFLOOD and GDW.")
    args = parser.parse_args()
    paths = DevpUSPaths.from_root(args.root)
    context = RunContext("extract_us_human_use", paths, run_name=args.run_name, dry_run=args.dry_run)
    outputs = extract_us_human_use(paths)
    context.write_manifest(outputs)
    for key, value in outputs.items():
        context.info("%s=%s", key, value)


if __name__ == "__main__":
    main()

