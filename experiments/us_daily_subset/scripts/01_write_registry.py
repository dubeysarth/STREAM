from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_us.paths import DevpUSPaths
from devp_us.registry import write_us_registry
from devp_us.runtime import RunContext, build_parser


def main() -> None:
    parser = build_parser("Write the CAMELS-US registry for devp_US.")
    args = parser.parse_args()
    paths = DevpUSPaths.from_root(args.root)
    context = RunContext("write_us_registry", paths, run_name=args.run_name, dry_run=args.dry_run)
    csv_path = write_us_registry(paths)
    context.write_manifest({"registry_csv": str(csv_path)})
    context.info("registry_csv=%s", csv_path)


if __name__ == "__main__":
    main()

