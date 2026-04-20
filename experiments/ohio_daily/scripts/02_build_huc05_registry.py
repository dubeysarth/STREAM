from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_ohio.paths import DevpOhioPaths
from devp_ohio.registry import build_huc05_registry, write_huc05_registry
from devp_ohio.runtime import RunContext, build_parser


def main() -> None:
    parser = build_parser("Build and write the HUC05 basin registry.")
    args = parser.parse_args()
    paths = DevpOhioPaths.from_root(args.root)
    context = RunContext("build_huc05_registry", paths, run_name=args.run_name, dry_run=args.dry_run)
    csv_path = write_huc05_registry(paths)
    registry = build_huc05_registry(paths)
    context.write_manifest({"registry_csv": str(csv_path), "num_basins": int(len(registry))})
    context.write_resolved_config({"registry_csv": str(csv_path)})
    context.info("wrote registry for %s basins", len(registry))


if __name__ == "__main__":
    main()
