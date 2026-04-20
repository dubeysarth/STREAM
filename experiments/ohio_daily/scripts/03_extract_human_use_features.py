from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_ohio.human_use import extract_huc05_human_use
from devp_ohio.paths import DevpOhioPaths
from devp_ohio.runtime import RunContext, build_parser


def main() -> None:
    parser = build_parser("Extract human-use features from LISFLOOD parameter maps.")
    args = parser.parse_args()
    paths = DevpOhioPaths.from_root(args.root)
    context = RunContext("extract_human_use_features", paths, run_name=args.run_name, dry_run=args.dry_run)
    outputs = extract_huc05_human_use(paths)
    context.write_manifest({key: str(value) for key, value in outputs.items()})
    context.write_resolved_config({key: str(value) for key, value in outputs.items()})
    for key, value in outputs.items():
        context.info("%s=%s", key, value)


if __name__ == "__main__":
    main()
