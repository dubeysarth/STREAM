"""Stage 03: prepare nested gauges."""

from stream.curation import run_nested_gauge_preparation
from _script_common import build_parser, load_config, summarize


def main() -> int:
    parser = build_parser("03_prepare_nested_gauges")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    summarize(
        "03_prepare_nested_gauges",
        config,
        config_path,
        [
            "Prepare nested-gauge relationships from snapped outlets and shared maskmaps.",
            "Persist hierarchy metadata for semi-distributed evaluation.",
        ],
    )
    if args.dry_run:
        return 0
    run_nested_gauge_preparation(config, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
