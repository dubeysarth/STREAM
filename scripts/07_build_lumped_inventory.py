"""Stage 07: build lumped inventory."""

from stream import LumpedAggregator

from _script_common import build_parser, load_config, summarize


def main() -> int:
    parser = build_parser("07_build_lumped_inventory")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    summarize(
        "07_build_lumped_inventory",
        config,
        config_path,
        ["LumpedAggregator.run()"],
    )
    if args.dry_run:
        return 0
    LumpedAggregator(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
