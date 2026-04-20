"""Stage 06: build monthly inventory."""

from stream import MonthlyAggregator

from _script_common import build_parser, load_config, summarize


def main() -> int:
    parser = build_parser("06_build_monthly_inventory")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    summarize(
        "06_build_monthly_inventory",
        config,
        config_path,
        ["MonthlyAggregator.run()"],
    )
    if args.dry_run:
        return 0
    MonthlyAggregator(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
