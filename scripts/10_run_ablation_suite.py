"""Stage 10: run ablation suite."""

from stream import AblationSuite

from _script_common import build_parser, load_config, summarize


def main() -> int:
    parser = build_parser("10_run_ablation_suite")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    summarize(
        "10_run_ablation_suite",
        config,
        config_path,
        ["AblationSuite.run()"],
    )
    if args.dry_run:
        return 0
    AblationSuite(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
