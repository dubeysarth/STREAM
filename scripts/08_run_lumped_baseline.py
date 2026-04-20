"""Stage 08: run lumped baseline."""

from stream import LumpedBaselineRunner

from _script_common import build_parser, load_config, summarize


def main() -> int:
    parser = build_parser("08_run_lumped_baseline")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    summarize(
        "08_run_lumped_baseline",
        config,
        config_path,
        [
            "LumpedBaselineRunner.train()",
            "LumpedBaselineRunner.evaluate()",
        ],
    )
    if args.dry_run:
        return 0
    runner = LumpedBaselineRunner(config)
    runner.train()
    runner.evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
