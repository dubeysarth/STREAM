"""Stage 09: run distributed or semidistributed baseline."""

from stream import DistributedBaselineRunner

from _script_common import build_parser, load_config, summarize


def main() -> int:
    parser = build_parser("09_run_distributed_baseline")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    summarize(
        "09_run_distributed_baseline",
        config,
        config_path,
        [
            "DistributedBaselineRunner.train()",
            "DistributedBaselineRunner.evaluate()",
        ],
    )
    if args.dry_run:
        return 0
    runner = DistributedBaselineRunner(config)
    runner.train()
    runner.evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
