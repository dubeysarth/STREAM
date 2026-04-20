from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_ohio.paths import DevpOhioPaths
from devp_ohio.runtime import RunContext, build_parser


def _nvidia_memory_mb() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        first = output.splitlines()[0]
        return int(first)
    except Exception:
        return None


def main() -> None:
    parser = build_parser("Benchmark parallel daily lumped runs for concurrency selection.")
    parser.add_argument("--parallel-counts", nargs="+", type=int, default=[4, 8, 12, 16])
    parser.add_argument("--dynamic-group", default="era5_full_selected")
    parser.add_argument("--loss-name", default="hydro_skill", choices=["rmse", "hydro_skill", "hydro_balance"])
    parser.add_argument("--history-length", type=int, default=365)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--limit-train-batches", type=int, default=64)
    parser.add_argument("--limit-eval-batches", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    paths = DevpOhioPaths.from_root(args.root)
    context = RunContext("benchmark_parallel_lumped", paths, run_name=args.run_name, dry_run=args.dry_run)
    rows = []

    for parallel_count in args.parallel_counts:
        procs = []
        start = time.time()
        peak_mem = _nvidia_memory_mb() or 0
        context.info("starting benchmark for parallel_count=%s", parallel_count)
        for run_idx in range(parallel_count):
            run_name = f"bench_p{parallel_count:02d}_r{run_idx:02d}"
            cmd = [
                sys.executable,
                str(paths.scripts_dir / "08_train_lumped.py"),
                "--root",
                str(paths.root.parent),
                "--run-name",
                run_name,
                "--frequency",
                "daily",
                "--dynamic-group",
                args.dynamic_group,
                "--loss-name",
                args.loss_name,
                "--history-length",
                str(args.history_length),
                "--max-epochs",
                str(args.max_epochs),
                "--limit-train-batches",
                str(args.limit_train_batches),
                "--limit-eval-batches",
                str(args.limit_eval_batches),
                "--device",
                args.device,
            ]
            stdout_path = paths.logs_dir / f"{run_name}.stdout.log"
            handle = stdout_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=paths.root.parent, stdout=handle, stderr=subprocess.STDOUT, env=os.environ.copy())
            procs.append((proc, handle, run_name, stdout_path))
        while True:
            alive = False
            peak_mem = max(peak_mem, _nvidia_memory_mb() or 0)
            for proc, _, _, _ in procs:
                if proc.poll() is None:
                    alive = True
                    break
            if not alive:
                break
            time.sleep(5)
        elapsed = time.time() - start
        return_codes = []
        for proc, handle, run_name, stdout_path in procs:
            rc = proc.wait()
            handle.close()
            return_codes.append(rc)
            rows.append(
                {
                    "parallel_count": parallel_count,
                    "run_name": run_name,
                    "elapsed_seconds_group": round(elapsed, 2),
                    "return_code": rc,
                    "peak_gpu_mem_mb_group": peak_mem,
                    "stdout_log": str(stdout_path),
                }
            )
        context.info(
            "completed benchmark for parallel_count=%s elapsed=%.2fs peak_gpu_mem_mb=%s failures=%s",
            parallel_count,
            elapsed,
            peak_mem,
            sum(code != 0 for code in return_codes),
        )

    output_csv = paths.reports_dir / "parallel_benchmark_lumped.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    context.write_manifest({"benchmark_csv": str(output_csv), "parallel_counts": args.parallel_counts})
    context.info("wrote parallel benchmark rows=%s to %s", len(rows), output_csv)


if __name__ == "__main__":
    main()
