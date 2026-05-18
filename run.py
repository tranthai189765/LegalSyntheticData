"""
Entry point – run the synthetic data generation pipeline.

Usage:
  python run.py --n 100
  python run.py --n 500 --output-dir output/btc_run
  python run.py --n 50 --task 3.3          # generate only task 3.3
  python run.py --n 200 --concurrency 5
  python run.py --list-tasks               # print all 22 tasks and exit

Per-task output structure:
  output/
    level_1/
      task_1_1.jsonl
      task_1_2.jsonl
      ...
    level_2/
      task_2_1.jsonl
      ...

Environment variables (can also be set in .env):
  FPT_API_KEY, FPT_MODEL, NEO4J_PASSWORD, BTC_PRIORITY_RATIO, ...
"""

import argparse
import asyncio
import sys
import os
import platform

from loguru import logger

# Fix Windows console encoding and asyncio cleanup warnings
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── logging setup ─────────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level} | {message}")
logger.add("output/pipeline.log", level="DEBUG", rotation="10 MB", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthetic Legal QA Data Generator (BTC-priority, Bloom-level tagged)"
    )
    parser.add_argument("--n", type=int, default=100,
                        help="Number of samples to generate (default: 100)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Root output directory (default: output). "
                             "Files go into output-dir/level_X/task_X_Y.jsonl")
    parser.add_argument("--task", type=str, default=None,
                        help="Generate only this task id, e.g. '3.3' (default: all tasks)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Number of parallel pipeline slots (overrides .env LLM_CONCURRENCY)")
    parser.add_argument("--list-tasks", action="store_true",
                        help="Print all available task IDs and exit")
    return parser.parse_args()


def list_tasks():
    from tasks.definitions import TASKS
    print(f"\nAvailable tasks ({len(TASKS)} total):\n")
    print(f"{'ID':<6} {'Level':<8} {'Name (VI)':<40} {'Weight'}")
    print("-" * 65)
    for t in TASKS:
        print(f"{t.id:<6} {t.level:<8} {t.name_vi:<40} {t.weight}")
    print()


async def main():
    args = parse_args()

    if args.list_tasks:
        list_tasks()
        return

    # Apply CLI overrides before importing pipeline (which reads config at import time)
    if args.concurrency:
        import config
        config.LLM_CONCURRENCY = args.concurrency

    from pipeline import Pipeline

    output_dir = args.output_dir

    # Validate task id if provided
    if args.task:
        from tasks.definitions import TASKS
        valid_ids = {t.id for t in TASKS}
        if args.task not in valid_ids:
            logger.error(f"Unknown task id '{args.task}'. Use --list-tasks to see valid IDs.")
            sys.exit(1)

    logger.info(f"Starting pipeline: n={args.n}  output_dir={output_dir}  task={args.task or 'all'}")

    import config as cfg
    logger.info(
        f"Config: model={cfg.FPT_MODEL}  btc_ratio={cfg.BTC_PRIORITY_RATIO}  "
        f"concurrency={cfg.LLM_CONCURRENCY}  max_retries={cfg.MAX_RETRIES}"
    )

    pipeline = Pipeline()
    stats = await pipeline.run(
        n_samples=args.n,
        output_dir=output_dir,
        task_filter=args.task,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(stats.summary())
    print(f"Output directory: {output_dir}/")
    print("  Per-task files: level_X/task_X_Y.jsonl")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
