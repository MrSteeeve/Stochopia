"""CLI contract: ``solve --task-dir /input --out /app/output``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import solve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="t1agent")
    sub = parser.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("solve", help="solve one Track 1 unit")
    p.add_argument("--task-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        stats = solve(args.task_dir, args.out)
    except Exception as exc:  # noqa: BLE001 - the harness only reads the output directory
        print(f"t1agent: fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"t1agent: {stats.task_id}: {stats.status}, {stats.requests_used} requests, {len(stats.attempts)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
