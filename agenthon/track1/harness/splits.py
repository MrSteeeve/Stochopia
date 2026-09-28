"""Deterministic dev / holdout split of the public units.

Holdout units are for measuring generalisation only: do not read their
failures while tuning prompts, and never use them as few-shot examples.

    python -m harness.splits --units-dir <track1-coding-public>/units
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

SPLITS_FILE = Path(__file__).with_name("splits.json")
HOLDOUT_FRACTION = 0.3
_SALT = "agenthon-t1-split-v1"
_ALWAYS_DEV = {"t1-EXAMPLE-bs-greeks-pde"}


def card(unit_dir: Path) -> dict:
    try:
        return tomllib.loads((unit_dir / "card.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def group(category: str, task_id: str) -> str:
    cat, tid = category.lower(), task_id.lower()
    if any(k in cat for k in ("derivative", "pricing", "volatility", "stochastic")) or "option" in tid:
        return "derivatives"
    if any(k in cat for k in ("fixed-income", "interest-rate", "fx-", "cross-currency")):
        return "rates_fx"
    if any(k in cat for k in ("risk", "credit", "dependence", "extreme")):
        return "risk_credit"
    if any(k in cat for k in ("backtest", "factor", "strategy", "cross-sectional", "performance", "portfolio", "alpha", "crypto", "cross-asset")):
        return "factor_strategy"
    if "execution" in cat:
        return "execution"
    return "other"


def make_splits(units_dir: Path) -> dict[str, list[str]]:
    by_difficulty: dict[str, list[str]] = {}
    for u in sorted(p for p in units_dir.iterdir() if (p / "instruction.md").is_file()):
        diff = str(card(u).get("metadata", {}).get("difficulty") or "unknown")
        by_difficulty.setdefault(diff, []).append(u.name)
    dev, holdout = [], []
    for names in by_difficulty.values():
        ranked = sorted(names, key=lambda n: hashlib.sha256(f"{_SALT}:{n}".encode()).hexdigest())
        k = round(len(ranked) * HOLDOUT_FRACTION)
        for i, n in enumerate(ranked):
            (holdout if i < k and n not in _ALWAYS_DEV else dev).append(n)
    return {"dev": sorted(dev), "holdout": sorted(holdout)}


def load_splits() -> dict[str, list[str]]:
    if SPLITS_FILE.is_file():
        return json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
    return {"dev": [], "holdout": []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units-dir", type=Path, required=True)
    args = ap.parse_args()
    splits = make_splits(args.units_dir)
    SPLITS_FILE.write_text(json.dumps(splits, indent=1) + "\n", encoding="utf-8")
    print(f"dev={len(splits['dev'])} holdout={len(splits['holdout'])} -> {SPLITS_FILE}")


if __name__ == "__main__":
    main()
