"""Rebuild the P1-M2 stock-day caches from their registered development files."""

import argparse
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path

from stochopia.common import atomic_write_json
from stochopia.data import reconstruct_szse_day, save_day_slice, validate_chronological_splits


def prepare_development_cache(root: Path) -> None:
    """Read the listed train/dev dates under raw/ and write cache/ and reconciliation.json."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    train_dates = manifest["train_dates"]
    dev_dates = manifest["dev_dates"]
    validate_chronological_splits(train_dates, dev_dates, manifest["test_dates"])
    reconciliation = []

    for date in [*train_dates, *dev_dates]:
        for symbol in manifest["symbols"]:
            day, report = reconstruct_szse_day(root / "raw" / str(date), symbol)
            if day.trading_date != date or any(stop <= start for start, stop in day.session_bounds):
                raise ValueError(f"Expected two continuous sessions for {symbol} on {date}")
            save_day_slice(root / "cache" / str(date) / f"{symbol}.npz", day)

            assert report is not None
            checks = {
                field: None if isinstance(value, float) and not isfinite(value) else value
                for field, value in asdict(report).items()
            }
            reconciliation.append({"date": date, "symbol": symbol, "n_events": len(day.seq_id), **checks})

    atomic_write_json(root / "reconciliation.json", reconciliation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    prepare_development_cache(parser.parse_args().root)
