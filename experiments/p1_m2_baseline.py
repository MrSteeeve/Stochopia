"""Train the P1 seed models and measure their development residuals.

Run with ``python -m experiments.p1_m2_baseline --root <market-data-root>``.
The split manifest is the sole source of train and dev paths. All weights,
paired paths and derived JSON are written to that external root.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
import torch

from experiments.p1_m2_metrics import TapeStats, aggregate_tapes, summarize_impact, tape_statistics
from stochopia.common import atomic_write_json
from stochopia.data import SZSEDaySlice, load_day_slice, validate_chronological_splits
from stochopia.harness.lob_rollout import ResidualProfile, RolloutArm, RolloutTrace, rollout_paired_window
from stochopia.neural.lob_interface import (
    ACTION_BINS,
    DT_BINS,
    LOT_BINS,
    REL_TICK_BINS,
    encode_events,
    make_batch,
    verify_world_model_contract,
    world_model_loss,
)
from stochopia.neural.primitives import SeedGRULOBModel


SYMBOLS = ("000001.SZ", "300001.SZ")
TRAIN_SEEDS = (103, 137, 211)
FIT_STEPS = 1600
TRAIN_LENGTH = 96
BATCH_SIZE = 16
CONTEXT = 128
CHILDREN = 8
PARTICIPATION = 0.10
WINDOW_MIDPOINTS = (37_800_000, 50_310_000)
SESSION_DURATION_MS = 7_200_000 + 7_020_000
HEADS = ("dt", "action", "rel_ticks", "volume")


@dataclass(frozen=True)
class EncodedDay:
    day: SZSEDaySlice
    ticks: torch.Tensor
    shares: torch.Tensor
    tokens: torch.Tensor


@dataclass(frozen=True)
class Window:
    data: EncodedDay
    session: int
    start: int
    end: int


def _load_days(root: Path, symbol: str, dates: list[int]) -> list[EncodedDay]:
    loaded = []
    for date in dates:
        day = load_day_slice(root / "cache" / str(date) / f"{symbol}.npz")
        if day.trading_date != date or day.wind_code != symbol:
            raise ValueError(f"Cache metadata does not match {symbol} on {date}")
        ticks = torch.tensor(day.pre_book_ticks)
        shares = torch.tensor(day.pre_book_shares)
        tokens = encode_events(
            torch.tensor(day.dt_ms), torch.tensor(day.action_type), torch.tensor(day.rel_ticks),
            torch.tensor(day.volume_shares), ticks,
        )
        loaded.append(EncodedDay(day, ticks, shares, tokens))
    return loaded


def _fit(days: list[EncodedDay], seed: int) -> tuple[SeedGRULOBModel, float, float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SeedGRULOBModel()
    optim = torch.optim.Adam(model.parameters(), lr=0.002)
    eligible = [(day, start, end) for day in days for start, end in day.day.session_bounds
                if end - start > TRAIN_LENGTH]
    started = perf_counter()
    for _ in range(FIT_STEPS):
        batch_ticks = []
        batch_shares = []
        batch_tokens = []
        for index in rng.integers(len(eligible), size=BATCH_SIZE):
            day, start, end = eligible[index]
            lo = int(rng.integers(start, end - TRAIN_LENGTH + 1))
            subset = slice(lo, lo + TRAIN_LENGTH)
            batch_ticks.append(day.ticks[subset])
            batch_shares.append(day.shares[subset])
            batch_tokens.append(day.tokens[subset])
        targets = torch.stack(batch_tokens)
        batch = make_batch(torch.stack(batch_ticks), torch.stack(batch_shares), targets)
        loss = world_model_loss(model.forward_sequence(batch), targets).total
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
    return model.eval(), perf_counter() - started, float(loss.detach())


def _marginal(days: list[EncodedDay]) -> list[np.ndarray]:
    observed = torch.cat([day.tokens for day in days]).numpy()
    marginals = []
    for axis, size in enumerate((DT_BINS, ACTION_BINS, REL_TICK_BINS, LOT_BINS)):
        counts = np.bincount(observed[:, axis], minlength=size) + 0.5
        marginals.append(-np.log(counts / counts.sum()))
    return marginals


def _calibrate(root: Path, train: list[EncodedDay], symbol: str) -> dict[str, int | float]:
    size_arrays = []
    daily_shares = []
    daily_price_changes = []
    for data in train:
        day = data.day
        aggressive = (day.action_type == 2) | (day.action_type == 3)
        size_arrays.append(day.volume_shares[aggressive])
        quotes = pq.read_table(root / "raw" / str(day.trading_date) / "行情.parquet", columns=["cum_volume"],
                               filters=[("wind_code", "==", symbol)])
        daily_shares.append(max(v for v in quotes["cum_volume"].to_pylist() if v is not None))
        daily_price_changes.append(day.mid_ticks_x2[-1] / 2 - day.prev_close_ticks)
    sizes = np.concatenate(size_arrays)
    frequency = len(sizes) / (len(train) * SESSION_DURATION_MS)
    spacing = 10 * max(1, round(1.0 / (10 * frequency * PARTICIPATION)))
    return {"child_shares": max(1, int(round(float(np.median(sizes))))), "spacing_ms": spacing,
            "train_aggressive_per_second": 1000 * frequency, "daily_shares": float(np.mean(daily_shares)),
            "sigma_ticks": float(np.std(daily_price_changes, ddof=1))}


def _windows(days: list[EncodedDay], horizon: int) -> list[Window]:
    windows = []
    for data in days:
        times = data.day.time_ms
        for session, (start, end) in enumerate(data.day.session_bounds):
            position = max(start + CONTEXT, int(np.searchsorted(times, WINDOW_MIDPOINTS[session])))
            cutoff = int(np.searchsorted(times, times[position] + horizon, side="right")) if position < end else end
            if not position < cutoff < end:
                raise ValueError(f"Incomplete dev window for {data.day.wind_code} {data.day.trading_date} session {session}")
            windows.append(Window(data, session, position, cutoff))
    return windows


def _replay_stats(window: Window, grid: np.ndarray) -> TapeStats:
    day = window.data.day
    times = day.time_ms[window.start:window.end] - day.time_ms[window.start]
    marks = np.searchsorted(times, grid, side="right") + window.start
    marks[0] = window.start  # both the model and replay start at the pre-event book
    return tape_statistics(np.column_stack((times, day.action_type[window.start:window.end])),
                           day.mid_ticks_x2[marks] / 2, day.spread_ticks[marks], day.pre_book_shares[marks, :, 0])


@torch.inference_mode()
def _validation(model: SeedGRULOBModel, window: Window, marginal: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    first = window.start - CONTEXT
    end = window.end
    data = window.data
    tokens = data.tokens[None, first:end]
    batch = make_batch(data.ticks[None, first:end], data.shares[None, first:end], tokens)
    loss = world_model_loss(model.forward_sequence(batch), tokens, reduction="none")
    observed = tokens[0, CONTEXT:].numpy()
    predicted = np.array([float(getattr(loss, name)[:, CONTEXT:].mean()) for name in HEADS])
    if not np.isfinite(predicted).all():
        raise ValueError(f"Nonfinite dev likelihood for {data.day.wind_code} {data.day.trading_date}")
    empirical = np.array([values[observed[:, axis]].mean() for axis, values in enumerate(marginal)])
    return predicted, empirical, len(observed)


def _json_value(value: object) -> object:
    """Render unavailable measurements as JSON null."""
    if isinstance(value, (float, np.floating)):
        return float(value) if isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _save_traces(path: Path, traces: dict[RolloutArm, RolloutTrace]) -> None:
    fields = ("grid_ms", "mid_ticks", "spread_ticks", "touch_shares", "events", "after_child_mid_ticks", "child_fill_shares")
    arrays = {f"{arm.value}_{field}": getattr(trace, field) for arm, trace in traces.items() for field in fields}
    np.savez_compressed(path, **arrays)


def _likelihood(rows: list[tuple[np.ndarray, np.ndarray, int]]) -> list[dict[str, float]]:
    weights = np.array([row[2] for row in rows])
    means = [np.average(np.array([row[i] for row in rows]), weights=weights, axis=0) for i in range(2)]
    return [{**dict(zip(HEADS, values.tolist())), "total": float(values.sum())} for values in means]


def _run_symbol(root: Path, symbol: str, train_dates: list[int], dev_dates: list[int]) -> None:
    torch.set_num_threads(1)
    train = _load_days(root, symbol, train_dates)
    dev = _load_days(root, symbol, dev_dates)
    marginal = _marginal(train)
    cfg = _calibrate(root, train, symbol)
    spacing = int(cfg["spacing_ms"])
    child_times = np.arange(1, CHILDREN + 1) * spacing
    grid = np.arange(2 * CHILDREN * 4 + 1) * spacing // 4
    windows = _windows(dev, int(grid[-1]))
    replay = [_replay_stats(window, grid) for window in windows]
    directory = root / "baseline" / symbol
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / "protocol.json", {**cfg, "fit_steps": FIT_STEPS, "batch_size": BATCH_SIZE,
                      "train_length": TRAIN_LENGTH, "context": CONTEXT, "train_seeds": list(TRAIN_SEEDS),
                      "grid_ms": grid.tolist(), "child_times_ms": child_times.tolist(),
                      "windows": [{"date": w.data.day.trading_date, "session": w.session, "start": w.start,
                                   "time_ms": int(w.data.day.time_ms[w.start]), "end": w.end} for w in windows]})
    nll = []
    generated = []
    fit = []
    paths = []
    start_all = perf_counter()
    for seed in TRAIN_SEEDS:
        model, elapsed, loss = _fit(train, seed)
        verify_world_model_contract(model)
        torch.save(model.state_dict(), directory / f"seed-{seed}.pt")
        fit.append({"seed": seed, "seconds": elapsed, "last_batch_nll": loss,
                    "params": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        print(f"{symbol} seed {seed} trained in {elapsed:.2f}s", flush=True)
        seed_paths = []
        seed_nll = []
        for number, window in enumerate(windows):
            seed_nll.append(_validation(model, window, marginal))
            traces = rollout_paired_window(model, window.data.day, window.start, child_times_ms=child_times,
                                           child_shares=int(cfg["child_shares"]), grid_ms=grid, seed=12300 + number,
                                           context_len=CONTEXT, max_events=4096)
            _save_traces(directory / f"seed-{seed}-{window.data.day.trading_date}-{window.session}.npz", traces)
            seed_paths.append(traces)
            trace = traces[RolloutArm.NO_INJECTION]
            generated.append(tape_statistics(trace.events, trace.mid_ticks, trace.spread_ticks, trace.touch_shares))
            if (number + 1) % 6 == 0:
                print(f"{symbol} seed {seed}: {number + 1}/{len(windows)} paired windows", flush=True)
        paths.append(seed_paths)
        nll.extend(seed_nll)
        fit[-1]["dev_nll"] = _likelihood(seed_nll)[0]

    profile = _summarize_paths(paths, generated, replay, cfg, grid, child_times, _likelihood(nll))
    result = asdict(profile)
    result.update(dev_targets=sum(entry[2] for entry in nll), fit=fit, seconds=perf_counter() - start_all)
    atomic_write_json(directory / "residuals.json", _json_value(result))


def _summarize_paths(paths: list[list[dict[RolloutArm, RolloutTrace]]], generated: list[TapeStats],
                     replay: list[TapeStats], cfg: dict[str, int | float], grid: np.ndarray,
                     children: np.ndarray, scores: list[dict[str, float]]) -> ResidualProfile:
    visible = (RolloutArm.BUY_VISIBLE, RolloutArm.SELL_VISIBLE)
    blind = (RolloutArm.BUY_BLIND, RolloutArm.SELL_BLIND)

    def paired(arms: tuple[RolloutArm, RolloutArm], field: str, *, relative: bool = False) -> np.ndarray:
        data = np.array([[[getattr(pair[arm], field) - (pair[arm].mid_ticks[0] if relative else 0)
                           for arm in arms] for pair in seed] for seed in paths])
        # [seed, window, side, time] -> [day, side, time] with two sessions per day.
        by_day = data.reshape(len(paths), -1, 2, *data.shape[2:])
        return by_day.mean(axis=(0, 2))

    actual = aggregate_tapes(replay)
    simulated = aggregate_tapes(generated)
    impact = summarize_impact(
        visible_mid=paired(visible, "mid_ticks", relative=True), blind_mid=paired(blind, "mid_ticks", relative=True),
        child_visible=paired(visible, "after_child_mid_ticks", relative=True),
        child_blind=paired(blind, "after_child_mid_ticks", relative=True),
        child_filled=paired(visible, "child_fill_shares"), sigma_ticks=float(cfg["sigma_ticks"]),
        daily_shares=float(cfg["daily_shares"]), schedule=np.isin(grid, children), generated=simulated, replay=actual,
        paths_per_day=2 * len(paths),
    )
    impact["visible_fill_fraction"] = float(impact["average_filled_shares"]) / (len(children) * cfg["child_shares"])
    return ResidualProfile(scores[0], scores[1], simulated, actual, impact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--symbol", choices=SYMBOLS)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_bytes())
    train = manifest["train_dates"]
    dev = manifest["dev_dates"]
    validate_chronological_splits(train, dev, manifest["test_dates"])
    if len(train) != 36 or len(dev) != 12 or len(manifest["test_dates"]) != 12:
        raise ValueError("Run the registered 36 train / 12 dev / 12 metadata-only test protocol")
    for symbol in (args.symbol,) if args.symbol else SYMBOLS:
        _run_symbol(args.root, symbol, train, dev)


if __name__ == "__main__":
    main()
