"""Train the P1 seed models and measure their development residuals.

Run with ``python -m experiments.p1_m2_baseline --root <market-data-root>``.
The split manifest is the sole source of train and dev paths. All weights,
paired paths and derived JSON are written to that external root.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
import torch

from experiments.p1_m2_metrics import (
    TapeStats, aggregate_tapes, day_paired_paths, market_bootstrap_payload, summarize_impact, tape_statistics,
)
from experiments.p1_m3_evidence import compact_evidence
from experiments.p1_m4_market_stats import market_metric_estimates
from stochopia.common import atomic_write_json
from stochopia.data import load_day_slice, validate_chronological_splits
from stochopia.harness.lob_rollout import ResidualProfile, RolloutArm, RolloutTrace, rollout_paired_window
from stochopia.neural.lob_interface import (
    ACTION_BINS,
    DT_BINS,
    LOBWorldModel,
    LOT_BINS,
    REL_TICK_BINS,
    make_batch,
    verify_world_model_contract,
    world_model_loss,
)
from stochopia.neural.lob_training import (
    P1_FIT_PROTOCOL,
    P1_TRAIN_SEEDS,
    EncodedDay,
    NonfiniteModelLoss,
    encode_day,
    fit_lob_world_model,
)
from stochopia.neural.primitives import SeedGRULOBModel


SYMBOLS = ("000001.SZ", "300001.SZ")
CONTEXT = 128
CHILDREN = 8
PARTICIPATION = 0.10
WINDOW_MIDPOINTS = (37_800_000, 50_310_000)
SESSION_DURATION_MS = 7_200_000 + 7_020_000
HEADS = ("dt", "action", "rel_ticks", "volume")


@dataclass(frozen=True)
class Window:
    data: EncodedDay
    session: int
    start: int
    end: int

    @property
    def max_events(self) -> int:
        return max(6_000, 8 * (self.end - self.start))


def _load_days(root: Path, symbol: str, dates: list[int]) -> list[EncodedDay]:
    loaded = []
    for date in dates:
        day = load_day_slice(root / "cache" / str(date) / f"{symbol}.npz")
        if day.trading_date != date or day.wind_code != symbol:
            raise ValueError(f"Cache metadata does not match {symbol} on {date}")
        loaded.append(encode_day(day))
    return loaded


def _marginal(days: list[EncodedDay]) -> list[np.ndarray]:
    observed = torch.cat([day.tokens for day in days]).numpy()
    marginals = []
    for axis, size in enumerate((DT_BINS, ACTION_BINS, REL_TICK_BINS, LOT_BINS)):
        counts = np.bincount(observed[:, axis], minlength=size) + 0.5
        marginals.append(-np.log(counts / counts.sum()))
    return marginals


def _calibrate(root: Path, train: list[EncodedDay], symbol: str) -> dict[str, int | float]:
    size_arrays = []
    touch_arrays = []
    daily_shares = []
    daily_price_changes = []
    for data in train:
        day = data.day
        aggressive = (day.action_type == 2) | (day.action_type == 3)
        size_arrays.append(day.volume_shares[aggressive])
        touch_arrays.append(day.pre_book_shares[:, :, 0].reshape(-1))
        quotes = pq.read_table(root / "raw" / str(day.trading_date) / "行情.parquet", columns=["cum_volume"],
                               filters=[("wind_code", "==", symbol)])
        daily_shares.append(max(v for v in quotes["cum_volume"].to_pylist() if v is not None))
        daily_price_changes.append(day.mid_ticks_x2[-1] / 2 - day.prev_close_ticks)
    sizes = np.concatenate(size_arrays)
    frequency = len(sizes) / (len(train) * SESSION_DURATION_MS)
    spacing = 10 * max(1, round(1.0 / (10 * frequency * PARTICIPATION)))
    median_touch = float(np.median(np.concatenate(touch_arrays)))
    median_order = int(round(float(np.median(sizes))))
    queue_scaled_size = 100 * round(0.08 * median_touch / 100)
    return {"child_shares": max(median_order, queue_scaled_size), "spacing_ms": spacing,
            "median_touch_shares": median_touch, "median_aggressive_shares": median_order,
            "queue_scaled_child_shares": queue_scaled_size,
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


def _replay_marks(window: Window, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    day = window.data.day
    times = day.time_ms[window.start:window.end] - day.time_ms[window.start]
    marks = np.searchsorted(times, grid, side="right") + window.start
    marks[0] = window.start  # both the model and replay start at the pre-event book
    return times, marks


def _replay_stats(window: Window, grid: np.ndarray) -> TapeStats:
    day = window.data.day
    times, marks = _replay_marks(window, grid)
    return tape_statistics(np.column_stack((times, day.action_type[window.start:window.end])),
                           day.mid_ticks_x2[marks] / 2, day.spread_ticks[marks], day.pre_book_shares[marks, :, 0])


@torch.inference_mode()
def _validation(model: LOBWorldModel, window: Window, marginal: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    first = window.start - CONTEXT
    end = window.end
    data = window.data
    tokens = data.tokens[None, first:end]
    batch = make_batch(data.ticks[None, first:end], data.shares[None, first:end], tokens)
    loss = world_model_loss(model.forward_sequence(batch), tokens, reduction="none")
    observed = tokens[0, CONTEXT:].numpy()
    predicted = np.array([float(getattr(loss, name)[:, CONTEXT:].mean()) for name in HEADS])
    if not np.isfinite(predicted).all():
        raise NonfiniteModelLoss(f"Nonfinite evaluation likelihood for {data.day.wind_code} {data.day.trading_date}")
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
    fields = ("grid_ms", "mid_ticks", "spread_ticks", "touch_shares", "events",
              "after_child_mid_ticks", "child_fill_shares", "exhausted", "budget_exhausted")
    arrays = {f"{arm.value}_{field}": getattr(trace, field) for arm, trace in traces.items() for field in fields}
    np.savez_compressed(path, **arrays)


def _likelihood(rows: list[tuple[np.ndarray, np.ndarray, int]]) -> list[dict[str, float]]:
    weights = np.array([row[2] for row in rows])
    means = [np.average(np.array([row[i] for row in rows]), weights=weights, axis=0) for i in range(2)]
    return [{**dict(zip(HEADS, values.tolist())), "total": float(values.sum())} for values in means]


def _window_exhausted(traces: dict[RolloutArm, RolloutTrace]) -> bool:
    """Count a free-book failure or an early event cap in any arm once."""
    return traces[RolloutArm.NO_INJECTION].exhausted or any(trace.budget_exhausted for trace in traces.values())


def evaluate_symbol(
    root: Path, symbol: str, train_dates: list[int], dev_dates: list[int], directory: Path,
    *, factory: Callable[[], LOBWorldModel] = SeedGRULOBModel, rollout_seed: int = 12300,
    checkpoints: dict[int, Path] | None = None, save_traces: bool = True,
) -> dict[str, object]:
    """Apply the same train/validation and injection protocol to any model factory."""
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
    replay_depth = []
    for window in windows:
        _, marks = _replay_marks(window, grid)
        replay_depth.append(window.data.day.pre_book_shares[marks, :, 0])
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / "protocol.json", {**cfg, "fit_protocol": asdict(P1_FIT_PROTOCOL),
                      "context": CONTEXT, "train_seeds": list(P1_TRAIN_SEEDS), "rollout_seed": rollout_seed,
                      "grid_ms": grid.tolist(), "child_times_ms": child_times.tolist(),
                      "windows": [{"date": w.data.day.trading_date, "session": w.session, "start": w.start,
                                   "time_ms": int(w.data.day.time_ms[w.start]), "end": w.end,
                                   "max_events": w.max_events} for w in windows]})
    nll = []
    generated = []
    fit = []
    paths = []
    start_all = perf_counter()
    for seed in P1_TRAIN_SEEDS:
        if checkpoints is None:
            fitted = fit_lob_world_model(factory, train, seed)
            model = fitted.model
            elapsed = fitted.seconds
            loss = fitted.last_nll
            torch.save(model.state_dict(), directory / f"seed-{seed}.pt")
            print(f"{symbol} seed {seed} trained in {elapsed:.2f}s", flush=True)
        else:
            model = factory().eval()
            model.load_state_dict(torch.load(checkpoints[seed], weights_only=True, map_location="cpu"))
            elapsed = None
            loss = None
        parameters, state_dim = verify_world_model_contract(model)
        fit.append({"seed": seed, "seconds": elapsed, "last_batch_nll": loss,
                    "params": parameters, "state_dim": state_dim})
        seed_paths = []
        seed_nll = []
        for number, window in enumerate(windows):
            seed_nll.append(_validation(model, window, marginal))
            traces = rollout_paired_window(model, window.data.day, window.start, child_times_ms=child_times,
                                           child_shares=int(cfg["child_shares"]), grid_ms=grid, seed=rollout_seed + number,
                                           context_len=CONTEXT, max_events=window.max_events)
            if save_traces:
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
    parameters = max(row["params"] for row in fit)
    state_dim = max(row["state_dim"] for row in fit)
    bootstrap = market_bootstrap_payload(paths, generated, replay, cfg=cfg, grid=grid,
                                         children=child_times, replay_touch_shares=np.stack(replay_depth))
    estimates = market_metric_estimates(bootstrap)
    result["market_bootstrap"] = bootstrap
    seed_nll = {row["seed"]: row["dev_nll"]["total"] for row in fit}
    exhausted = sum(_window_exhausted(pair) for seed in paths for pair in seed)
    evidence = compact_evidence(profile, parameters, state_dim, exhausted_windows=exhausted,
                                train_seed_nll=seed_nll, l1_depth_mae=estimates["l1_depth_mae"])
    evidence["paired_windows"] = len(windows)
    evidence["paired_seed_windows"] = sum(len(seed) for seed in paths)
    result["evidence"] = evidence
    daily_metrics = {}
    for day_number, date in enumerate(dev_dates):
        indices = slice(2 * day_number, 2 * day_number + 2)
        flat_indices = [trial * len(windows) + index for trial in range(len(paths))
                        for index in range(indices.start, indices.stop)]
        daily_profile = _summarize_paths(
            [trial[indices] for trial in paths], [generated[i] for i in flat_indices], replay[indices],
            cfg, grid, child_times, _likelihood([nll[i] for i in flat_indices]),
        )
        daily_exhausted = sum(_window_exhausted(pair) for seed in paths for pair in seed[indices])
        daily_metrics[str(date)] = compact_evidence(daily_profile, parameters, state_dim,
                                                   exhausted_windows=daily_exhausted)
    result["day_metrics"] = daily_metrics
    result.update(dev_targets=sum(entry[2] for entry in nll), fit=fit, seconds=perf_counter() - start_all)
    atomic_write_json(directory / "residuals.json", _json_value(result))
    return result


def _summarize_paths(paths: list[list[dict[RolloutArm, RolloutTrace]]], generated: list[TapeStats],
                     replay: list[TapeStats], cfg: dict[str, int | float], grid: np.ndarray,
                     children: np.ndarray, scores: list[dict[str, float]]) -> ResidualProfile:
    visible = (RolloutArm.BUY_VISIBLE, RolloutArm.SELL_VISIBLE)
    blind = (RolloutArm.BUY_BLIND, RolloutArm.SELL_BLIND)

    actual = aggregate_tapes(replay)
    simulated = aggregate_tapes(generated)
    impact = summarize_impact(
        visible_mid=day_paired_paths(paths, visible, "mid_ticks", relative=True),
        blind_mid=day_paired_paths(paths, blind, "mid_ticks", relative=True),
        child_visible=day_paired_paths(paths, visible, "after_child_mid_ticks", relative=True),
        child_blind=day_paired_paths(paths, blind, "after_child_mid_ticks", relative=True),
        child_filled=day_paired_paths(paths, visible, "child_fill_shares"), sigma_ticks=float(cfg["sigma_ticks"]),
        daily_shares=float(cfg["daily_shares"]), schedule=np.isin(grid, children), generated=simulated, replay=actual,
        paths_per_day=2 * len(paths),
    )
    impact["visible_fill_fraction"] = float(impact["average_filled_shares"]) / (len(children) * cfg["child_shares"])
    return ResidualProfile(scores[0], scores[1], simulated, actual, impact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--symbol", choices=SYMBOLS)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_bytes())
    train = manifest["train_dates"]
    dev = manifest["dev_dates"]
    validate_chronological_splits(train, dev, manifest["test_dates"])
    if len(train) != 36 or len(dev) != 12 or len(manifest["test_dates"]) != 12:
        raise ValueError("Run the registered 36 train / 12 dev / 12 metadata-only test protocol")
    output = args.output or args.root / "baseline-m3"
    for symbol in (args.symbol,) if args.symbol else SYMBOLS:
        evaluate_symbol(args.root, symbol, train, dev, output / symbol)


if __name__ == "__main__":
    main()
