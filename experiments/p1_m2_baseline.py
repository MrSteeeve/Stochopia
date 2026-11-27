"""Train the P1 seed models and measure their development residuals.

Run with ``python -m experiments.p1_m2_baseline --root <market-data-root>``.
The split manifest is the sole source of train and dev paths. All weights,
paired paths and derived JSON are written to that external root.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.p1_m2_metrics import (
    TapeStats, aggregate_tapes, day_paired_paths, market_bootstrap_payload, summarize_impact, tape_statistics,
)
from experiments.p1_m3_evidence import compact_evidence
from experiments.p2_m1_rollout_eval import (
    average_replicates, pool_continuous_days, stateless_projection, window_replicates,
)
from experiments.p1_m4_market_stats import market_metric_estimates
from stochopia.common import atomic_write_json
from stochopia.data import validate_chronological_splits
from stochopia.harness.lob_rollout import ResidualProfile, RolloutArm, RolloutTrace, rollout_paired_window
from stochopia.measure.continuous_impact import P1_METRIC_SCHEMA, P2_METRIC_SCHEMA
from stochopia.neural.lob_interface import (
    LOBWorldModel,
    make_batch,
    verify_world_model_contract,
    world_model_loss,
)
from stochopia.neural.lob_training import (
    P1_FIT_PROTOCOL,
    P1_TRAIN_SEEDS,
    NonfiniteModelLoss,
    fit_lob_world_model,
    world_model_nll_events,
)
from stochopia.neural.primitives import SeedGRULOBModel


# Re-export the registered P1 window helpers used by existing callers.
from experiments.p1_m2_windows import (
    CHILDREN, CONTEXT, HEADS, PARTICIPATION, SESSION_DURATION_MS, SYMBOLS, WINDOW_MIDPOINTS,
    Window, _calibrate, _load_days, _marginal, _replay_marks, _replay_stats, _windows,
)


@torch.inference_mode()
def _validation(
    model: LOBWorldModel, window: Window, marginal: list[np.ndarray], *, stateless: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    first = window.start - CONTEXT
    end = window.end
    data = window.data
    device = next(model.parameters(), torch.empty(0)).device
    tokens = data.tokens[None, first:end].to(device)
    batch = make_batch(data.ticks[None, first:end].to(device), data.shares[None, first:end].to(device), tokens)
    if stateless:
        loss = world_model_nll_events(model, batch, stateless=True)
    else:
        loss = world_model_loss(model.forward_sequence(batch), tokens, reduction="none")
    observed = data.tokens[window.start:end].cpu().numpy()
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


def _verify_trained_model(model: LOBWorldModel, *, max_state_dim: int | None = None) -> tuple[int, int]:
    """Classify a trained-weight tensor contract failure at the model boundary."""
    try:
        if max_state_dim is None:
            return verify_world_model_contract(model)
        return verify_world_model_contract(model, max_state_dim=max_state_dim)
    except (AssertionError, TypeError, ValueError) as exc:
        raise RuntimeError("Trained model failed tensor contract") from exc


def evaluate_symbol(
    root: Path, symbol: str, train_dates: list[int], dev_dates: list[int], directory: Path,
    *, factory: Callable[[], LOBWorldModel] = SeedGRULOBModel, rollout_seed: int = 12300,
    checkpoints: dict[int, Path] | None = None, save_traces: bool = True,
    device: str | torch.device = "cpu", num_replicates: int = 1,
    metric_schema_version: str = P1_METRIC_SCHEMA, rollout_batch_windows: int = 2,
    max_state_dim: int | None = None,
) -> dict[str, object]:
    """Apply the same train/validation and injection protocol to any model factory."""
    p2 = metric_schema_version == P2_METRIC_SCHEMA
    if metric_schema_version not in (P1_METRIC_SCHEMA, P2_METRIC_SCHEMA):
        raise ValueError("Unknown metric schema version")
    if type(num_replicates) is not int or num_replicates < 1 or (not p2 and num_replicates != 1):
        raise ValueError("num_replicates requires a positive integer and P2 for K > 1")
    if type(rollout_batch_windows) is not int or rollout_batch_windows < 1:
        raise ValueError("rollout_batch_windows requires a positive integer")
    max_state_dim = max_state_dim if max_state_dim is not None else (512 if p2 else 128)
    if type(max_state_dim) is not int or max_state_dim < 1:
        raise ValueError("max_state_dim requires a positive integer")
    if not p2 and max_state_dim != 128:
        raise ValueError("The historical P1 state cap remains 128")
    torch.set_num_threads(1)
    selected = torch.device(device)
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
    evaluation_protocol = {
        "metric_schema_version": metric_schema_version, "num_replicates": num_replicates,
        "rollout_batch_windows": rollout_batch_windows, "max_state_dim": max_state_dim,
        "rollout_protocol": "rollout_protocol_v2",
    } if p2 else {}
    atomic_write_json(directory / "protocol.json", {**cfg, **evaluation_protocol, "fit_protocol": asdict(P1_FIT_PROTOCOL),
                      "context": CONTEXT, "train_seeds": list(P1_TRAIN_SEEDS), "rollout_seed": rollout_seed,
                      "grid_ms": grid.tolist(), "child_times_ms": child_times.tolist(),
                      "windows": [{"date": w.data.day.trading_date, "session": w.session, "start": w.start,
                                   "time_ms": int(w.data.day.time_ms[w.start]), "end": w.end,
                                   "max_events": w.max_events} for w in windows]})
    nll = []
    stateless_nll = []
    window_depth_errors = []
    absolute_drifts = []
    continuous_rows = []
    exhaustion = []
    generated = []
    fit = []
    paths = []
    start_all = perf_counter()
    for seed in P1_TRAIN_SEEDS:
        if checkpoints is None:
            fitted = fit_lob_world_model(factory, train, seed, device=selected)
            model = fitted.model
            elapsed = fitted.seconds
            loss = fitted.last_nll
            torch.save(model.state_dict(), directory / f"seed-{seed}.pt")
            print(f"{symbol} seed {seed} trained in {elapsed:.2f}s", flush=True)
        else:
            model = factory().eval()
            model.load_state_dict(torch.load(checkpoints[seed], weights_only=True, map_location="cpu"))
            model.to(selected)
            elapsed = None
            loss = None
        parameters, state_dim = _verify_trained_model(model, max_state_dim=max_state_dim)
        fit.append({"seed": seed, "seconds": elapsed, "last_batch_nll": loss,
                    "params": parameters, "state_dim": state_dim})
        seed_paths = []
        seed_nll = []
        if p2:
            batched = window_replicates(
                model, windows, child_times=child_times, child_shares=int(cfg["child_shares"]),
                grid=grid, rollout_seed=rollout_seed, context_len=CONTEXT,
                num_replicates=num_replicates, batch_windows=rollout_batch_windows,
            )
        for number, window in enumerate(windows):
            nll_row = _validation(model, window, marginal)
            seed_nll.append(nll_row)
            name = f"seed-{seed}-{window.data.day.trading_date}-{window.session}"
            if p2:
                stateless_nll.append(_validation(model, window, marginal, stateless=True))
                replicas = next(batched)
                traces, tape, depth_error, absolute_drift, exhausted, continuous = average_replicates(
                    replicas, replay_depth[number],
                )
                window_depth_errors.append(depth_error)
                absolute_drifts.append(absolute_drift)
                exhaustion.append(exhausted)
                continuous_rows.append(continuous)
                if save_traces:
                    for replica, pair in enumerate(replicas):
                        _save_traces(directory / f"{name}-rep-{replica}.npz", pair)
            else:
                traces = rollout_paired_window(model, window.data.day, window.start, child_times_ms=child_times,
                                               child_shares=int(cfg["child_shares"]), grid_ms=grid, seed=rollout_seed + number,
                                               context_len=CONTEXT, max_events=window.max_events)
                if save_traces:
                    _save_traces(directory / f"{name}.npz", traces)
                trace = traces[RolloutArm.NO_INJECTION]
                tape = tape_statistics(trace.events, trace.mid_ticks, trace.spread_ticks, trace.touch_shares)
                exhaustion.append(_window_exhausted(traces))
            seed_paths.append(traces)
            generated.append(tape)
            if (number + 1) % 6 == 0:
                print(f"{symbol} seed {seed}: {number + 1}/{len(windows)} paired windows", flush=True)
        paths.append(seed_paths)
        nll.extend(seed_nll)
        fit[-1]["dev_nll"] = _likelihood(seed_nll)[0]

    continuous_panel = pool_continuous_days(continuous_rows, len(windows)) if p2 else None
    profile = _summarize_paths(
        paths, generated, replay, cfg, grid, child_times, _likelihood(nll),
        metric_schema_version=metric_schema_version, num_replicates=num_replicates,
        absolute_drifts=absolute_drifts if p2 else None, continuous_panel=continuous_panel,
    )
    result = asdict(profile)
    parameters = max(row["params"] for row in fit)
    state_dim = max(row["state_dim"] for row in fit)
    bootstrap = market_bootstrap_payload(
        paths, generated, replay, cfg=cfg, grid=grid, children=child_times,
        replay_touch_shares=np.stack(replay_depth), num_replicates=num_replicates,
        metric_schema_version=metric_schema_version,
        window_depth_errors=np.asarray(window_depth_errors).reshape(len(paths), -1) if p2 else None,
        generated_absolute_drifts=absolute_drifts if p2 else None, continuous_panel=continuous_panel,
    )
    if p2:
        bootstrap.update(evaluation_protocol)
    estimates = market_metric_estimates(bootstrap)
    result["market_bootstrap"] = bootstrap
    seed_nll = {row["seed"]: row["dev_nll"]["total"] for row in fit}
    exhausted = sum(exhaustion)
    evidence = compact_evidence(profile, parameters, state_dim, exhausted_windows=exhausted,
                                train_seed_nll=seed_nll, l1_depth_mae=estimates["l1_depth_mae"])
    evidence["paired_windows"] = len(windows)
    evidence["paired_seed_windows"] = sum(len(seed) for seed in paths)
    if p2:
        evidence.update(evaluation_protocol)
        evidence.update(stateless_projection(profile.model_nll, _likelihood(stateless_nll)[0], HEADS))
        evidence["paired_replica_windows"] = evidence["paired_seed_windows"] * num_replicates
    result["evidence"] = evidence
    daily_metrics = {}
    for day_number, date in enumerate(dev_dates):
        indices = slice(2 * day_number, 2 * day_number + 2)
        flat_indices = [trial * len(windows) + index for trial in range(len(paths))
                        for index in range(indices.start, indices.stop)]
        daily_profile = _summarize_paths(
            [trial[indices] for trial in paths], [generated[i] for i in flat_indices], replay[indices],
            cfg, grid, child_times, _likelihood([nll[i] for i in flat_indices]),
            metric_schema_version=metric_schema_version, num_replicates=num_replicates,
            absolute_drifts=[absolute_drifts[i] for i in flat_indices] if p2 else None,
            continuous_panel=None if continuous_panel is None else {
                key: marks[day_number:day_number + 1] for key, marks in continuous_panel.items()
            },
        )
        daily_exhausted = sum(exhaustion[i] for i in flat_indices)
        daily_row = compact_evidence(daily_profile, parameters, state_dim, exhausted_windows=daily_exhausted)
        if p2:
            stateless_daily = _likelihood([stateless_nll[i] for i in flat_indices])[0]
            daily_row.update(stateless_projection(daily_profile.model_nll, stateless_daily, HEADS))
        daily_metrics[str(date)] = daily_row
    result["day_metrics"] = daily_metrics
    result.update(dev_targets=sum(entry[2] for entry in nll), fit=fit, seconds=perf_counter() - start_all)
    atomic_write_json(directory / "residuals.json", _json_value(result))
    return result


def _summarize_paths(paths: list[list[dict[RolloutArm, RolloutTrace]]], generated: list[TapeStats],
                     replay: list[TapeStats], cfg: dict[str, int | float], grid: np.ndarray,
                     children: np.ndarray, scores: list[dict[str, float]], *,
                     metric_schema_version: str = P1_METRIC_SCHEMA, num_replicates: int = 1,
                     absolute_drifts: list[float] | None = None,
                     continuous_panel: dict[str, np.ndarray] | None = None) -> ResidualProfile:
    visible = (RolloutArm.BUY_VISIBLE, RolloutArm.SELL_VISIBLE)
    blind = (RolloutArm.BUY_BLIND, RolloutArm.SELL_BLIND)

    actual = aggregate_tapes(replay)
    simulated = aggregate_tapes(generated, absolute_drifts=absolute_drifts)
    impact = summarize_impact(
        visible_mid=day_paired_paths(paths, visible, "mid_ticks", relative=True),
        blind_mid=day_paired_paths(paths, blind, "mid_ticks", relative=True),
        child_visible=day_paired_paths(paths, visible, "after_child_mid_ticks", relative=True),
        child_blind=day_paired_paths(paths, blind, "after_child_mid_ticks", relative=True),
        child_filled=day_paired_paths(paths, visible, "child_fill_shares"), sigma_ticks=float(cfg["sigma_ticks"]),
        daily_shares=float(cfg["daily_shares"]), schedule=np.isin(grid, children), generated=simulated, replay=actual,
        paths_per_day=2 * len(paths) * num_replicates, metric_schema_version=metric_schema_version,
        grid_ms=grid, planned_child_shares=float(cfg["child_shares"]), continuous_panel=continuous_panel,
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
