"""SZSE likelihood, rollout evaluation and aligned day-cluster market inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from stochopia.common import atomic_write_json
from stochopia.harness.lob_rollout import ResidualProfile, RolloutArm, RolloutTrace, rollout_paired_window
from stochopia.market.rollout_eval import (
    average_replicates, decode_quote_panel, exhausted_replicate_rate, pool_continuous_days,
    stateless_projection, window_replicates,
)
from stochopia.market.tape_metrics import (
    FLOW_LAGS, PRICE_LAGS, TapeStats, aggregate_tapes, day_paired_paths, finite_mean,
    log_slope, market_bootstrap_payload, summarize_impact, tape_statistics,
)
from stochopia.market.windows import (
    CHILDREN, CONTEXT, HEADS, Window, calibrate_market, load_market_days, marginal_nll,
    replay_marks, replay_stats, select_windows,
)
from stochopia.measure.impact import (
    C2_PASS_METRIC, CONTINUOUS_IMPACT_METRICS, LEGACY_METRIC_SCHEMA, CONTINUOUS_METRIC_SCHEMA,
)
from stochopia.neural.lob_interface import (
    LOBWorldModel, make_batch, verify_world_model_contract, world_model_loss,
)
from stochopia.neural.lob_training import (
    DEFAULT_FIT_PROTOCOL, DEFAULT_TRAIN_SEEDS, NonfiniteModelLoss, fit_lob_world_model,
    world_model_nll_events,
)
from stochopia.neural.primitives import SeedGRULOBModel


# --- Projection of closed-loop evidence ---

MARKET_METRICS = (
    "lob_impact_score",
    "end_over_peak_ratio",
    "free_abs_drift_ticks",
    "trade_sign_acf_exp_gap",
    "hurst_gap",
    "visible_over_blind_ratio",
    "impact_exponent_gap",
    "event_count_rel_gap",
    "l1_depth_mae",
) + CONTINUOUS_IMPACT_METRICS + (C2_PASS_METRIC,)


def _absolute_gap(left: object, right: object) -> float | None:
    if isinstance(left, Real) and isinstance(right, Real) and isfinite(left) and isfinite(right):
        return abs(float(left) - float(right))
    return None


def mechanism_readouts(
    generated: Mapping[str, object], historical: Mapping[str, object], impact: Mapping[str, object],
    l1_depth_mae: float | None,
) -> dict[str, float | None]:
    """Project pooled market readouts using the same units for original and bootstrap samples.

    ACF and Hurst gaps are absolute differences to replay. The event gap is
    |generated / observed - 1|. The impact-exponent gap is |delta - 0.5|.
    Recovery and visible/blind are raw ratios. Depth MAE is in shares.
    Undefined gaps and ratios stay unavailable.
    """
    n_observed = historical["events"]
    count_difference = _absolute_gap(generated["events"], n_observed)
    event_gap = None
    if count_difference is not None and n_observed > 0:
        event_gap = count_difference / n_observed
    score = impact["score"]
    measurements = impact["measurements"]
    result = {
        "lob_impact_score": score["score"],
        "end_over_peak_ratio": measurements["end_over_peak"],
        "free_abs_drift_ticks": generated["mean_absolute_drift_ticks"],
        "trade_sign_acf_exp_gap": _absolute_gap(generated["gamma_flow"], historical["gamma_flow"]),
        "hurst_gap": _absolute_gap(generated["hurst"], historical["hurst"]),
        "visible_over_blind_ratio": score["visible_to_blind_ratio"],
        "impact_exponent_gap": _absolute_gap(measurements["delta"], 0.5),
        "event_count_rel_gap": event_gap,
        "l1_depth_mae": l1_depth_mae,
    }
    if "continuous" in impact:
        continuous = impact["continuous"]
        for metric in (*CONTINUOUS_IMPACT_METRICS, C2_PASS_METRIC):
            result[metric] = continuous[metric]
    return result


def compact_evidence(
    profile: ResidualProfile, parameters: int, state_dim: int, *, exhausted_windows: int | float | None = None,
    train_seed_nll: Mapping[int, float] | None = None, l1_depth_mae: float | None = None,
) -> dict[str, object]:
    def residual(key: str) -> float | None:
        generated = profile.generated[key]
        observed = profile.historical[key]
        return None if generated is None or observed is None else float(generated - observed)

    impact = profile.impact
    metrics = impact["measurements"]
    score = impact["score"]
    evidence = {
        **mechanism_readouts(profile.generated, profile.historical, impact, l1_depth_mae),
        "parameters": parameters,
        "state_dim": state_dim,
        "val_nll_total": profile.model_nll["total"],
        "exhausted_windows": exhausted_windows,
        "train_seed_nll": None if train_seed_nll is None else {str(seed): nll for seed, nll in train_seed_nll.items()},
        "claims": {key: claim["status"] for key, claim in score["claims"].items()},
        "lob_impact_score_reason": score["reason"],
        "nll_dt": profile.model_nll["dt"],
        "nll_action": profile.model_nll["action"],
        "nll_rel_ticks": profile.model_nll["rel_ticks"],
        "nll_volume": profile.model_nll["volume"],
        "event_count_delta": residual("events"),
        "zero_wait_rate_delta": residual("burst_rate"),
        "sign_acf_delta": residual("gamma_flow"),
        "hurst_delta": residual("hurst"),
        "top_depth_delta": residual("touch_shares"),
        "idle_abs_drift": profile.generated["mean_absolute_drift_ticks"],
        "impact_peak": impact["visible_peak_for_amplitude_ticks"],
        "blind_ratio": impact["score"]["visible_to_blind_ratio"],
        "impact_y": metrics["y"],
        "impact_delta": metrics["delta"],
        "recovery_ratio": metrics["end_over_peak"],
    }
    if "continuous" in impact:
        continuous = impact["continuous"]
        evidence["impact_identifiability_state"] = continuous["impact_identifiability_state"]
        evidence["reference_scale_ticks"] = continuous["reference_scale_ticks"]
        evidence["reference_participation"] = continuous["reference_participation"]
        if "available_quote_support" in continuous:
            evidence["available_quote_support"] = continuous["available_quote_support"]
    return evidence


# --- Paired day-cluster statistics ---

PATH_FIELDS = ("visible_mid", "blind_mid", "child_visible", "child_blind", "child_filled")
TAPE_FIELDS = ("events", "flow_acf", "price_msd", "drift_ticks")


def _pool_tapes(tapes: Mapping[str, np.ndarray], days: np.ndarray) -> dict[str, float | None]:
    """Average finite per-window terms over resampled days and their windows.

    ACF and price MSD are [day, window, lag]. Refit the original logarithmic
    slopes after pooling, keeping the original three-positive-lag threshold.
    Signed drift is for the composite score; absolute drift is for the guard.
    """
    acf = finite_mean(tapes["flow_acf"][days], axis=(0, 1))
    price_msd = finite_mean(tapes["price_msd"][days], axis=(0, 1))
    flow_slope = log_slope(FLOW_LAGS, acf)
    price_slope = log_slope(PRICE_LAGS, price_msd)
    drift = tapes["drift_ticks"][days]
    absolute_drift = np.abs(drift)
    if "absolute_drift_ticks" in tapes:
        absolute_drift = tapes["absolute_drift_ticks"][days]
    return {
        "events": float(finite_mean(tapes["events"][days])),
        "gamma_flow": -flow_slope if flow_slope is not None else None,
        "hurst": price_slope / 2 if price_slope is not None else None,
        "drift_ticks": float(finite_mean(drift)),
        "mean_absolute_drift_ticks": float(finite_mean(absolute_drift)),
    }


@dataclass(frozen=True)
class MarketClusters:
    """One model's paths, tapes and common schedule prepared for resampling."""

    paths: Mapping[str, np.ndarray]
    generated: Mapping[str, np.ndarray]
    historical: Mapping[str, np.ndarray]
    depth_errors: np.ndarray | None
    schedule: np.ndarray
    paths_per_day: int
    sigma_ticks: float
    daily_shares: float
    metric_schema_version: str = LEGACY_METRIC_SCHEMA
    grid_ms: np.ndarray | None = None
    planned_child_shares: float | None = None
    num_replicates: int = 1
    continuous_panel: Mapping[str, np.ndarray] | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> MarketClusters:
        """Decode JSON null as unavailable rather than a zero physical mark."""
        depth = payload["l1_depth_errors"]
        schema = payload.get("metric_schema_version", LEGACY_METRIC_SCHEMA)
        if schema not in (LEGACY_METRIC_SCHEMA, CONTINUOUS_METRIC_SCHEMA):
            raise ValueError("unknown market metric schema")
        replicates = payload.get("num_replicates", 1)
        grid = None
        child_shares = None
        continuous = None
        generated = {name: np.asarray(payload["generated_tapes"][name], dtype=float) for name in TAPE_FIELDS}
        if schema == CONTINUOUS_METRIC_SCHEMA:
            grid = np.asarray(payload["grid_ms"], dtype=float)
            child_shares = float(payload["planned_child_shares"])
            if "continuous_panel" in payload:
                continuous = decode_quote_panel(
                    payload["continuous_panel"], len(payload["visible_mid"]), len(grid),
                    int(payload["paths_per_day"]),
                )
            if type(replicates) is not int or replicates < 1:
                raise ValueError("Continuous-schema replicas must be a positive integer")
            if "absolute_drift_ticks" in payload["generated_tapes"]:
                generated["absolute_drift_ticks"] = np.asarray(
                    payload["generated_tapes"]["absolute_drift_ticks"], dtype=float,
                )
        elif replicates != 1:
            raise ValueError("Legacy bootstrap payload cannot contain multiple replicates")
        return cls(
            paths={name: np.asarray(payload[name], dtype=float) for name in PATH_FIELDS},
            generated=generated,
            historical={name: np.asarray(payload["historical_tapes"][name], dtype=float) for name in TAPE_FIELDS},
            depth_errors=None if depth is None else np.asarray(depth, dtype=float),
            schedule=np.asarray(payload["schedule"], dtype=bool),
            paths_per_day=int(payload["paths_per_day"]),
            sigma_ticks=float(payload["sigma_ticks"]),
            daily_shares=float(payload["daily_shares"]),
            metric_schema_version=schema,
            grid_ms=grid,
            planned_child_shares=child_shares,
            num_replicates=replicates,
            continuous_panel=continuous,
        )

    @property
    def days(self) -> int:
        return len(self.paths["visible_mid"])

    def estimate(self, days: np.ndarray) -> dict[str, float | None]:
        """Run the original scorer, including C2 eligibility, on a day panel."""
        readouts, _ = self._measure(days, law_admitted=None)
        return readouts

    def _measure(
        self, days: np.ndarray, *, law_admitted: bool | None,
    ) -> tuple[dict[str, float | None], bool]:
        """Recompute on one multiset of day indices.

        Pool paths and child marks to derive Y, exponent and recovery. The
        resampling targets score uncertainty conditional on the full-panel C2
        result. Reapplying the 3-SE detection cutoff inside each draw would
        change the estimand and cause law detection to censor score draws.
        The continuous schema explicitly changes this bootstrap estimand: independent C2
        adjudication in each draw and continuous metrics without that gate.
        """
        generated = _pool_tapes(self.generated, days)
        historical = _pool_tapes(self.historical, days)
        continuous = None
        if self.continuous_panel is not None:
            continuous = {name: values[days] for name, values in self.continuous_panel.items()}
        impact = summarize_impact(
            **{key: marks[days] for key, marks in self.paths.items()},
            sigma_ticks=self.sigma_ticks,
            daily_shares=self.daily_shares,
            schedule=self.schedule,
            generated=generated,
            replay=historical,
            paths_per_day=self.paths_per_day,
            include_peak_interval=False,
            law_admitted=law_admitted,
            metric_schema_version=self.metric_schema_version,
            grid_ms=self.grid_ms,
            planned_child_shares=self.planned_child_shares,
            continuous_panel=continuous,
        )
        depth = None if self.depth_errors is None else float(finite_mean(self.depth_errors[days]))
        admitted = impact["measurements"]["law_identified"] is True
        return mechanism_readouts(generated, historical, impact, depth), admitted

    def resample(self, draws: np.ndarray, metrics: Sequence[str]) -> dict[str, np.ndarray]:
        """Return one pooled value per [draw, day] row.

        Legacy payloads retain original-panel admission. The continuous schema calls
        identify_law independently for each resampled complete-day panel.
        """
        admitted = None
        if self.metric_schema_version == LEGACY_METRIC_SCHEMA:
            _, admitted = self._measure(np.arange(self.days), law_admitted=None)
        samples = {metric: np.full(len(draws), np.nan) for metric in metrics}
        for i, days in enumerate(draws):
            result, _ = self._measure(days, law_admitted=admitted)
            for metric, values in samples.items():
                value = result[metric]
                if value is not None:
                    values[i] = value
        return samples

    def continuous_mark_support(self, metric: str) -> np.ndarray:
        """Return [day, required clock marks] support for paired continuous metrics."""
        if self.continuous_panel is None:
            raise ValueError("available quote support requires a continuous panel")
        visible = self.continuous_panel["visible"]
        if metric == "visible_minus_blind_auc":
            return np.isfinite(self.continuous_panel["joint_difference"])
        if metric in ("impact_curve_mae", "stylized_impact_mae"):
            return np.isfinite(visible)
        if metric in ("post_stop_relaxation_delta", "bounded_relaxation_ratio"):
            last = int(np.flatnonzero(self.schedule)[-1])
            return np.isfinite(visible[:, [last, -1]])
        raise ValueError(f"unknown continuous mark metric {metric}")

    def day_support(self, metric: str) -> np.ndarray:
        """Return [day] support for the physical terms underlying a pooled metric.

        Individual dates can support the impact score even though C2 eligibility
        and the score itself are estimated only on a pooled panel. Response
        ratios need a finite path and positive denominator on a supported day.
        Tape-exponent gaps need at least three positive finite lags in each tape.
        """
        day_children = self.paths["child_visible"]
        if metric == C2_PASS_METRIC:
            # Admission is defined (pass or not admitted) for every draw;
            # complete strict marks are reported separately from the indicator.
            return np.ones(self.days, dtype=bool)
        if metric in CONTINUOUS_IMPACT_METRICS:
            if self.continuous_panel is not None:
                return self.continuous_mark_support(metric).all(axis=1)
            visible = self.paths["visible_mid"]
            if metric == "visible_minus_blind_auc":
                blind = self.paths["blind_mid"]
                return np.isfinite(visible).all(axis=(1, 2)) & np.isfinite(blind).all(axis=(1, 2))
            if metric in ("impact_curve_mae", "stylized_impact_mae"):
                return np.isfinite(visible).all(axis=(1, 2))
            last = int(np.flatnonzero(self.schedule)[-1])
            return np.isfinite(visible[:, :, [last, -1]]).all(axis=(1, 2))
        if metric == "impact_exponent_gap":
            return np.isfinite(day_children).all(axis=(1, 2))
        if metric in ("lob_impact_score", "end_over_peak_ratio", "visible_over_blind_ratio"):
            peak = (day_children[:, 0, -1] - day_children[:, 1, -1]) / 2
            if metric == "visible_over_blind_ratio":
                blind = self.paths["child_blind"]
                blind_peak = (blind[:, 0, -1] - blind[:, 1, -1]) / 2
                return np.isfinite(peak) & np.isfinite(blind_peak) & (blind_peak > 0)
            last = int(np.flatnonzero(self.schedule)[-1])
            post = np.isfinite(self.paths["visible_mid"][:, :, last:]).all(axis=(1, 2))
            recovery = np.isfinite(peak) & (peak > 0) & post
            if metric == "end_over_peak_ratio":
                return recovery
            fills = finite_mean(self.paths["child_filled"].sum(axis=-1), axis=1)
            drift = np.isfinite(self.generated["drift_ticks"]).any(axis=1)
            complete_execution = np.isfinite(day_children).all(axis=(1, 2))
            return recovery & complete_execution & (fills > 0) & drift
        if metric in ("trade_sign_acf_exp_gap", "hurst_gap"):
            field = "flow_acf" if metric == "trade_sign_acf_exp_gap" else "price_msd"
            generated = finite_mean(self.generated[field], axis=1)
            historical = finite_mean(self.historical[field], axis=1)
            return ((generated > 0).sum(axis=1) >= 3) & ((historical > 0).sum(axis=1) >= 3)
        if metric == "event_count_rel_gap":
            generated = finite_mean(self.generated["events"], axis=1)
            historical = finite_mean(self.historical["events"], axis=1)
            return np.isfinite(generated) & (historical > 0)
        if metric == "free_abs_drift_ticks":
            return np.isfinite(self.generated["drift_ticks"]).any(axis=1)
        if metric == "l1_depth_mae":
            return np.zeros(self.days, dtype=bool) if self.depth_errors is None else np.isfinite(self.depth_errors)
        raise ValueError(f"unknown pooled market metric {metric}")

    def score_support(self, dates: tuple[int, ...], samples: np.ndarray) -> dict[str, object]:
        """Identify missing original day marks and undefined composite draws.

        Day masks operate on [day, side, time] arrays. Draw indices are zero
        based. When all draws are undefined the count suffices.
        """
        day_children = self.paths["child_visible"]
        execution = np.isfinite(day_children).all(axis=(1, 2))
        last = int(np.flatnonzero(self.schedule)[-1])
        last_marks = np.isfinite(day_children[:, :, -1]).all(axis=1)
        post_marks = np.isfinite(self.paths["visible_mid"][:, :, last:]).all(axis=(1, 2))
        recovery = last_marks & post_marks
        missing = np.flatnonzero(~np.isfinite(samples))
        return {
            "incomplete_execution_dates": [date for date, valid in zip(dates, execution, strict=True) if not valid],
            "incomplete_recovery_dates": [date for date, valid in zip(dates, recovery, strict=True) if not valid],
            "undefined_draw_count": len(missing),
            "undefined_draw_indices": missing.tolist() if len(missing) < len(samples) else None,
        }


def market_metric_estimates(payload: Mapping[str, object]) -> dict[str, float | None]:
    """Estimate on every registered day in its original order."""
    clusters = MarketClusters.from_payload(payload)
    return clusters.estimate(np.arange(clusters.days))


# --- Candidate evaluation on historical market days ---

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


def window_exhausted(traces: dict[RolloutArm, RolloutTrace]) -> bool:
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
    metric_schema_version: str = LEGACY_METRIC_SCHEMA, rollout_batch_windows: int = 2,
    max_state_dim: int | None = None,
) -> dict[str, object]:
    """Apply the same train/validation and injection protocol to any model factory."""
    continuous_schema = metric_schema_version == CONTINUOUS_METRIC_SCHEMA
    if metric_schema_version not in (LEGACY_METRIC_SCHEMA, CONTINUOUS_METRIC_SCHEMA):
        raise ValueError("Unknown metric schema version")
    if type(num_replicates) is not int or num_replicates < 1 or (not continuous_schema and num_replicates != 1):
        raise ValueError("num_replicates requires a positive integer and the continuous schema for K > 1")
    if type(rollout_batch_windows) is not int or rollout_batch_windows < 1:
        raise ValueError("rollout_batch_windows requires a positive integer")
    max_state_dim = max_state_dim if max_state_dim is not None else (512 if continuous_schema else 128)
    if type(max_state_dim) is not int or max_state_dim < 1:
        raise ValueError("max_state_dim requires a positive integer")
    if not continuous_schema and max_state_dim != 128:
        raise ValueError("The historical schema state cap remains 128")
    torch.set_num_threads(1)
    selected = torch.device(device)
    train = load_market_days(root, symbol, train_dates)
    dev = load_market_days(root, symbol, dev_dates)
    marginal = marginal_nll(train)
    cfg = calibrate_market(root, train, symbol)
    spacing = int(cfg["spacing_ms"])
    child_times = np.arange(1, CHILDREN + 1) * spacing
    grid = np.arange(2 * CHILDREN * 4 + 1) * spacing // 4
    windows = select_windows(dev, int(grid[-1]))
    replay = [replay_stats(window, grid) for window in windows]
    replay_depth = []
    for window in windows:
        _, marks = replay_marks(window, grid)
        replay_depth.append(window.data.day.pre_book_shares[marks, :, 0])
    directory.mkdir(parents=True, exist_ok=True)
    evaluation_protocol = {
        "metric_schema_version": metric_schema_version, "num_replicates": num_replicates,
        "rollout_batch_windows": rollout_batch_windows, "max_state_dim": max_state_dim,
        "rollout_protocol": "rollout_protocol_v2",
    } if continuous_schema else {}
    atomic_write_json(directory / "protocol.json", {**cfg, **evaluation_protocol, "fit_protocol": asdict(DEFAULT_FIT_PROTOCOL),
                      "context": CONTEXT, "train_seeds": list(DEFAULT_TRAIN_SEEDS), "rollout_seed": rollout_seed,
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
    replica_exhaustion = []
    generated = []
    fit = []
    paths = []
    start_all = perf_counter()
    for seed in DEFAULT_TRAIN_SEEDS:
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
        if continuous_schema:
            batched = window_replicates(
                model, windows, child_times=child_times, child_shares=int(cfg["child_shares"]),
                grid=grid, rollout_seed=rollout_seed, context_len=CONTEXT,
                num_replicates=num_replicates, batch_windows=rollout_batch_windows,
            )
        for number, window in enumerate(windows):
            nll_row = _validation(model, window, marginal)
            seed_nll.append(nll_row)
            name = f"seed-{seed}-{window.data.day.trading_date}-{window.session}"
            if continuous_schema:
                stateless_nll.append(_validation(model, window, marginal, stateless=True))
                replicas = next(batched)
                traces, tape, depth_error, absolute_drift, exhausted, continuous = average_replicates(
                    replicas, replay_depth[number],
                )
                window_depth_errors.append(depth_error)
                absolute_drifts.append(absolute_drift)
                exhaustion.append(exhausted)
                replica_exhaustion.append(exhausted_replicate_rate(replicas))
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
                exhaustion.append(window_exhausted(traces))
            seed_paths.append(traces)
            generated.append(tape)
            if (number + 1) % 6 == 0:
                print(f"{symbol} seed {seed}: {number + 1}/{len(windows)} paired windows", flush=True)
        paths.append(seed_paths)
        nll.extend(seed_nll)
        fit[-1]["dev_nll"] = _likelihood(seed_nll)[0]

    continuous_panel = pool_continuous_days(continuous_rows, len(windows)) if continuous_schema else None
    profile = _summarize_paths(
        paths, generated, replay, cfg, grid, child_times, _likelihood(nll),
        metric_schema_version=metric_schema_version, num_replicates=num_replicates,
        absolute_drifts=absolute_drifts if continuous_schema else None, continuous_panel=continuous_panel,
    )
    result = asdict(profile)
    parameters = max(row["params"] for row in fit)
    state_dim = max(row["state_dim"] for row in fit)
    bootstrap = market_bootstrap_payload(
        paths, generated, replay, cfg=cfg, grid=grid, children=child_times,
        replay_touch_shares=np.stack(replay_depth), num_replicates=num_replicates,
        metric_schema_version=metric_schema_version,
        window_depth_errors=np.asarray(window_depth_errors).reshape(len(paths), -1) if continuous_schema else None,
        generated_absolute_drifts=absolute_drifts if continuous_schema else None, continuous_panel=continuous_panel,
    )
    if continuous_schema:
        bootstrap.update(evaluation_protocol)
    estimates = market_metric_estimates(bootstrap)
    result["market_bootstrap"] = bootstrap
    seed_nll = {row["seed"]: row["dev_nll"]["total"] for row in fit}
    exhausted = sum(exhaustion)
    evidence = compact_evidence(profile, parameters, state_dim, exhausted_windows=exhausted,
                                train_seed_nll=seed_nll, l1_depth_mae=estimates["l1_depth_mae"])
    evidence["paired_windows"] = len(windows)
    evidence["paired_seed_windows"] = sum(len(seed) for seed in paths)
    if continuous_schema:
        evidence.update(evaluation_protocol)
        evidence["exhausted_replicate_rate"] = float(np.mean(replica_exhaustion))
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
            absolute_drifts=[absolute_drifts[i] for i in flat_indices] if continuous_schema else None,
            continuous_panel=None if continuous_panel is None else {
                key: marks[day_number:day_number + 1] for key, marks in continuous_panel.items()
            },
        )
        daily_exhausted = sum(exhaustion[i] for i in flat_indices)
        daily_row = compact_evidence(daily_profile, parameters, state_dim, exhausted_windows=daily_exhausted)
        if continuous_schema:
            stateless_daily = _likelihood([stateless_nll[i] for i in flat_indices])[0]
            daily_row.update(stateless_projection(daily_profile.model_nll, stateless_daily, HEADS))
            daily_row["exhausted_replicate_rate"] = float(np.mean([replica_exhaustion[i] for i in flat_indices]))
        daily_metrics[str(date)] = daily_row
    result["day_metrics"] = daily_metrics
    result.update(dev_targets=sum(entry[2] for entry in nll), fit=fit, seconds=perf_counter() - start_all)
    atomic_write_json(directory / "residuals.json", _json_value(result))
    return result


def _summarize_paths(paths: list[list[dict[RolloutArm, RolloutTrace]]], generated: list[TapeStats],
                     replay: list[TapeStats], cfg: dict[str, int | float], grid: np.ndarray,
                     children: np.ndarray, scores: list[dict[str, float]], *,
                     metric_schema_version: str = LEGACY_METRIC_SCHEMA, num_replicates: int = 1,
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
