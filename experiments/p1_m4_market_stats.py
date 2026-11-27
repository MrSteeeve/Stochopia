"""Repool market statistics on aligned day-cluster bootstrap draws."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .p1_m2_metrics import FLOW_LAGS, PRICE_LAGS, _finite_mean, _log_slope, summarize_impact
from .p1_m3_evidence import mechanism_readouts
from .p2_m1_quote_support import decode_quote_panel
from stochopia.measure.impact import (
    C2_PASS_METRIC, CONTINUOUS_IMPACT_METRICS, P1_METRIC_SCHEMA, P2_METRIC_SCHEMA,
)


PATH_FIELDS = ("visible_mid", "blind_mid", "child_visible", "child_blind", "child_filled")
TAPE_FIELDS = ("events", "flow_acf", "price_msd", "drift_ticks")


def _pool_tapes(tapes: Mapping[str, np.ndarray], days: np.ndarray) -> dict[str, float | None]:
    """Average finite per-window terms over resampled days and their windows.

    ACF and price MSD are [day, window, lag]. Refit the original logarithmic
    slopes after pooling, keeping the original three-positive-lag threshold.
    Signed drift is for the composite score; absolute drift is for the guard.
    """
    acf = _finite_mean(tapes["flow_acf"][days], axis=(0, 1))
    price_msd = _finite_mean(tapes["price_msd"][days], axis=(0, 1))
    flow_slope = _log_slope(FLOW_LAGS, acf)
    price_slope = _log_slope(PRICE_LAGS, price_msd)
    drift = tapes["drift_ticks"][days]
    absolute_drift = np.abs(drift)
    if "absolute_drift_ticks" in tapes:
        absolute_drift = tapes["absolute_drift_ticks"][days]
    return {
        "events": float(_finite_mean(tapes["events"][days])),
        "gamma_flow": -flow_slope if flow_slope is not None else None,
        "hurst": price_slope / 2 if price_slope is not None else None,
        "drift_ticks": float(_finite_mean(drift)),
        "mean_absolute_drift_ticks": float(_finite_mean(absolute_drift)),
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
    metric_schema_version: str = P1_METRIC_SCHEMA
    grid_ms: np.ndarray | None = None
    planned_child_shares: float | None = None
    num_replicates: int = 1
    continuous_panel: Mapping[str, np.ndarray] | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> MarketClusters:
        """Decode JSON null as unavailable rather than a zero physical mark."""
        depth = payload["l1_depth_errors"]
        schema = payload.get("metric_schema_version", P1_METRIC_SCHEMA)
        if schema not in (P1_METRIC_SCHEMA, P2_METRIC_SCHEMA):
            raise ValueError("unknown market metric schema")
        replicates = payload.get("num_replicates", 1)
        grid = None
        child_shares = None
        continuous = None
        generated = {name: np.asarray(payload["generated_tapes"][name], dtype=float) for name in TAPE_FIELDS}
        if schema == P2_METRIC_SCHEMA:
            grid = np.asarray(payload["grid_ms"], dtype=float)
            child_shares = float(payload["planned_child_shares"])
            if "continuous_panel" in payload:
                continuous = decode_quote_panel(
                    payload["continuous_panel"], len(payload["visible_mid"]), len(grid),
                    int(payload["paths_per_day"]),
                )
            if type(replicates) is not int or replicates < 1:
                raise ValueError("P2 replicas must be a positive integer")
            if "absolute_drift_ticks" in payload["generated_tapes"]:
                generated["absolute_drift_ticks"] = np.asarray(
                    payload["generated_tapes"]["absolute_drift_ticks"], dtype=float,
                )
        elif replicates != 1:
            raise ValueError("P1 bootstrap payload cannot contain multiple replicates")
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
        P2 explicitly changes this bootstrap estimand: independent C2
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
        depth = None if self.depth_errors is None else float(_finite_mean(self.depth_errors[days]))
        admitted = impact["measurements"]["law_identified"] is True
        return mechanism_readouts(generated, historical, impact, depth), admitted

    def resample(self, draws: np.ndarray, metrics: Sequence[str]) -> dict[str, np.ndarray]:
        """Return one pooled value per [draw, day] row.

        Legacy payloads retain P1's original-panel admission. P2 calls
        identify_law independently for each resampled complete-day panel.
        """
        admitted = None
        if self.metric_schema_version == P1_METRIC_SCHEMA:
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
            raise ValueError("available quote support requires the P2 continuous panel")
        visible = self.continuous_panel["visible"]
        if metric == "visible_minus_blind_auc":
            return np.isfinite(self.continuous_panel["joint_difference"])
        if metric == "impact_curve_mae":
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
            if metric == "impact_curve_mae":
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
            fills = _finite_mean(self.paths["child_filled"].sum(axis=-1), axis=1)
            drift = np.isfinite(self.generated["drift_ticks"]).any(axis=1)
            complete_execution = np.isfinite(day_children).all(axis=(1, 2))
            return recovery & complete_execution & (fills > 0) & drift
        if metric in ("trade_sign_acf_exp_gap", "hurst_gap"):
            field = "flow_acf" if metric == "trade_sign_acf_exp_gap" else "price_msd"
            generated = _finite_mean(self.generated[field], axis=1)
            historical = _finite_mean(self.historical[field], axis=1)
            return ((generated > 0).sum(axis=1) >= 3) & ((historical > 0).sum(axis=1) >= 3)
        if metric == "event_count_rel_gap":
            generated = _finite_mean(self.generated["events"], axis=1)
            historical = _finite_mean(self.historical["events"], axis=1)
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
