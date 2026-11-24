"""Repool market statistics on aligned day-cluster bootstrap draws."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .p1_m2_metrics import FLOW_LAGS, PRICE_LAGS, _finite_mean, _log_slope, summarize_impact
from .p1_m3_evidence import mechanism_readouts


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
    return {
        "events": float(_finite_mean(tapes["events"][days])),
        "gamma_flow": -flow_slope if flow_slope is not None else None,
        "hurst": price_slope / 2 if price_slope is not None else None,
        "drift_ticks": float(_finite_mean(drift)),
        "mean_absolute_drift_ticks": float(_finite_mean(np.abs(drift))),
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

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> MarketClusters:
        """Decode JSON null as unavailable rather than a zero physical mark."""
        depth = payload["l1_depth_errors"]
        return cls(
            paths={name: np.asarray(payload[name], dtype=float) for name in PATH_FIELDS},
            generated={name: np.asarray(payload["generated_tapes"][name], dtype=float) for name in TAPE_FIELDS},
            historical={name: np.asarray(payload["historical_tapes"][name], dtype=float) for name in TAPE_FIELDS},
            depth_errors=None if depth is None else np.asarray(depth, dtype=float),
            schedule=np.asarray(payload["schedule"], dtype=bool),
            paths_per_day=int(payload["paths_per_day"]),
            sigma_ticks=float(payload["sigma_ticks"]),
            daily_shares=float(payload["daily_shares"]),
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
        """
        generated = _pool_tapes(self.generated, days)
        historical = _pool_tapes(self.historical, days)
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
        )
        depth = None if self.depth_errors is None else float(_finite_mean(self.depth_errors[days]))
        admitted = impact["measurements"]["law_identified"] is True
        return mechanism_readouts(generated, historical, impact, depth), admitted

    def resample(self, draws: np.ndarray, metrics: Sequence[str]) -> dict[str, np.ndarray]:
        """Return one pooled value per [draw, day] row, with original C2 admission."""
        _, admitted = self._measure(np.arange(self.days), law_admitted=None)
        samples = {metric: np.full(len(draws), np.nan) for metric in metrics}
        for i, days in enumerate(draws):
            result, _ = self._measure(days, law_admitted=admitted)
            for metric, values in samples.items():
                value = result[metric]
                if value is not None:
                    values[i] = value
        return samples

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
