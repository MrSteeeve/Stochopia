"""Project P1 readouts into scalar evidence suitable for paired inference."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real

from stochopia.harness.lob_rollout import ResidualProfile


M4_MARKET_METRICS = (
    "lob_impact_score",
    "end_over_peak_ratio",
    "free_abs_drift_ticks",
    "trade_sign_acf_exp_gap",
    "hurst_gap",
    "visible_over_blind_ratio",
    "impact_exponent_gap",
    "event_count_rel_gap",
    "l1_depth_mae",
)


def _absolute_gap(left: object, right: object) -> float | None:
    if isinstance(left, Real) and isinstance(right, Real) and isfinite(left) and isfinite(right):
        return abs(float(left) - float(right))
    return None


def mechanism_readouts(
    generated: Mapping[str, object], historical: Mapping[str, object], impact: Mapping[str, object],
    l1_depth_mae: float | None,
) -> dict[str, float | None]:
    """Project pooled M4 readouts using the same units for original and bootstrap samples.

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
    return {
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


def compact_evidence(
    profile: ResidualProfile, parameters: int, state_dim: int, *, exhausted_windows: int | None = None,
    train_seed_nll: Mapping[int, float] | None = None, l1_depth_mae: float | None = None,
) -> dict[str, object]:
    def residual(key: str) -> float | None:
        generated = profile.generated[key]
        observed = profile.historical[key]
        return None if generated is None or observed is None else float(generated - observed)

    impact = profile.impact
    metrics = impact["measurements"]
    score = impact["score"]
    return {
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
