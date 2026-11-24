"""Project P1 readouts into scalar evidence suitable for paired inference."""

from __future__ import annotations

from stochopia.harness.lob_rollout import ResidualProfile


def compact_evidence(profile: ResidualProfile, parameters: int, state_dim: int) -> dict[str, float | int | None]:
    def residual(key: str) -> float | None:
        generated = profile.generated[key]
        observed = profile.historical[key]
        return None if generated is None or observed is None else float(generated - observed)

    impact = profile.impact
    metrics = impact["measurements"]
    return {
        "parameters": parameters,
        "state_dim": state_dim,
        "val_nll_total": profile.model_nll["total"],
        "lob_impact_score": impact["score"]["score"],
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
