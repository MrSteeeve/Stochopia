"""Impact-law estimators for metaorder injection experiments.

Inputs are response paths ``[n_units, n_events]`` on the merged event clock of
an injection schedule (see :func:`stochopia.sim.world.injection_schedule`),
one row per independent unit (a trading day). Existence is decided before
magnitude: an exponent is reported only when the peak response is resolved
and the fitted exponent is interior to the search grid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import fsum, isclose, isfinite, log2
from numbers import Real
from statistics import mean
from typing import TypeGuard

import numpy as np

from stochopia.measure.continuous_impact import (
    C2_PASS_METRIC,
    CONTINUOUS_IMPACT_METRICS,
    IDENTIFIED_POWERLAW,
    NO_RESPONSE,
    NOISY_UNIDENTIFIED,
    P1_METRIC_SCHEMA,
    P2_METRIC_SCHEMA,
    RESPONSE_EPSILON_TICKS,
    continuous_impact_readouts,
    counterfactual_impact_mae,
    impact_identifiability_state,
    reference_impact_curve,
)

DELTA_GRID = np.round(np.arange(0.05, 1.5 + 1e-9, 0.01), 2)


def antisymmetrise(buy: np.ndarray, sell: np.ndarray) -> np.ndarray:
    """Combine buy and sell arms; direction-independent drift cancels."""
    return 0.5 * (np.asarray(buy, dtype=float) - np.asarray(sell, dtype=float))


def child_impact(paths: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """Response right after each injected child, ``[n_units, n_children]``."""
    return np.asarray(paths, dtype=float)[..., np.asarray(schedule, dtype=bool)]


def fit_power_law(q: np.ndarray, impact: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of ``impact = a * q ** delta`` on raw signed values.

    For each delta on the grid the best amplitude is closed-form, so the fit
    needs no logarithms and no sign selection. Returns ``(delta, a)``.
    """
    q = np.asarray(q, dtype=float)
    impact = np.asarray(impact, dtype=float)
    basis = q[None, :] ** DELTA_GRID[:, None]
    amp = (basis * impact).sum(axis=1) / (basis**2).sum(axis=1)
    misfit = ((impact[None, :] - amp[:, None] * basis) ** 2).sum(axis=1)
    best = int(np.argmin(misfit))
    return float(DELTA_GRID[best]), float(amp[best])


@dataclass(frozen=True)
class ImpactLaw:
    identified: bool
    reason: str
    delta: float  # nan unless identified
    amplitude: float  # nan unless identified
    peak: float  # mean response after the last child
    peak_se: float


def identify_law(impact: np.ndarray, *, min_peak_se: float = 3.0) -> ImpactLaw:
    """Decide whether an impact law exists and, if so, fit it.

    ``impact`` is ``[n_units, n_children]``; participation is measured in
    children executed, ``q = 1..n_children``. The law is fitted on the unit
    mean, which is the pooled least-squares fit because every unit shares q.
    """
    impact = np.asarray(impact, dtype=float)
    if impact.ndim != 2 or impact.shape[0] < 2 or impact.shape[1] < 3:
        raise ValueError("impact must be [n_units >= 2, n_children >= 3]")
    n_units, n_children = impact.shape
    last = impact[:, -1]
    peak = float(last.mean())
    peak_se = float(last.std(ddof=1) / np.sqrt(n_units))
    nan = float("nan")
    if not peak > min_peak_se * peak_se:
        return ImpactLaw(False, "peak not resolved", nan, nan, peak, peak_se)
    delta, amp = fit_power_law(np.arange(1, n_children + 1), impact.mean(axis=0))
    if delta in (DELTA_GRID[0], DELTA_GRID[-1]):
        return ImpactLaw(False, "exponent on grid bound", nan, nan, peak, peak_se)
    return ImpactLaw(True, "ok", delta, amp, peak, peak_se)


def delta_bootstrap_ci(
    impact: np.ndarray, *, seed: int, n_resamples: int = 1000, confidence: float = 0.95
) -> tuple[float, float]:
    """Percentile CI for delta, resampling units (days) with replacement."""
    impact = np.asarray(impact, dtype=float)
    n_units, n_children = impact.shape
    q = np.arange(1, n_children + 1)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = impact[rng.integers(0, n_units, size=n_units)]
        deltas[i] = fit_power_law(q, sample.mean(axis=0))[0]
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(deltas, [tail, 1.0 - tail])
    return float(lo), float(hi)


@dataclass(frozen=True)
class Relaxation:
    end_over_peak: float
    slope: float  # change per execution duration, as a fraction of the peak


def relaxation(mean_path: np.ndarray, schedule: np.ndarray, *, tail: int = 20) -> Relaxation:
    """Post-execution behaviour of a mean response path.

    The peak is the response right after the last child. The end level is the
    mean over the final ``tail`` events. The slope is a least-squares line
    through the post-execution path, per execution duration.
    """
    mean_path = np.asarray(mean_path, dtype=float)
    schedule = np.asarray(schedule, dtype=bool)
    children = np.flatnonzero(schedule)
    last = int(children[-1])
    post = mean_path[last:]
    if post.size < max(tail, 3):
        raise ValueError("schedule has no post-execution window to measure")
    peak = float(mean_path[last])
    duration = last + 1
    x = np.arange(post.size) / duration
    slope = float(np.polyfit(x, post, 1)[0] / peak)
    return Relaxation(end_over_peak=float(post[-tail:].mean() / peak), slope=slope)


# --- LOB-Impact measurements and claim ladder ---


DEFAULT_IMPACT_WEIGHTS = (0.40, 0.25, 0.35)
_AMPLITUDE_BAND = (0.5, 1.0)
_EXPONENT_BAND = (0.4, 0.6)
_RELAXATION_BAND = (0.5, 2 / 3)
_DRIFT_LIMIT_TICKS = 20.0


class ImpactClaimStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class ImpactClaim:
    status: ImpactClaimStatus
    reason: str


@dataclass(frozen=True, kw_only=True)
class ImpactMetrics:
    """Same-stock LOB-Impact readouts under one frozen injection protocol.

    ``y`` is volatility-normalized amplitude, not the fitted kernel
    coefficient. The peak interval uses ticks and independent days.
    ``relaxation_slope`` is a fraction of peak per execution duration.
    ``visible_to_blind_ratio`` permits a published ratio when raw peaks
    are unavailable. The observed peaks take precedence when supplied,
    with a positive blind peak required to form the ratio.
    """

    y: float | None = None
    delta: float | None = None
    end_over_peak: float | None = None
    idle_drift_ticks: float | None = None
    law_identified: bool | None = None
    gamma_flow: float | None = None
    hurst: float | None = None
    relaxation_slope: float | None = None
    visible_peak_ticks: float | None = None
    blind_peak_ticks: float | None = None
    visible_to_blind_ratio: float | None = None
    peak_interval_ticks: tuple[float, float] | None = None
    is_ow_baseline: bool = False


@dataclass(frozen=True, kw_only=True)
class ImpactReference:
    """Same-stock replay, same-schedule OW kernel and preregistered zero band."""

    gamma_flow: float | None = None
    hurst: float | None = None
    idle_drift_ticks: float | None = None
    ow_end_over_peak: float | None = None
    near_zero_peak_band_ticks: tuple[float, float] | None = None


def _is_finite_number(value: object) -> TypeGuard[float]:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def _visible_to_blind_ratio(metrics: ImpactMetrics) -> float | None:
    visible = metrics.visible_peak_ticks
    blind = metrics.blind_peak_ticks
    if _is_finite_number(visible) and _is_finite_number(blind):
        if blind <= 0:
            return None
        ratio = visible / blind
    else:
        ratio = metrics.visible_to_blind_ratio
    return float(ratio) if _is_finite_number(ratio) else None


def _matches_replay(measured: float | None, replay: float | None, tolerance: float) -> bool | None:
    if not _is_finite_number(measured) or not _is_finite_number(replay):
        return None
    return replay - tolerance <= measured <= replay + tolerance


def _inside_band(value: float | None, band: tuple[float, float]) -> bool | None:
    return band[0] <= value <= band[1] if _is_finite_number(value) else None


def _claim(*criteria: tuple[str, bool | None]) -> ImpactClaim:
    failed = [name for name, passed in criteria if passed is False]
    if failed:
        return ImpactClaim(ImpactClaimStatus.FAIL, "criterion violated: " + ", ".join(failed))
    missing = [name for name, passed in criteria if passed is None]
    if missing:
        return ImpactClaim(ImpactClaimStatus.INCONCLUSIVE, "measurement unavailable: " + ", ".join(missing))
    return ImpactClaim(ImpactClaimStatus.PASS, "criterion satisfied")


def _claim_existence(metrics: ImpactMetrics, reference: ImpactReference) -> ImpactClaim:
    if metrics.law_identified is True:
        return ImpactClaim(ImpactClaimStatus.PASS, "impact law identified")
    band = reference.near_zero_peak_band_ticks
    interval = metrics.peak_interval_ticks
    if band is not None and interval is not None:
        if not all(_is_finite_number(bound) for bound in band) or not band[0] <= 0 <= band[1]:
            raise ValueError("the near-zero peak band must be finite and contain zero")
        if not all(_is_finite_number(endpoint) for endpoint in interval) or interval[0] > interval[1]:
            raise ValueError("the peak interval must be finite and ordered")
        if band[0] <= interval[0] and interval[1] <= band[1]:
            return ImpactClaim(ImpactClaimStatus.FAIL, "peak interval establishes a near zero response")
    return ImpactClaim(ImpactClaimStatus.INCONCLUSIVE, "impact law unidentified with insufficient evidence of near zero")


def _claim_relaxation(metrics: ImpactMetrics, reference: ImpactReference) -> ImpactClaim:
    if metrics.is_ow_baseline:
        return ImpactClaim(ImpactClaimStatus.INCONCLUSIVE, "OW serves as the relaxation reference")
    raw_ratio = metrics.end_over_peak
    ow_ratio = reference.ow_end_over_peak
    slope = metrics.relaxation_slope
    drift = metrics.idle_drift_ticks
    below_ow = raw_ratio <= ow_ratio if _is_finite_number(raw_ratio) and _is_finite_number(ow_ratio) else None
    return _claim(
        ("end-to-peak ratio at or below the OW reference", below_ow),
        ("nonpositive relaxation slope", slope <= 0 if _is_finite_number(slope) else None),
        ("absolute idle drift below 20 ticks", abs(drift) < _DRIFT_LIMIT_TICKS if _is_finite_number(drift) else None),
    )


def _claims(metrics: ImpactMetrics, reference: ImpactReference, visible_blind_ratio: float | None) -> dict[str, ImpactClaim]:
    """Apply the paper's §2.1 claim criteria to the raw readouts."""
    claims = {
        "C1": _claim(
            ("flow exponent versus replay", _matches_replay(metrics.gamma_flow, reference.gamma_flow, 0.10)),
            ("Hurst exponent versus replay", _matches_replay(metrics.hurst, reference.hurst, 0.02)),
            ("idle drift versus replay", _matches_replay(metrics.idle_drift_ticks, reference.idle_drift_ticks, 5.0)),
        ),
        "C2": _claim_existence(metrics, reference),
        "C3": _claim(("visible-to-blind peak ratio of at least three",
                      visible_blind_ratio >= 3.0 if visible_blind_ratio is not None else None)),
    }
    if metrics.law_identified is True:
        claims["C4"] = _claim(
            ("square-root exponent", _inside_band(metrics.delta, _EXPONENT_BAND)),
            ("volatility-normalized amplitude", _inside_band(metrics.y, _AMPLITUDE_BAND)),
        )
        claims["C5"] = _claim_relaxation(metrics, reference)
    else:
        unavailable = ImpactClaim(ImpactClaimStatus.INCONCLUSIVE, "impact law not identified")
        claims["C4"] = unavailable
        claims["C5"] = unavailable
    return claims


# --- Composite scores and panel coverage ---


@dataclass(frozen=True, kw_only=True)
class ImpactScore:
    claims: dict[str, ImpactClaim]
    score: float | None = None
    weighted_error: float | None = None
    amplitude_error: float | None = None
    exponent_error: float | None = None
    relaxation_error: float | None = None
    scored_end_over_peak: float | None = None
    visible_to_blind_ratio: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ImpactModelScore:
    mean_score: float | None
    coverage: float
    per_symbol: dict[str, ImpactScore]


@dataclass(frozen=True)
class ImpactPanelScore:
    common_symbols: tuple[str, ...]
    common_coverage: float
    models: dict[str, ImpactModelScore]


def interval_distance(value: float, lower: float, upper: float) -> float:
    """Return zero inside [lower, upper], otherwise distance to its nearer endpoint."""
    if lower > upper:
        raise ValueError("the interval must be ordered")
    return max(lower - value, 0.0, value - upper)


def _validate_weights(weights: tuple[float, float, float]) -> None:
    if len(weights) != 3 or not all(_is_finite_number(weight) and weight >= 0 for weight in weights):
        raise ValueError("the score requires three finite nonnegative weights")
    if not isclose(fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("the score weights must sum to one")


def _score_relaxation(metrics: ImpactMetrics, drift_correction: bool) -> float | None:
    ratio = metrics.end_over_peak
    if not _is_finite_number(ratio):
        return None
    if not drift_correction:
        return float(ratio)
    drift = metrics.idle_drift_ticks
    if not _is_finite_number(drift):
        return None
    return max(1.0, ratio) if abs(drift) >= _DRIFT_LIMIT_TICKS else float(ratio)


def _score_impact(
    metrics: ImpactMetrics, reference: ImpactReference, weights: tuple[float, float, float], drift_correction: bool,
) -> ImpactScore:
    ratio = _visible_to_blind_ratio(metrics)
    claims = _claims(metrics, reference, ratio)
    if metrics.law_identified is not True:
        return ImpactScore(claims=claims, visible_to_blind_ratio=ratio, reason="impact law not identified")

    y = metrics.y
    delta = metrics.delta
    scored_relaxation = _score_relaxation(metrics, drift_correction)
    e_y = interval_distance(log2(y), -1.0, 0.0) if _is_finite_number(y) and y > 0 else None
    e_delta = interval_distance(delta, *_EXPONENT_BAND) / 0.10 if _is_finite_number(delta) else None
    e_relaxation = None
    if scored_relaxation is not None:
        e_relaxation = interval_distance(scored_relaxation, *_RELAXATION_BAND) / (1 / 3)

    error = score = reason = None
    if e_y is not None and e_delta is not None and e_relaxation is not None:
        error = fsum((weights[0] * e_y, weights[1] * e_delta, weights[2] * e_relaxation))
        score = 100.0 / (1.0 + error)
    else:
        missing = []
        if e_y is None:
            missing.append("positive volatility-normalized amplitude")
        if e_delta is None:
            missing.append("impact exponent")
        if e_relaxation is None:
            missing.append("end-to-peak ratio with the requested drift treatment")
        reason = "unable to score: " + ", ".join(missing)
    return ImpactScore(
        claims=claims, score=score, weighted_error=error, amplitude_error=e_y, exponent_error=e_delta,
        relaxation_error=e_relaxation, scored_end_over_peak=scored_relaxation, visible_to_blind_ratio=ratio, reason=reason,
    )


def score_impact(
    metrics: ImpactMetrics,
    reference: ImpactReference = ImpactReference(),
    *,
    weights: tuple[float, float, float] = DEFAULT_IMPACT_WEIGHTS,
    drift_correction: bool = True,
) -> ImpactScore:
    """Score one stock on 100 / (1 + weighted interval errors), before rounding.

    The amplitude error uses log2(Y), the power-law error divides the
    exponent distance by 0.10, and the relaxation error divides the
    end-over-peak distance by 1/3. At 20 ticks of absolute idle drift,
    correction floors the ratio at 1.0. The C1–C5 claims use raw metrics.
    """
    _validate_weights(weights)
    return _score_impact(metrics, reference, weights, drift_correction)


def score_impact_panel(
    models: Mapping[str, Mapping[str, ImpactMetrics | None]],
    expected_symbols: Sequence[str],
    *,
    references: Mapping[str, ImpactReference] | None = None,
    weights: tuple[float, float, float] = DEFAULT_IMPACT_WEIGHTS,
    drift_correction: bool = True,
) -> ImpactPanelScore:
    """Take each model's mean on the intersection of scoreable registered symbols.

    Each model's coverage counts its independently scoreable symbols over
    the registered cohort. Common coverage uses the same denominator and
    the intersection as its numerator. Disjoint models have no mean score.
    """
    _validate_weights(weights)
    symbols = tuple(expected_symbols)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("the expected cohort must name one or more distinct symbols")
    references = references if references is not None else {}
    default_reference = ImpactReference()
    scored: dict[str, dict[str, ImpactScore]] = {}
    complete_scores: dict[str, dict[str, float]] = {}
    common = set(symbols) if models else set()
    for model, observations in models.items():
        per_symbol = {}
        scoreable = {}
        for symbol in symbols:
            metrics = observations.get(symbol)
            if metrics is None:
                missing = ImpactClaim(ImpactClaimStatus.INCONCLUSIVE, "stock measurement missing")
                claims = {claim: missing for claim in ("C1", "C2", "C3", "C4", "C5")}
                result = ImpactScore(claims=claims, reason=missing.reason)
            else:
                reference = references.get(symbol, default_reference)
                result = _score_impact(metrics, reference, weights, drift_correction)
            per_symbol[symbol] = result
            if result.score is not None:
                scoreable[symbol] = result.score
        scored[model] = per_symbol
        complete_scores[model] = scoreable
        common.intersection_update(scoreable)

    common_symbols = tuple(symbol for symbol in symbols if symbol in common)
    by_model = {}
    for model, per_symbol in scored.items():
        model_scores = complete_scores[model]
        mean_score = mean(model_scores[symbol] for symbol in common_symbols) if common_symbols else None
        by_model[model] = ImpactModelScore(mean_score, len(model_scores) / len(symbols), per_symbol)
    return ImpactPanelScore(common_symbols, len(common_symbols) / len(symbols), by_model)
