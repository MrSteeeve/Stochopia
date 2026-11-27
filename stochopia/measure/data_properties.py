"""Frozen model-independent structure measurements on historical session windows.

All prices are integer ticks and quantities shares. Each physical session
contributes one scalar per diagnostic, then we report equal-window median and
interquartile range. Reconstruction agreement is a separate data-quality audit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


PropertySpec = dict[str, dict[str, dict[str, float]]]
_EPSILON_VARIANCE = 1e-8
_EPSILON_IQR = 1e-6


def _value(window: Any, name: str) -> Any:
    """Allow published day slices and equivalent simple array mappings."""
    return window[name] if isinstance(window, Mapping) else getattr(window, name)


def _session_arrays(window: Any) -> Iterable[dict[str, np.ndarray]]:
    """Split a day slice into same-session windows without crossing its clock reset."""
    fields = ("action_type", "dt_ms", "rel_ticks", "volume_shares", "pre_book_ticks", "pre_book_shares")
    arrays = {name: np.asarray(_value(window, name)) for name in fields}
    count = len(arrays["action_type"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("event fields must align on an identical axis")
    if isinstance(window, Mapping):
        bounds = window.get("session_bounds", ((0, count),))
    else:
        bounds = getattr(window, "session_bounds", ((0, count),))
    for start, end in bounds:
        if not 0 <= start < end <= count:
            raise ValueError("invalid historical session bounds")
        yield {name: value[start:end] for name, value in arrays.items()}


def _sign_acf(signs: np.ndarray) -> np.ndarray:
    """Biased finite-sample autocorrelation for lags 1..32, in event time."""
    if len(signs) < 33:
        raise ValueError("z_mem requires at least 33 aggressive events per window")
    centered = signs.astype(float) - float(signs.mean())
    variance = float(centered @ centered)
    if variance <= _EPSILON_VARIANCE:
        # Constant direction has no centered variation. Its bias is measured separately.
        return np.zeros(32)
    return np.array([centered[lag:] @ centered[:-lag] / variance for lag in range(1, 33)])


def _acf_slope(acf: np.ndarray) -> float:
    """OLS slope on log(lag), log(acf), using only actually positive lags."""
    keep = acf > 0
    if keep.sum() < 2:
        return 0.0  # no estimable positive decay segment
    return float(np.polyfit(np.log(np.arange(1, 33)[keep]), np.log(acf[keep]), 1)[0])


def _flow_vr(signs: np.ndarray) -> float:
    """Variance per step of overlapping 32-event sums relative to 4-event sums."""
    def normalized_variance(length: int) -> float:
        sums = np.lib.stride_tricks.sliding_window_view(signs, length).sum(axis=1)
        return float(np.var(sums) / length)

    return normalized_variance(32) / max(normalized_variance(4), _EPSILON_VARIANCE)


def _window_properties(window: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    action = window["action_type"]
    dt = window["dt_ms"].astype(float)
    volume = window["volume_shares"].astype(float)
    ticks = window["pre_book_ticks"]
    quantities = window["pre_book_shares"].astype(float)
    if ticks.shape != (len(action), 2, 10) or quantities.shape != ticks.shape:
        raise ValueError("book arrays must be [event, bid/ask, ten levels]")
    if not np.isfinite(quantities).all() or not np.isfinite(dt).all() or (dt < 0).any():
        raise ValueError("historical quantities and waiting times must be finite")
    aggressive = (action == 2) | (action == 3)
    signs = 1 - 2 * (action[aggressive] % 2).astype(np.int64)
    acf = _sign_acf(signs)
    target_depth = quantities[aggressive, 1 - (action[aggressive] % 2), 0]
    valid_trade = target_depth > 0
    if not valid_trade.any():
        raise ValueError("no aggressive trade has positive consumed-side touch depth")

    touch = ticks[:, :, 0]
    mid_valid = (touch[:, 0] > 0) & (touch[:, 1] > touch[:, 0])
    midpoint_twice = touch.sum(axis=1)
    adjacent = mid_valid[1:] & mid_valid[:-1]
    if not adjacent.any():
        raise ValueError("no consecutive valid two-sided quote observations")
    # The last unknown post-event book is not manufactured.
    shifts = float((np.diff(midpoint_twice)[adjacent] != 0).mean())
    cancellations = action >= 4
    side = action[cancellations] % 2
    cancel_book = ticks[cancellations, side, :]
    cancel_price = cancel_book[:, :1] - (1 - 2 * side[:, None]) * window["rel_ticks"][cancellations, None]
    top_three = ((cancel_book[:, :3] == cancel_price) & (cancel_book[:, :3] > 0)).any(axis=1)
    canceled_top = float(volume[cancellations][top_three].sum())
    passive_added = float(volume[action < 2].sum())
    if passive_added <= 0:
        raise ValueError("cancel-to-limit ratio requires positive added limit quantity")
    # Cancel distance is physical ticks (not a rank token). The denominator is
    # all newly placed passive shares, as specified for this diagnostic.
    depth = quantities.sum(axis=2)
    imbalance = (depth[:, 0] - depth[:, 1]) / np.maximum(depth.sum(axis=1), 1.0)
    return {
        "z_mem": {
            "sign_acf_sum_1_32": float(acf.sum()),
            "sign_acf_log_slope": _acf_slope(acf),
            "flow_vr_32_4": _flow_vr(signs),
        },
        "z_thin": {
            "trade_to_top1_depth": float(np.median(volume[aggressive][valid_trade] / target_depth[valid_trade])),
            "midprice_shift_rate": shifts,
            "cancel_to_limit_ratio": canceled_top / passive_added,
        },
        "z_burst": {
            "zero_wait_prob": float((dt == 0).mean()),
            "interarrival_cv2": float(np.var(dt) / (float(dt.mean()) + 1e-9) ** 2),
        },
        "z_asym": {
            "book_imbalance_bias": abs(float(imbalance.mean())),
            "flow_sign_bias": abs(float(signs.mean())),
        },
    }


def measure_data_properties(windows: Iterable[Any]) -> PropertySpec:
    """Compute model-independent diagnostics on training windows only.

    [N,2,10] books are measured before each event. Buy aggression consumes
    the ask queue, and sell aggression consumes the bid queue. Every session
    contributes one diagnostic value (not one weight per event).
    """
    rows = [_window_properties(session) for window in windows for session in _session_arrays(window)]
    if not rows:
        raise ValueError("at least one same-session historical window is required")
    output = {}
    for dimension, measures in rows[0].items():
        output[dimension] = {}
        for name in measures:
            values = np.array([row[dimension][name] for row in rows])
            q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
            output[dimension][name] = {"median": float(median), "intraday_iqr": float(q75 - q25)}
    return output


def compute_property_drift(z_ref: PropertySpec, z_target: PropertySpec) -> dict[str, float]:
    """Signed median offsets in reference IQR units; overall RMS across measures."""
    drift = {}
    if z_ref.keys() != z_target.keys():
        raise ValueError("property dimensions differ")
    for dimension, measures in z_ref.items():
        if measures.keys() != z_target[dimension].keys():
            raise ValueError("property measures differ")
        for name, reference in measures.items():
            target = z_target[dimension][name]
            delta = target["median"] - reference["median"]
            drift[f"{dimension}.{name}"] = float(delta / max(reference["intraday_iqr"], _EPSILON_IQR))
    drift["total_rms"] = float(np.sqrt(np.mean(np.square(list(drift.values())))))
    return drift


def measure_reconstruction_quality(reconstructed_windows: Iterable[Any]) -> dict[str, float]:
    """Physical quality only, independent of the model-independent Z dictionary.

    Each record has observed/reconstructed_ticks and observed/reconstructed_shares
    with shape [snapshot,2,10]. Touch match means both prices agree; depth MAE
    uses raw shares, and invalid spread means a missing/crossed reconstructed touch.
    Pool by snapshot, not by file. Never infer reference quotes from reconstruction.
    """
    matches = []
    errors = []
    invalid = []
    for window in reconstructed_windows:
        actual = np.asarray(_value(window, "observed_ticks"))
        replay = np.asarray(_value(window, "reconstructed_ticks"))
        actual_depth = np.asarray(_value(window, "observed_shares"), dtype=float)
        replay_depth = np.asarray(_value(window, "reconstructed_shares"), dtype=float)
        if actual.ndim != 3 or actual.shape[1:] != (2, 10) or not all(
            x.shape == actual.shape for x in (replay, actual_depth, replay_depth)
        ):
            raise ValueError("reconstruction panel requires aligned [snapshot,2,10] arrays")
        matches.extend((actual[:, :, 0] == replay[:, :, 0]).all(axis=1))
        errors.extend(np.abs(actual_depth - replay_depth).mean(axis=(1, 2)))
        bid = replay[:, 0, 0]
        ask = replay[:, 1, 0]
        invalid.extend((bid <= 0) | (ask <= bid))
    if not matches:
        raise ValueError("no reconstruction observations")
    return {
        "top1_quote_match_rate": float(np.mean(matches)),
        "top10_depth_mae": float(np.mean(errors)),
        "invalid_spread_rate": float(np.mean(invalid)),
    }
