"""Hand-computable model-independent data diagnostics and independent quality."""

from __future__ import annotations

import numpy as np
import pytest

from stochopia.measure.data_properties import (
    compute_property_drift, measure_data_properties, measure_reconstruction_quality,
)


def _window() -> dict[str, np.ndarray]:
    # 40 alternating aggressive orders, then one top-three cancel and one add.
    prices = np.tile(np.array([[100 - i for i in range(10)], [101 + i for i in range(10)]]), (42, 1, 1))
    prices[21:] += 1
    quantity = np.full((42, 2, 10), 100)
    quantity[:, 0] = 200
    return {
        "action_type": np.array([2, 3] * 20 + [4, 0]),
        "dt_ms": np.array([0, 2] * 21),
        "volume_shares": np.array([50] * 40 + [100, 200]),
        "rel_ticks": np.array([0] * 40 + [2, 0]),
        "pre_book_ticks": prices,
        "pre_book_shares": quantity,
    }


def test_window_statistics_use_physical_depth_clock_and_finite_negative_acf() -> None:
    result = measure_data_properties([_window()])
    scalars = {
        ("z_mem", "sign_acf_sum_1_32"): -0.4,
        ("z_mem", "flow_vr_32_4"): 0.0,
        ("z_thin", "trade_to_top1_depth"): 0.375,
        ("z_thin", "midprice_shift_rate"): 1 / 41,
        ("z_thin", "cancel_to_limit_ratio"): 0.5,
        ("z_burst", "zero_wait_prob"): 0.5,
        ("z_burst", "interarrival_cv2"): 1.0,
        ("z_asym", "book_imbalance_bias"): 1 / 3,
        ("z_asym", "flow_sign_bias"): 0.0,
    }
    for (dimension, metric), expected in scalars.items():
        assert result[dimension][metric]["median"] == pytest.approx(expected)
        assert result[dimension][metric]["intraday_iqr"] == 0.0
    assert np.isfinite(result["z_mem"]["sign_acf_log_slope"]["median"])
    second = _window()
    second["dt_ms"] = np.full(42, 2)
    pooled = measure_data_properties([_window(), second])
    assert pooled["z_burst"]["zero_wait_prob"] == {"median": 0.25, "intraday_iqr": 0.25}
    drift = compute_property_drift(pooled, result)
    assert drift["z_burst.zero_wait_prob"] == 1.0
    assert compute_property_drift(result, result)["total_rms"] == 0.0


def test_reconstruction_quality_is_not_a_data_property() -> None:
    window = _window()
    actual = window["pre_book_ticks"][:2]
    replay = actual.copy()
    replay[1, 1, 0] = 99
    shares = window["pre_book_shares"][:2]
    reconstructed_shares = shares.copy()
    reconstructed_shares[0] += 20
    quality = measure_reconstruction_quality([{
        "observed_ticks": actual,
        "reconstructed_ticks": replay,
        "observed_shares": shares,
        "reconstructed_shares": reconstructed_shares,
    }])
    assert quality == {"top1_quote_match_rate": 0.5, "top10_depth_mae": 10.0, "invalid_spread_rate": 0.5}
    assert not set(quality).intersection(measure_data_properties([window]))
