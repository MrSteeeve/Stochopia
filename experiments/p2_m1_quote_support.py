"""Support audit for P2's available-quote continuous impact panel."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


QUOTE_PANEL_FIELDS = (
    "visible", "joint_difference", "visible_counts", "joint_counts",
    "execution_peak", "execution_peak_counts",
)


def decode_quote_panel(
    payload: Mapping[str, object], days: int, marks: int, paths_per_day: int,
) -> dict[str, np.ndarray]:
    """Decode [day, grid] means/counts and [day] exact execution marks.

    Each physical mark's count is the number of available Monte Carlo
    replica/seed/session pairs; a finite mean is allowed iff at least one
    valid pair was observed. Joint differences require all four arms.
    """
    panel = {name: np.asarray(payload[name], dtype=float) for name in QUOTE_PANEL_FIELDS}
    for field, count_field in (
        ("visible", "visible_counts"), ("joint_difference", "joint_counts"),
        ("execution_peak", "execution_peak_counts"),
    ):
        values = panel[field]
        counts = panel[count_field]
        shape = (days,) if field == "execution_peak" else (days, marks)
        if values.shape != shape or counts.shape != shape:
            raise ValueError("available quote panel must align day and clock dimensions")
        if (not np.isfinite(counts).all() or (counts < 0).any() or (counts > paths_per_day).any()
                or (counts != np.floor(counts)).any()):
            raise ValueError("invalid available-quote replica counts")
        if not np.array_equal(np.isfinite(values), counts > 0):
            raise ValueError("finite available quote means must correspond to positive support counts")
    if (panel["joint_counts"] > panel["visible_counts"]).any():
        raise ValueError("joint visible/blind support cannot exceed visible support")
    return panel


def quote_support(panel: Mapping[str, np.ndarray], paths_per_day: int) -> dict[str, float | int]:
    """Describe observed support across [day,grid]; no count is a bootstrap unit."""
    n_days = len(panel["execution_peak"])
    expected = n_days * paths_per_day
    visible = panel["visible_counts"].sum(axis=0) / expected
    joint = panel["joint_counts"].sum(axis=0) / expected
    return {
        "days": n_days,
        "expected_pairs_per_grid_mark": expected,
        "visible_min_fraction": float(visible.min()),
        "visible_mean_fraction": float(visible.mean()),
        "joint_min_fraction": float(joint.min()),
        "joint_mean_fraction": float(joint.mean()),
        "execution_peak_fraction": float(panel["execution_peak_counts"].sum() / expected),
        "visible_complete_days": int(np.isfinite(panel["visible"]).all(axis=1).sum()),
        "joint_complete_days": int(np.isfinite(panel["joint_difference"]).all(axis=1).sum()),
    }


def common_quote_panels(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Retain common day/time supports across models for paired curve inference.

    The inputs are already within-window replica means; counts describe
    each model's observed replica support, not a claim of cross-model
    replicate-level completeness. A day/time with no mark in either model
    contributes to neither curve mean on that mark.
    """
    first = {}
    second = {}
    for field, count_field in (
        ("visible", "visible_counts"), ("joint_difference", "joint_counts"),
        ("execution_peak", "execution_peak_counts"),
    ):
        shared = np.isfinite(left[field]) & np.isfinite(right[field])
        first[field] = np.where(shared, left[field], np.nan)
        second[field] = np.where(shared, right[field], np.nan)
        first[count_field] = np.where(shared, left[count_field], 0)
        second[count_field] = np.where(shared, right[count_field], 0)
    return first, second
