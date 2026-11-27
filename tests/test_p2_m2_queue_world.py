"""Generation-only manipulation and true matching tests; no model fitting."""

from __future__ import annotations

from importlib.util import find_spec
from sys import version_info

import numpy as np
import pytest

from stochopia.measure.data_properties import measure_data_properties
from stochopia.neural.lob_interface import make_batch, valid_action_mask, valid_rel_tick_mask
from stochopia.neural.lob_training import encode_day
from stochopia.sim.queue_reactive_world import (
    GRID_MS, counterfactual_truth, generate_dataset, standard_worlds, tier0_manipulation_check,
)


def test_standard_queue_manipulations_are_measured_and_book_targets_legal() -> None:
    recipes = standard_worlds()
    if version_info >= (3, 13) and find_spec("jaxlob") is None:
        with pytest.raises(ImportError, match="requires the harness-lob extra"):
            generate_dataset(recipes["W1_thick_short"], train_days=1, eval_days=0)
        return
    summaries = {}
    for name, recipe in recipes.items():
        dataset = generate_dataset(recipe, eval_days=1)
        summaries[name] = measure_data_properties(dataset.train)
        day = dataset.evaluation[0]
        assert day.session_bounds == ((0, 256), (256, 512))
        assert day.trading_date > dataset.train[-1].trading_date
        assert tuple(window.start_index for window in dataset.windows) == (128, 384)
        assert tuple(window.window_id for window in dataset.windows) == (0, 1)
        assert (day.spread_ticks > 0).all()
        encoded = encode_day(day)
        batch = make_batch(encoded.ticks.reshape(2, 256, 2, 10), encoded.shares.reshape(2, 256, 2, 10),
                           encoded.tokens.reshape(2, 256, 4))
        actions = batch.targets[..., 1]
        action_mask = valid_action_mask(batch.observation.book_ticks, batch.observation.book_shares)
        relative_mask = valid_rel_tick_mask(batch.observation, actions)
        assert action_mask.gather(-1, actions[..., None]).all()
        assert relative_mask.gather(-1, batch.targets[..., 2, None]).all()

    def median(world: str, dimension: str, metric: str) -> float:
        return summaries[world][dimension][metric]["median"]

    for short, long in (("W1_thick_short", "W2_thick_long"), ("W3_thin_short", "W4_thin_long")):
        assert median(long, "z_mem", "sign_acf_sum_1_32") - median(short, "z_mem", "sign_acf_sum_1_32") > 2
        assert median(long, "z_mem", "flow_vr_32_4") > median(short, "z_mem", "flow_vr_32_4")
    for name in recipes:
        expected_thick = "thick" in name
        shift = median(name, "z_thin", "midprice_shift_rate")
        assert shift < 0.05 if expected_thick else shift > 0.2
    assert median("W6_thin_hawkes", "z_burst", "interarrival_cv2") > median("W3_thin_short", "z_burst", "interarrival_cv2")
    assert median("W6_thin_hawkes", "z_burst", "zero_wait_prob") > median("W3_thin_short", "z_burst", "zero_wait_prob")


def test_exact_generator_crn_truth_is_reproducible_on_shared_history() -> None:
    recipes = standard_worlds()
    if version_info >= (3, 13) and find_spec("jaxlob") is None:
        with pytest.raises(ImportError, match="requires the harness-lob extra"):
            generate_dataset(recipes["W3_thin_short"], train_days=1, eval_days=1)
        return
    poisson = generate_dataset(recipes["W3_thin_short"], train_days=1, eval_days=1)
    hawkes = generate_dataset(recipes["W6_thin_hawkes"], train_days=1, eval_days=1)
    # Holding all mark random coordinates fixed isolates the arrival-clock knob.
    np.testing.assert_array_equal(poisson.evaluation[0].action_type, hawkes.evaluation[0].action_type)
    np.testing.assert_array_equal(poisson.evaluation[0].pre_book_ticks, hawkes.evaluation[0].pre_book_ticks)
    assert not np.array_equal(poisson.evaluation[0].dt_ms, hawkes.evaluation[0].dt_ms)
    first, support = counterfactual_truth(poisson, num_replicates=2, return_support=True)
    second = counterfactual_truth(poisson, num_replicates=2)
    np.testing.assert_array_equal(first, second)
    assert first.shape == support.shape == (2, len(GRID_MS))
    assert np.isfinite(first).all()
    assert (support > 0).all()
    assert (first[:, 0] == 0.0).all()
    assert np.max(np.abs(first)) > 0


def test_tier0_flow_manipulation_does_not_change_independent_price_kernel_truth() -> None:
    rows = tier0_manipulation_check()
    short, long = rows["short"], rows["long"]
    assert long["sign_acf_sum_1_32"]["median"] - short["sign_acf_sum_1_32"]["median"] > 2
    np.testing.assert_allclose(short["fixed_kernel_truth"], long["fixed_kernel_truth"], atol=1e-12)
    assert not np.allclose(short["independent_kernel_truth"], long["independent_kernel_truth"])
