"""Focused Task 0 regression: public sampling, finite CRN, reset and readout contracts."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.p2_m1_rollout_eval import exhausted_replicate_rate
from stochopia.harness.lob_rollout import RolloutArm, RolloutWindow
from stochopia.harness.lob_rollout_batch import (
    NonfiniteRolloutProbability, _sample_uniforms, rollout_batch,
)
from stochopia.measure.continuous_impact import (
    continuous_impact_readouts, counterfactual_impact_mae, reference_impact_curve,
)
from stochopia.neural.lob_interface import (
    LOBBatch, LOBWorldModel, StepObservation, _probe, verify_world_model_contract, world_model_loss,
)
from stochopia.neural.lob_training import (
    compute_memory_horizon_profile, evaluate_world_model_nll, world_model_nll_events,
)
from stochopia.neural.primitives import SeedGRULOBModel, parallel_affine_scan
from test_lob_rollout import _day
from test_p2_m1_batched_rollout import _no_jax_on_new_python, _replica
from test_p2_m1_stateless_nll import CurrentOnly


def test_public_only_candidate_contract_and_batched_rollouts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Remove even the base compatibility alias to prove no private dependency.
    monkeypatch.delattr(LOBWorldModel, "_logits")
    model = CurrentOnly()
    verify_world_model_contract(model)
    assert not hasattr(model, "_logits")
    day = _day()
    windows = [RolloutWindow(day, 2, 0, 8), RolloutWindow(replace(day, trading_date=20170104), 2, 0, 8)]
    kwargs = dict(child_times_ms=(10,), child_shares=90, grid_ms=(0, 10, 20),
                  seed=1, context_len=2, num_replicates=2)
    if _no_jax_on_new_python():
        with pytest.raises(ImportError, match="requires the harness-lob extra"):
            rollout_batch(model, windows, **kwargs)
        return
    result = rollout_batch(model, windows, **kwargs)
    assert len(result) == 2
    assert all(len(window) == 2 for window in result)
    assert all(set(replica) == set(RolloutArm) for window in result for replica in window)


@pytest.mark.parametrize("value", [-torch.inf, torch.nan])
def test_invalid_probability_mass_has_named_failure(value: float) -> None:
    model = CurrentOnly()
    with torch.no_grad():
        model.heads[0].weight.zero_()
        model.heads[0].bias.fill_(value)
    batch = _probe(torch.device("cpu"), 103, 2)
    sampler, _ = model.forward_step(batch.observation.at((slice(None), 0)))
    with pytest.raises(NonfiniteRolloutProbability, match="head 0"):
        _sample_uniforms(sampler, torch.full((2, 4), 0.5, dtype=torch.float64))


@pytest.mark.parametrize("mode", ["static", "batch", "full"])
def test_equal_batch_time_scan_and_static_matrix_gate(mode: str) -> None:
    rng = torch.Generator().manual_seed(7)
    inputs = torch.randn((8, 8, 10, 6), generator=rng, dtype=torch.float64)
    shape = {"static": (10, 6), "batch": (8, 1, 10, 6), "full": (8, 8, 10, 6)}[mode]
    gates = torch.rand(shape, generator=rng, dtype=torch.float64, requires_grad=True)
    actual = parallel_affine_scan(gates, inputs, chunk_size=3)
    expanded = torch.broadcast_to(gates, inputs.shape)
    carry = torch.zeros_like(inputs[:, 0])
    expected = []
    for t in range(inputs.size(1)):
        carry = expanded[:, t] * carry + inputs[:, t]
        expected.append(carry)
    expected_tensor = torch.stack(expected, dim=1)
    torch.testing.assert_close(actual, expected_tensor)
    actual_grad = torch.autograd.grad(actual.sum(), gates)[0]
    expected_grad = torch.autograd.grad(expected_tensor.sum(), gates)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    with pytest.raises(ValueError, match="ambiguous gates"):
        parallel_affine_scan(torch.ones(8, 10, 6), inputs)


def test_reset_horizons_preserve_original_support_and_match_explicit_blocks() -> None:
    torch.manual_seed(33)
    model = SeedGRULOBModel(state_dim=16)
    probe = _probe(torch.device("cpu"), 3, 2)
    targets = probe.targets.repeat(1, 8, 1)[:, :67]
    fields = []
    for field in probe.observation:
        if field is None:
            fields.append(torch.ones(2, 67, 2, dtype=torch.bool))
        else:
            copies = [1, 8] + [1] * (field.ndim - 2)
            fields.append(field.repeat(*copies)[:, :67])
    batch = LOBBatch(StepObservation(*fields), targets)
    supervised = slice(5, 64)
    event_count = targets[:, supervised].shape[0] * targets[:, supervised].shape[1]
    stateful = evaluate_world_model_nll(model, [batch], supervised=supervised)["total"]
    profile = compute_memory_horizon_profile(model, [batch], supervised=supervised)
    assert set(profile) == {1, 4, 16, 64}
    for interval in profile:
        actual = world_model_nll_events(model, batch, reset_interval=interval)
        with torch.no_grad():
            explicit = []
            for start in range(0, targets.shape[1], interval):
                block = slice(start, start + interval)
                piece = LOBBatch(batch.observation.at((slice(None), block)), targets[:, block])
                explicit.append(world_model_loss(model.forward_sequence(piece), piece.targets, reduction="none"))
        for i, term in enumerate(actual):
            expected = torch.cat([loss[i] for loss in explicit], dim=1)
            torch.testing.assert_close(term, expected, atol=2e-6, rtol=0)
            assert term[:, supervised].numel() == event_count
        expected_total = sum(float(head[:, supervised].double().sum()) for head in actual[1:]) / event_count
        assert profile[interval] == pytest.approx(expected_total - stateful, abs=1e-12)
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    torch.testing.assert_close(world_model_nll_events(model, batch, stateless=True).total,
                               world_model_nll_events(model, batch, reset_interval=1).total)


def test_stylized_reference_legacy_and_counterfactual_are_distinct() -> None:
    grid = np.array([0., 2., 4., 8.])
    kwargs = dict(scale_ticks=8., participation=0.25)
    transient = reference_impact_curve(grid, 4., **kwargs, permanent_fraction=0.)
    stylized = reference_impact_curve(grid, 4., **kwargs)
    np.testing.assert_allclose(stylized[:3], transient[:3])
    assert stylized[3] == pytest.approx(4. * (0.5 + 0.5 * (np.sqrt(2.) - 1.)))
    reads = continuous_impact_readouts(visible_curve_ticks=stylized, blind_curve_ticks=None,
                                       grid_ms=grid, schedule=np.array([0, 1, 1, 0], dtype=bool),
                                       c2_pass=False, execution_peak_ticks=4., **kwargs)
    assert reads["stylized_impact_mae"] == 0.
    assert reads["impact_curve_mae"] > 0.
    assert reads["reference_curve_ticks"] == list(transient)
    truth = stylized + np.array([0, -1, -2, 3])
    assert counterfactual_impact_mae(stylized, truth) == 1.5
    with pytest.raises(ValueError, match="aligned"):
        counterfactual_impact_mae(stylized, truth[:2])
    with pytest.raises(ValueError, match="finite"):
        counterfactual_impact_mae(stylized, np.array([0., 2., np.nan, 4.]))


def test_replicate_exhaustion_counts_any_arm_and_is_k_independent() -> None:
    replicas = [_replica(i) for i in range(3)]
    assert exhausted_replicate_rate(replicas) == pytest.approx(2 / 3)
    assert exhausted_replicate_rate(replicas * 4) == pytest.approx(2 / 3)
    replicas[0][RolloutArm.SELL_BLIND] = replace(replicas[0][RolloutArm.SELL_BLIND], budget_exhausted=True)
    assert exhausted_replicate_rate(replicas) == 1
