"""Controlled probe capacity, causal interfaces, and operator treatment checks."""

from __future__ import annotations

import pytest
import torch

from stochopia.neural.lob_interface import (
    StepObservation, _probe, state_num_elements_per_sample, verify_world_model_contract, world_model_loss,
)
from stochopia.neural.probe_features import advect_level_state
from stochopia.neural.probes import PROBE_NAMES, _SurpriseGate, make_probe, model_factory


PARAM_COUNTS = {
    "gru-base": 24244,
    "gru-multiscale": 24244,
    "gru-surprise": 24307,
    "lru-base": 22361,
    "lru-multiscale": 22361,
    "lru-surprise": 22424,
    "flat-gru": 24244,
    "matrix-noshift": 25722,
    "matrix-advect": 25722,
    "flat-readoutgate": 26227,
    "selective-ssm": 22212,
    "gru-hawkes": 24383,
}


@pytest.mark.parametrize("name", PROBE_NAMES)
def test_probe_causal_contract_has_exact_capacity_and_finite_gradients(name: str) -> None:
    torch.manual_seed(103)
    model = model_factory(name)()
    assert verify_world_model_contract(model, max_state_dim=60) == (PARAM_COUNTS[name], 60)
    batch = _probe(torch.device("cpu"), 721, batch_size=5)
    loss = world_model_loss(model.forward_sequence(batch), batch.targets).total
    loss.backward()
    assert torch.isfinite(loss).item()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all().item()
    with pytest.raises(ValueError, match="max_state_dim"):
        verify_world_model_contract(model, max_state_dim=59)


def _same_seed(name: str) -> torch.nn.Module:
    torch.manual_seed(211)
    return make_probe(name)


def test_geometry_and_readout_controls_share_parent_weights() -> None:
    flat = _same_seed("flat-gru")
    gate = _same_seed("flat-readoutgate")
    for key, value in flat.state_dict().items():
        torch.testing.assert_close(value, gate.state_dict()[key])
    batch = _probe(torch.device("cpu"), 42, batch_size=5)
    original_state = None
    gated_state = None
    for t in range(batch.targets.size(1)):
        obs = batch.observation.at((slice(None), t))
        ordinary, original_state = flat.step_context(obs, original_state)
        modified, gated_state = gate.step_context(obs, gated_state)
        torch.testing.assert_close(original_state, gated_state, rtol=0, atol=0)
        assert not torch.equal(ordinary, modified)

    no_shift = _same_seed("matrix-noshift")
    advect = _same_seed("matrix-advect")
    assert no_shift.state_dict().keys() == advect.state_dict().keys()
    for key, value in no_shift.state_dict().items():
        torch.testing.assert_close(value, advect.state_dict()[key])
    obs = batch.observation.at((slice(None), 0))
    _, matrix = advect.step_context(obs, None)
    assert matrix.shape == (5, 10, 6)
    torch.testing.assert_close(matrix[:, 0, [2, 5]], obs.book_ticks[..., 0].to(matrix.dtype))


def test_tick_transport_preserves_absolute_side_coordinates() -> None:
    latent = torch.arange(1, 61, dtype=torch.float32).reshape(1, 10, 6)
    previous = torch.tensor([[100.0, 102.0]])
    current = previous + 1
    shifted = advect_level_state(latent, previous, current)
    # Bid price moves up by one: old slots become farther away. Ask vice versa.
    torch.testing.assert_close(shifted[:, 1:, :3], latent[:, :-1, :3])
    torch.testing.assert_close(shifted[:, :-1, 3:], latent[:, 1:, 3:])
    assert shifted[:, 0, :3].eq(0).all().item()
    assert shifted[:, -1, 3:].eq(0).all().item()
    assert advect_level_state(latent, previous, previous + 10).eq(0).all().item()


def test_multiscale_retention_modulates_previous_volume() -> None:
    model = make_probe("lru-multiscale")
    obs = _probe(torch.device("cpu"), 12, batch_size=1).observation.at((slice(None), 0))
    tokens = obs.previous_event.clone()
    low = tokens.clone()
    high = tokens.clone()
    low[:, 3] = 0
    high[:, 3] = 11
    x = torch.zeros(1, 60)
    g_low = model._retention(x, obs._replace(previous_event=low))
    g_high = model._retention(x, obs._replace(previous_event=high))
    assert (g_low[:, 1:] > g_low[:, :-1]).all().item()
    assert (g_high < g_low).all().item()


def test_surprise_gate_uses_lagged_flow_prediction() -> None:
    gate = _SurpriseGate()
    prior = torch.zeros(1, 60)
    prior[:, 0] = 2
    with torch.no_grad():
        gate.predict.weight.zero_()
        gate.predict.bias.zero_()
        gate.predict.weight[0, 0] = 1
    proposal = torch.ones_like(prior)
    aligned = gate(proposal, prior, torch.ones(1, 1))
    surprising = gate(proposal, prior, -torch.ones(1, 1))
    # Identical proposal, but prior predicts the positive sign and not the negative.
    torch.testing.assert_close(aligned[..., :30], proposal[..., :30])
    assert (surprising[..., 30:] > aligned[..., 30:]).all().item()


def test_hawkes_state_is_positive_and_waiting_time_ages_it() -> None:
    model = make_probe("gru-hawkes")
    obs = _probe(torch.device("cpu"), 12, batch_size=1).observation.at((slice(None), 0))
    tokens = torch.tensor([[0, 2, 0, 0]])
    event = obs._replace(previous_event=tokens)
    _, state = model.step_context(event, None)
    assert state_num_elements_per_sample(state) == 60
    assert state[0].shape == (1, 48)
    assert state[1].shape == (1, 12)
    assert state[1].ge(0).all().item()
    assert state[1][0, 4:6].gt(0).all().item()
    _, instant = model.step_context(event, state)
    torch.testing.assert_close(instant[1], 2 * state[1])
    late = obs._replace(previous_event=torch.tensor([[15, 2, 0, 0]]))
    _, aged = model.step_context(late, state)
    torch.testing.assert_close(aged[1], state[1], rtol=1e-6, atol=1e-6)
    assert 0 < float(0.95 * model.alpha_logit.detach().sigmoid()) < 0.95


def test_factory_rejects_unregistered_variants() -> None:
    with pytest.raises(ValueError, match="unknown P2 probe"):
        make_probe("unregistered")
