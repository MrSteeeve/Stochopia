"""Sequence model: shapes, learning signal, and rollout controls."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from stochopia.neural.seq import (  # noqa: E402
    _RHO_MAX,
    _RHO_MIN,
    FlowReturnModel,
    ModelSpec,
    _scan,
    make_model,
    next_event_loss,
    rollout,
    train,
)
from stochopia.sim.world import injection_schedule  # noqa: E402


def _toy_days(n_days=4, n=600, seed=0):
    """Return depends on the sign: +1 trades move the price up by 1."""
    rng = np.random.default_rng(seed)
    days = []
    for _ in range(n_days):
        signs = rng.choice([-1.0, 1.0], size=n)
        days.append(np.stack([signs, signs + rng.normal(0, 0.1, size=n)], axis=1))
    return days


def _model(seed=0):
    torch.manual_seed(seed)
    return FlowReturnModel(ModelSpec(hidden=16, layers=1))


def test_forward_shapes():
    logits, params, state = _model()(torch.zeros(3, 7, 2))
    assert logits.shape == (3, 7) and params.shape == (3, 7, 1, 2, 2)
    assert state.shape == (1, 3, 16)
    flagged = FlowReturnModel(ModelSpec(hidden=16, layers=1, own_flag=True))
    logits, params, _ = flagged(torch.zeros(3, 7, 3))
    assert logits.shape == (3, 7) and params.shape == (3, 7, 2, 2, 2)


def test_training_learns_sign_conditional_return():
    model = _model()
    days = _toy_days()
    probe = torch.as_tensor(days[0][None, :200], dtype=torch.float32)
    before = next_event_loss(model, probe).item()
    history = train(model, days, steps=300, batch=16, window=32, lr=1e-2, seed=0, device="cpu")
    assert history[-1] < before - 0.5
    _, params, _ = model(torch.as_tensor(days[0][None, :50], dtype=torch.float32))
    sell_mean, buy_mean = params[0, -1, 0, 0, 0].item(), params[0, -1, 0, 1, 0].item()
    assert buy_mean == pytest.approx(1.0, abs=0.2) and sell_mean == pytest.approx(-1.0, abs=0.2)


def test_rollout_of_trained_model_returns_raw_units_from_the_right_head():
    """Injected buys must draw from the buy head, sells from the sell head,
    and returns must come back in raw units when ``return_scale`` is not 1."""
    torch.manual_seed(0)
    model = FlowReturnModel(ModelSpec(hidden=16, layers=1, return_scale=2.5))
    days = _toy_days()
    train(model, days, steps=400, batch=16, window=32, lr=1e-2, seed=0, device="cpu")
    schedule = np.ones(8, dtype=bool)  # every event is an injected trade
    context = np.stack([d[:30] for d in _toy_days(n_days=64, n=30, seed=5)])
    u, z = _draws(64, schedule.size)
    _, buy = rollout(model, context, schedule, +1, uniforms=u, normals=z, device="cpu")
    _, sell = rollout(model, context, schedule, -1, uniforms=u, normals=z, device="cpu")
    assert buy.mean() == pytest.approx(1.0, abs=0.2)
    assert sell.mean() == pytest.approx(-1.0, abs=0.2)
    assert buy.std() == pytest.approx(0.1, abs=0.1)


def _draws(batch, length, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((batch, length)), rng.standard_normal((batch, length))


def test_rollout_injects_side_and_is_reproducible():
    model = _model()
    schedule = injection_schedule(n_children=3, spacing=2, n_relax=4)
    context = np.stack(_toy_days(n_days=5, n=20))
    u, z = _draws(5, schedule.size)
    signs, rets = rollout(model, context, schedule, +1, uniforms=u, normals=z, device="cpu")
    again, rets2 = rollout(model, context, schedule, +1, uniforms=u, normals=z, device="cpu")
    assert signs.shape == rets.shape == (5, schedule.size)
    assert np.all(signs[:, schedule] == 1.0) and set(np.unique(signs)) <= {-1.0, 1.0}
    assert np.array_equal(signs, again) and np.array_equal(rets, rets2)
    sell, _ = rollout(model, context, schedule, -1, uniforms=u, normals=z, device="cpu")
    assert np.all(sell[:, schedule] == -1.0)


def test_rollout_clamped_uses_supplied_background():
    model = _model()
    schedule = injection_schedule(n_children=3, spacing=2, n_relax=4)
    context = np.stack(_toy_days(n_days=5, n=20))
    u, z = _draws(5, schedule.size)
    background = np.random.default_rng(1).choice([-1.0, 1.0], size=(5, int((~schedule).sum())))
    signs, _ = rollout(
        model, context, schedule, +1, uniforms=u, normals=z, clamp_signs=background, device="cpu"
    )
    assert np.array_equal(signs[:, ~schedule], background)
    assert np.all(signs[:, schedule] == 1.0)


def test_rollout_does_not_modify_context():
    model = _model()
    schedule = injection_schedule(n_children=2, spacing=1, n_relax=2)
    context = np.stack(_toy_days(n_days=3, n=20))
    snapshot = context.copy()
    u, z = _draws(3, schedule.size)
    rollout(model, context, schedule, +1, uniforms=u, normals=z, device="cpu")
    assert np.array_equal(context, snapshot)


def _own_days(n_days=4, n=600, seed=0):
    """Own trades are always buys and move the price by 3; background trades
    are balanced and move it by 1."""
    rng = np.random.default_rng(seed)
    days, flags = [], []
    for _ in range(n_days):
        own = (rng.random(n) < 0.4).astype(np.float32)
        signs = np.where(own > 0, 1.0, rng.choice([-1.0, 1.0], size=n))
        rets = np.where(own > 0, 3.0, 1.0) * signs + rng.normal(0, 0.1, size=n)
        days.append(np.stack([signs, rets], axis=1))
        flags.append(own)
    return days, flags


def _trained_own_model():
    torch.manual_seed(0)
    model = FlowReturnModel(ModelSpec(hidden=16, layers=1, own_flag=True))
    days, flags = _own_days()
    train(
        model, days, steps=500, batch=16, window=32, lr=1e-2, seed=0, device="cpu",
        own_flags=lambda day, start, length, rng: flags[day][start : start + length],
    )
    return model


def test_train_requires_own_flags_exactly_for_flagged_models():
    days, flags = _own_days(n_days=1, n=100)
    with pytest.raises(ValueError):
        train(_model(), days, steps=1, batch=2, window=8, lr=1e-3, seed=0, device="cpu",
              own_flags=lambda *a: flags[0][:9])
    flagged = FlowReturnModel(ModelSpec(hidden=16, layers=1, own_flag=True))
    with pytest.raises(ValueError):
        train(flagged, days, steps=1, batch=2, window=8, lr=1e-3, seed=0, device="cpu")


def test_own_trades_use_their_own_return_head_and_do_not_bias_the_sign_head():
    model = _trained_own_model()
    schedule = injection_schedule(n_children=6, spacing=1, n_relax=0)
    context = np.stack([d[:30] for d in _own_days(n_days=64, n=30, seed=5)[0]])
    u, z = _draws(64, schedule.size)
    signs, rets = rollout(model, context, schedule, +1, uniforms=u, normals=z, device="cpu")
    # injected trades are flagged as own: their returns come from the own head
    assert rets[:, schedule].mean() == pytest.approx(3.0, abs=0.4)
    background = rets[:, ~schedule] * signs[:, ~schedule]
    assert background.mean() == pytest.approx(1.0, abs=0.3)
    # own trades are 40% of training events and all buys. A sign head trained
    # on every event would give a mean sign near +0.4; trained on background
    # trades only it stays balanced.
    assert abs(signs[:, ~schedule].mean()) < 0.2


# --- linear state space cells -----------------------------------------------


@pytest.mark.parametrize("cell", ["ssm", "lru"])
def test_linear_state_model_forward_shapes(cell):
    model = make_model(ModelSpec(hidden=16, layers=1, cell=cell))
    logits, params, state = model(torch.zeros(3, 7, 2))
    assert logits.shape == (3, 7) and params.shape == (3, 7, 1, 2, 2)
    assert state.shape == (3, 16)
    flagged = make_model(ModelSpec(hidden=16, layers=1, cell=cell, own_flag=True))
    logits, params, _ = flagged(torch.zeros(3, 7, 3))
    assert logits.shape == (3, 7) and params.shape == (3, 7, 2, 2, 2)


def test_linear_state_model_rejects_unknown_cell():
    with pytest.raises(ValueError):
        make_model(ModelSpec(hidden=8, layers=1, cell="lstm"))


def test_scan_matches_naive_recurrence():
    torch.manual_seed(0)
    x = torch.randn(3, 40, 5)
    rho = torch.rand(5) * 0.9 + 0.05
    h = _scan(x, rho, None)
    prev = torch.zeros(3, 5)
    ref = torch.empty_like(h)
    for t in range(40):
        prev = rho * prev + (1 - rho) * x[:, t]
        ref[:, t] = prev
    assert torch.allclose(h, ref, atol=1e-5)


def test_scan_is_finite_and_correct_at_the_clamp_extremes():
    """The chunked cumsum exponentiates rho; at the clamp bounds the
    intermediate magnitudes are largest, so this pins the numerical safety
    claimed in the ``_scan`` docstring."""
    torch.manual_seed(3)
    x = torch.randn(2, 64, 3) * 10
    for value in (_RHO_MIN, _RHO_MAX):
        rho = torch.full((3,), value)
        h = _scan(x, rho, None)
        assert torch.isfinite(h).all()
        prev = torch.zeros(2, 3)
        ref = torch.empty_like(h)
        for t in range(64):
            prev = rho * prev + (1 - rho) * x[:, t]
            ref[:, t] = prev
        assert torch.allclose(h, ref, atol=1e-2)


def test_scan_state_continues_the_recurrence():
    torch.manual_seed(1)
    x = torch.randn(2, 33, 4)  # 33 is not a multiple of the chunk: exercises padding
    rho = torch.rand(4) * 0.9 + 0.05
    h = _scan(x, rho, None)
    h1 = _scan(x[:, :10], rho, None)
    h2 = _scan(x[:, 10:], rho, h1[:, -1])
    assert torch.allclose(torch.cat([h1, h2], dim=1), h, atol=1e-5)


@pytest.mark.parametrize("cell", ["ssm", "lru"])
def test_linear_state_model_single_steps_match_the_sequence(cell):
    torch.manual_seed(2)
    model = make_model(ModelSpec(hidden=8, layers=1, cell=cell))
    model.eval()
    x = torch.randn(2, 12, 2)
    with torch.no_grad():
        logits_seq, params_seq, state = model(x)
        s = None
        for t in range(12):
            logits_t, params_t, s = model(x[:, t : t + 1], s)
    assert torch.allclose(logits_seq[:, -1:], logits_t, atol=1e-5)
    assert torch.allclose(params_seq[:, -1:], params_t, atol=1e-5)
    assert torch.allclose(state, s, atol=1e-5)


@pytest.mark.parametrize("cell", ["ssm", "lru"])
def test_training_learns_sign_conditional_return_linear_state(cell):
    torch.manual_seed(0)
    model = make_model(ModelSpec(hidden=16, layers=1, cell=cell))
    days = _toy_days()
    probe = torch.as_tensor(days[0][None, :200], dtype=torch.float32)
    before = next_event_loss(model, probe).item()
    train(model, days, steps=300, batch=16, window=32, lr=1e-2, seed=0, device="cpu")
    assert next_event_loss(model, probe).item() < before - 0.5


@pytest.mark.parametrize("cell", ["ssm", "lru"])
def test_rollout_injects_side_linear_state(cell):
    torch.manual_seed(0)
    model = make_model(ModelSpec(hidden=16, layers=1, cell=cell))
    schedule = injection_schedule(n_children=3, spacing=2, n_relax=4)
    context = np.stack(_toy_days(n_days=5, n=20))
    u, z = _draws(5, schedule.size)
    signs, rets = rollout(model, context, schedule, +1, uniforms=u, normals=z, device="cpu")
    assert signs.shape == rets.shape == (5, schedule.size)
    assert np.all(signs[:, schedule] == 1.0) and set(np.unique(signs)) <= {-1.0, 1.0}


@pytest.mark.parametrize("cell", ["ssm", "lru"])
def test_linear_state_readout_branch(cell):
    """Pin which readout each cell uses: ssm reads the state directly, lru
    reads gelu(readout(h)). If a branch silently changes, the heads no longer
    match the hand-computed path."""
    torch.manual_seed(4)
    model = make_model(ModelSpec(hidden=8, layers=1, cell=cell))
    model.eval()
    x = torch.randn(2, 10, 2)
    with torch.no_grad():
        logits, _, _ = model(x)
        rho = torch.sigmoid(model.decay_logit).clamp(_RHO_MIN, _RHO_MAX)
        h = _scan(model.embed(x), rho, None)
        out = h if cell == "ssm" else torch.nn.functional.gelu(model.readout(h))
        expect = model.sign_head(out).squeeze(-1)
    assert torch.allclose(logits, expect, atol=1e-6)


@pytest.mark.parametrize("cell", ["ssm", "lru"])
def test_rollout_injects_side_linear_state_with_own_flag(cell):
    """The CLI allows recipe self with --cell ssm/lru; the linear state shape
    ``[batch, hidden]`` must survive the rollout's state passing."""
    torch.manual_seed(0)
    model = make_model(ModelSpec(hidden=16, layers=1, cell=cell, own_flag=True))
    schedule = injection_schedule(n_children=3, spacing=2, n_relax=4)
    context = np.stack([d[:30] for d in _own_days(n_days=5, n=30, seed=5)[0]])
    u, z = _draws(5, schedule.size)
    signs, rets = rollout(model, context, schedule, +1, uniforms=u, normals=z, device="cpu")
    assert signs.shape == rets.shape == (5, schedule.size)
    assert np.all(signs[:, schedule] == 1.0)
