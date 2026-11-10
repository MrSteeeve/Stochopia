# Vendored world model r02s01 (frozen copy; do not edit the model body).
# Source: OpenHyra artifacts/stochopia-world-model-20261108/deepseek/programs/r02s01.py
# Produced by the OpenHyra world-model loop on 2026-11-08 (generator: DeepSeek V4.1 Flash,
# round 2 slot 1, parent r01s00). Scores vs the EWMA-GBM/random-walk reference under
# stochopia.market-world-model-forecast.v1: dev +0.0478 (LCB +0.0258), test +0.0783 (LCB +0.0280).
# Reviewed 2026-11-08: numpy only; reads/writes only its --input/--output directories.
"""World model: persistent latent stress state driving stress-dependent
cross-index dependence and asymmetric fat-tailed shocks, with path-level
stochastic volatility, leverage, and Q quote dynamics (AR(1) log IV with
exact horizon variance; robust OU carries).

Contract: python world_model.py --input IN --output OUT
"""

import argparse
import json
from pathlib import Path

import numpy as np

Q_DEFAULTS = (0.20, 0.10, 0.10)
NU = 7.0


def _ewma_var(returns, lam=0.94):
    if returns.shape[0] < 22:
        return np.var(returns, axis=0) + 1e-10
    var = np.var(returns[:20], axis=0)
    for row in returns[20:]:
        var = lam * var + (1.0 - lam) * row ** 2
    return var


def _skew_t(rng, size, nu=NU, down=1.15, up=0.94):
    """Standardised (unit-variance, zero-mean) skewed Student-t draws."""
    z = rng.standard_normal(size)
    chi = rng.chisquare(nu, size=size)
    t = z * np.sqrt(nu / chi) * np.sqrt((nu - 2.0) / nu)
    t = np.where(t < 0.0, t * down, t * up)
    axes = tuple(range(t.ndim - 1))
    t = t - t.mean(axis=axes, keepdims=True)
    sd = t.std(axis=axes, keepdims=True)
    return t / np.maximum(sd, 1e-12)


def _ar1_log_iv(liv):
    """AR(1) fit on log IV: returns (b, a, resid_sd, trend_per_day)."""
    x, y = liv[:-1], liv[1:]
    xm, ym = x.mean(), y.mean()
    den = float(((x - xm) ** 2).sum())
    b = float(((x - xm) * (y - ym)).sum() / den) if den > 1e-12 else 0.96
    b = float(np.clip(b, 0.90, 0.995))
    a = float(ym - b * xm)
    resid = y - (a + b * x)
    sd_e = float(np.std(resid, ddof=1)) if resid.size > 3 else 0.06
    sd_e = float(np.clip(sd_e, 0.02, 0.18))
    idx = np.arange(liv.size, dtype=float)
    slope = float(np.polyfit(idx, liv, 1)[0]) if liv.size >= 12 else 0.0
    return b, a, sd_e, slope


def forecast(history, query):
    rng = np.random.default_rng(int(query["seed"]))
    n = int(query["n_paths"])
    horizon = int(query["horizon"])

    raw = np.asarray(history["returns"], dtype=float)
    mask = np.all(np.isfinite(raw), axis=1)
    returns = raw[mask] if mask.sum() >= 40 else np.nan_to_num(raw)
    eps = 1e-12
    r2 = returns ** 2

    # --- volatility term structure: HAR-style geometric blend, mean reversion
    var_fast = _ewma_var(returns)
    var_mid = r2[-21:].mean(axis=0) if returns.shape[0] >= 21 else var_fast
    var_long = r2[-252:].mean(axis=0) if returns.shape[0] >= 252 else r2.mean(axis=0)
    var_fast = np.minimum(var_fast, 4.0 * var_long + eps)
    v0 = np.exp(
        0.50 * np.log(var_fast + eps)
        + 0.30 * np.log(var_mid + eps)
        + 0.20 * np.log(var_long + eps)
    )
    v0 = np.clip(v0, (0.002) ** 2, (0.25) ** 2)
    kappa = 0.96
    steps = np.arange(horizon)
    var_h = var_long[None, :] + (v0 - var_long)[None, :] * (kappa ** steps)[:, None]
    var_h = np.maximum(var_h, (0.003) ** 2)
    sigma_h = np.sqrt(var_h)  # (horizon, 2)

    # --- baseline dependence, shrunk toward the typical equity-pair level
    lookback = min(252, returns.shape[0])
    corr_hat = 0.85
    if lookback >= 10:
        c = float(np.corrcoef(returns[-lookback:, 0], returns[-lookback:, 1])[0, 1])
        if np.isfinite(c):
            corr_hat = c
    rho = float(np.clip(0.6 * corr_hat + 0.4 * 0.85, 0.20, 0.95))

    # --- persistent per-path latent stress state (negative = risk-off)
    stress = _skew_t(rng, (n, 1), down=1.10, up=0.95)
    stress = np.clip(stress, -3.0, 3.0)

    # correlation regime rises in stressed (risk-off) paths
    rho_path = np.clip(rho + 0.12 * np.maximum(0.0, -stress), 0.20, 0.97)  # (n,1)

    # --- asymmetric fat-tailed shocks with stress-dependent dependence
    common = _skew_t(rng, (n, horizon, 1), down=1.22, up=0.88)
    idio = _skew_t(rng, (n, horizon, 2), down=1.20, up=0.90)

    rho_h = rho_path[:, None, :] + 0.10 * np.maximum(0.0, -common - 0.30)
    rho_h = np.clip(rho_h, 0.20, 0.97)
    wc = np.sqrt(rho_h)
    wi = np.sqrt(1.0 - rho_h)
    z = wc * common + wi * idio

    # --- persistent path-level stochastic volatility (common + idiosyncratic),
    #     with a stress tilt: risk-off paths carry higher vol
    log_sv = (
        0.18 * rng.standard_normal((n, 1))
        + 0.12 * rng.standard_normal((n, 2))
        - 0.08 * stress
    )
    sv = np.exp(log_sv)
    sv = sv / np.sqrt(np.mean(sv ** 2))
    sv = np.clip(sv, 0.45, 3.0)

    # --- very mild momentum tilt in the drift
    drift = np.zeros(2)
    if returns.shape[0] >= 21:
        mom = returns[-21:].sum(axis=0)
        drift = np.clip(0.05 * mom, -0.006, 0.006) / horizon

    paths = np.empty((n, horizon, 2), dtype=float)
    lev = np.zeros((n, 2))
    for h in range(horizon):
        mult = sv * np.exp(-0.28 * lev)
        step = drift[None, :] + sigma_h[h][None, :] * mult * z[:, h, :]
        paths[:, h, :] = np.clip(step, -0.5, 0.5)
        lev = 0.90 * lev + 0.10 * z[:, h, :]

    # --- Q quotes ---------------------------------------------------------
    q = np.empty((n, 3), dtype=float)
    hz = float(horizon)

    tot = paths.sum(axis=1)  # (n, 2)
    sd_ret = tot.std(axis=0)
    sd_ret[sd_ret < 1e-9] = 1.0
    z_ret = (tot - tot.mean(axis=0)) / sd_ret

    rv = np.sqrt(np.mean(paths[:, :, 1] ** 2, axis=1) * 252.0)
    log_rv = np.log(np.maximum(rv, 1e-6))
    sd_rv = log_rv.std()
    z_rv = (log_rv - log_rv.mean()) / (sd_rv if sd_rv > 1e-9 else 1.0)

    z_ind = rng.standard_normal((n, 3))

    # trailing annualised CSI1000 realised vol, used for the IV risk premium
    if returns.shape[0] >= 21:
        rv21 = float(np.sqrt(np.mean(returns[-21:, 1] ** 2) * 252.0))
    else:
        rv21 = float(np.sqrt(np.mean(returns[:, 1] ** 2) * 252.0))
    rv21 = max(rv21, 1e-4)

    # --- implied volatility: AR(1) on log IV with exact horizon variance
    iv = np.asarray(history["iv_1m"], dtype=float)
    ivf = iv[np.isfinite(iv)]
    if ivf.size >= 15:
        liv = np.log(np.maximum(ivf, 1e-3))
        b, a, sd_e, slope = _ar1_log_iv(liv)
        lvl = float(liv[-1])
        mu_ar = a * (1.0 - b ** hz) / (1.0 - b) + (b ** hz) * lvl
        # damped reversion: guards against a stale (too high) historical anchor
        mu_iv = lvl + 0.55 * (mu_ar - lvl)
        # variance-risk-premium: rich implied vol tends to bleed lower
        vrp = float(np.clip(lvl - np.log(rv21), -0.6, 0.6))
        mu_iv -= 0.20 * vrp * (1.0 - b ** hz)
        # damped local trend, capped
        mu_iv += float(np.clip(0.25 * slope * hz, -0.06, 0.06))
        mu_iv = float(np.clip(mu_iv, lvl - 0.40, lvl + 0.40))
        var_h_iv = sd_e ** 2 * (1.0 - b ** (2.0 * hz)) / (1.0 - b ** 2)
        sd_iv = float(np.clip(0.82 * np.sqrt(var_h_iv), 0.025, 0.13))
    else:
        mu_iv, sd_iv = float(np.log(Q_DEFAULTS[0])), 0.10

    rho_iv = 0.40
    z_iv = rho_iv * z_rv + np.sqrt(1.0 - rho_iv ** 2) * z_ind[:, 0]
    q[:, 0] = np.clip(np.exp(mu_iv + sd_iv * z_iv), 0.03, 1.5)

    # --- futures carry: robust OU anchor, widens (falls) in drawdowns
    carry = np.asarray(history["carry"], dtype=float)
    rho_c = -0.30
    for j in range(2):
        col = carry[:, j]
        cf = col[np.isfinite(col)]
        if cf.size >= 5:
            level = float(cf[-1])
            recent = cf[-min(60, cf.size):]
            lr = float(np.median(recent))
            mu_c = lr + (0.97 ** hz) * (level - lr)
            lag = min(21, max(1, cf.size - 1))
            ch = cf[lag:] - cf[:-lag]
            if ch.size >= 3:
                sd_c = float(np.std(ch)) * np.sqrt(21.0 / lag)
            else:
                sd_c = 0.03
            sd_c = float(np.clip(0.92 * sd_c, 0.008, 0.10))
            mu_c -= 0.08 * sd_c
        else:
            mu_c, sd_c = Q_DEFAULTS[1 + j], 0.03
        z_c = rho_c * z_ret[:, j] + np.sqrt(1.0 - rho_c ** 2) * z_ind[:, 1 + j]
        q[:, 1 + j] = np.clip(mu_c + sd_c * z_c, -0.6, 0.6)

    return paths, q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    with np.load(source / "history.npz", allow_pickle=False) as data:
        history = {name: data[name] for name in data.files}
    query = json.loads((source / "query.json").read_text())
    returns, q = forecast(history, query)
    np.savez(Path(args.output) / "forecast.npz", returns=returns, q=q)


if __name__ == "__main__":
    main()
