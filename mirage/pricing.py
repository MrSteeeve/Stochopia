"""定价引擎：Black-Scholes、障碍期权解析解、雪球/自动赎回蒙特卡洛，以及产品级公允价值与希腊值。

本模块只做纯数学计算，不涉及任何 LLM 调用，且只依赖标准库（math、random）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from .products import ClientProfile, MarketState, ProductSpec

MARKUP = 1.03
HEDGE_COST_RATIO = 1.02
MC_PATHS = 10000
MC_SEED = 42

_VANILLA_TYPES = ("vanilla_call", "vanilla_put")
_BARRIER_TYPES = ("barrier_call", "barrier_put")


class PricingError(Exception):
    """定价参数非法或计算失败。"""


def _reference_spot(product: ProductSpec, current_spot: float) -> float:
    """返回合约发行时参考现货；旧 ProductSpec 默认以首次计价现货为参考。"""
    reference = current_spot if product.reference_spot is None else product.reference_spot
    if not isinstance(reference, (int, float)) or isinstance(reference, bool):
        raise PricingError(f"reference_spot 必须是正的有限数字，实际为 {reference!r}")
    reference = float(reference)
    if not math.isfinite(reference) or reference <= 0.0:
        raise PricingError(f"reference_spot 必须是正的有限数字，实际为 {reference!r}")
    return reference


def _remaining_months(product: ProductSpec) -> int:
    """存续产品的剩余月数；elapsed_months 由环境维护，但定价边界仍做防御性校验。"""
    elapsed = product.elapsed_months
    if isinstance(elapsed, bool) or not isinstance(elapsed, int):
        raise PricingError(f"elapsed_months 必须是整数，实际为 {elapsed!r}")
    if elapsed < 0 or elapsed > product.maturity_months:
        raise PricingError(
            "elapsed_months 必须在 0 到 maturity_months 之间，"
            f"实际为 {elapsed!r}"
        )
    return int(product.maturity_months) - elapsed


def _barrier_direction(product: ProductSpec) -> str:
    """获取发行时固定的障碍方向，兼容未携带新字段的旧式直接构造。"""
    direction = product.barrier_direction
    if direction is None:
        barrier_pct = product.barrier_pct if product.barrier_pct is not None else 1.0
        direction = "up" if barrier_pct > 1.0 else "down"
    if direction not in ("up", "down"):
        raise PricingError(f"未知障碍方向：{direction}")
    return direction


def _barrier_is_touched(spot: float, barrier: float, direction: str) -> bool:
    if direction == "down":
        return spot <= barrier
    if direction == "up":
        return spot >= barrier
    raise PricingError(f"未知障碍方向：{direction}")


# ---------------------------------------------------------------------------
# 正态分布辅助函数
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    """标准正态分布累积分布函数，基于 math.erf 计算。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """标准正态分布概率密度函数。"""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Black-Scholes 解析解
# ---------------------------------------------------------------------------


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """欧式看涨期权 Black-Scholes 价格（连续分红 q）。T<=0 或 sigma<=0 时退化为内在价值。"""
    if T <= 0 or sigma <= 0:
        if T <= 0:
            return max(S - K, 0.0)
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """欧式看跌期权 Black-Scholes 价格（连续分红 q）。T<=0 或 sigma<=0 时退化为内在价值。"""
    if T <= 0 or sigma <= 0:
        if T <= 0:
            return max(K - S, 0.0)
        return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> dict:
    """解析希腊值：delta、gamma、vega（每 1 个 vol 点）、theta（每日历日）。"""
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    gamma = disc_q * _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega_annual = S * disc_q * _norm_pdf(d1) * math.sqrt(T)
    vega = vega_annual / 100.0

    if option_type == "call":
        delta = disc_q * _norm_cdf(d1)
        theta_annual = (
            -S * disc_q * _norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
            - r * K * disc_r * _norm_cdf(d2)
            + q * S * disc_q * _norm_cdf(d1)
        )
    elif option_type == "put":
        delta = disc_q * (_norm_cdf(d1) - 1.0)
        theta_annual = (
            -S * disc_q * _norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
            + r * K * disc_r * _norm_cdf(-d2)
            - q * S * disc_q * _norm_cdf(-d1)
        )
    else:
        raise PricingError(f"未知期权类型：{option_type}")

    theta = theta_annual / 365.0
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


# ---------------------------------------------------------------------------
# 障碍期权：Reiner-Rubinstein (1991) 解析解，零返还
# ---------------------------------------------------------------------------


def barrier_option(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    barrier: float,
    barrier_type: str,
    option_type: str,
    barrier_direction: str,
    *,
    already_touched: bool = False,
) -> float:
    """Reiner-Rubinstein (1991) 障碍期权解析解，零返还（rebate=0）。

    barrier_type 取值 {"knock_in", "knock_out"}；barrier_direction 是发行时
    固定的 {"down", "up"}，不会随重估时现货位置反转。already_touched
    用于继承存续期已触碰状态；当前现货在障碍边界上也视为触碰。
    """
    if barrier_type not in ("knock_in", "knock_out"):
        raise PricingError(f"未知障碍类型：{barrier_type}")
    if option_type not in ("call", "put"):
        raise PricingError(f"未知期权类型：{option_type}")
    if barrier_direction not in ("down", "up"):
        raise PricingError(f"未知障碍方向：{barrier_direction}")

    vanilla = bs_call(S, K, T, r, sigma, q) if option_type == "call" else bs_put(S, K, T, r, sigma, q)

    # 已经触碰的边界情况（含 barrier == S）。方向绝不由当前 S 重推。
    is_down = barrier_direction == "down"
    already_breached = already_touched or _barrier_is_touched(
        S, barrier, barrier_direction
    )
    if already_breached:
        return vanilla if barrier_type == "knock_in" else 0.0

    if T <= 0 or sigma <= 0:
        # 未触碰、且已到期或无波动率：不会再触碰障碍
        return 0.0 if barrier_type == "knock_in" else vanilla

    phi = 1.0 if option_type == "call" else -1.0
    eta = 1.0 if is_down else -1.0

    mu = (r - q - 0.5 * sigma * sigma) / (sigma * sigma)
    lam = math.sqrt(mu * mu + 2.0 * r / (sigma * sigma))
    sqT = sigma * math.sqrt(T)

    def _x1():
        return math.log(S / K) / sqT + (1.0 + mu) * sqT

    def _x2():
        return math.log(S / barrier) / sqT + (1.0 + mu) * sqT

    def _y1():
        return math.log(barrier * barrier / (S * K)) / sqT + (1.0 + mu) * sqT

    def _y2():
        return math.log(barrier / S) / sqT + (1.0 + mu) * sqT

    def _z():
        return math.log(barrier / S) / sqT + lam * sqT

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    x1 = _x1()
    x2 = _x2()
    y1 = _y1()
    y2 = _y2()

    A = phi * S * disc_q * _norm_cdf(phi * x1) - phi * K * disc_r * _norm_cdf(phi * x1 - phi * sqT)
    B = phi * S * disc_q * _norm_cdf(phi * x2) - phi * K * disc_r * _norm_cdf(phi * x2 - phi * sqT)
    C = (
        phi
        * S
        * disc_q
        * (barrier / S) ** (2.0 * (mu + 1.0))
        * _norm_cdf(eta * y1)
        - phi * K * disc_r * (barrier / S) ** (2.0 * mu) * _norm_cdf(eta * y1 - eta * sqT)
    )
    D = (
        phi
        * S
        * disc_q
        * (barrier / S) ** (2.0 * (mu + 1.0))
        * _norm_cdf(eta * y2)
        - phi * K * disc_r * (barrier / S) ** (2.0 * mu) * _norm_cdf(eta * y2 - eta * sqT)
    )

    # 依据 8 种 up/down × in/out × call/put 组合，用 A/B/C/D 拼出结果
    # 参考 Reiner & Rubinstein (1991)，rebate = 0
    down = is_down
    if barrier_type == "knock_in":
        if down and option_type == "call":
            if K >= barrier:
                price = C
            else:
                price = A - B + D
        elif (not down) and option_type == "call":
            if K >= barrier:
                price = A
            else:
                price = B - C + D
        elif down and option_type == "put":
            if K >= barrier:
                price = B - C + D
            else:
                price = A
        else:  # up put
            if K >= barrier:
                price = A - B + D
            else:
                price = C
    else:  # knock_out
        if down and option_type == "call":
            if K >= barrier:
                price = A - C
            else:
                price = B - D
        elif (not down) and option_type == "call":
            if K >= barrier:
                price = 0.0
            else:
                price = A - B + C - D
        elif down and option_type == "put":
            if K >= barrier:
                price = A - B + C - D
            else:
                price = 0.0
        else:  # up put
            if K >= barrier:
                price = B - D
            else:
                price = A - C

    price = max(price, 0.0)
    return price


# ---------------------------------------------------------------------------
# 蒙特卡洛：雪球 / 自动赎回
# ---------------------------------------------------------------------------


def _mc_stats(
    S: float,
    T_months: int,
    r: float,
    sigma: float,
    q: float,
    payoff_kind: str,
    *,
    coupon_rate: float,
    knock_in_pct: float | None = None,
    knock_out_pct: float | None = None,
    autocall_pct: float | None = None,
    protected: bool = False,
    n_paths: int = MC_PATHS,
    seed: int = MC_SEED,
    drift: float | None = None,
    discount: bool = True,
    fixing_ratio: float = 1.0,
    already_knocked_in: bool = False,
    elapsed_months: int = 0,
) -> dict:
    """月度步长 GBM 蒙特卡洛核心：雪球 / 自动赎回结构定价与统计量。

    fixing_ratio = 期初定盘价 / 当前现货。合约的敲入/敲出/赎回水平与
    亏损结算均锚定期初定盘价的绝对水平；计算 delta 等需要扰动现货时，
    传入偏离 1 的 fixing_ratio 使触发水平保持绝对不变。

    T_months 是合约总期限，elapsed_months 是已存续月数；仅模拟剩余路径，
    但票息仍按发行起累计的总存续月数计算。already_knocked_in
    使后续路径继承历史敲入，不会因重估重置。
    """
    if payoff_kind not in ("snowball", "autocallable"):
        raise PricingError(f"未知 MC 产品类型：{payoff_kind}")

    if isinstance(elapsed_months, bool) or not isinstance(elapsed_months, int):
        raise PricingError(f"elapsed_months 必须是整数，实际为 {elapsed_months!r}")
    total_months = int(T_months)
    if elapsed_months < 0 or elapsed_months > total_months:
        raise PricingError(
            f"elapsed_months 必须在 0 到 T_months 之间，实际为 {elapsed_months!r}"
        )
    if not math.isfinite(fixing_ratio) or fixing_ratio <= 0.0:
        raise PricingError(f"fixing_ratio 必须是正的有限数字，实际为 {fixing_ratio!r}")

    dt = 1.0 / 12.0
    n_steps = total_months - elapsed_months
    mu = drift if drift is not None else r
    rng = random.Random(seed)

    fair_values: list[float] = []
    knock_in_count = 0
    knock_out_count = 0
    life_months: list[float] = []
    loss_fracs: list[float] = []

    drift_term = (mu - q - 0.5 * sigma * sigma) * dt
    vol_term = sigma * math.sqrt(dt)

    for _ in range(n_paths):
        ratio = 1.0
        min_ratio = 1.0
        cashflow = None
        term_t = None
        payment_delay_months = None

        for t in range(1, n_steps + 1):
            z = rng.gauss(0.0, 1.0)
            ratio = ratio * math.exp(drift_term + vol_term * z)
            min_ratio = min(min_ratio, ratio)

            if payoff_kind == "snowball":
                if knock_out_pct is not None and ratio >= knock_out_pct * fixing_ratio:
                    term_t = elapsed_months + t
                    payment_delay_months = t
                    cashflow = 1.0 + coupon_rate * (term_t / 12.0)
                    knock_out_count += 1
                    break
            else:  # autocallable
                if autocall_pct is not None and ratio >= autocall_pct * fixing_ratio:
                    term_t = elapsed_months + t
                    payment_delay_months = t
                    cashflow = 1.0 + coupon_rate * (term_t / 12.0)
                    knock_out_count += 1
                    break

        if cashflow is None:
            term_t = total_months
            payment_delay_months = n_steps
            if payoff_kind == "snowball":
                knocked_in = (
                    (
                        already_knocked_in
                        or (
                            knock_in_pct is not None
                            and min_ratio <= knock_in_pct * fixing_ratio
                        )
                    )
                    and ratio < fixing_ratio
                )
                if knocked_in:
                    cashflow = ratio / fixing_ratio
                    knock_in_count += 1
                else:
                    cashflow = 1.0 + coupon_rate * (total_months / 12.0)
            else:  # autocallable
                knocked_in = (
                    already_knocked_in
                    or (
                        knock_in_pct is not None
                        and min_ratio <= knock_in_pct * fixing_ratio
                    )
                ) and not protected
                if knocked_in:
                    cashflow = min(ratio / fixing_ratio, 1.0)
                    knock_in_count += 1
                else:
                    cashflow = 1.0 + coupon_rate * (total_months / 12.0)

        t_years = payment_delay_months / 12.0
        pv = cashflow * (math.exp(-r * t_years) if discount else 1.0)
        fair_values.append(pv)
        life_months.append(float(term_t))
        loss_fracs.append(max(0.0, 1.0 - cashflow))

    n = n_paths
    # 逐路径二阶矩 -> pv 标准误占名义比（MC 模型不确定性带的基础量）。
    # 仅复用已抽取的 fair_values / loss_fracs，不新增随机抽样，确定性不变。
    pv_mean = sum(fair_values) / n
    pv_sq = sum(v * v for v in fair_values) / n
    pv_var = max(pv_sq - pv_mean * pv_mean, 0.0)
    pv_std = math.sqrt(pv_var)
    pv_se = pv_std / math.sqrt(n)
    sorted_losses = sorted(loss_fracs)
    p95_idx = min(int(0.95 * n), n - 1)
    return {
        "fair_value": pv_mean,
        "knock_in_prob": knock_in_count / n,
        "knock_out_prob": knock_out_count / n,
        "expected_life_months": sum(life_months) / n,
        "expected_loss_frac": sum(loss_fracs) / n,
        "pv_std_frac": pv_std,
        "pv_se_frac": pv_se,
        "p95_loss_frac": sorted_losses[p95_idx] if sorted_losses else 0.0,
        "max_loss_frac": max(loss_fracs) if loss_fracs else 0.0,
    }


def autocallable_price(
    S: float,
    T_months: int,
    r: float,
    sigma: float,
    q: float,
    coupon_rate: float,
    barrier_pct: float | None,
    autocall_pct: float = 1.05,
    n_paths: int = 10000,
    seed: int = 42,
    fixing_ratio: float = 1.0,
    already_knocked_in: bool = False,
    elapsed_months: int = 0,
    protected: bool = False,
) -> dict:
    """自动赎回票据蒙特卡洛定价（fair_value 为单位名义本金的贴现现值）。"""
    stats = _mc_stats(
        S,
        T_months,
        r,
        sigma,
        q,
        "autocallable",
        coupon_rate=coupon_rate,
        knock_in_pct=barrier_pct,
        autocall_pct=autocall_pct,
        n_paths=n_paths,
        seed=seed,
        fixing_ratio=fixing_ratio,
        already_knocked_in=already_knocked_in,
        elapsed_months=elapsed_months,
        protected=protected,
    )
    return {
        "fair_value": stats["fair_value"],
        "knock_in_prob": stats["knock_in_prob"],
        "expected_life_months": stats["expected_life_months"],
        "expected_loss_frac": stats["expected_loss_frac"],
    }


def snowball_price(
    S: float,
    T_months: int,
    r: float,
    sigma: float,
    q: float,
    coupon_rate: float,
    knock_in_pct: float = 0.75,
    knock_out_pct: float = 1.03,
    n_paths: int = 10000,
    seed: int = 42,
    fixing_ratio: float = 1.0,
    already_knocked_in: bool = False,
    elapsed_months: int = 0,
) -> dict:
    """雪球结构蒙特卡洛定价（fair_value 为单位名义本金的贴现现值）。"""
    stats = _mc_stats(
        S,
        T_months,
        r,
        sigma,
        q,
        "snowball",
        coupon_rate=coupon_rate,
        knock_in_pct=knock_in_pct,
        knock_out_pct=knock_out_pct,
        n_paths=n_paths,
        seed=seed,
        fixing_ratio=fixing_ratio,
        already_knocked_in=already_knocked_in,
        elapsed_months=elapsed_months,
    )
    return {
        "fair_value": stats["fair_value"],
        "knock_in_prob": stats["knock_in_prob"],
        "knock_out_prob": stats["knock_out_prob"],
        "expected_life_months": stats["expected_life_months"],
        "expected_loss_frac": stats["expected_loss_frac"],
    }


# ---------------------------------------------------------------------------
# 产品级定价：期望收益、公允价值 + 希腊值、一站式评估
# ---------------------------------------------------------------------------


def _snowball_kwargs(product: ProductSpec) -> dict:
    return {
        "knock_in_pct": product.barrier_pct if product.barrier_pct is not None else 0.75,
        "knock_out_pct": 1.03,
    }


def _autocallable_kwargs(product: ProductSpec) -> dict:
    return {
        "barrier_pct": product.barrier_pct,
        "autocall_pct": 1.05,
    }


def expected_payoff(product: ProductSpec, market: MarketState) -> float | None:
    """真实世界漂移（mu = r + trend_alpha）下的未贴现期望收益（人民币）。custom 返回 None。"""
    if product.product_type == "custom":
        return None

    S = market.spot
    reference = _reference_spot(product, S)
    sigma = market.volatility
    q = market.dividend_yield
    T = _remaining_months(product) / 12.0
    mu = market.risk_free_rate + market.trend_alpha
    K = product.strike_pct * reference
    m = product.notional / reference

    if product.product_type in _VANILLA_TYPES:
        if product.product_type == "vanilla_call":
            e_points = math.exp(mu * T) * bs_call(S, K, T, mu, sigma, q)
        else:
            e_points = math.exp(mu * T) * bs_put(S, K, T, mu, sigma, q)
        if product.principal_protected:
            return product.notional * (
                1.0 + product.participation_rate * e_points / reference
            )
        return m * product.participation_rate * e_points

    if product.product_type in _BARRIER_TYPES:
        barrier = (
            product.barrier_pct * reference
            if product.barrier_pct is not None
            else reference
        )
        option_type = "call" if product.product_type == "barrier_call" else "put"
        barrier_type = product.barrier_type or "knock_out"
        barrier_direction = _barrier_direction(product)
        e_points = math.exp(mu * T) * barrier_option(
            S,
            K,
            T,
            mu,
            sigma,
            q,
            barrier,
            barrier_type,
            option_type,
            barrier_direction,
            already_touched=product.barrier_touched,
        )
        if product.principal_protected:
            return product.notional * (
                1.0 + product.participation_rate * e_points / reference
            )
        return m * product.participation_rate * e_points

    if product.product_type == "snowball":
        kwargs = _snowball_kwargs(product)
        stats = _mc_stats(
            S,
            int(product.maturity_months),
            market.risk_free_rate,
            sigma,
            q,
            "snowball",
            coupon_rate=product.coupon_rate or 0.0,
            knock_in_pct=kwargs["knock_in_pct"],
            knock_out_pct=kwargs["knock_out_pct"],
            drift=mu,
            discount=False,
            fixing_ratio=reference / S,
            already_knocked_in=product.knock_in_active,
            elapsed_months=product.elapsed_months,
        )
        return stats["fair_value"] * product.notional

    if product.product_type == "autocallable":
        kwargs = _autocallable_kwargs(product)
        stats = _mc_stats(
            S,
            int(product.maturity_months),
            market.risk_free_rate,
            sigma,
            q,
            "autocallable",
            coupon_rate=product.coupon_rate or 0.0,
            knock_in_pct=kwargs["barrier_pct"],
            autocall_pct=kwargs["autocall_pct"],
            protected=product.principal_protected,
            drift=mu,
            discount=False,
            fixing_ratio=reference / S,
            already_knocked_in=product.knock_in_active,
            elapsed_months=product.elapsed_months,
        )
        return stats["fair_value"] * product.notional

    raise PricingError(f"未知产品类型：{product.product_type}")


def _vanilla_price_points(product: ProductSpec, market: MarketState) -> float:
    S = market.spot
    reference = _reference_spot(product, S)
    K = product.strike_pct * reference
    T = _remaining_months(product) / 12.0
    r = market.risk_free_rate
    sigma = market.volatility
    q = market.dividend_yield
    if product.product_type == "vanilla_call":
        return bs_call(S, K, T, r, sigma, q)
    return bs_put(S, K, T, r, sigma, q)


def _barrier_price_points(product: ProductSpec, market: MarketState) -> float:
    S = market.spot
    reference = _reference_spot(product, S)
    K = product.strike_pct * reference
    T = _remaining_months(product) / 12.0
    r = market.risk_free_rate
    sigma = market.volatility
    q = market.dividend_yield
    barrier = (
        product.barrier_pct * reference
        if product.barrier_pct is not None
        else reference
    )
    option_type = "call" if product.product_type == "barrier_call" else "put"
    barrier_type = product.barrier_type or "knock_out"
    return barrier_option(
        S,
        K,
        T,
        r,
        sigma,
        q,
        barrier,
        barrier_type,
        option_type,
        _barrier_direction(product),
        already_touched=product.barrier_touched,
    )


def price_product(product: ProductSpec, market: MarketState) -> dict | None:
    """风险中性公允价值 + 希腊值。custom 返回 None。

    希腊值口径：
    - vanilla / barrier：解析或中心差分希腊值已乘以 participation_rate。
    - MC 产品（snowball / autocallable）：票息结构不含参与率杠杆，希腊值不做
      participation_rate 缩放（parse_product_spec 也强制其 participation_rate == 1.0）。
    - gamma/theta 对 MC 产品取 0.0（未做二阶差分）。
    - vega_pct 恒定存在，为每 1 个 vol 点对应的名义本金占比，供波动率约束使用。
    """
    if product.product_type == "custom":
        return None

    S = market.spot
    reference = _reference_spot(product, S)
    sigma = market.volatility
    r = market.risk_free_rate
    q = market.dividend_yield
    T = _remaining_months(product) / 12.0
    K = product.strike_pct * reference
    m = product.notional / reference
    participation = product.participation_rate
    details: dict = {}

    if product.product_type in _VANILLA_TYPES:
        price_points = _vanilla_price_points(product, market)
        option_type = "call" if product.product_type == "vanilla_call" else "put"
        raw_greeks = bs_greeks(S, K, T, r, sigma, q, option_type)
        greeks = {
            # 下游直接乘 notional 得到 dollar-delta，因此这里返回
            # 现货相对变动的名义占比：S/reference * dOption/dS。
            "delta": raw_greeks["delta"] * participation * S / reference,
            "gamma": raw_greeks["gamma"] * participation,
            "vega": raw_greeks["vega"] * participation,
            "theta": raw_greeks["theta"] * participation,
        }
        greeks["vega_pct"] = greeks["vega"] / reference

        if product.principal_protected:
            fair_value = product.notional * (
                math.exp(-r * T) + participation * price_points / reference
            )
        else:
            fair_value = m * participation * price_points
        details = {"price_points": price_points}

    elif product.product_type in _BARRIER_TYPES:
        price_points = _barrier_price_points(product, market)
        barrier_level = (
            product.barrier_pct * reference
            if product.barrier_pct is not None
            else reference
        )
        option_type = "call" if product.product_type == "barrier_call" else "put"
        barrier_type = product.barrier_type or "knock_out"
        barrier_direction = _barrier_direction(product)

        def _g_points(s: float, vol: float) -> float:
            # 合约条款（行权价、障碍位）在产品设计时已固定为绝对水平，
            # 计算希腊值时只扰动现货价格，不随现货重新折算。
            return barrier_option(
                s,
                K,
                T,
                r,
                vol,
                q,
                barrier_level,
                barrier_type,
                option_type,
                barrier_direction,
                already_touched=product.barrier_touched,
            )

        # delta 在绝对价格上做中心差分（对"价格/现货"差分会混入 −price/S 项）
        bump_s = 0.01 * S
        delta_frac = (_g_points(S + bump_s, sigma) - _g_points(S - bump_s, sigma)) / (2.0 * bump_s)

        vol_bump = 0.01
        p_vol_up = _g_points(S, sigma + vol_bump) / reference
        p_vol_down = _g_points(S, max(sigma - vol_bump, 1e-6)) / reference
        vega_pct = (p_vol_up - p_vol_down) / 2.0
        vega_points = vega_pct * reference

        greeks = {
            # 与 vanilla 一致，将绝对价格导数转成相对现货变动占名义比。
            "delta": delta_frac * participation * S / reference,
            "gamma": 0.0,
            "vega": vega_points * participation,
            "theta": 0.0,
            "vega_pct": vega_pct * participation,
        }

        if product.principal_protected:
            fair_value = product.notional * (
                math.exp(-r * T) + participation * price_points / reference
            )
        else:
            fair_value = m * participation * price_points
        details = {"price_points": price_points}

    elif product.product_type in ("snowball", "autocallable"):
        if product.product_type == "snowball":
            kwargs = _snowball_kwargs(product)

            def _pv_frac(s: float, vol: float, seed: int = MC_SEED) -> float:
                # 触发水平锚定期初定盘价 S 的绝对水平：扰动现货 s 时通过
                # fixing_ratio = S / s 保持绝对水平不变（否则 delta 恒为 0）。
                return snowball_price(
                    s,
                    int(product.maturity_months),
                    r,
                    vol,
                    q,
                    product.coupon_rate or 0.0,
                    knock_in_pct=kwargs["knock_in_pct"],
                    knock_out_pct=kwargs["knock_out_pct"],
                    n_paths=MC_PATHS,
                    seed=seed,
                    fixing_ratio=reference / s,
                    already_knocked_in=product.knock_in_active,
                    elapsed_months=product.elapsed_months,
                )["fair_value"]

            mc_stats = snowball_price(
                S,
                int(product.maturity_months),
                r,
                sigma,
                q,
                product.coupon_rate or 0.0,
                knock_in_pct=kwargs["knock_in_pct"],
                knock_out_pct=kwargs["knock_out_pct"],
                n_paths=MC_PATHS,
                seed=MC_SEED,
                fixing_ratio=reference / S,
                already_knocked_in=product.knock_in_active,
                elapsed_months=product.elapsed_months,
            )
        else:
            kwargs = _autocallable_kwargs(product)

            def _pv_frac(s: float, vol: float, seed: int = MC_SEED) -> float:
                # 同雪球：赎回/敲入水平锚定期初定盘价，扰动现货靠 fixing_ratio。
                return autocallable_price(
                    s,
                    int(product.maturity_months),
                    r,
                    vol,
                    q,
                    product.coupon_rate or 0.0,
                    barrier_pct=kwargs["barrier_pct"],
                    autocall_pct=kwargs["autocall_pct"],
                    n_paths=MC_PATHS,
                    seed=seed,
                    fixing_ratio=reference / s,
                    already_knocked_in=product.knock_in_active,
                    elapsed_months=product.elapsed_months,
                    protected=product.principal_protected,
                )["fair_value"]

            mc_stats = autocallable_price(
                S,
                int(product.maturity_months),
                r,
                sigma,
                q,
                product.coupon_rate or 0.0,
                barrier_pct=kwargs["barrier_pct"],
                autocall_pct=kwargs["autocall_pct"],
                n_paths=MC_PATHS,
                seed=MC_SEED,
                fixing_ratio=reference / S,
                already_knocked_in=product.knock_in_active,
                elapsed_months=product.elapsed_months,
                protected=product.principal_protected,
            )

        pv_frac = mc_stats["fair_value"]
        fair_value = product.notional * pv_frac

        bump_s = 0.01 * S
        p_up = _pv_frac(S + bump_s, sigma)
        p_down = _pv_frac(S - bump_s, sigma)
        delta_frac = (p_up - p_down) / (0.02)

        vol_bump = 0.01
        p_vol_up = _pv_frac(S, sigma + vol_bump)
        p_vol_down = _pv_frac(S, max(sigma - vol_bump, 1e-6))
        vega_pct = (p_vol_up - p_vol_down) / 2.0
        vega_points = vega_pct * S

        greeks = {
            "delta": delta_frac,
            "gamma": 0.0,
            "vega": vega_points,
            "theta": 0.0,
            "vega_pct": vega_pct,
        }
        details = {"pv_frac": pv_frac, **{k: v for k, v in mc_stats.items() if k != "fair_value"}}

    else:
        raise PricingError(f"未知产品类型：{product.product_type}")

    hedging_cost = HEDGE_COST_RATIO * fair_value
    return {
        "fair_value": fair_value,
        "greeks": greeks,
        "hedging_cost": hedging_cost,
        "pricing_details": details,
    }


def evaluate_product(product: ProductSpec, market: MarketState) -> dict | None:
    """一站式产品评估：公允价值、希腊值、对客户报价、期望收益与损失比例。custom 返回 None。

    返回字典包含：fair_value、greeks、hedging_cost、pricing_details、
    client_price、expected_payoff 及 loss_frac（预期损失（期望口径））。
    """
    if product.product_type == "custom":
        return None

    result = price_product(product, market)
    if result is None:
        return None

    fair_value = result["fair_value"]
    result["client_price"] = MARKUP * fair_value
    result["expected_payoff"] = expected_payoff(product, market)

    if product.principal_protected:
        if product.product_type in ("snowball",):
            loss_frac = result["pricing_details"].get("expected_loss_frac", 0.0)
        elif product.product_type == "autocallable" and product.barrier_pct is not None:
            loss_frac = result["pricing_details"].get("expected_loss_frac", 0.0)
        else:
            loss_frac = 0.0
    elif product.product_type in _VANILLA_TYPES or product.product_type in _BARRIER_TYPES:
        loss_frac = result["client_price"] / product.notional
    elif product.product_type == "snowball":
        loss_frac = result["pricing_details"].get("expected_loss_frac", 0.0)
    elif product.product_type == "autocallable":
        if product.barrier_pct is not None:
            loss_frac = result["pricing_details"].get("expected_loss_frac", 0.0)
        else:
            loss_frac = 0.0
    else:
        loss_frac = 0.0

    result["loss_frac"] = loss_frac
    return result


# ---------------------------------------------------------------------------
# 真实世界漂移下超越门槛收益的概率
# ---------------------------------------------------------------------------


def hurdle_hit_prob(
    product: ProductSpec,
    market: MarketState,
    hurdle_annual: float,
    client_price: float,
) -> float | None:
    """真实世界漂移（μ = r + trend_alpha）下实现年化收益 ≥ hurdle_annual 的概率。

    - custom → None
    - vanilla_call / vanilla_put：闭合公式（对数正态尾概率）
    - barrier_call / barrier_put / autocallable / snowball：月度步长 MC（漂移 μ，不贴现）

    对于 barrier 产品，敲入/敲出采用月度步长近似连续监测（未实施
    Broadie-Glasserman-Kou 精确修正，属销售端合理近似）。

    定义：
        realized_annual = (payoff_yuan − client_price) / notional × 12 / t_months
    其中 t_months 为从当前估值日到提前赎回/到期的剩余月数。
    """
    if (
        not isinstance(product.participation_rate, (int, float))
        or isinstance(product.participation_rate, bool)
        or not math.isfinite(product.participation_rate)
        or product.participation_rate <= 0.0
    ):
        raise PricingError(
            "hurdle_hit_prob 要求 participation_rate > 0，"
            f"实际为 {product.participation_rate!r}"
        )

    if product.product_type == "custom":
        return None

    S0 = market.spot
    reference = _reference_spot(product, S0)
    sigma = market.volatility
    q = market.dividend_yield
    T = _remaining_months(product) / 12.0
    mu = market.risk_free_rate + market.trend_alpha
    K = product.strike_pct * reference
    participation = product.participation_rate
    protected = product.principal_protected
    notional = product.notional

    # payoff_frac ≥ required_frac ⟺ realized_annual ≥ hurdle_annual（定期到期精确）
    required_frac = client_price / notional + hurdle_annual * T

    if product.product_type in _VANILLA_TYPES:
        is_call = product.product_type == "vanilla_call"

        if is_call:
            if protected:
                # payoff_frac = 1 + participation * max(S_T − K, 0) / reference
                excess = required_frac - 1.0
                if excess <= 0.0:
                    return 1.0  # 保本地板已超过门槛
                S_T_star = K + excess * reference / participation
            else:
                # payoff_frac = participation * max(S_T − K, 0) / reference
                if required_frac <= 0.0:
                    return 1.0
                S_T_star = K + required_frac * reference / participation

            if S_T_star <= 0.0:
                return 1.0
            if T <= 0 or sigma <= 0:
                return 1.0 if S0 >= S_T_star else 0.0
            d = (math.log(S_T_star / S0) - (mu - q - 0.5 * sigma * sigma) * T) / (
                sigma * math.sqrt(T)
            )
            return _norm_cdf(-d)

        else:  # vanilla_put
            if protected:
                # payoff_frac = 1 + participation * max(K − S_T, 0) / reference
                excess = required_frac - 1.0
                if excess <= 0.0:
                    return 1.0
                max_payoff_frac = 1.0 + participation * K / reference
                if required_frac > max_payoff_frac + 1e-12:
                    return 0.0
                S_T_star = K - excess * reference / participation
                if S_T_star <= 0.0:
                    return 0.0
            else:
                # payoff_frac = participation * max(K − S_T, 0) / reference
                if required_frac <= 0.0:
                    return 1.0
                max_payoff_frac = participation * K / reference
                if required_frac > max_payoff_frac + 1e-12:
                    return 0.0
                S_T_star = K - required_frac * reference / participation
                if S_T_star <= 0.0:
                    return 0.0

            if T <= 0 or sigma <= 0:
                return 1.0 if S0 <= S_T_star else 0.0
            d = (math.log(S_T_star / S0) - (mu - q - 0.5 * sigma * sigma) * T) / (
                sigma * math.sqrt(T)
            )
            return _norm_cdf(d)

    # barrier / autocallable / snowball → 月度步长 MC
    return _mc_hit_prob(product, market, hurdle_annual, client_price, mu)


def _mc_hit_prob(
    product: ProductSpec,
    market: MarketState,
    hurdle_annual: float,
    client_price: float,
    drift: float,
) -> float:
    """月度步长 GBM 模拟，计算指定漂移下实现年化 ≥ hurdle_annual 的路径占比。

    与 _mc_stats 共享路径生成逻辑（GBM 月步长、敲入/敲出规则、票息结算），
    但逐路径计算实现年化收益而非贴现现值。barrier 产品障碍监测为月度步长近似。
    """
    S0 = market.spot
    reference = _reference_spot(product, S0)
    fixing_ratio = reference / S0
    sigma = market.volatility
    q = market.dividend_yield
    total_months = int(product.maturity_months)
    elapsed_months = product.elapsed_months
    remaining_months = _remaining_months(product)
    notional = product.notional
    participation = product.participation_rate
    protected = product.principal_protected

    dt = 1.0 / 12.0
    drift_term = (drift - q - 0.5 * sigma * sigma) * dt
    vol_term = sigma * math.sqrt(dt)

    rng = random.Random(MC_SEED)
    hits = 0

    for _ in range(MC_PATHS):
        ratio = 1.0  # S_t / S0
        cashflow_frac: float = 0.0  # payoff / notional
        term_t: int = remaining_months

        if product.product_type in _BARRIER_TYPES:
            barrier_pct_val = product.barrier_pct if product.barrier_pct is not None else 1.0
            barrier_ratio = barrier_pct_val * fixing_ratio
            barrier_type = product.barrier_type or "knock_out"
            is_call = product.product_type == "barrier_call"
            K_frac = product.strike_pct
            barrier_direction = _barrier_direction(product)

            touched = product.barrier_touched or _barrier_is_touched(
                S0, barrier_pct_val * reference, barrier_direction
            )
            for _t in range(1, remaining_months + 1):
                if barrier_type == "knock_out" and touched:
                    break
                z = rng.gauss(0.0, 1.0)
                ratio = ratio * math.exp(drift_term + vol_term * z)
                if not touched:
                    touched = _barrier_is_touched(
                        ratio, barrier_ratio, barrier_direction
                    )
                    # knock_out：已触碰即死，无需继续模拟到期
                    if barrier_type == "knock_out" and touched:
                        break

            # 对 knock_in：ratio 为到期价格比；knock_out 触碰后：payoff=0/floor，不依赖 ratio
            terminal_ref_ratio = ratio / fixing_ratio
            if is_call:
                opt_frac = participation * max(terminal_ref_ratio - K_frac, 0.0)
            else:
                opt_frac = participation * max(K_frac - terminal_ref_ratio, 0.0)

            active = (barrier_type == "knock_out" and not touched) or (
                barrier_type == "knock_in" and touched
            )
            option_part = opt_frac if active else 0.0
            cashflow_frac = (1.0 + option_part) if protected else option_part
            term_t = remaining_months

        elif product.product_type == "snowball":
            ko_pct = 1.03
            ki_pct = product.barrier_pct if product.barrier_pct is not None else 0.75
            coupon = product.coupon_rate or 0.0
            min_ratio = 1.0
            knocked_out = False

            for t_step in range(1, remaining_months + 1):
                z = rng.gauss(0.0, 1.0)
                ratio = ratio * math.exp(drift_term + vol_term * z)
                min_ratio = min(min_ratio, ratio)
                if ratio >= ko_pct * fixing_ratio:
                    cashflow_frac = 1.0 + coupon * (
                        (elapsed_months + t_step) / 12.0
                    )
                    term_t = t_step
                    knocked_out = True
                    break

            if not knocked_out:
                term_t = remaining_months
                knocked_in = (
                    product.knock_in_active or min_ratio <= ki_pct * fixing_ratio
                ) and ratio < fixing_ratio
                cashflow_frac = (
                    ratio / fixing_ratio
                    if knocked_in
                    else 1.0 + coupon * (total_months / 12.0)
                )

        elif product.product_type == "autocallable":
            autocall_pct = 1.05
            ki_pct = product.barrier_pct
            coupon = product.coupon_rate or 0.0
            min_ratio = 1.0
            knocked_out = False

            for t_step in range(1, remaining_months + 1):
                z = rng.gauss(0.0, 1.0)
                ratio = ratio * math.exp(drift_term + vol_term * z)
                min_ratio = min(min_ratio, ratio)
                if ratio >= autocall_pct * fixing_ratio:
                    cashflow_frac = 1.0 + coupon * (
                        (elapsed_months + t_step) / 12.0
                    )
                    term_t = t_step
                    knocked_out = True
                    break

            if not knocked_out:
                term_t = remaining_months
                knocked_in = (
                    product.knock_in_active
                    or (
                        ki_pct is not None
                        and min_ratio <= ki_pct * fixing_ratio
                    )
                ) and not protected
                cashflow_frac = (
                    min(ratio / fixing_ratio, 1.0)
                    if knocked_in
                    else 1.0 + coupon * (total_months / 12.0)
                )

        else:
            continue

        if term_t <= 0:
            realized_annual = (
                math.inf
                if cashflow_frac >= client_price / notional
                else -math.inf
            )
        else:
            realized_annual = (cashflow_frac - client_price / notional) * 12.0 / term_t
        if realized_annual >= hurdle_annual - 1e-12:
            hits += 1

    return hits / MC_PATHS


# ---------------------------------------------------------------------------
# 到期情景表
# ---------------------------------------------------------------------------


def build_payoff_table(product: ProductSpec, spot: float) -> str | None:
    """到期情景表（每 1 元名义本金的到期现金流，未贴现）。

    纯函数，刻意不接受 MarketState，防止路径依赖信息或市场观点泄漏。
    custom → None。

    指数涨跌情景：−20%,−10%,−5%,0%,+5%,+10%,+20%。
    - vanilla：单列终值
    - barrier：双列（未触发障碍 / 已触发障碍）+ 注释
    - autocallable / snowball：情景文字描述（含敲出/敲入场景）
    """
    if product.product_type == "custom":
        return None

    moves = [-0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20]
    move_labels = ["-20%", "-10%", "-5%", "0%", "+5%", "+10%", "+20%"]

    K = product.strike_pct * spot
    participation = product.participation_rate
    protected = product.principal_protected
    T_months = int(product.maturity_months)

    title = "【到期情景表｜每 1 元名义本金的到期现金流，未贴现】"

    def _opt_frac(S_T: float, is_call: bool) -> float:
        if is_call:
            return participation * max(S_T - K, 0.0) / spot
        return participation * max(K - S_T, 0.0) / spot

    if product.product_type in _VANILLA_TYPES:
        is_call = product.product_type == "vanilla_call"
        lines = [title, "", f"{'指数涨跌':>8}  {'到期收益':>10}", "-" * 24]
        for label, move in zip(move_labels, moves):
            S_T = spot * (1.0 + move)
            pf = (1.0 + _opt_frac(S_T, is_call)) if protected else _opt_frac(S_T, is_call)
            lines.append(f"{label:>8}  {pf:>10.4f}")
        return "\n".join(lines)

    if product.product_type in _BARRIER_TYPES:
        is_call = product.product_type == "barrier_call"
        barrier_type = product.barrier_type or "knock_out"

        lines = [
            title,
            "",
            f"{'指数涨跌':>8}  {'未触发障碍':>12}  {'已触发障碍':>12}",
            "-" * 40,
        ]
        for label, move in zip(move_labels, moves):
            S_T = spot * (1.0 + move)
            opt = _opt_frac(S_T, is_call)

            if barrier_type == "knock_out":
                # 未触发：期权有效
                no_touch = (1.0 + opt) if protected else opt
                # 已触发：期权死亡
                touch = 1.0 if protected else 0.0
            else:  # knock_in
                # 未触发：期权无效
                no_touch = 1.0 if protected else 0.0
                # 已触发：期权有效
                touch = (1.0 + opt) if protected else opt

            lines.append(f"{label:>8}  {no_touch:>12.4f}  {touch:>12.4f}")

        lines.append("")
        lines.append("注：障碍为路径观察，终值不能确定是否触发")
        return "\n".join(lines)

    if product.product_type in ("autocallable", "snowball"):
        coupon = product.coupon_rate or 0.0
        lines = [title, ""]

        lines.append(
            f"提前敲出（第 m 月赎回）：1+票息×m/12"
            f"（示例 m=3）：{1.0 + coupon * 3 / 12:.4f}"
        )
        lines.append(
            f"提前敲出（第 m 月赎回）：1+票息×m/12"
            f"（示例 m=6）：{1.0 + coupon * 6 / 12:.4f}"
        )
        lines.append(f"到期未敲入：{1.0 + coupon * T_months / 12:.4f}（票息全收）")

        if product.product_type == "snowball":
            lines.append("到期已敲入：按标的终值结算（雪球=终值/期初）")
            lines.append(f"  示例：指数-20% → 终值收益约 0.8000")
        else:
            lines.append("到期已敲入：按标的终值结算（autocall=min(终值/期初,1)）")
            lines.append(f"  示例：指数-20% → 终值收益约 0.8000")

        return "\n".join(lines)

    return None


# ---------------------------------------------------------------------------
# 最高情景收益摘要行（报价展示）
# ---------------------------------------------------------------------------


def best_case_line(
    product: ProductSpec,
    market: MarketState,
    client_price: float,
) -> str | None:
    """交易台报价中的最高情景收益摘要行（风险中性口径）。

    使用风险中性漂移（drift = r，不含 trend_alpha），防止内幕信息泄漏。custom → None。

    - snowball / autocallable：最高年化 = coupon_rate；概率为 RN 下实现票息情景的概率
      （1 − 敲入概率，即敲出 + 到期未敲入路径）。
      格式：最高情景收益：票息年化 {coupon:.1%}｜风险中性实现概率约 {p:.0%}
    - knock_out barrier（封顶结构）：以障碍位为封顶情景，年化净收益 + RN 实现概率。
    - 无封顶（vanilla / knock_in barrier）：以 ±20% 情景为示意，标注「上不封顶」。
    """
    if product.product_type == "custom":
        return None

    S0 = market.spot
    reference = _reference_spot(product, S0)
    r = market.risk_free_rate
    sigma = market.volatility
    q = market.dividend_yield
    remaining_months = _remaining_months(product)
    T = remaining_months / 12.0
    T_months = int(product.maturity_months)
    participation = product.participation_rate
    protected = product.principal_protected
    notional = product.notional
    K = product.strike_pct * reference

    # 风险中性市场态（trend_alpha = 0，防止漏出真实漂移）
    rn_market = MarketState(
        round_num=market.round_num,
        index_name=market.index_name,
        spot=S0,
        volatility=sigma,
        risk_free_rate=r,
        dividend_yield=q,
        recent_trend=market.recent_trend,
        vix_level=market.vix_level,
        trend_alpha=0.0,
    )

    # ---- coupon 产品 ------------------------------------------------
    if product.product_type in ("snowball", "autocallable"):
        coupon = product.coupon_rate or 0.0

        if product.product_type == "snowball":
            kwargs = _snowball_kwargs(product)
            rn_stats = _mc_stats(
                S0,
                T_months,
                r,
                sigma,
                q,
                "snowball",
                coupon_rate=coupon,
                knock_in_pct=kwargs["knock_in_pct"],
                knock_out_pct=kwargs["knock_out_pct"],
                discount=False,
                fixing_ratio=reference / S0,
                already_knocked_in=product.knock_in_active,
                elapsed_months=product.elapsed_months,
            )
        else:
            kwargs = _autocallable_kwargs(product)
            rn_stats = _mc_stats(
                S0,
                T_months,
                r,
                sigma,
                q,
                "autocallable",
                coupon_rate=coupon,
                knock_in_pct=kwargs["barrier_pct"],
                autocall_pct=kwargs["autocall_pct"],
                protected=protected,
                discount=False,
                fixing_ratio=reference / S0,
                already_knocked_in=product.knock_in_active,
                elapsed_months=product.elapsed_months,
            )

        # 票息情景 = 敲出 + 到期未敲入 = 1 − 敲入概率
        p_coupon = 1.0 - rn_stats["knock_in_prob"]
        return f"最高情景收益：票息年化 {coupon:.1%}｜风险中性实现概率约 {p_coupon:.0%}"

    # ---- 封顶结构（knock_out barrier）--------------------------------
    is_capped = (
        product.product_type in _BARRIER_TYPES
        and product.barrier_type == "knock_out"
    )

    if is_capped:
        barrier_S = (product.barrier_pct or 1.0) * reference
        is_call = product.product_type == "barrier_call"

        if is_call:
            opt_at_cap = participation * max(barrier_S - K, 0.0) / reference
        else:
            opt_at_cap = participation * max(K - barrier_S, 0.0) / reference

        cap_payoff_frac = (1.0 + opt_at_cap) if protected else opt_at_cap
        cap_annual = (
            (cap_payoff_frac - client_price / notional)
            * 12.0
            / max(remaining_months, 1)
        )

        # P_RN(realized_annual ≥ cap_annual)，以风险中性市场态调用 hurdle_hit_prob
        p = hurdle_hit_prob(product, rn_market, cap_annual, client_price)
        if p is None:
            p = 0.0
        return f"最高情景收益（封顶年化）：{cap_annual:.1%}｜风险中性实现概率约 {p:.0%}"

    # ---- 无封顶：vanilla 或 knock_in barrier --------------------------
    is_call = product.product_type in ("vanilla_call", "barrier_call")
    move = 0.20 if is_call else -0.20
    S_T_illus = S0 * (1.0 + move)

    if is_call:
        opt_frac = participation * max(S_T_illus - K, 0.0) / reference
    else:
        opt_frac = participation * max(K - S_T_illus, 0.0) / reference

    illus_payoff_frac = (1.0 + opt_frac) if protected else opt_frac
    illus_annual = (
        (illus_payoff_frac - client_price / notional)
        * 12.0
        / max(remaining_months, 1)
    )

    # RN 概率：P(S_T ≥ 1.2*S0) for call, P(S_T ≤ 0.8*S0) for put
    if product.product_type in _VANILLA_TYPES:
        p = hurdle_hit_prob(product, rn_market, illus_annual, client_price)
        if p is None:
            p = 0.0
    else:
        # barrier knock_in：用对数正态尾概率近似（忽略敲入条件）
        if T > 0 and sigma > 0:
            d = (
                math.log(S_T_illus / S0) - (r - q - 0.5 * sigma * sigma) * T
            ) / (sigma * math.sqrt(T))
            p = _norm_cdf(-d) if is_call else _norm_cdf(d)
        else:
            p = 0.0

    return (
        f"最高情景收益（年化）：{illus_annual:.1%}"
        f"｜风险中性实现概率约 {p:.0%}（上不封顶，以 ±20% 情景示意）"
    )


# ===========================================================================
# v2 定价经济学核心：MC 诊断、报价经济学、连续损失度量、系数标定
# ---------------------------------------------------------------------------
# 与旧路径（MARKUP / HEDGE_COST_RATIO / evaluate_product 的 client_price）并存，
# benchmark.py 仍用旧路径，下一波才切换。本节全部为纯函数 / 确定性计算。
# ===========================================================================


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# MC 诊断：pv 标准误、事件概率、损失分位（vanilla/barrier/snowball/autocallable 统一口径）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCDiagnostics:
    """单位名义本金口径的蒙特卡洛诊断量（风险中性漂移、贴现现值）。

    所有量都不含 notional（对名义线性的 fair value 也以占比表示），
    故可按"去掉 notional 的结构键"跨候选缓存。
    - pv_mean_frac：贴现现值均值占名义比（≈ fair_value / notional）。
    - pv_std_frac / pv_se_frac：逐路径 pv 的标准差 / 标准误占名义比。
    - expected_loss_frac：期望损失占名义比（par 票据用 1−终值，期权票据为 0）。
    - event_probs：路径事件概率（vanilla 空；barrier touch；snowball/autocallable KI/KO）。
    - expected_life_months：期望存续月份。
    - p95_loss_frac：损失 95 分位占名义比。
    """

    pv_mean_frac: float
    pv_std_frac: float
    pv_se_frac: float
    expected_loss_frac: float
    event_probs: dict
    expected_life_months: float
    p95_loss_frac: float


# 结构键 -> MCDiagnostics 的进程内缓存；oracle 枚举时同结构不同名义共享一次算力。
_DIAG_CACHE: dict = {}


def _diag_cache_key(product: ProductSpec, market: MarketState, n_paths: int, seed: int) -> tuple:
    """去掉 notional / target_client / pitch / hedging_plan 的结构+市场+MC 参数键。"""
    return (
        product.product_type,
        int(product.maturity_months),
        product.strike_pct,
        product.barrier_pct,
        product.barrier_type,
        product.barrier_direction,
        product.coupon_rate,
        product.participation_rate,
        product.principal_protected,
        product.reference_spot,
        product.barrier_touched,
        product.knock_in_active,
        product.elapsed_months,
        market.spot,
        market.volatility,
        market.risk_free_rate,
        market.dividend_yield,
        int(n_paths),
        int(seed),
    )


def _mc_vanilla_barrier_stats(
    product: ProductSpec, market: MarketState, n_paths: int, seed: int
) -> MCDiagnostics:
    """vanilla / barrier 的诊断 MC：月度步长 GBM，风险中性漂移，贴现终值。

    barrier 触碰采用与 _mc_hit_prob 一致的月度步长近似（knock_out 触碰即提前终止）。
    损失口径：期权票据（非保本 vanilla/barrier）的最大损失是权利金（由 ClientLossMeasure
    的 premium 项承担），故此处逐路径损失记 0；保本票据地板为 1，损失恒 0。
    """
    S0 = market.spot
    reference = _reference_spot(product, S0)
    fixing_ratio = reference / S0
    sigma = market.volatility
    q = market.dividend_yield
    r = market.risk_free_rate
    total_months = int(product.maturity_months)
    remaining_months = _remaining_months(product)
    T = remaining_months / 12.0
    participation = product.participation_rate
    protected = product.principal_protected
    K_frac = product.strike_pct
    disc = math.exp(-r * T)

    dt = 1.0 / 12.0
    drift_term = (r - q - 0.5 * sigma * sigma) * dt
    vol_term = sigma * math.sqrt(dt)

    is_barrier = product.product_type in _BARRIER_TYPES
    is_call = product.product_type in ("vanilla_call", "barrier_call")
    if is_barrier:
        barrier_pct_val = product.barrier_pct if product.barrier_pct is not None else 1.0
        barrier_ratio = barrier_pct_val * fixing_ratio
        barrier_type = product.barrier_type or "knock_out"
        barrier_direction = _barrier_direction(product)

    rng = random.Random(seed)
    pvs: list[float] = []
    losses: list[float] = []
    touch_count = 0

    for _ in range(n_paths):
        ratio = 1.0
        touched = is_barrier and (
            product.barrier_touched
            or _barrier_is_touched(
                S0, barrier_pct_val * reference, barrier_direction
            )
        )
        for _t in range(1, remaining_months + 1):
            if is_barrier and barrier_type == "knock_out" and touched:
                break
            z = rng.gauss(0.0, 1.0)
            ratio = ratio * math.exp(drift_term + vol_term * z)
            if is_barrier and not touched:
                touched = _barrier_is_touched(
                    ratio, barrier_ratio, barrier_direction
                )
                if barrier_type == "knock_out" and touched:
                    break

        if is_barrier and touched:
            touch_count += 1

        terminal_ref_ratio = ratio / fixing_ratio
        if is_call:
            opt = participation * max(terminal_ref_ratio - K_frac, 0.0)
        else:
            opt = participation * max(K_frac - terminal_ref_ratio, 0.0)

        if is_barrier:
            active = (barrier_type == "knock_out" and not touched) or (
                barrier_type == "knock_in" and touched
            )
            opt_part = opt if active else 0.0
        else:
            opt_part = opt

        payoff_frac = (1.0 + opt_part) if protected else opt_part
        pvs.append(payoff_frac * disc)
        losses.append(max(0.0, 1.0 - payoff_frac) if protected else 0.0)

    n = n_paths
    pv_mean = sum(pvs) / n
    pv_sq = sum(v * v for v in pvs) / n
    pv_std = math.sqrt(max(pv_sq - pv_mean * pv_mean, 0.0))
    pv_se = pv_std / math.sqrt(n)
    sorted_losses = sorted(losses)
    p95 = sorted_losses[min(int(0.95 * n), n - 1)]
    event_probs = {"touch": touch_count / n} if is_barrier else {}

    return MCDiagnostics(
        pv_mean_frac=pv_mean,
        pv_std_frac=pv_std,
        pv_se_frac=pv_se,
        expected_loss_frac=sum(losses) / n,
        event_probs=event_probs,
        expected_life_months=float(total_months),
        p95_loss_frac=p95,
    )


def mc_diagnostics(
    product: ProductSpec,
    market: MarketState,
    *,
    n_paths: int = 4096,
    seed: int,
) -> MCDiagnostics:
    """统一 payoff sampler 产出的 MC 诊断量，按去掉 notional 的结构键缓存。

    snowball / autocallable 复用 _mc_stats（含二阶矩与事件计数）；
    vanilla / barrier 走 _mc_vanilla_barrier_stats（复用 _mc_hit_prob 的月度触碰逻辑）。
    风险中性漂移、贴现现值；同 seed 同结果（common random numbers）。
    """
    key = _diag_cache_key(product, market, n_paths, seed)
    cached = _DIAG_CACHE.get(key)
    if cached is not None:
        return cached

    ptype = product.product_type
    if ptype == "custom":
        raise PricingError("custom 产品不支持 MC 诊断")

    if ptype in ("snowball", "autocallable"):
        if ptype == "snowball":
            kw = _snowball_kwargs(product)
            stats = _mc_stats(
                market.spot,
                int(product.maturity_months),
                market.risk_free_rate,
                market.volatility,
                market.dividend_yield,
                "snowball",
                coupon_rate=product.coupon_rate or 0.0,
                knock_in_pct=kw["knock_in_pct"],
                knock_out_pct=kw["knock_out_pct"],
                n_paths=n_paths,
                seed=seed,
                fixing_ratio=_reference_spot(product, market.spot) / market.spot,
                already_knocked_in=product.knock_in_active,
                elapsed_months=product.elapsed_months,
            )
        else:
            kw = _autocallable_kwargs(product)
            stats = _mc_stats(
                market.spot,
                int(product.maturity_months),
                market.risk_free_rate,
                market.volatility,
                market.dividend_yield,
                "autocallable",
                coupon_rate=product.coupon_rate or 0.0,
                knock_in_pct=kw["barrier_pct"],
                autocall_pct=kw["autocall_pct"],
                protected=product.principal_protected,
                n_paths=n_paths,
                seed=seed,
                fixing_ratio=_reference_spot(product, market.spot) / market.spot,
                already_knocked_in=product.knock_in_active,
                elapsed_months=product.elapsed_months,
            )
        diag = MCDiagnostics(
            pv_mean_frac=stats["fair_value"],
            pv_std_frac=stats["pv_std_frac"],
            pv_se_frac=stats["pv_se_frac"],
            expected_loss_frac=stats["expected_loss_frac"],
            event_probs={
                "knock_in": stats["knock_in_prob"],
                "knock_out": stats["knock_out_prob"],
            },
            expected_life_months=stats["expected_life_months"],
            p95_loss_frac=stats["p95_loss_frac"],
        )
    elif ptype in _VANILLA_TYPES or ptype in _BARRIER_TYPES:
        diag = _mc_vanilla_barrier_stats(product, market, n_paths, seed)
    else:
        raise PricingError(f"未知产品类型：{ptype}")

    _DIAG_CACHE[key] = diag
    return diag


# ---------------------------------------------------------------------------
# 报价经济学：dealer_margin = N·(r_c·suitability − r_h)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuotePolicy:
    """报价系数（draft_codex §3 公式 + draft_opus §3.2 适配度）。

    a_* 为对客加成率系数、b_* 为对冲成本系数、client_cap / hedge_cap 为封顶。
    suit_* 为 suitability 四因子的权重（加权乘积的指数）与两个失配地板。
    系数须在开发集上标定后冻结，held-out 后不得再调。
    """

    a_f: float = 0.030
    a_v: float = 0.25
    a_p: float = 0.002
    a_b: float = 0.75
    b_f: float = 0.020
    b_v: float = 0.75
    b_p: float = 0.004
    b_b: float = 1.00
    b_l: float = 0.003
    b_q: float = 0.015
    client_cap: float = 0.08
    hedge_cap: float = 0.10
    diagnostic_paths: int = 4096
    suit_w_type: float = 1.0
    suit_w_protection: float = 1.0
    suit_w_maturity: float = 1.0
    suit_w_hurdle: float = 1.0
    suit_whitelist_miss: float = 0.20
    suit_protection_miss: float = 0.10


@dataclass(frozen=True)
class QuoteEconomics:
    """一次报价的经济结果分解。dealer_margin 允许为负且不截断。"""

    fair_value: float
    client_price: float
    hedging_cost: float
    dealer_margin: float
    margin_rate: float
    risk_adjusted_margin: float
    suitability: float
    breakdown: dict


def _suitability(
    product: ProductSpec, client: ClientProfile, pricing: dict, policy: QuotePolicy
) -> tuple[float, dict]:
    """客户适配度 ∈ [0,1]（draft_opus §3.2 四因子加权乘积）。

    四因子：产品类型白名单命中、保本偏好匹配、期限贴合度、hurdle_hit_prob
    相对 min_hit_prob 的余量。任一失配把加成率压向 0，使乱塞产品无利可图。
    """
    # 1) 白名单命中
    if client.allowed_product_types is None or product.product_type in client.allowed_product_types:
        s_type = 1.0
    else:
        s_type = policy.suit_whitelist_miss

    # 2) 保本偏好匹配（要求保本却不保本，或保守偏好却不保本 → 失配）
    needs_protection = client.principal_protection_required or client.risk_appetite == "conservative"
    if needs_protection and not product.principal_protected:
        s_protect = policy.suit_protection_miss
    else:
        s_protect = 1.0

    # 3) 期限贴合度（落在授权期限内为 1，超出线性衰减）
    mm = client.max_maturity_months
    if mm > 0 and product.maturity_months > mm:
        s_maturity = max(0.0, 1.0 - (product.maturity_months - mm) / mm)
    else:
        s_maturity = 1.0

    # 4) hurdle_hit_prob 相对 min_hit_prob 的余量（pricing 未带 hurdle 时中性 1.0）
    hh = pricing.get("hurdle_hit_prob")
    if hh is None or client.min_hit_prob <= 0:
        s_hurdle = 1.0
    elif hh >= client.min_hit_prob:
        s_hurdle = 1.0
    else:
        s_hurdle = max(0.0, hh / client.min_hit_prob)

    suit = (
        s_type ** policy.suit_w_type
        * s_protect ** policy.suit_w_protection
        * s_maturity ** policy.suit_w_maturity
        * s_hurdle ** policy.suit_w_hurdle
    )
    suit = _clip(suit, 0.0, 1.0)
    detail = {
        "s_type": s_type,
        "s_protection": s_protect,
        "s_maturity": s_maturity,
        "s_hurdle": s_hurdle,
    }
    return suit, detail


def _path_dependence(event_probs: dict) -> float:
    """路径依赖度 P（draft_codex §3）：vanilla 为 0，否则 2·√mean(p(1−p))，clip 到 [0,1]。"""
    if not event_probs:
        return 0.0
    vals = [p * (1.0 - p) for p in event_probs.values()]
    mean_val = sum(vals) / len(vals)
    return _clip(2.0 * math.sqrt(mean_val), 0.0, 1.0)


def quote_economics(
    pricing: dict,
    diag: MCDiagnostics,
    *,
    stress_loss: float,
    post_notional: float,
    capacity_notional: float,
    policy: QuotePolicy,
    product: ProductSpec,
    client: ClientProfile,
) -> QuoteEconomics:
    """按 draft_codex §3 计算报价经济学，r_c 额外乘 draft_opus §3.2 的 suitability。

    client_price = F + N·r_c·suit；hedging_cost = F + N·r_h；
    dealer_margin = N·(r_c·suit − r_h)（允许为负，不截断）。
    Q²（容量冲击）进 r_h，破除"永远上满 100% 名义"哑策略。
    """
    N = product.notional
    F = pricing["fair_value"]
    f = F / N if N else 0.0
    V = abs(pricing["greeks"].get("vega_pct", 0.0))
    P = _path_dependence(diag.event_probs)
    B = 2.0 * 1.96 * diag.pv_se_frac
    L = stress_loss / N if N else 0.0
    Q = post_notional / capacity_notional if capacity_notional > 0 else 0.0

    r_c = _clip(policy.a_f * f + policy.a_v * V + policy.a_p * P + policy.a_b * B, 0.0, policy.client_cap)
    r_h = _clip(
        policy.b_f * f
        + policy.b_v * V
        + policy.b_p * P
        + policy.b_b * B
        + policy.b_l * L
        + policy.b_q * Q * Q,
        0.0,
        policy.hedge_cap,
    )

    suit, suit_detail = _suitability(product, client, pricing, policy)
    r_c_eff = r_c * suit

    client_price = F + N * r_c_eff
    hedging_cost = F + N * r_h
    dealer_margin = N * (r_c_eff - r_h)
    margin_rate = dealer_margin / N if N else 0.0
    risk_adjusted_margin = dealer_margin / max(stress_loss, 0.01 * N)

    breakdown = {
        "f": f,
        "V": V,
        "P": P,
        "B": B,
        "L": L,
        "Q": Q,
        "r_c": r_c,
        "r_h": r_h,
        "r_c_effective": r_c_eff,
        "markup_amount": N * r_c_eff,
        "hedge_amount": N * r_h,
        "suitability": suit,
        **{f"suit_{k}": v for k, v in suit_detail.items()},
    }

    return QuoteEconomics(
        fair_value=F,
        client_price=client_price,
        hedging_cost=hedging_cost,
        dealer_margin=dealer_margin,
        margin_rate=margin_rate,
        risk_adjusted_margin=risk_adjusted_margin,
        suitability=suit,
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# 客户连续损失度量（draft_codex §7）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClientLossMeasure:
    """连续 loss proxy（不称数学 worst-case）。observed = max(三者)。"""

    expected_loss_frac: float
    premium_at_risk_frac: float
    stress_loss_frac: float
    observed_loss_frac: float
    worst_stress_id: str


def client_loss_measure(
    product: ProductSpec,
    pricing: dict,
    stress_loss: float,
    *,
    worst_stress_id: str,
) -> ClientLossMeasure:
    """expected / premium / stress 三路损失取 max 作为连续观测损失。

    - expected：MC 期望损失（pricing_details.expected_loss_frac，缺省 0）。
    - premium：pricing.loss_frac（vanilla/barrier 非保本即权利金占比；保本为 0）。
    - stress：stress_loss / notional。
    """
    # premium 分量刻意读 pricing["loss_frac"]，即 evaluate_product 用固定 MARKUP
    # 常数算出的权利金占比，而不是 quote_economics 用当轮 QuotePolicy 算出的
    # client_price。这是有意的口径选择：CLIENT_LOSS_BUDGET_V2 是硬约束（对客户
    # 损失设上限），若这一项随 QuotePolicy 标定/敏感性缩放而漂移，同一份产品在
    # 系数重新标定前后会得到不同的硬约束判定，约束语义就不稳定了。用policy-无关
    # 的基准定价口径，使这条硬约束的含义在 QuotePolicy 冻结、重标定或做敏感性
    # 分析时都保持不变。
    expected = float(pricing.get("pricing_details", {}).get("expected_loss_frac", 0.0))
    premium = float(pricing.get("loss_frac", 0.0))
    stress = stress_loss / product.notional if product.notional else 0.0
    observed = max(expected, premium, stress)
    return ClientLossMeasure(
        expected_loss_frac=expected,
        premium_at_risk_frac=premium,
        stress_loss_frac=stress,
        observed_loss_frac=observed,
        worst_stress_id=worst_stress_id,
    )


# ---------------------------------------------------------------------------
# 系数标定：mirage.benchmark_cli 的 calibrate-margin 子命令接线于此
# ---------------------------------------------------------------------------

# 与 benchmark._stress_loss 一致的冻结压力网格，但用 fair value 差避免 quote 公式循环。
_STRESS_SCENARIOS = (("spot_down_20", 0.8, 0.0), ("spot_up_20", 1.2, 0.0), ("vol_up_10", 1.0, 0.10))


def _fair_value_stress_loss(product: ProductSpec, market: MarketState) -> tuple[float, str]:
    """压力网格下 fair value 的最大跌幅（人民币）及最坏情景 id。"""
    base = price_product(product, market)
    if base is None:
        return 0.0, ""
    base_fv = base["fair_value"]
    worst_loss = 0.0
    worst_id = ""
    for stress_id, spot_factor, vol_add in _STRESS_SCENARIOS:
        # 压力仅扰动现货/波动率，不改写合约条款。对旧式
        # reference_spot=None 的发行时产品，在压力副本上显式冻结
        # 基准市场现货。这不仅锁定 strike/barrier，也锁定雪球/
        # autocall 的隐式 1.03/1.05 敲出水平。
        stressed_product = replace(
            product,
            reference_spot=_reference_spot(product, market.spot),
            barrier_direction=(
                _barrier_direction(product)
                if product.product_type in _BARRIER_TYPES
                else None
            ),
        )
        stressed_market = replace(
            market,
            spot=market.spot * spot_factor,
            volatility=max(market.volatility + vol_add, 1e-6),
        )
        stressed = price_product(stressed_product, stressed_market)
        if stressed is None:
            continue
        loss = max(base_fv - stressed["fair_value"], 0.0)
        if loss > worst_loss:
            worst_loss = loss
            worst_id = stress_id
    return worst_loss, worst_id


def scale_quote_policy_markup(policy: QuotePolicy, factor: float) -> QuotePolicy:
    """按整体缩放因子缩放对客加成系数 a_*（对冲系数 b_* 保持不变）。

    只缩放 a_* 才能单调调节"正 margin 占比"：等比缩放 a_* 与 b_* 不改变
    dealer_margin 符号，无法命中 20–60% 目标带。calibrate_quote_policy 的网格
    搜索与标定后系数的 0.8/1.0/1.2 敏感性分析（benchmark_cli calibrate-margin）
    共用这一个缩放定义。
    """
    return replace(
        policy,
        a_f=policy.a_f * factor,
        a_v=policy.a_v * factor,
        a_p=policy.a_p * factor,
        a_b=policy.a_b * factor,
    )


def _ranking_nondegenerate(infos: list[tuple[ProductSpec, float]]) -> bool:
    """结构间 margin 排序是否非退化：最优 margin 不是"最大名义 vanilla"。"""
    if not infos:
        return False
    best_product = max(infos, key=lambda item: item[1])[0]
    vanillas = [p for p, _ in infos if p.product_type in _VANILLA_TYPES]
    if not vanillas:
        return True
    max_vanilla_notional = max(p.notional for p in vanillas)
    best_is_max_vanilla = (
        best_product.product_type in _VANILLA_TYPES
        and abs(best_product.notional - max_vanilla_notional) < 1e-6
    )
    return not best_is_max_vanilla


def evaluate_quote_policy(
    policy: QuotePolicy,
    dev_cases: list[tuple[ProductSpec, MarketState, ClientProfile]],
    *,
    seed: int = MC_SEED,
    capacity_fn=None,
) -> dict:
    """在开发案例上跑一遍给定 QuotePolicy：正 margin 占比 + 结构排序是否非退化。

    calibrate_quote_policy 的网格搜索每格评估，以及标定后系数的 0.8/1.0/1.2
    敏感性分析（benchmark_cli calibrate-margin）共用这一段逻辑，避免两处定义
    漂移。mc_diagnostics 按结构键缓存，同一批 dev_cases 在不同 policy/factor
    间重复调用不会重复跑蒙特卡洛。capacity_fn(client) -> capacity_notional，
    默认取 client.capital。
    """
    margins: list[float] = []
    infos: list[tuple[ProductSpec, float]] = []
    for product, market, client in dev_cases:
        pricing = evaluate_product(product, market)
        if pricing is None:
            continue
        diag = mc_diagnostics(product, market, n_paths=policy.diagnostic_paths, seed=seed)
        stress_loss, _ = _fair_value_stress_loss(product, market)
        capacity = capacity_fn(client) if capacity_fn is not None else client.capital
        qe = quote_economics(
            pricing,
            diag,
            stress_loss=stress_loss,
            post_notional=product.notional,
            capacity_notional=capacity,
            policy=policy,
            product=product,
            client=client,
        )
        margins.append(qe.dealer_margin)
        infos.append((product, qe.dealer_margin))
    n = len(margins)
    positive_rate = sum(1 for m in margins if m > 0) / n if n else 0.0
    return {
        "positive_margin_rate": positive_rate,
        "ranking_nondegenerate": _ranking_nondegenerate(infos),
        "n_cases": n,
    }


def calibrate_quote_policy(
    dev_cases: list[tuple[ProductSpec, MarketState, ClientProfile]],
    *,
    base_policy: QuotePolicy | None = None,
    factors: tuple[float, ...] | None = None,
    target: tuple[float, float] = (0.20, 0.60),
    seed: int = MC_SEED,
    capacity_fn=None,
) -> tuple[QuotePolicy, dict]:
    """在开发案例上网格搜索 QuotePolicy 的整体缩放因子并冻结。

    目标：20–60% 可行案例为正 margin，且结构间 margin 排序非退化
    （最大名义 vanilla 不恒为最优）。capacity_fn(client) -> capacity_notional，
    默认取 client.capital。返回 (标定后 QuotePolicy, 标定报告 dict)。
    """
    base = base_policy or QuotePolicy()
    if factors is None:
        factors = tuple(value / 10 for value in range(1, 31))  # 0.1 .. 3.0
    if not dev_cases:
        raise PricingError("标定需要至少一个开发案例")

    rows: list[dict] = []
    for factor in factors:
        policy = scale_quote_policy_markup(base, factor)
        result = evaluate_quote_policy(policy, dev_cases, seed=seed, capacity_fn=capacity_fn)
        rows.append({"factor": float(factor), **result})

    lo, hi = target
    mid = (lo + hi) / 2.0
    in_band = [
        r for r in rows if lo <= r["positive_margin_rate"] <= hi and r["ranking_nondegenerate"]
    ]
    pool = in_band or [r for r in rows if lo <= r["positive_margin_rate"] <= hi] or rows
    selected = min(pool, key=lambda r: (abs(r["positive_margin_rate"] - mid), r["factor"]))
    calibrated = scale_quote_policy_markup(base, selected["factor"])
    report = {
        "selected_factor": selected["factor"],
        "selected_positive_margin_rate": selected["positive_margin_rate"],
        "ranking_nondegenerate": selected["ranking_nondegenerate"],
        "target": list(target),
        "within_target": selected in in_band,
        "grid": rows,
        "warning": "freeze this policy before evaluating held-out episodes",
    }
    return calibrated, report
