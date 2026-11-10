import copy
import importlib
import numpy as np
import pytest


def modules():
    market = importlib.import_module('stochopia.research.liquidity_market')
    execution = importlib.import_module('stochopia.research.liquidity_execution')
    return market, execution


def test_environment_exists():
    from importlib.util import find_spec
    assert find_spec('stochopia.research.liquidity_market') is not None
    assert find_spec('stochopia.research.liquidity_execution') is not None


def panel(scenario='normal', **config):
    market, _ = modules()
    return market.generate_panel(seed=12, n_paths=4, config={'days':2, 'steps_per_day':4, **config}, scenario=scenario)


def risk(**changes):
    return dict(buyer_capital=1e6, seller_capital=1e6, fee_bps=1., borrow_rate=.03,
                gross_limit=2., **changes)


def strategy(targets, rebalance_every=1):
    return {'track':'synthetic', 'policy':{'targets':targets, 'rebalance_every':rebalance_every}, 'params':{}}


def test_paired_factors_root_and_price_separation():
    base = panel()
    for scenario, factor in [('spread_bad','spread'), ('depth_bad','depth'), ('recovery_bad','recovery')]:
        other = panel(scenario)
        for name in base.data:
            np.testing.assert_array_equal(other.data[name][:,0], base.data[name][:,0])
        for name in base.paths.spot:
            np.testing.assert_array_equal(other.paths.spot[name], base.paths.spot[name])
        for name in ['spread','depth','recovery','vol']:
            if name != factor:
                np.testing.assert_array_equal(base.data[name], other.data[name])
        assert not np.array_equal(base.data[factor], other.data[factor])
    assert np.corrcoef(base.data['spread'][:,1:].ravel(), base.data['depth'][:,1:].ravel())[0,1] != pytest.approx(1)
    assert base.dt == 1/(252*78)
    assert len(base.paths.spot) == 11


def test_prices_and_vol_interventions_pair_quotes_at_inception():
    market, execution = modules()
    base=panel(); baseline=execution.instrument_prices(base)
    for scene in ['price_up','price_down','vol_high','vol_low']:
        other=panel(scene)
        for name, quote in execution.instrument_prices(other).items():
            np.testing.assert_array_equal(quote[:,0], baseline[name][:,0])
        assert not np.array_equal(other.data['basket'], base.data['basket'])


def test_cash_control_and_real_costs():
    _, execution = modules()
    flat=panel(sigma=0, factor_scale=0)
    zero=execution.execute_strategy(strategy({'basket':'0'}), flat, risk())
    assert np.all(zero['pnl']==0) and not zero['funding_violation'].any()
    bought=execution.execute_strategy(strategy({'basket':'0.2'}, 100), flat, risk())
    np.testing.assert_allclose(bought['pnl'], -bought['costs'], atol=1e-7)
    assert np.all(bought['costs']>0)
    assert not bought['funding_violation'].any()
    assert np.all(bought['final_inventory']==0)


def test_capacity_terminal_residual_and_funding_failure():
    _, execution=modules()
    thin=panel(depth=.001, factor_scale=0)
    result=execution.execute_strategy(strategy({'basket':'0.8'}), thin, risk())
    assert np.all(result['unfinished']) and np.all(result['funding_violation'])
    assert np.all(result['turnover']<.8)
    rich=execution.execute_strategy(strategy({'basket':'5'},100), panel(depth=10), risk())
    assert np.all(rich['funding_violation'])
    assert np.all(rich['peak_funding']<=1e6+1e-5)
    short=execution.execute_strategy(strategy({'put':'-1'},100), panel(depth=10), risk())
    assert short['funding_violation'].any()


def test_strategy_adapted_history_and_rejects_future_or_fake_assets():
    _, execution=modules()
    original=panel()
    changed=copy.deepcopy(original)
    for value in changed.paths.spot.values(): value[:,5:] *= 1.5
    for value in changed.paths.external.values(): value[:,5:] *= 2
    artifact=strategy({'basket':'(clip (lin 1 (avg_arith (external depth) daily) 0) 0 0.2)'})
    a=execution.execute_strategy(artifact,original,risk())
    b=execution.execute_strategy(artifact,changed,risk())
    np.testing.assert_array_equal(a['weights'][:,:5], b['weights'][:,:5])
    for expr in ['(terminal (spot basket))', '(ret (spot basket) maturity maturity)', '(hit_time (spot basket) 1 up)']:
        with pytest.raises(ValueError): execution.execute_strategy(strategy({'basket':expr}),original,risk())
    with pytest.raises(ValueError): execution.execute_strategy(strategy({'liquidity':'1'}),original,risk())


def test_contract_external_midterm_negative_and_root_financing():
    _, execution=modules()
    source='(contract (meta (underlyings basket) (maturity 8) (schedules (obs 2 4))) (premium (both (coupon_on obs -0.01) (pay (clip (lin -1 (external spread) 0.1) -0.2 0.2) maturity))))'
    artifact={'track':'contract','contract_sexpr':source,'params':{},'notional':1e6}
    cf=execution.contract_cashflows(artifact,panel())
    assert np.all(cf[:,0]==0)
    assert np.all(cf[:,2]==-1e4) and np.all(cf[:,4]==-1e4)
    assert np.all(cf[:,-1]>0)
    for text in [source.replace('(maturity 8)', '(maturity 9)'), source.replace('(external spread)', '(spot spread)'), source.replace('(clip (lin -1 (external spread) 0.1) -0.2 0.2)', '(lin -1 (external spread) 0.1)')]:
        with pytest.raises(ValueError): execution.contract_cashflows({**artifact,'contract_sexpr':text},panel())


def test_recovery_controls_persistent_impact_and_common_factor_quotes():
    _, execution=modules()
    artifact=strategy({'basket':'0.2'},100)
    base=panel(sigma=0,factor_scale=0)
    slow=panel('recovery_bad',sigma=0,factor_scale=0)
    a=execution.execute_strategy(artifact,base,risk())
    b=execution.execute_strategy(artifact,slow,risk())
    assert np.all(b['costs']>a['costs'])
    stressed=panel('common_bad')
    normal=panel()
    assert np.all(stressed.data['spread'][:,1:]>normal.data['spread'][:,1:])
    assert np.all(stressed.data['depth'][:,1:]<normal.data['depth'][:,1:])
    assert np.all(stressed.data['recovery'][:,1:]<normal.data['recovery'][:,1:])
    for name in execution.instrument_prices(normal):
        np.testing.assert_array_equal(execution.instrument_prices(normal)[name],execution.instrument_prices(stressed)[name])


def test_contract_future_coupon_rejected_and_cashflows_prefix_independent():
    _, execution=modules()
    source='(contract (meta (underlyings basket) (maturity 8)) (premium (coupon_on daily (clip (external depth) -1 1))))'
    original=panel(); changed=copy.deepcopy(original)
    changed.paths.external['depth'][:,5:]*=4
    artifact={'track':'contract','contract_sexpr':source,'notional':100.,'params':{}}
    a=execution.contract_cashflows(artifact,original)
    b=execution.contract_cashflows(artifact,changed)
    np.testing.assert_array_equal(a[:,:5],b[:,:5])
    with pytest.raises(ValueError):
        execution.contract_cashflows({**artifact,'contract_sexpr':source.replace('(external depth)','(terminal (spot basket))')},original)


def test_option_expiry_settles_intrinsic_without_fake_liquidation():
    _, execution=modules()
    original=panel(depth=10)
    artifact=strategy({'put':'0.001'},100)
    result=execution.execute_strategy(artifact,original,risk())
    assert not result['unfinished'].any()
    assert not result['funding_violation'].any()
    assert np.all(result['final_inventory']==0)
    assert np.all(result['costs']>0)


def test_empty_cash_strategy_and_reject_inception_business_payment():
    _, execution=modules()
    result=execution.execute_strategy(strategy({}),panel(),risk())
    assert result['weights'].shape==(4,9,0)
    assert np.all(result['pnl']==0) and np.all(result['costs']==0)
    assert np.all(result['peak_funding']==0) and not result['funding_violation'].any()
    prefix='(contract (meta (underlyings basket) (maturity 8) (schedules (obs 0 2))) '
    for body in ['(funded 1 1 (pay 0 maturity))','(premium (coupon_on obs 0.01))']:
        with pytest.raises(ValueError):
            execution.contract_cashflows({'track':'contract','contract_sexpr':prefix+body+')','notional':100,'params':{}},panel())


def test_fixed_reference_metrics_separate_service_factors():
    market,_=modules()
    normal=panel(depth=.025)
    reference=market.reference_metrics(normal,1e6)
    assert reference['cost'].shape==(4,3,3)
    assert reference['unfinished'].shape==(4,3,3)
    assert np.all(market.reference_metrics(panel('spread_bad',depth=.025),1e6)['cost']>reference['cost'])
    assert np.any(market.reference_metrics(panel('depth_bad',depth=.025),1e6)['unfinished']>reference['unfinished'])
    assert np.all(market.reference_metrics(panel('recovery_bad',depth=.025),1e6)['recovery_half_life']>reference['recovery_half_life'])
