import numpy as np
import pytest

from stochopia.research.tree_market import TreeMarket
from stochopia.research.path_demand import PathDemand, PathCashflowEnv, evaluate_cashflows, demand_cases


def test_shared_volatility_q_martingale_and_p_equals_q():
    tree = TreeMarket({'months': 3, 'sharpe': 0.})
    assert np.array_equal(tree.p_leaf, tree.q_leaf)
    assert np.sum(tree.q_leaf * tree.paths[:, -1]) == pytest.approx(np.exp(tree.rate * .25))
    assert len(tree.prefixes) == 15


def test_adapted_payments_and_exact_ex_dividend_valuation():
    tree = TreeMarket({'months': 2, 'rate': 0., 'sharpe': 0.})
    table = np.zeros_like(tree.paths); table[:, 1] = 20.; table[:, 2] = 80.
    assert tree.price(table) == pytest.approx(100.)
    assert np.allclose(tree.conditional_values(table), [100., 80., 0.])
    table[0, 1] = 21.
    with pytest.raises(ValueError, match='adapted'):
        tree.price(table)


def test_coupons_withdrawals_and_exit_not_paid_twice():
    tree = TreeMarket({'months': 2, 'rate': 0.})
    table = np.zeros_like(tree.paths); table[:, 1] = 20.; table[:, 2] = 80.
    demand = PathDemand(budget=100., withdrawal_ratios={1:.2}, terminal_total_ratio=.99,
                        max_drawdown=.01, min_success_probability=1., exit_discount=.01,
                        exit_scenarios=({'month':1,'probability':1.},))
    env = PathCashflowEnv(tree, demand, table)
    env.reset(0, demand_cases(tree, demand)[0]); first = env.step(); second = env.step()
    assert first['withdrawal_paid'] == 20.
    assert first['exit_proceeds'] == pytest.approx(79.2)
    assert first['adjusted_wealth'] == pytest.approx(99.2)
    assert second['contract_payment'] == 0.
    assert second['adjusted_wealth'] == pytest.approx(99.2)
    assert first['drawdown'] == pytest.approx(.008)
    metrics = evaluate_cashflows(tree, demand, table)
    assert metrics['joint_success_probability'] == pytest.approx(1.)


def test_partial_sales_is_registered_control_not_default():
    tree = TreeMarket({'months':2,'rate':0.})
    table = np.zeros_like(tree.paths); table[:,2] = 100.
    d = dict(budget=100.,withdrawal_ratios={1:.2},terminal_total_ratio=1.,
             max_drawdown=.01,min_success_probability=1.,exit_scenarios=({'month':None,'probability':1.},))
    assert not evaluate_cashflows(tree,PathDemand(**d),table)['feasible']
    assert evaluate_cashflows(tree,PathDemand(**d,partial_sales=True),table)['feasible']


def test_event_condition_uses_observed_market_only_and_payments_never_see_event():
    tree = TreeMarket({'months':2})
    demand = PathDemand(withdrawal_ratios={},exit_scenarios=({'month':None,'probability':.5},
       {'month':1,'probability':.5,'condition':{'month':1,'direction':'below','level':1.}}))
    cases = demand_cases(tree,demand)
    assert sum(x['weight_p'] for x in cases) == pytest.approx(1.)
    assert all(x['exit_month'] is None for x in cases if tree.paths[x['leaf_index'],1] > 1.)
    future = PathDemand(withdrawal_ratios={},exit_scenarios=({'month':1,'probability':1.,
          'condition':{'month':2,'direction':'above','level':1.}},))
    with pytest.raises(ValueError,match='observed'):
        demand_cases(tree,future)


def test_existing_dsl_can_pay_at_month_one_without_new_language():
    from structurerdsl.program import Program
    from stochopia.research.dsl_cashflows import table_from_artifact
    tree=TreeMarket({'months':2,'rate':0.})
    program=Program.from_sexpr('(contract (meta (underlyings u) (maturity 2) (schedules (obs 1 2))) (premium (both (pay 0.2 (at obs 0)) (pay 0.8 maturity))))')
    table,metadata=table_from_artifact(tree,{'contract':{'program':program.to_json(),'notional':100.,'params':{}}})
    assert np.allclose(table,[0.,20.,80.])
    assert tree.price(table)==pytest.approx(100.)
    assert metadata['node_count']>0


def test_cost_includes_reserve_and_unused_budget_is_not_free_wealth():
    tree=TreeMarket({'months':2,'rate':0.})
    d=PathDemand(budget=100.,withdrawal_ratios={1:.2},terminal_total_ratio=.5,
      max_drawdown=0.,min_success_probability=1.,exit_scenarios=({'month':None,'probability':1.},))
    table=np.zeros_like(tree.paths);m=evaluate_cashflows(tree,d,table)
    assert m['contract_purchase_cost']==0.
    assert m['cash_reserve']==pytest.approx(50.)
    assert m['cost']==pytest.approx(50.)
    assert m['initial_funded_wealth']==pytest.approx(50.)
    assert m['feasible']
    assert all(r['terminal_adjusted_wealth']==50. for r in m['case_results'])


@pytest.mark.parametrize('partial',[False,True])
def test_batch_and_monthly_account_replay_agree(partial):
    tree=TreeMarket({'months':3})
    table=np.zeros_like(tree.paths);table[:,1]=10.;table[:,3]=70.+15.*tree.paths[:,-1]
    d=PathDemand(budget=100.,withdrawal_ratios={1:.1,2:.05},terminal_total_ratio=.99,
        max_drawdown=.2,min_success_probability=.5,partial_sales=partial,
        exit_scenarios=({'month':None,'probability':.7},{'month':2,'probability':.3}))
    env=PathCashflowEnv(tree,d,table);metrics=evaluate_cashflows(tree,d,table)
    for case,result in zip(env.cases,metrics['case_results']):
        env.reset(case['leaf_index'],case)
        for _ in range(tree.months):env.step()
        assert env.records[-1]['adjusted_wealth']==pytest.approx(result['terminal_adjusted_wealth'])
        assert env.max_dd==pytest.approx(result['max_drawdown'])
        assert env.shortfall==pytest.approx(result['cash_shortfall'])


def test_original_demand_payload_keeps_legacy_control_semantics():
    d=PathDemand.from_dict({'schema':'stochopia-path-demand.v1','budget':100.})
    assert d.funding_mode=='fixed_budget_leg_control'
    assert PathDemand().to_dict()['schema']=='stochopia-path-demand.v2'


def test_partial_sale_control_does_not_assume_monotone_success_or_claim_minimum():
    tree=TreeMarket({'months':2,'rate':0.,'sharpe':0.})
    table=np.zeros_like(tree.paths)
    table[np.argmax(tree.paths[:,-1]),-1]=100./tree.q
    d=PathDemand(budget=100.,withdrawal_ratios={1:.5},terminal_total_ratio=.5,
        max_drawdown=.5,min_success_probability=.4,partial_sales=True,
        exit_scenarios=({'month':None,'probability':1.},))
    from stochopia.research.path_demand import _batch_account
    rates=[]
    for reserve in (0.,25.,50.):
        env=PathCashflowEnv(tree,d,table,cash_reserve=reserve)
        result=_batch_account(tree,d,table,env.values,env.contract_purchase_cost,reserve,env.cases)
        rates.append(sum(c['weight_p']*s for c,s in zip(env.cases,result['success'])))
    assert rates[0]>.4 and rates[1]<.4 and rates[2]>.4
    m=evaluate_cashflows(tree,d,table)
    assert m['minimum_required_reserve'] is None and m['cost']==pytest.approx(100.)
    assert m['reserve_fit_status']=='partial-sale-fixed-budget-control-not-minimum'
