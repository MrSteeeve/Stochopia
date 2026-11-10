"""Path demand and minimum operational funding, without LLM/nested MC.

Primary cost is the Q contract purchase plus required cash reserve. Unallocated
budget stays outside this service account. A legacy fixed-budget/leg-price
control is explicit and cannot be confused with this funding objective.
"""
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class PathDemand:
    budget: float = 1_000_000.
    withdrawal_ratios: dict = field(default_factory=lambda:{3:.03,6:.03,9:.03})
    terminal_total_ratio: float = 1.02
    max_drawdown: float = .15
    min_success_probability: float = .95
    exit_scenarios: tuple = ({'month':None,'probability':.7},{'month':6,'probability':.15},{'month':9,'probability':.15})
    exit_discount: float = .01
    partial_sales: bool = False
    partial_sale_discount: float = 0.
    max_payment_ratio: float = 4.
    funding_mode: str = 'minimum_service_capital'

    def to_dict(self):
        return {'schema':'stochopia-path-demand.v2',**asdict(self)}

    @classmethod
    def from_dict(cls,value):
        fields = {k:v for k,v in value.items() if k!='schema'}
        if value.get('schema')=='stochopia-path-demand.v1' and 'funding_mode' not in fields:
            fields['funding_mode']='fixed_budget_leg_control'
        if 'withdrawal_ratios' in fields:
            fields['withdrawal_ratios'] = {int(k):v for k,v in fields['withdrawal_ratios'].items()}
        return cls(**fields)


def demand_cases(tree,demand):
    """Events affect client needs/exit, never contract information or payments."""
    if not (demand.budget > 0 and 0 <= demand.exit_discount < 1 and 0 <= demand.partial_sale_discount < 1):
        raise ValueError('positive budget and discounts in [0,1) are required')
    if not 0 <= demand.min_success_probability <= 1 or not 0 <= demand.max_drawdown <= 1:
        raise ValueError('success and drawdown thresholds must lie in [0,1]')
    scenarios = list(demand.exit_scenarios)
    if abs(sum(s['probability'] for s in scenarios)-1.) > 1e-9:
        raise ValueError('client event probabilities must sum to one')
    for s in scenarios:
        exit_month = s['month']
        if exit_month is not None and not 1 <= exit_month <= tree.months:
            raise ValueError('exit month outside the market grid')
        condition = s.get('condition')
        if condition and (exit_month is None or not 1 <= condition['month'] <= exit_month):
            raise ValueError('event condition must use observed information')
        if condition and condition['direction'] not in ('below','above'):
            raise ValueError('event direction must be below or above')
    rows=[]
    for leaf,path in enumerate(tree.paths):
        for index,scenario in enumerate(scenarios):
            probability = float(scenario['probability'])
            if probability < 0:raise ValueError('event probabilities must be nonnegative')
            if probability == 0:continue
            exit_month= scenario['month']; cond=scenario.get('condition')
            if cond:
                observed = path[cond['month']]
                eligible = observed < cond['level'] if cond['direction']=='below' else observed > cond['level']
                if not eligible:exit_month=None
            withdrawals=np.zeros(tree.months+1)
            for month,ratio in scenario.get('withdrawal_ratios',demand.withdrawal_ratios).items():
                month=int(month)
                if not 1 <= month <= tree.months or ratio < 0:
                    raise ValueError('withdrawal needs must be nonnegative and on the grid')
                withdrawals[month]=float(ratio)*demand.budget
            rows.append({'leaf_index':leaf,'event_index':index,'exit_month':exit_month,
                'weight_p':tree.p_leaf[leaf]*probability,'weight_q':tree.q_leaf[leaf]*probability,
                'withdrawals':withdrawals})
    return rows


def _batch_account(tree,demand,table,values,purchase,reserve,cases):
    leaves=np.array([c['leaf_index'] for c in cases]);count=len(cases)
    exits=np.array([-1 if c['exit_month'] is None else c['exit_month'] for c in cases])
    needs=np.array([c['withdrawals'] for c in cases]);cash=np.full(count,reserve)
    quantity=np.ones(count);withdrawn=np.zeros(count);shortfall=np.zeros(count)
    peak=np.full(count,purchase+reserve);dd=np.zeros(count);wealth=peak.copy()
    for t in range(1,tree.months+1):
        cash+=quantity*table[leaves,t];value=values[leaves,t]
        if demand.partial_sales:
            bid=(1-demand.partial_sale_discount)*value
            sell=np.minimum(quantity,np.divide(np.maximum(needs[:,t]-cash,0.),bid,
                                  out=np.zeros(count),where=bid>0))
            quantity-=sell;cash+=sell*bid
        paid=np.minimum(cash,needs[:,t]);cash-=paid;withdrawn+=paid;shortfall+=needs[:,t]-paid
        leaving=exits==t;cash[leaving]+=quantity[leaving]*value[leaving]*(1-demand.exit_discount);quantity[leaving]=0.
        wealth=cash+withdrawn+quantity*value;peak=np.maximum(peak,wealth)
        now=np.divide(peak-wealth,peak,out=np.zeros(count),where=peak>0)
        dd=np.maximum(dd,now)
    tolerance=demand.budget*1e-9
    passed=(shortfall<=tolerance)&(dd<=demand.max_drawdown+1e-9)&(wealth>=demand.budget*demand.terminal_total_ratio-tolerance)
    return {'success':passed,'max_drawdown':dd,'cash_shortfall':shortfall,'terminal_adjusted_wealth':wealth}


def _reserve_requirements(tree,demand,table,values,purchase,cases):
    """Exact per-case cash lower bounds, for the no-partial-sale branch."""
    thresholds=[]
    for case in cases:
        leaf=case['leaf_index'];exit_month=case['exit_month']
        received=np.cumsum(table[leaf]);base=received+values[leaf];base[0]=purchase
        liquid=received.copy()
        if exit_month is not None:
            bid=values[leaf,exit_month]*(1-demand.exit_discount)
            received[exit_month+1:]=received[exit_month]
            liquid=received.copy();liquid[exit_month+1:]+=bid
            base[exit_month:]=received[exit_month]+bid
        required=max(0.,float(np.max(np.cumsum(case['withdrawals'])-liquid)),
                     demand.budget*demand.terminal_total_ratio-base[-1])
        if demand.max_drawdown<1:
            right=(1-demand.max_drawdown)*base[None,:]-base[:,None]
            worst=float(np.max(np.where(np.tril(np.ones_like(right),k=-1).astype(bool),right,0.)))
            if demand.max_drawdown==0:
                if worst>demand.budget*1e-9:required=float('inf')
            else:required=max(required,worst/demand.max_drawdown)
        thresholds.append(required)
    return np.array(thresholds)


def _fit_reserve(tree,demand,table,values,purchase,cases):
    if demand.funding_mode=='fixed_budget_leg_control':
        return max(0.,demand.budget-purchase),None,'legacy-fixed-reserve'
    if demand.funding_mode!='minimum_service_capital':raise ValueError('unknown funding mode')
    weights=np.array([c['weight_p'] for c in cases]);available=max(0.,demand.budget-purchase)
    if not demand.partial_sales:
        thresholds=_reserve_requirements(tree,demand,table,values,purchase,cases)
        if demand.min_success_probability==0:required=0.
        elif demand.min_success_probability==1:required=float(thresholds.max())
        else:
            order=np.argsort(thresholds);cdf=np.cumsum(weights[order])
            index=min(len(order)-1,int(np.searchsorted(cdf,demand.min_success_probability-1e-12)))
            required=float(thresholds[order[index]])
        return min(required,available),required,'exact-per-case-reserve-quantile' if required<=available+1e-7 else 'infeasible-within-budget'
    # With relative drawdown, joint success need not be monotone in reserve.
    # Keep this exploratory liquidity control at the fixed full budget, and
    # do not report its reserve/capital as a minimum.
    return available,None,'partial-sale-fixed-budget-control-not-minimum'



class PathCashflowEnv:
    """Issued adapted contract, with just the next observed month per step."""
    environment_version='stochopia-path-cashflow-tree.v2'

    def __init__(self,tree,demand,table,*,cash_reserve=None):
        self.tree,self.demand=tree,demand;self.table=np.asarray(table,dtype=float)
        self.values=tree.conditional_values(self.table);self.contract_purchase_cost=tree.price(self.table)
        if np.min(self.table)<-1e-7 or np.max(self.table)>demand.budget*demand.max_payment_ratio+1e-7:
            raise ValueError('research cashflows must be nonnegative and within the declared payment bound')
        if self.contract_purchase_cost>demand.budget+1e-7:raise ValueError('contract purchase exceeds service budget')
        self.cases=demand_cases(tree,demand)
        reserve,required,fit=_fit_reserve(tree,demand,self.table,self.values,self.contract_purchase_cost,self.cases)
        self.cash_reserve=reserve if cash_reserve is None else float(cash_reserve)
        self.required_reserve=required;self.reserve_fit_status=fit
        if self.cash_reserve<0 or self.cash_reserve+self.contract_purchase_cost>demand.budget+1e-7:
            raise ValueError('purchase plus operational reserve exceeds service budget')
        self.cost=self.contract_purchase_cost+self.cash_reserve if demand.funding_mode=='minimum_service_capital' else self.contract_purchase_cost
        self.initial_funded_wealth=self.contract_purchase_cost+self.cash_reserve

    def reset(self,leaf_index,case):
        self.leaf=int(leaf_index);self.case=case;self.month=0;self.cash=self.cash_reserve
        self.quantity=1.;self.withdrawn=0.;self.peak=self.initial_funded_wealth
        self.max_dd=0.;self.shortfall=0.;self.records=[]
        return {'month':0,'spot_ratio':1.,'cash':self.cash,'fair_value':self.contract_purchase_cost,'adjusted_wealth':self.peak}

    def step(self):
        if self.month>=self.tree.months:raise ValueError('episode is complete')
        self.month+=1;t=self.month;payment=self.quantity*self.table[self.leaf,t];self.cash+=payment
        value=self.values[self.leaf,t];need=float(self.case['withdrawals'][t]);sold=0.;sale_cash=0.
        if self.demand.partial_sales and self.cash<need and self.quantity*value>0:
            unit_bid=(1-self.demand.partial_sale_discount)*value
            sold=min(self.quantity,(need-self.cash)/unit_bid);sale_cash=sold*unit_bid
            self.quantity-=sold;self.cash+=sale_cash
        paid=min(self.cash,need);self.cash-=paid;self.withdrawn+=paid;self.shortfall+=need-paid
        exit_cash=0.
        if self.case['exit_month']==t:
            exit_cash=self.quantity*value*(1-self.demand.exit_discount);self.cash+=exit_cash;self.quantity=0.
        wealth=self.cash+self.withdrawn+self.quantity*value;self.peak=max(self.peak,wealth)
        dd=max(0.,1-wealth/self.peak) if self.peak>0 else 0.;self.max_dd=max(self.max_dd,dd)
        row={'month':t,'spot_ratio':float(self.tree.paths[self.leaf,t]),'contract_payment':payment,
            'withdrawal_required':need,'withdrawal_paid':paid,'withdrawal_shortfall':need-paid,
            'partial_sale_quantity':sold,'partial_sale_proceeds':sale_cash,'exit_proceeds':exit_cash,
            'cash':self.cash,'remaining_quantity':self.quantity,'remaining_fair_value':self.quantity*value,
            'adjusted_wealth':wealth,'drawdown':dd}
        self.records.append(row);return row


def evaluate_cashflows(tree,demand,table):
    try:env=PathCashflowEnv(tree,demand,table)
    except ValueError as exc:
        return {'status':'rejected','feasible':False,'cost':None,'satisfaction_probability':0.,
                'joint_success_probability':0.,'drawdown_loss':None,'error':str(exc)}
    result=_batch_account(tree,demand,env.table,env.values,env.contract_purchase_cost,env.cash_reserve,env.cases)
    rows=[{'leaf_index':c['leaf_index'],'event_index':c['event_index'],'weight_p':float(c['weight_p']),
        'weight_q':float(c['weight_q']),'success':bool(result['success'][i]),
        'max_drawdown':float(result['max_drawdown'][i]),'cash_shortfall':float(result['cash_shortfall'][i]),
        'terminal_adjusted_wealth':float(result['terminal_adjusted_wealth'][i])} for i,c in enumerate(env.cases)]
    success=sum(r['weight_p']*r['success'] for r in rows);successq=sum(r['weight_q']*r['success'] for r in rows)
    tail=.05;cvar=0.
    for row in sorted(rows,key=lambda r:r['max_drawdown'],reverse=True):
        mass=min(tail,row['weight_p']);cvar+=mass*row['max_drawdown'];tail-=mass
        if tail<=1e-12:break
    return {'schema':'stochopia-path-demand-evaluation.v2','status':'ok','cost':env.cost,
        'contract_purchase_cost':env.contract_purchase_cost,'cash_reserve':env.cash_reserve,
        'initial_funded_wealth':env.initial_funded_wealth,'funding_mode':demand.funding_mode,
        'minimum_required_reserve':env.required_reserve if env.required_reserve is not None and np.isfinite(env.required_reserve) else None,
        'reserve_fit_status':env.reserve_fit_status,'satisfaction_probability':success,'joint_success_probability':success,
        'q_satisfaction_probability':successq,'drawdown_loss':cvar/.05,'feasible':success>=demand.min_success_probability-1e-9,
        'expected_cash_shortfall':sum(r['weight_p']*r['cash_shortfall'] for r in rows),
        'case_count':len(rows),'case_results':rows,'market_scope':tree.describe(),
        'rules':{'unallocated_budget':'outside service account; no free reinvestment utility',
                 'reserve_cash':'zero interest; part of service capital cost',
                 'order':'contract payment -> use available cash, permitted partial sale for withdrawal deficit -> full exit -> flow-adjusted wealth',
                 'partial_sales':demand.partial_sales,'exit_discount':demand.exit_discount,
                 'partial_sale_discount':demand.partial_sale_discount,'monitoring':'monthly',
                 'monetary_comparison_tolerance':demand.budget*1e-9}}
