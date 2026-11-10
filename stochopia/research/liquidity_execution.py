"""Observable DSL weights, cash-constrained market execution and DSL payments."""
import numpy as np
from scipy.special import ndtr
from structurerdsl.ast import Node, walk, EXPR_OPS
from structurerdsl.check import check, ARITY
from structurerdsl.interp import Env, eval_expr, run
from structurerdsl.program import Program
from structurerdsl.sexpr import parse


def instrument_prices(panel):
    """ATM (strike 1) basket European options; BS quote approximation, not Q law."""
    prices = {name:value.detach().cpu().numpy() for name,value in panel.paths.spot.items()}
    s = prices['basket']; tau = np.maximum((panel.steps-np.arange(panel.steps+1))*panel.dt,0.)[None,:]
    sigma = panel.paths.external['vol'].detach().cpu().numpy()
    total = sigma*np.sqrt(tau); r = panel.paths.rate
    d1 = (np.log(s)+(r+.5*sigma**2)*tau)/np.maximum(total,1e-14); d2 = d1-total
    discount = np.exp(-r*tau)
    call = s*ndtr(d1)-discount*ndtr(d2); put = discount*ndtr(-d2)-s*ndtr(-d1)
    prices['call'] = np.where(total>0,call,np.maximum(s-discount,0))
    prices['put'] = np.where(total>0,put,np.maximum(discount-s,0))
    return prices


def _observables(body, panel):
    for node in walk(body):
        if node.op in {'spot','div'} and str(node.args[0]) not in panel.paths.spot:
            raise ValueError('unknown stock observable '+str(node.args[0]))
        if node.op == 'external' and str(node.args[0]) not in panel.paths.external:
            raise ValueError('unknown external observable '+str(node.args[0]))


def strategy_weights(artifact, panel):
    if artifact.get('track') != 'synthetic': raise ValueError('synthetic strategy required')
    targets = artifact['policy']['targets']; prices = instrument_prices(panel)
    names = tuple(sorted(targets))
    if not names: return names,np.zeros((panel.paths.n_paths,panel.steps+1,0))
    if not set(names)<=set(prices): raise ValueError('targets must be stock, basket, call or put')
    params = {k:float(v) for k,v in artifact.get('params',{}).items()}
    program = Program(tuple(panel.paths.spot),panel.steps,{},Node('premium',(Node('pay',(0,'maturity')),)))
    env = Env(program,panel.paths,params)
    outputs=[]
    for name in names:
        expression = parse(str(targets[name]))
        for node in walk(expression):
            if node.op not in EXPR_OPS-{'terminal','ret','hit_time'}:
                raise ValueError('unsupported or future strategy expression: '+node.op)
            if node.op in ARITY and not ARITY[node.op][0]<=len(node.args)<=ARITY[node.op][1]:
                raise ValueError('invalid expression arity')
        _observables(expression,panel)
        try: outputs.append(eval_expr(expression,env).detach().cpu().numpy())
        except (KeyError,RuntimeError,IndexError,TypeError) as error: raise ValueError(str(error)) from error
    weights = np.stack(outputs,axis=2)
    if not np.isfinite(weights).all(): raise ValueError('weights must be finite')
    return names,weights


def execute_strategy(artifact, panel, risk_config):
    """Weights are instrument market value / initial buyer capital.

    Short stock proceeds are restricted plus 50% margin. Short options restrict
    their premium proceeds plus 100% basket notional. Borrow is charged on short
    stocks. Cash never receives external financing; unfundable orders are cut
    and sticky failure is reported. Terminal options settle legally; stocks are
    sold through the same finite capacity and costs, with residuals unfinished.
    """
    required={'buyer_capital','seller_capital','fee_bps','borrow_rate','gross_limit'}
    if not required<=set(risk_config): raise ValueError('explicit risk_config required: '+str(sorted(required)))
    cfg = dict(short_margin_rate=.5, option_margin_rate=1., impact_bps=2., option_spread_bps=20., trace=False)
    cfg.update(risk_config)
    B=float(cfg['buyer_capital']); seller=float(cfg['seller_capital'])
    if min(B,seller,cfg['gross_limit'])<=0 or min(cfg[k] for k in ['fee_bps','borrow_rate','impact_bps','short_margin_rate','option_margin_rate','option_spread_bps'])<0:
        raise ValueError('invalid capital, cost or margin inputs')
    names,weights = strategy_weights(artifact,panel); prices=instrument_prices(panel)
    P,N,K=weights.shape; N-=1
    if not names:
        result=dict(pnl=np.zeros(P),costs=np.zeros(P),peak_funding=np.zeros(P),funding_violation=np.zeros(P,dtype=bool),turnover=np.zeros(P),final_inventory=np.zeros((P,0)),unfinished=np.zeros(P,dtype=bool),weights=weights,instrument_names=names,ending_cash=np.full(P,B))
        if cfg['trace']: result['trace']=dict(cash=np.full((P,N+1),B),funding=np.zeros((P,N+1)))
        return result
    interval=artifact['policy'].get('rebalance_every',78)
    if not isinstance(interval,int) or interval<1: raise ValueError('rebalance_every must be positive integer')
    marks=np.stack([prices[n] for n in names],axis=2)
    option=np.array([n in {'call','put'} for n in names]); inventory=np.zeros((P,K)); memory=np.zeros_like(inventory)
    cash=np.full(P,B); costs=np.zeros(P); turnover=np.zeros(P); violation=np.zeros(P,dtype=bool); peak=np.zeros(P)
    cash_trace=[]; funding_trace=[]
    external={k:v.detach().cpu().numpy() for k,v in panel.paths.external.items()}

    def funding(q, money, t):
        mark=marks[:,t]; short=np.maximum(-q,0)
        collateral=np.sum(short*mark*(1+cfg['short_margin_rate'])*(~option),axis=1)
        collateral+=np.sum(short*(mark+cfg['option_margin_rate']*prices['basket'][:,t,None])*option,axis=1)
        need=np.maximum(B-money+collateral,0)
        # Options gross is underlying notional (avoids cheap-premium leverage).
        gross=np.sum(np.abs(q)*np.where(option,prices['basket'][:,t,None],mark),axis=1)
        return need,gross

    for t in range(N+1):
        mark=marks[:,t]
        if t:
            borrow=np.sum(np.maximum(-inventory,0)*mark*(~option),axis=1)*cfg['borrow_rate']*panel.dt
            cash-=borrow; costs+=borrow
            memory*=1-np.stack([external.get('recovery_'+name,external['recovery'])[:,t] for name in names],axis=1)
        if t==N:
            cash+=np.sum(inventory*mark*option,axis=1); inventory[:,option]=0
        need,gross=funding(inventory,cash,t)
        peak=np.maximum(peak,need); violation|=(need>B+1e-7)|(gross>B*cfg['gross_limit']+1e-7)
        if t%interval==0 or t==N:
            desired=np.zeros_like(inventory) if t==N else weights[:,t]*B/np.maximum(mark,1e-12)
            # One stock-capacity pool shared between basket and its constituents.
            remaining=np.stack([external[f'depth_s{i}'][:,t]*B for i in range(10)],axis=1)
            for j,name in enumerate(names):
                if t==N and option[j]: continue
                ref=prices['basket'][:,t] if option[j] else mark[:,j]
                if name=='basket':
                    components=np.stack([prices[f's{i}'][:,t]/10 for i in range(10)],axis=1)
                    capacity=np.min(remaining/components,axis=1)
                elif name.startswith('s'):
                    stock=int(name[1:]); capacity=remaining[:,stock]/ref
                else: capacity=external['depth'][:,t]*B*.25/ref
                requested=desired[:,j]-inventory[:,j]; delta=np.clip(requested,-capacity,capacity)
                spread=external.get('spread_'+name,external['spread'])[:,t]
                if option[j]: spread=np.maximum(spread,cfg['option_spread_bps']/1e4)
                # Temporary impact state is signed; crossing always costs, and
                # the magnitude of previous impact remains until recovery.
                def transaction(dq):
                    impact=cfg['impact_bps']/1e4*(np.abs(dq)/np.maximum(capacity,1e-12))
                    cost=np.abs(dq)*(mark[:,j]*(spread/2+cfg['fee_bps']/1e4)+ref*(impact+np.abs(memory[:,j])))
                    q=inventory.copy(); q[:,j]+=dq
                    money=cash-dq*mark[:,j]-cost
                    f,g=funding(q,money,t)
                    return cost,q,money,f,g
                cost,q,money,f,g=transaction(delta)
                bad=(f>B+1e-7)|(g>B*cfg['gross_limit']+1e-7)
                violation|=bad
                if bad.any():
                    # Finance feasible partial fill. Existing breaches cannot
                    # be erased by later profit; unwinding is still permitted.
                    low=np.zeros(P); high=np.ones(P)
                    oldf,oldg=funding(inventory,cash,t)
                    for _ in range(24):
                        mid=(low+high)/2; _,_,_,mf,mg=transaction(delta*mid)
                        feasible=(mf<=np.maximum(B,oldf)+1e-7)&(mg<=np.maximum(B*cfg['gross_limit'],oldg)+1e-7)
                        low=np.where(feasible,mid,low); high=np.where(feasible,high,mid)
                    delta*=np.where(bad,low,1.)
                    cost,q,money,f,g=transaction(delta)
                inventory=q; cash=money; costs+=cost; peak=np.maximum(peak,f)
                turnover+=np.abs(delta)*ref/B
                memory[:,j]+=np.sign(delta)*cfg['impact_bps']/1e4*np.abs(delta)/np.maximum(capacity,1e-12)
                if name=='basket': remaining=np.maximum(remaining-np.abs(delta[:,None])*components,0)
                elif name.startswith('s'): remaining[:,stock]=np.maximum(remaining[:,stock]-np.abs(delta)*ref,0)
        if cfg['trace']:
            cash_trace.append(cash.copy()); funding_trace.append(funding(inventory,cash,t)[0])
    unfinished=np.any(np.abs(inventory)>1e-8,axis=1); violation|=unfinished
    result=dict(pnl=cash-B,costs=costs,peak_funding=peak,funding_violation=violation,
                turnover=turnover,final_inventory=inventory,unfinished=unfinished,
                weights=weights,instrument_names=names,ending_cash=cash)
    if cfg['trace']: result['trace']=dict(cash=np.stack(cash_trace,axis=1),funding=np.stack(funding_trace,axis=1))
    return result


def contract_cashflows(artifact,panel):
    """Client-perspective payments; inception purchase price handled by caller.

    Only premium roots and zero inception business payments are accepted.
    Negative payments are retained and require caller-side buyer financing.
    The native checker enforces adaptation and bounded client loss.
    """
    if artifact.get('track')!='contract': raise ValueError('contract track required')
    program=Program.from_sexpr(artifact['contract_sexpr'])
    if program.maturity!=panel.steps: raise ValueError('contract maturity must match panel steps')
    if not set(program.underlyings)<=set(panel.paths.spot): raise ValueError('unknown contract underlying')
    if program.body.op!='premium': raise ValueError('this pilot requires a premium root wrapper')
    _observables(program.body,panel)
    params={k:float(v) for k,v in artifact.get('params',{}).items()}
    report=check(program,params=params,max_years=panel.steps*panel.dt)
    if not report.valid: raise ValueError('invalid DSL: '+'; '.join(report.errors))
    notional=float(artifact['notional'])
    if not np.isfinite(notional) or notional<=0: raise ValueError('positive finite notional required')
    cf=run(program,panel.paths,params).cashflows.detach().cpu().numpy()*notional
    if np.any(np.abs(cf[:,0])>1e-9): raise ValueError('inception business payments are outside this pilot')
    if not np.isfinite(cf).all(): raise ValueError('non-finite contract cashflows')
    return cf
