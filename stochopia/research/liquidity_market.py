"""Synthetic five-minute reduced-form market; no historical order-book claim."""
from dataclasses import dataclass
import numpy as np
import torch
from structurerdsl.path import Paths


@dataclass
class MarketPanel:
    paths: Paths
    data: dict
    scenario: str
    seed: int
    config: dict
    dt: float = 1 / (252 * 78)

    @property
    def steps(self):
        return self.paths.n_steps


def generate_panel(seed=0, n_paths=64, config=None, scenario='normal'):
    """depth = executable stock notional / initial capital per five-minute step.

    spread is full bid/ask width / mid; recovery is the fraction of temporary
    impact decaying each step. Separate AR innovations drive each liquidity
    factor. Interventions use common random numbers and never alter column 0.
    """
    cfg = dict(days=20, steps_per_day=78, sigma=.2, rate=0., drift=0.,
               spread=.001, depth=.35, recovery=.2, factor_scale=.2,
               price_shift=.04)
    cfg.update(config or {})
    n_paths, steps = int(n_paths), int(cfg['days']) * int(cfg['steps_per_day'])
    dt = 1/(252*78)
    if steps < 1 or n_paths < 1 or cfg['sigma'] < 0 or min(cfg[k] for k in ['spread','depth','recovery']) <= 0:
        raise ValueError('positive paths, horizon and liquidity inputs required; sigma >= 0')
    scenarios = {'normal','spread_bad','spread_good','depth_bad','depth_good',
                 'recovery_bad','recovery_good','common_bad','common_good','price_up','price_down','vol_high','vol_low'}
    if scenario not in scenarios:
        raise ValueError('unknown scenario: '+scenario)
    rng = np.random.default_rng(seed)
    # Draw prices first; changing a liquidity factor leaves diffusion unchanged.
    innovations = np.sqrt(.4)*rng.normal(size=(n_paths,steps,1)) + np.sqrt(.6)*rng.normal(size=(n_paths,steps,10))
    shocks = rng.normal(size=(n_paths,steps,4,11))
    factors = np.zeros((n_paths,steps+1,4,11))
    for t in range(1,steps+1):
        factors[:,t] = .96*factors[:,t-1] + np.sqrt(1-.96**2)*shocks[:,t-1]
    mixed = .75*factors[:,:,:,0,None] + .45*factors[:,:,:,1:]
    data = {}
    multipliers = {'spread_bad':('spread',2.),'spread_good':('spread',.5),
                   'depth_bad':('depth',.5),'depth_good':('depth',2.),
                   'recovery_bad':('recovery',.5),'recovery_good':('recovery',2.)}
    for i, name in enumerate(['spread','depth','recovery']):
        values = cfg[name]*np.exp(np.clip(cfg['factor_scale']*mixed[:,:,i],-2,2))
        if scenario in multipliers and multipliers[scenario][0] == name:
            values[:,1:] *= multipliers[scenario][1]
        if scenario in {'common_bad','common_good'}:
            bad = scenario == 'common_bad'
            values[:,1:] *= (2. if bad else .5) if name == 'spread' else (.5 if bad else 2.)
        if name == 'recovery':
            values = np.clip(values, .001, .95)
        for k in range(10): data[f'{name}_s{k}'] = values[:,:,k].copy()
        data[name] = values.mean(axis=2)
    vol = cfg['sigma']*np.exp(.08*mixed[:,:,3])
    if scenario == 'vol_high': vol[:,1:] *= 1.5
    if scenario == 'vol_low': vol[:,1:] *= .65
    increments = (cfg['drift']-.5*vol[:,1:]**2)*dt + vol[:,1:]*np.sqrt(dt)*innovations
    logspot = np.concatenate([np.zeros((n_paths,1,10)), np.cumsum(increments,axis=1)],axis=1)
    if scenario in {'price_up','price_down'}:
        logspot += (1 if scenario=='price_up' else -1)*cfg['price_shift']*np.arange(steps+1)[None,:,None]/steps
    for k in range(10): data[f's{k}'] = np.exp(logspot[:,:,k])
    data['basket'] = np.mean([data[f's{k}'] for k in range(10)],axis=0)
    # Basket volatility approximation for synthetic BS marks, public and causal.
    data['vol'] = np.sqrt(.4+.6/10)*vol.mean(axis=2)
    data['variance'] = data['vol']**2
    data['liquidity'] = (data['spread']/cfg['spread'] + cfg['depth']/data['depth'] + cfg['recovery']/data['recovery'])/3
    spot = {k:torch.from_numpy(data[k]) for k in [*(f's{k}' for k in range(10)), 'basket']}
    external = {k:torch.from_numpy(v) for k,v in data.items() if k not in spot}
    paths = Paths(spot=spot, external=external, t_years=torch.arange(steps+1,dtype=torch.float64)*dt, rate=cfg['rate'])
    return MarketPanel(paths, data, scenario, int(seed), cfg, dt)


def reference_metrics(panel, capital):
    """Fixed frozen-state service probes, not candidate returns or ADV measures.

    For each post-root observed state, request q capital of basket uniformly in
    d steps, then the equal opposite leg. Hold that state fixed for the probe;
    costs/capital and unserved requested fraction are averaged over states.
    Temporary impact uses the execution model's 2bp coefficient, without fee.
    """
    if not np.isfinite(capital) or capital<=0: raise ValueError('positive finite reference capital required')
    q_fractions=np.array([.01,.1,.5]); deadlines=np.array([1,6,24])
    stock=np.stack([panel.data[f's{i}'][:,1:] for i in range(10)],axis=2)
    depth=np.stack([panel.data[f'depth_s{i}'][:,1:] for i in range(10)],axis=2)
    capacity=np.min(depth/(stock/10),axis=2)*panel.data['basket'][:,1:]*capital
    spread=panel.data['spread'][:,1:]; recovery=panel.data['recovery'][:,1:]
    cost=np.empty((panel.paths.n_paths,3,3)); unfinished=np.empty_like(cost)
    for i,q in enumerate(q_fractions):
        for j,d in enumerate(deadlines):
            filled=np.minimum(q*capital/d,capacity)
            impact=2e-4*filled/np.maximum(capacity,1e-12)
            memory=np.zeros_like(capacity); expense=np.zeros_like(capacity)
            for direction in [1,-1]:
                for _ in range(d):
                    memory*=1-recovery
                    expense+=filled/capital*(spread/2+impact+np.abs(memory))
                    memory+=direction*impact
            cost[:,i,j]=expense.mean(axis=1)
            unfinished[:,i,j]=np.maximum(1-d*filled/(q*capital),0).mean(axis=1)
    half_life=(np.log(.5)/np.log1p(-recovery)).mean(axis=1)
    return dict(q_fractions=q_fractions,deadlines=deadlines,cost=cost,unfinished=unfinished,
                recovery_half_life=half_life,mean_cost=cost.mean(axis=0),mean_unfinished=unfinished.mean(axis=0))
