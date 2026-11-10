"""Exact monthly binomial valuation with shared P/Q diffusion inputs.

The stock price recombines; histories remain distinct for path-dependent
payments. Optimal values on this tree are not continuous-GBM certificates.
"""
from itertools import product
import math
import numpy as np


class TreeMarket:
    def __init__(self, config=None):
        self.config = {'months':12, 'sigma':.2, 'rate':.02, 'dividend':0., 'sharpe':.3, **(config or {})}
        self.months = int(self.config['months']); self.dt = 1 / 12
        self.rate = float(self.config['rate']); self.sigma = float(self.config['sigma'])
        dividend = float(self.config['dividend'])
        if self.months < 1 or self.months > 16 or self.sigma <= 0:
            raise ValueError('finite research tree needs 1..16 months and positive volatility')
        self.up = math.exp(self.sigma * math.sqrt(self.dt)); self.down = 1 / self.up
        self.q = (math.exp((self.rate - dividend) * self.dt) - self.down) / (self.up - self.down)
        drift = self.rate - dividend + float(self.config['sharpe']) * self.sigma
        self.p = (math.exp(drift * self.dt) - self.down) / (self.up - self.down)
        if not (0 < self.p < 1 and 0 < self.q < 1):
            raise ValueError('binomial P/Q probabilities must lie strictly between zero and one')
        self.prefixes = tuple(prefix for t in range(self.months + 1) for prefix in product((0,1),repeat=t))
        self.node_index = {prefix:i for i,prefix in enumerate(self.prefixes)}
        self.node_times = np.array([len(prefix) for prefix in self.prefixes],dtype=int)
        self.children = np.full((len(self.prefixes),2),-1,dtype=int)
        for i,prefix in enumerate(self.prefixes):
            if len(prefix) < self.months:
                self.children[i] = [self.node_index[prefix+(bit,)] for bit in (0,1)]
        def mass(prob):
            return np.array([prob**sum(prefix) * (1-prob)**(len(prefix)-sum(prefix)) for prefix in self.prefixes])
        self.q_mass, self.p_mass = mass(self.q), mass(self.p)
        leaves = list(product((0,1),repeat=self.months))
        self.path_nodes = np.array([[self.node_index[leaf[:t]] for t in range(self.months+1)] for leaf in leaves])
        ups = np.array([[sum(leaf[:t]) for t in range(self.months+1)] for leaf in leaves])
        self.paths = self.up**ups * self.down**(np.arange(self.months+1)[None,:]-ups)
        self.q_leaf = self.q_mass[self.path_nodes[:,-1]]; self.p_leaf = self.p_mass[self.path_nodes[:,-1]]
        self._representative_leaf = np.empty(len(self.prefixes),dtype=int)
        for t in range(self.months+1):
            nodes, first = np.unique(self.path_nodes[:,t],return_index=True)
            self._representative_leaf[nodes] = first

    def node_cashflows(self, table):
        table = np.asarray(table,dtype=float)
        if table.shape != self.paths.shape or not np.isfinite(table).all():
            raise ValueError('cashflow table must be finite with one column per observed month')
        if np.any(np.abs(table[:,0]) > 1e-8):
            raise ValueError('initial purchase is separate; root cashflow must be zero')
        maximum = np.full(len(self.prefixes),-np.inf); minimum = np.full(len(self.prefixes),np.inf)
        np.maximum.at(maximum,self.path_nodes.ravel(),table.ravel())
        np.minimum.at(minimum,self.path_nodes.ravel(),table.ravel())
        if np.any(maximum-minimum > 1e-7):
            raise ValueError('cashflows must be adapted to the observed market prefix')
        return table[self._representative_leaf,self.node_times]

    def conditional_node_values(self, table):
        payments = self.node_cashflows(table); values = np.zeros(len(self.prefixes))
        discount = math.exp(-self.rate * self.dt)
        for i in range(len(self.prefixes)-1,-1,-1):
            left,right = self.children[i]
            if left >= 0:
                values[i] = discount * ((1-self.q)*(payments[left]+values[left]) + self.q*(payments[right]+values[right]))
        return values

    def conditional_values(self, table):
        """Remaining Q value after this month's payment, at every prefix."""
        return self.conditional_node_values(table)[self.path_nodes]

    def price(self, table):
        return float(self.conditional_node_values(table)[0])

    def describe(self):
        return {'schema':'stochopia-shared-vol-tree.v1', **self.config,
                'q_up_probability':self.q, 'p_up_probability':self.p,
                'p_equals_q':self.p==self.q, 'stock_nodes':(self.months+1)*(self.months+2)//2,
                'history_nodes':len(self.prefixes), 'terminal_paths':len(self.paths),
                'monitoring':'monthly', 'scope':'finite binomial model; shared diffusion sigma, not identical finite-step P/Q variances'}
