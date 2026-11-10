"""Compile existing monthly StructurerDSL into adapted research payments.

No language extension: this experiment consumes dated payments already present
in StructurerDSL. It does not alter the legacy three-month adapter.
"""
import torch
from structurerdsl.program import Program
from structurerdsl.check import check
from structurerdsl.interp import run
from structurerdsl.path import Paths
from structurerdsl.ast import walk


CONTRACT_VERSION='stochopia.dsl-tree-contract.v1'


def table_from_artifact(tree,artifact):
    payload=artifact['contract'];program=Program.from_json(payload['program'])
    if len(program.underlyings)!=1 or program.maturity!=tree.months:
        raise ValueError('contract must use one stock and this experiment horizon')
    params={key:float(value) for key,value in payload.get('params',{}).items()}
    if 'issue_price' in params:
        raise ValueError('environment sets the initial funding movement')
    params['issue_price']=0.
    if any(node.op in {'external','anytime','callable'} for node in walk(program.body)):
        raise ValueError('client events are not contract observables; discretionary exercise is outside this protocol')
    report=check(program,params=params,max_years=tree.months/12)
    if not report.valid:
        raise ValueError('invalid DSL: '+'; '.join(report.errors))
    paths=Paths(spot={program.underlyings[0]:torch.tensor(tree.paths,dtype=torch.float64)},
        t_years=torch.arange(tree.months+1,dtype=torch.float64)/12,rate=tree.rate)
    result=run(program,paths,params)
    table=result.cashflows.detach().numpy()*float(payload['notional'])
    # Inception financing is separately settled by the Q purchase price.
    table[:,0]=0.
    tree.node_cashflows(table)
    return table, {'node_count':sum(1 for _ in walk(program.body)),
                   'contract_version':CONTRACT_VERSION,'monitoring':'monthly'}
