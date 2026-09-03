#!/usr/bin/env python3
from __future__ import annotations

import argparse, itertools, json
from pathlib import Path
import numpy as np
from outosynapsi import SpectralTree, complete_binary_tree, edge_flow


def alloc(flow,tax,budget,exponent):
    s=(np.asarray(flow,float)+tax)**exponent
    return budget*s/s.sum()

def path_cost(tree,pairs,power):
    return float(np.mean([sum(1.0/(tree.couplings[e]**power) for e in tree.path_edge_indices(s,t)) for s,t in pairs]))
def objective(flow,w,tax,power):
    return float(np.sum((np.asarray(flow)+tax)/(w**power)))
def summary(xs):
    x=np.asarray(xs,float); return {'mean':float(x.mean()),'min':float(x.min()),'max':float(x.max())}

def online_p2(n_nodes,edges,pairs,seed,steps,eta,tax,budget,shuffle=False):
    topo=SpectralTree(n_nodes,edges,np.ones(len(edges)))
    paths=[topo.path_edge_indices(s,t) for s,t in pairs]
    rng=np.random.default_rng(seed); g=np.full(len(edges),budget/len(edges))
    for _ in range(steps):
        q=np.zeros(len(edges)); q[paths[int(rng.integers(len(paths)))]]=1
        if shuffle: q=q[rng.permutation(len(edges))]
        u=2*(q+tax)/(g**3)
        g += eta*(u-u.mean()); g += (budget-g.sum())/len(edges)
        if np.any(g<=0): raise RuntimeError('coupling became non-positive')
    return g

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='results/GATE2.json'); ap.add_argument('--seeds',type=int,default=40); ap.add_argument('--steps',type=int,default=10000); ap.add_argument('--learning-rate',type=float,default=0.0005); ap.add_argument('--standing-tax',type=float,default=0.05); args=ap.parse_args()
    n,edges=complete_binary_tree(4); E=len(edges); B=float(E); topo=SpectralTree(n,edges,np.ones(E))
    A=[15,16,17,18]; C=[27,28,29,30]; train=[]
    for i,s in enumerate(A): train += [(s,C[i]),(s,C[(i+1)%4])]
    test=[p for p in itertools.product(A,C) if p not in train]; F=edge_flow(topo,train)
    leaves=list(range(15,31)); allpairs=list(itertools.combinations(leaves,2))
    uniform=np.ones(E); sqrtw=alloc(F,args.standing_tax,B,.5); cubew=alloc(F,args.standing_tax,B,1/3); prop=alloc(F,args.standing_tax,B,1)
    def metrics(w):
        t=SpectralTree(n,edges,w); return {'connes_task_distance_p1':path_cost(t,test,1),'laplacian_resistance_task_p2':path_cost(t,test,2),'all_leaf_resistance_p2':path_cost(t,allpairs,2),'laplacian_objective_p2':objective(F,w,args.standing_tax,2)}
    learned=[]; shuffled=[]; cos=[]
    for seed in range(args.seeds):
        w=online_p2(n,edges,train,seed,args.steps,args.learning_rate,args.standing_tax,B); ws=online_p2(n,edges,train,seed,args.steps,args.learning_rate,args.standing_tax,B,True)
        learned.append(metrics(w)); shuffled.append(metrics(ws)); cos.append(float(np.dot(w,cubew)/(np.linalg.norm(w)*np.linalg.norm(cubew))))
    ls={k:summary([r[k] for r in learned]) for k in learned[0]}; ss={k:summary([r[k] for r in shuffled]) for k in shuffled[0]}; ls['cosine_to_laplacian_oracle']=summary(cos)
    fixed=metrics(uniform); dm=metrics(sqrtw); lm=metrics(cubew); tp=metrics(prop); ratio=ls['laplacian_objective_p2']['mean']/lm['laplacian_objective_p2']
    passed=ratio<=1.001 and ls['cosine_to_laplacian_oracle']['mean']>=.999 and lm['laplacian_objective_p2']<dm['laplacian_objective_p2'] and ss['laplacian_objective_p2']['mean']>=.98*fixed['laplacian_objective_p2']
    result={'gate':2,'classification':'DIRAC_METRIC_AND_D_SQUARED_TRANSPORT_REQUIRE_DIFFERENT_BUDGET_OPTIMA' if passed else 'DIRAC_VS_LAPLACIAN_BOUNDARY_NOT_ESTABLISHED','passed':passed,'interpretation':'Connes path cost scales as 1/g; a Laplacian-like transport with g^2 conductance has tree resistance scaling as 1/g^2. Fixed-budget optima are sqrt(flow+tax) versus cube_root(flow+tax).','frozen':fixed,'gate1_dirac_sqrt_allocation':dm,'laplacian_cube_root_oracle':lm,'traffic_proportional':tp,'online_laplacian_plasticity':ls,'shuffled_traffic_attacker':ss,'learned_objective_over_oracle':float(ratio),'scope':'Operator-power boundary; not a claim that biological transport is literally D^2.'}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); raise SystemExit(0 if passed else 1)
if __name__=='__main__': main()
