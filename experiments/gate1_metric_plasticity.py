#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from outosynapsi import SpectralTree, analytic_budget_oracle, complete_binary_tree, edge_flow, mean_distance, regularized_metric_objective, stochastic_metric_plasticity, traffic_proportional_allocation


def summary(values):
    x = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(x)), "min": float(np.min(x)), "max": float(np.max(x))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/GATE1.json")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--learning-rate", type=float, default=0.002)
    ap.add_argument("--standing-tax", type=float, default=0.05)
    args = ap.parse_args()

    n_nodes, edges = complete_binary_tree(4)
    n_edges = len(edges); budget = float(n_edges)
    uniform = np.ones(n_edges)
    topology = SpectralTree(n_nodes, edges, uniform)
    source = [15,16,17,18]; target = [27,28,29,30]
    train = []
    for i, s in enumerate(source):
        train += [(s,target[i]), (s,target[(i+1)%4])]
    test = [p for p in itertools.product(source,target) if p not in train]
    F_train = edge_flow(topology, train)
    F_test = edge_flow(topology, test)
    flow_match_error = float(np.max(np.abs(F_train-F_test)))
    if flow_match_error > 1e-12:
        raise AssertionError("train/test edge-flow split changed")

    leaves = list(range(15,31))
    all_pairs = list(itertools.combinations(leaves,2))
    oracle_w = analytic_budget_oracle(F_train, args.standing_tax, budget)
    traffic_w = traffic_proportional_allocation(F_train, args.standing_tax, budget)

    def metrics(w):
        tree = SpectralTree(n_nodes, edges, w)
        return {
            "heldout_task_distance": mean_distance(tree,test),
            "all_leaf_distance": mean_distance(tree,all_pairs),
            "objective": regularized_metric_objective(tree,F_test,args.standing_tax),
            "min_coupling": float(np.min(w)),
            "max_coupling": float(np.max(w)),
        }

    frozen = metrics(uniform); traffic = metrics(traffic_w); oracle = metrics(oracle_w)
    learned_rows=[]; shuffled_rows=[]; cosines=[]
    for seed in range(args.seeds):
        learned_w = stochastic_metric_plasticity(n_nodes,edges,train,seed=seed,steps=args.steps,learning_rate=args.learning_rate,standing_tax=args.standing_tax,total_budget=budget)
        shuffled_w = stochastic_metric_plasticity(n_nodes,edges,train,seed=seed,steps=args.steps,learning_rate=args.learning_rate,standing_tax=args.standing_tax,total_budget=budget,shuffle_traffic=True)
        learned_rows.append(metrics(learned_w)); shuffled_rows.append(metrics(shuffled_w))
        cosines.append(float(np.dot(learned_w,oracle_w)/(np.linalg.norm(learned_w)*np.linalg.norm(oracle_w))))

    keys=["heldout_task_distance","all_leaf_distance","objective","min_coupling","max_coupling"]
    learned={k:summary([r[k] for r in learned_rows]) for k in keys}
    shuffled={k:summary([r[k] for r in shuffled_rows]) for k in keys}
    learned["cosine_to_oracle"]=summary(cosines)
    learned_oracle_ratio=learned["objective"]["mean"]/oracle["objective"]
    learned_vs_frozen=learned["heldout_task_distance"]["mean"]/frozen["heldout_task_distance"]
    shuffled_vs_frozen=shuffled["heldout_task_distance"]["mean"]/frozen["heldout_task_distance"]
    requirements={
      "heldout_flow_split_exact": flow_match_error < 1e-12,
      "learned_objective_within_0p1pct_oracle": learned_oracle_ratio <= 1.001,
      "learned_task_distance_le_0p65_frozen": learned_vs_frozen <= 0.65,
      "learned_cosine_to_oracle_ge_0p999": learned["cosine_to_oracle"]["mean"] >= 0.999,
      "shuffled_traffic_does_not_contract_task_metric": shuffled_vs_frozen >= 0.98,
      "traffic_proportional_overcontracts_background": traffic["all_leaf_distance"] >= 2.0*frozen["all_leaf_distance"],
    }
    passed=all(requirements.values())
    result={
      "gate":1,
      "classification":"TRAFFIC_DRIVEN_RULE_DEFORMS_SPECTRAL_METRIC_TO_NEAR_ORACLE_UNDER_BUDGET" if passed else "METRIC_PLASTICITY_GATE_FAILED",
      "passed":passed,
      "tree":{"nodes":n_nodes,"edges":n_edges,"total_coupling_budget":budget},
      "task":{"train_pairs":train,"heldout_pairs":test,"edge_flow_match_max_error":flow_match_error,"standing_body_tax":args.standing_tax},
      "objective":"J(g)=E_task[d_D(s,t)] + lambda * sum_e 1/g_e, subject to sum_e g_e=B",
      "analytic_oracle":"g_e* = B sqrt(F_e+lambda) / sum_j sqrt(F_j+lambda)",
      "online_rule":"g_e += eta * ((q_e+lambda)/g_e^2 - mean_j((q_j+lambda)/g_j^2)); q_e is local path traffic",
      "frozen_uniform":frozen,
      "traffic_proportional":traffic,
      "analytic_metric_oracle":oracle,
      "stochastic_metric_plasticity":learned,
      "shuffled_traffic_attacker":shuffled,
      "comparisons":{"learned_objective_over_oracle":float(learned_oracle_ratio),"learned_heldout_distance_over_frozen":float(learned_vs_frozen),"shuffled_heldout_distance_over_frozen":float(shuffled_vs_frozen)},
      "requirements":requirements,
      "scope":"Literal metric deformation inside the chosen finite spectral triple; not a biological, novelty, or necessity claim."
    }
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2)); raise SystemExit(0 if passed else 1)

if __name__ == "__main__":
    main()
