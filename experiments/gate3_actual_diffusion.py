#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from outosynapsi import (
    SpectralTree,
    analytic_budget_oracle,
    complete_binary_tree,
    edge_flow,
    traffic_proportional_allocation,
)
from outosynapsi.dynamics import (
    cubic_budget_oracle,
    directional_volume_resistance,
    expected_pair_scores,
    mfpt_matrix,
    source_side_table,
    stochastic_mfpt_plasticity,
)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return correlation(rankdata(np.asarray(a)), rankdata(np.asarray(b)))


def log_linear_predictor(
    metric: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, float]:
    metric = np.maximum(np.asarray(metric, dtype=float), 1e-15)
    target = np.maximum(np.asarray(target, dtype=float), 1e-15)
    test_mask = ~train_mask

    x_train = np.column_stack(
        [np.ones(int(np.sum(train_mask))), np.log(metric[train_mask])]
    )
    beta = np.linalg.lstsq(
        x_train, np.log(target[train_mask]), rcond=None
    )[0]
    predicted = np.exp(
        beta[0] + beta[1] * np.log(metric[test_mask])
    )
    y = target[test_mask]
    log_y = np.log(y)
    log_pred = np.log(predicted)
    log_r2 = 1.0 - float(
        np.sum((log_y - log_pred) ** 2)
        / np.sum((log_y - np.mean(log_y)) ** 2)
    )
    mape = float(np.mean(np.abs(predicted - y) / y))

    return {
        "spearman_all": spearman(metric, target),
        "log_pearson_all": correlation(np.log(metric), np.log(target)),
        "heldout_log_r2": log_r2,
        "heldout_mape": mape,
        "fit_intercept": float(beta[0]),
        "fit_exponent": float(beta[1]),
    }


def summarize(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def allocation(score: np.ndarray, exponent: float, budget: float) -> np.ndarray:
    value = np.asarray(score, dtype=float) ** float(exponent)
    return float(budget) * value / float(np.sum(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/GATE3.json")
    parser.add_argument("--seed", type=int, default=240903)
    parser.add_argument("--geometry-samples", type=int, default=96)
    parser.add_argument("--fit-geometries", type=int, default=64)
    parser.add_argument("--geometry-log-sigma", type=float, default=0.7)
    parser.add_argument("--online-seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--standing-tax", type=float, default=0.05)
    parser.add_argument("--body-tax", type=float, default=0.05)
    args = parser.parse_args()

    n_nodes, edges = complete_binary_tree(depth=4)
    n_edges = len(edges)
    budget = float(n_edges)
    uniform = np.ones(n_edges, dtype=float)
    topology = SpectralTree(n_nodes, edges, uniform)
    side = source_side_table(topology)

    source_leaves = [15, 16, 17, 18]
    target_leaves = [27, 28, 29, 30]
    train_pairs: list[tuple[int, int]] = []
    for i, source in enumerate(source_leaves):
        train_pairs.extend(
            [
                (source, target_leaves[i]),
                (source, target_leaves[(i + 1) % 4]),
            ]
        )
    test_pairs = [
        pair
        for pair in itertools.product(source_leaves, target_leaves)
        if pair not in train_pairs
    ]
    leaves = list(range(15, 31))
    body_pairs = [
        (source, target)
        for source in leaves
        for target in leaves
        if source != target
    ]

    task_rows, task_coefficient = expected_pair_scores(
        topology, train_pairs, side_table=side
    )
    _, test_coefficient = expected_pair_scores(
        topology, test_pairs, side_table=side
    )
    body_rows, body_coefficient = expected_pair_scores(
        topology, body_pairs, side_table=side
    )
    coefficient_match_error = float(
        np.max(np.abs(task_coefficient - test_coefficient))
    )

    # Gate-1 and Gate-2 allocations are carried forward exactly as attackers.
    flow = edge_flow(topology, train_pairs)
    gate1_sqrt = analytic_budget_oracle(
        flow, args.standing_tax, budget
    )
    gate2_cube = allocation(
        flow + float(args.standing_tax), 1.0 / 3.0, budget
    )
    traffic_proportional = traffic_proportional_allocation(
        flow, args.standing_tax, budget
    )

    # Actual diffusion objective.  The local generator uses edge rates g_e^2.
    # On a tree, one-way MFPT has coefficient equal to source-side volume.
    dynamic_coefficient = (
        task_coefficient
        + float(args.body_tax) * body_coefficient
    )
    dynamic_oracle = cubic_budget_oracle(
        dynamic_coefficient, budget
    )

    def propagation_metrics(weights: np.ndarray) -> dict[str, float]:
        tree = SpectralTree(n_nodes, edges, weights)
        h = mfpt_matrix(tree, conductance_power=2.0)
        heldout = float(
            np.mean([h[source, target] for source, target in test_pairs])
        )
        body = float(
            np.mean([h[source, target] for source, target in body_pairs])
        )
        return {
            "heldout_mfpt": heldout,
            "all_leaf_mfpt": body,
            "dynamic_objective": heldout + float(args.body_tax) * body,
            "min_coupling": float(np.min(weights)),
            "max_coupling": float(np.max(weights)),
        }

    allocation_results = {
        "frozen_uniform": propagation_metrics(uniform),
        "gate1_dirac_sqrt": propagation_metrics(gate1_sqrt),
        "gate2_resistance_cube": propagation_metrics(gate2_cube),
        "traffic_proportional": propagation_metrics(
            traffic_proportional
        ),
        "dynamic_mfpt_oracle": propagation_metrics(dynamic_oracle),
    }

    online_rows: list[dict[str, float]] = []
    online_cosines: list[float] = []
    for seed in range(args.online_seeds):
        learned = stochastic_mfpt_plasticity(
            n_nodes,
            edges,
            train_pairs,
            body_pairs,
            seed=seed,
            steps=args.steps,
            learning_rate=args.learning_rate,
            body_tax=args.body_tax,
            total_budget=budget,
        )
        online_rows.append(propagation_metrics(learned))
        online_cosines.append(
            float(
                np.dot(learned, dynamic_oracle)
                / (
                    np.linalg.norm(learned)
                    * np.linalg.norm(dynamic_oracle)
                )
            )
        )

    online_summary = {
        key: summarize([row[key] for row in online_rows])
        for key in (
            "heldout_mfpt",
            "all_leaf_mfpt",
            "dynamic_objective",
            "min_coupling",
            "max_coupling",
        )
    }
    online_summary["cosine_to_dynamic_oracle"] = summarize(
        online_cosines
    )

    # Independent predictor benchmark over unseen coupling geometries.
    rng = np.random.default_rng(args.seed)
    rows: list[tuple[float, ...]] = []
    for geometry_index in range(args.geometry_samples):
        weights = np.exp(
            rng.normal(
                0.0, args.geometry_log_sigma, size=n_edges
            )
        )
        weights *= budget / float(np.sum(weights))
        tree = SpectralTree(n_nodes, edges, weights)
        h = mfpt_matrix(tree, conductance_power=2.0)

        for source, target in body_pairs:
            path = tree.path_edge_indices(source, target)
            connes = tree.connes_distance(source, target)
            resistance = float(
                sum(1.0 / weights[ei] ** 2 for ei in path)
            )
            hop_count = float(len(path))
            bottleneck = float(
                max(1.0 / weights[ei] ** 2 for ei in path)
            )
            directional = directional_volume_resistance(
                tree,
                source,
                target,
                conductance_power=2.0,
                side_table=side,
            )
            rows.append(
                (
                    float(geometry_index),
                    float(h[source, target]),
                    connes,
                    resistance,
                    hop_count,
                    bottleneck,
                    directional,
                )
            )

    array = np.asarray(rows, dtype=float)
    geometry_index = array[:, 0].astype(int)
    propagation_time = array[:, 1]
    fit_mask = geometry_index < int(args.fit_geometries)

    predictors = {
        "connes_distance": log_linear_predictor(
            array[:, 2], propagation_time, fit_mask
        ),
        "resistance_distance": log_linear_predictor(
            array[:, 3], propagation_time, fit_mask
        ),
        "hop_count": log_linear_predictor(
            array[:, 4], propagation_time, fit_mask
        ),
        "bottleneck_inverse_conductance": log_linear_predictor(
            array[:, 5], propagation_time, fit_mask
        ),
        "directional_volume_resistance": log_linear_predictor(
            array[:, 6], propagation_time, fit_mask
        ),
    }

    oracle_objective = allocation_results[
        "dynamic_mfpt_oracle"
    ]["dynamic_objective"]
    learned_objective = online_summary[
        "dynamic_objective"
    ]["mean"]

    requirements = {
        "train_and_heldout_dynamic_coefficients_match": (
            coefficient_match_error < 1e-12
        ),
        "directional_volume_predictor_is_exact": (
            predictors["directional_volume_resistance"][
                "heldout_log_r2"
            ]
            > 0.999999
            and predictors["directional_volume_resistance"][
                "heldout_mape"
            ]
            < 1e-10
        ),
        "plain_resistance_beats_connes_on_heldout_geometries": (
            predictors["resistance_distance"]["heldout_log_r2"]
            > predictors["connes_distance"]["heldout_log_r2"] + 0.02
        ),
        "connes_distance_is_not_the_diffusion_time": (
            predictors["connes_distance"]["heldout_log_r2"] < 0.90
        ),
        "online_dynamic_rule_within_0p3pct_oracle": (
            learned_objective / oracle_objective <= 1.003
        ),
        "online_couplings_match_dynamic_oracle": (
            online_summary["cosine_to_dynamic_oracle"]["mean"]
            >= 0.9998
        ),
        "dynamic_oracle_beats_gate2_objective": (
            oracle_objective
            < allocation_results["gate2_resistance_cube"][
                "dynamic_objective"
            ]
        ),
        "traffic_proportional_breaks_background_dynamics": (
            allocation_results["traffic_proportional"][
                "all_leaf_mfpt"
            ]
            > 10.0
            * allocation_results["frozen_uniform"][
                "all_leaf_mfpt"
            ]
        ),
    }
    passed = all(requirements.values())

    result = {
        "gate": 3,
        "classification": (
            "LOCAL_DIFFUSION_REVEALS_DIRECTIONAL_VOLUME_GEOMETRY"
            if passed
            else "ACTUAL_DYNAMICS_GATE_FAILED"
        ),
        "passed": passed,
        "dynamics": (
            "continuous-time local diffusion / random walk with "
            "adjacent transition rate c_e=g_e^2"
        ),
        "measurement": (
            "source-to-target mean first-passage time from the "
            "generator; no Connes or resistance distance enters "
            "the propagation solve"
        ),
        "tree": {
            "nodes": n_nodes,
            "edges": n_edges,
            "coupling_budget": budget,
        },
        "task": {
            "train_pairs": train_pairs,
            "heldout_pairs": test_pairs,
            "all_leaf_ordered_pairs": len(body_pairs),
            "dynamic_coefficient_match_max_error": (
                coefficient_match_error
            ),
            "body_tax": float(args.body_tax),
        },
        "predictor_benchmark": {
            "random_geometries": int(args.geometry_samples),
            "fit_geometries": int(args.fit_geometries),
            "heldout_geometries": int(
                args.geometry_samples - args.fit_geometries
            ),
            "ordered_leaf_pairs_per_geometry": len(body_pairs),
            "total_propagation_problems": int(len(rows)),
            "lognormal_sigma": float(args.geometry_log_sigma),
            "predictors": predictors,
        },
        "allocations_on_actual_dynamics": allocation_results,
        "online_dynamic_plasticity": {
            "seeds": int(args.online_seeds),
            "updates_per_seed": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "rule": (
                "g_e += eta * [2(score_task + body_tax*score_body)"
                "/g_e^3 - mean_edges]; score is source-side volume "
                "on each currently traversed edge"
            ),
            "summary": online_summary,
            "objective_over_oracle": float(
                learned_objective / oracle_objective
            ),
        },
        "exact_tree_identity": (
            "For this symmetric continuous-time generator on a tree, "
            "MFPT(s->t)=sum_{e in path} |S_e(s)|/g_e^2, where "
            "|S_e(s)| is the number of vertices on the source side "
            "when edge e is cut."
        ),
        "requirements": requirements,
        "scope": (
            "This gate establishes an operator-specific propagation geometry "
            "for one local diffusion process. It does not make Connes distance "
            "a universal dynamical distance, nor claim that biological "
            "synapses use this generator."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
