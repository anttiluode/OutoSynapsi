#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from outosynapsi import SpectralTree, complete_binary_tree
from outosynapsi.dynamics import (
    mfpt_matrix,
    mfpt_to_target,
    source_side_table,
)


HIDDEN_POWERS = (0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
ALPHAS = np.arange(0.0, 1.5001, 0.05)
BETAS = np.arange(0.5, 3.5001, 0.05)
CALIBRATION_PROBES = 16
HELDOUT_PROBES = 32
MEASUREMENT_LOG_NOISE = 0.02
BODY_TAX = 0.05
GEOMETRY_SIGMA = 0.7


def summarize(values):
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def random_couplings(rng, n_edges, budget):
    w = np.exp(rng.normal(0.0, GEOMETRY_SIGMA, size=n_edges))
    return float(budget) * w / float(np.sum(w))


def edge_coefficient(tree, side, pairs, alpha):
    coeff = np.zeros(len(tree.edges), dtype=float)
    for source, target in pairs:
        for ei in tree.path_edge_indices(source, target):
            coeff[ei] += (
                float(side[source, ei]) ** float(alpha)
            ) / len(pairs)
    return coeff


def model_allocation(
    tree,
    side,
    train_pairs,
    body_pairs,
    alpha,
    beta,
    body_tax,
    budget,
):
    coeff = edge_coefficient(tree, side, train_pairs, alpha)
    coeff += float(body_tax) * edge_coefficient(
        tree, side, body_pairs, alpha
    )
    score = coeff ** (1.0 / (float(beta) + 1.0))
    return float(budget) * score / float(np.sum(score))


def actual_objective(
    n_nodes,
    edges,
    weights,
    hidden_power,
    heldout_pairs,
    body_pairs,
    body_tax,
):
    tree = SpectralTree(n_nodes, edges, weights)
    h = mfpt_matrix(tree, conductance_power=hidden_power)
    task = float(
        np.mean([h[s, t] for s, t in heldout_pairs])
    )
    body = float(np.mean([h[s, t] for s, t in body_pairs]))
    return {
        "heldout_mfpt": task,
        "all_leaf_mfpt": body,
        "objective": task + float(body_tax) * body,
    }


def exact_oracle(
    tree,
    side,
    train_pairs,
    body_pairs,
    hidden_power,
    body_tax,
    budget,
):
    coeff = edge_coefficient(tree, side, train_pairs, 1.0)
    coeff += float(body_tax) * edge_coefficient(
        tree, side, body_pairs, 1.0
    )
    score = coeff ** (1.0 / (float(hidden_power) + 1.0))
    return float(budget) * score / float(np.sum(score))


def path_logs(tree, side, weights, source, target):
    path = tree.path_edge_indices(source, target)
    log_side = np.log(side[source, path])[None, None, :]
    log_weight = np.log(weights[path])[None, None, :]
    return log_side, log_weight


def candidate_log_metrics(tree, side, weights, source, target):
    log_side, log_weight = path_logs(
        tree, side, weights, source, target
    )
    aa = ALPHAS[:, None, None]
    bb = BETAS[None, :, None]
    values = np.exp(aa * log_side - bb * log_weight)
    return np.log(np.sum(values, axis=2))


def fit_geometry(tree, side, samples):
    observed = np.log(
        np.asarray([row["arrival_time"] for row in samples], dtype=float)
    )
    predictions = np.stack(
        [
            candidate_log_metrics(
                tree,
                side,
                row["couplings"],
                row["source"],
                row["target"],
            )
            for row in samples
        ],
        axis=2,
    )
    offsets = np.mean(
        observed[None, None, :] - predictions, axis=2
    )
    residual = observed[None, None, :] - (
        predictions + offsets[:, :, None]
    )
    mse = np.mean(residual * residual, axis=2)
    ia, ib = np.unravel_index(np.argmin(mse), mse.shape)
    return {
        "alpha": float(ALPHAS[ia]),
        "beta": float(BETAS[ib]),
        "log_scale": float(offsets[ia, ib]),
        "train_log_mse": float(mse[ia, ib]),
    }


def prediction_score(tree, side, samples, fit):
    alpha_index = int(np.argmin(np.abs(ALPHAS - fit["alpha"])))
    beta_index = int(np.argmin(np.abs(BETAS - fit["beta"])))
    observed = np.asarray(
        [row["arrival_time"] for row in samples], dtype=float
    )
    predicted = np.asarray(
        [
            math.exp(
                fit["log_scale"]
                + candidate_log_metrics(
                    tree,
                    side,
                    row["couplings"],
                    row["source"],
                    row["target"],
                )[alpha_index, beta_index]
            )
            for row in samples
        ],
        dtype=float,
    )
    log_y = np.log(observed)
    log_pred = np.log(predicted)
    r2 = 1.0 - float(
        np.sum((log_y - log_pred) ** 2)
        / np.sum((log_y - np.mean(log_y)) ** 2)
    )
    mape = float(np.mean(np.abs(predicted - observed) / observed))
    return {"heldout_log_r2": r2, "heldout_mape": mape}


def generate_samples(
    tree,
    pairs,
    hidden_power,
    count,
    seed,
    noise,
):
    rng = np.random.default_rng(int(seed))
    samples = []
    for _ in range(int(count)):
        weights = random_couplings(
            rng, len(tree.edges), len(tree.edges)
        )
        source, target = pairs[int(rng.integers(0, len(pairs)))]
        body = SpectralTree(tree.n_nodes, tree.edges, weights)
        arrival = mfpt_to_target(
            body,
            source,
            target,
            conductance_power=hidden_power,
        )
        arrival *= math.exp(float(rng.normal(0.0, noise)))
        samples.append(
            {
                "couplings": weights,
                "source": int(source),
                "target": int(target),
                "arrival_time": float(arrival),
            }
        )
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/GATE4.json")
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    n_nodes, edges = complete_binary_tree(depth=4)
    n_edges = len(edges)
    budget = float(n_edges)
    topology = SpectralTree(
        n_nodes, edges, np.ones(n_edges, dtype=float)
    )
    side = source_side_table(topology)

    source_leaves = [15, 16, 17, 18]
    target_leaves = [27, 28, 29, 30]
    train_pairs = []
    for i, source in enumerate(source_leaves):
        train_pairs.extend(
            [
                (source, target_leaves[i]),
                (source, target_leaves[(i + 1) % 4]),
            ]
        )
    heldout_pairs = [
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

    world_rows = []
    fixed_attackers = {
        "uniform": np.ones(n_edges, dtype=float),
        "fixed_connes_alpha0_beta1": model_allocation(
            topology, side, train_pairs, body_pairs,
            0.0, 1.0, BODY_TAX, budget
        ),
        "fixed_resistance_alpha0_beta2": model_allocation(
            topology, side, train_pairs, body_pairs,
            0.0, 2.0, BODY_TAX, budget
        ),
        "fixed_gate3_alpha1_beta2": model_allocation(
            topology, side, train_pairs, body_pairs,
            1.0, 2.0, BODY_TAX, budget
        ),
    }
    attacker_rows = {name: [] for name in fixed_attackers}

    for hidden_power in HIDDEN_POWERS:
        oracle_weights = exact_oracle(
            topology,
            side,
            train_pairs,
            body_pairs,
            hidden_power,
            BODY_TAX,
            budget,
        )
        oracle = actual_objective(
            n_nodes,
            edges,
            oracle_weights,
            hidden_power,
            heldout_pairs,
            body_pairs,
            BODY_TAX,
        )

        for name, weights in fixed_attackers.items():
            row = actual_objective(
                n_nodes,
                edges,
                weights,
                hidden_power,
                heldout_pairs,
                body_pairs,
                BODY_TAX,
            )
            attacker_rows[name].append(
                {
                    "hidden_power": float(hidden_power),
                    "objective_over_oracle": float(
                        row["objective"] / oracle["objective"]
                    ),
                }
            )

        for seed in range(args.seeds):
            calibration = generate_samples(
                topology,
                body_pairs,
                hidden_power,
                CALIBRATION_PROBES,
                10000 + seed + int(hidden_power * 1000),
                MEASUREMENT_LOG_NOISE,
            )
            fit = fit_geometry(topology, side, calibration)

            heldout = generate_samples(
                topology,
                body_pairs,
                hidden_power,
                HELDOUT_PROBES,
                20000 + seed + int(hidden_power * 1000),
                0.0,
            )
            prediction = prediction_score(
                topology, side, heldout, fit
            )

            inferred_weights = model_allocation(
                topology,
                side,
                train_pairs,
                body_pairs,
                fit["alpha"],
                fit["beta"],
                BODY_TAX,
                budget,
            )
            inferred = actual_objective(
                n_nodes,
                edges,
                inferred_weights,
                hidden_power,
                heldout_pairs,
                body_pairs,
                BODY_TAX,
            )

            world_rows.append(
                {
                    "hidden_power": float(hidden_power),
                    "seed": int(seed),
                    **fit,
                    **prediction,
                    "objective_over_oracle": float(
                        inferred["objective"] / oracle["objective"]
                    ),
                    "heldout_mfpt": inferred["heldout_mfpt"],
                    "all_leaf_mfpt": inferred["all_leaf_mfpt"],
                    "objective": inferred["objective"],
                    "oracle_objective": oracle["objective"],
                }
            )

    # Probe-budget audit: does the identification improve as scalar
    # consequence samples increase?
    budget_sweep = {}
    for probe_budget in (4, 8, 16, 32):
        alpha_error = []
        beta_error = []
        objective_ratio = []
        for hidden_power in HIDDEN_POWERS:
            oracle_weights = exact_oracle(
                topology,
                side,
                train_pairs,
                body_pairs,
                hidden_power,
                BODY_TAX,
                budget,
            )
            oracle = actual_objective(
                n_nodes,
                edges,
                oracle_weights,
                hidden_power,
                heldout_pairs,
                body_pairs,
                BODY_TAX,
            )
            for seed in range(8):
                calibration = generate_samples(
                    topology,
                    body_pairs,
                    hidden_power,
                    probe_budget,
                    (
                        30000
                        + seed
                        + int(hidden_power * 1000)
                        + probe_budget * 100
                    ),
                    MEASUREMENT_LOG_NOISE,
                )
                fit = fit_geometry(topology, side, calibration)
                inferred_weights = model_allocation(
                    topology,
                    side,
                    train_pairs,
                    body_pairs,
                    fit["alpha"],
                    fit["beta"],
                    BODY_TAX,
                    budget,
                )
                inferred = actual_objective(
                    n_nodes,
                    edges,
                    inferred_weights,
                    hidden_power,
                    heldout_pairs,
                    body_pairs,
                    BODY_TAX,
                )
                alpha_error.append(abs(fit["alpha"] - 1.0))
                beta_error.append(
                    abs(fit["beta"] - hidden_power)
                )
                objective_ratio.append(
                    inferred["objective"] / oracle["objective"]
                )

        budget_sweep[str(probe_budget)] = {
            "worlds": len(objective_ratio),
            "mean_abs_alpha_error": float(
                np.mean(alpha_error)
            ),
            "mean_abs_beta_error": float(np.mean(beta_error)),
            "mean_objective_over_oracle": float(
                np.mean(objective_ratio)
            ),
            "worst_objective_over_oracle": float(
                np.max(objective_ratio)
            ),
        }

    per_power = {}
    for hidden_power in HIDDEN_POWERS:
        rows = [
            row
            for row in world_rows
            if row["hidden_power"] == hidden_power
        ]
        per_power[str(hidden_power)] = {
            "alpha": summarize([row["alpha"] for row in rows]),
            "beta": summarize([row["beta"] for row in rows]),
            "heldout_log_r2": summarize(
                [row["heldout_log_r2"] for row in rows]
            ),
            "heldout_mape": summarize(
                [row["heldout_mape"] for row in rows]
            ),
            "objective_over_oracle": summarize(
                [row["objective_over_oracle"] for row in rows]
            ),
        }

    learned_summary = {
        "alpha_absolute_error": summarize(
            [abs(row["alpha"] - 1.0) for row in world_rows]
        ),
        "beta_absolute_error": summarize(
            [
                abs(row["beta"] - row["hidden_power"])
                for row in world_rows
            ]
        ),
        "heldout_log_r2": summarize(
            [row["heldout_log_r2"] for row in world_rows]
        ),
        "heldout_mape": summarize(
            [row["heldout_mape"] for row in world_rows]
        ),
        "objective_over_oracle": summarize(
            [row["objective_over_oracle"] for row in world_rows]
        ),
    }

    attacker_summary = {}
    for name, rows in attacker_rows.items():
        attacker_summary[name] = {
            "mean_objective_over_oracle": float(
                np.mean(
                    [row["objective_over_oracle"] for row in rows]
                )
            ),
            "worst_objective_over_oracle": float(
                np.max(
                    [row["objective_over_oracle"] for row in rows]
                )
            ),
            "per_power": rows,
        }

    requirements = {
        "mean_beta_error_le_0p01": (
            learned_summary["beta_absolute_error"]["mean"] <= 0.01
        ),
        "mean_alpha_error_le_0p03": (
            learned_summary["alpha_absolute_error"]["mean"] <= 0.03
        ),
        "heldout_prediction_log_r2_ge_0p995": (
            learned_summary["heldout_log_r2"]["mean"] >= 0.995
        ),
        "mean_learned_objective_within_0p1pct_oracle": (
            learned_summary["objective_over_oracle"]["mean"]
            <= 1.001
        ),
        "worst_learned_objective_within_0p3pct_oracle": (
            learned_summary["objective_over_oracle"]["max"]
            <= 1.003
        ),
        "learned_beats_every_fixed_geometry_on_mean": all(
            learned_summary["objective_over_oracle"]["mean"]
            < attacker_summary[name][
                "mean_objective_over_oracle"
            ]
            for name in attacker_summary
        ),
        "eight_scalar_probes_are_already_near_oracle": (
            budget_sweep["8"]["mean_objective_over_oracle"]
            <= 1.001
        ),
    }
    passed = all(requirements.values())

    result = {
        "gate": 4,
        "classification": (
            "SCALAR_CONSEQUENCES_IDENTIFY_EFFECTIVE_GEOMETRY"
            if passed
            else "GEOMETRY_IDENTIFICATION_GATE_FAILED"
        ),
        "passed": passed,
        "question": (
            "Can the body infer which geometry governs consequence "
            "without being told the hidden conductance exponent?"
        ),
        "hidden_worlds": {
            "conductance_law": "c_e = g_e^p",
            "hidden_powers": list(HIDDEN_POWERS),
            "actual_directed_mfpt_geometry": (
                "sum |S_e(source)| / g_e^p"
            ),
        },
        "observer": {
            "per_calibration_observation": (
                "known local topology + current couplings + chosen "
                "(source,target) + one scalar arrival time"
            ),
            "calibration_probes": CALIBRATION_PROBES,
            "measurement_log_noise_sigma": MEASUREMENT_LOG_NOISE,
            "not_observed": (
                "generator matrix, hidden exponent p, full MFPT map, "
                "analytic tree identity"
            ),
        },
        "model_family": {
            "form": (
                "T_hat = scale * sum_on_path "
                "|S_e(source)|^alpha / g_e^beta"
            ),
            "alpha_grid": [
                float(ALPHAS[0]),
                float(ALPHAS[-1]),
                0.05,
            ],
            "beta_grid": [
                float(BETAS[0]),
                float(BETAS[-1]),
                0.05,
            ],
            "important_boundary": (
                "The learner selects a geometry inside this supplied "
                "two-parameter family; it does not invent the functional "
                "form from scratch."
            ),
        },
        "worlds_evaluated": len(world_rows),
        "per_hidden_power": per_power,
        "learned_summary": learned_summary,
        "fixed_geometry_attackers": attacker_summary,
        "probe_budget_sweep": budget_sweep,
        "requirements": requirements,
        "scope": (
            "This is system identification of an effective propagation "
            "geometry from scalar consequences. It demonstrates selection "
            "within a supplied metric family, not open-ended mathematical "
            "discovery or a biological mechanism."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
