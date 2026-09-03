from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

from .spectral_tree import SpectralTree


def continuous_time_generator(
    tree: SpectralTree,
    *,
    conductance_power: float = 2.0,
) -> np.ndarray:
    """Symmetric local continuous-time random-walk generator.

    Adjacent vertices u,v exchange mass at rate c_e = g_e**conductance_power.
    No metric quantity is used in the dynamics.
    """
    q = np.zeros((tree.n_nodes, tree.n_nodes), dtype=float)
    conductance = tree.couplings ** float(conductance_power)
    for ei, edge in enumerate(tree.edges):
        c = float(conductance[ei])
        q[edge.u, edge.v] = c
        q[edge.v, edge.u] = c
    np.fill_diagonal(q, -np.sum(q, axis=1))
    return q


def mfpt_matrix(
    tree: SpectralTree,
    *,
    conductance_power: float = 2.0,
) -> np.ndarray:
    """Mean first-passage times of the local continuous-time dynamics."""
    q = continuous_time_generator(tree, conductance_power=conductance_power)
    n = tree.n_nodes
    out = np.zeros((n, n), dtype=float)
    for target in range(n):
        keep = np.arange(n) != target
        reduced = q[np.ix_(keep, keep)]
        h = np.linalg.solve(reduced, -np.ones(n - 1, dtype=float))
        out[keep, target] = h
    return out


def source_side_size(tree: SpectralTree, source: int, edge_index: int) -> int:
    """Number of vertices on source's side after deleting one tree edge."""
    source = int(source)
    blocked = int(edge_index)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(tree.n_nodes)]
    for ei, edge in enumerate(tree.edges):
        adjacency[edge.u].append((edge.v, ei))
        adjacency[edge.v].append((edge.u, ei))

    seen = {source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v, ei in adjacency[u]:
            if ei == blocked or v in seen:
                continue
            seen.add(v)
            queue.append(v)
    return len(seen)


def source_side_table(tree: SpectralTree) -> np.ndarray:
    out = np.zeros((tree.n_nodes, len(tree.edges)), dtype=int)
    for source in range(tree.n_nodes):
        for edge_index in range(len(tree.edges)):
            out[source, edge_index] = source_side_size(
                tree, source, edge_index
            )
    return out


def directional_volume_resistance(
    tree: SpectralTree,
    source: int,
    target: int,
    *,
    conductance_power: float = 2.0,
    side_table: np.ndarray | None = None,
) -> float:
    """Tree hitting-time quantity sum |S_e(source)| / g_e**power.

    For the symmetric continuous-time generator used here this equals the
    source->target mean first-passage time exactly on a tree.
    """
    side = source_side_table(tree) if side_table is None else side_table
    total = 0.0
    for ei in tree.path_edge_indices(source, target):
        total += float(side[source, ei]) / float(
            tree.couplings[ei] ** conductance_power
        )
    return float(total)


def pair_score_vector(
    tree: SpectralTree,
    pair: tuple[int, int],
    *,
    side_table: np.ndarray | None = None,
) -> np.ndarray:
    """Coefficient vector whose dot with 1/g^2 is pair MFPT."""
    side = source_side_table(tree) if side_table is None else side_table
    source, target = pair
    score = np.zeros(len(tree.edges), dtype=float)
    for ei in tree.path_edge_indices(source, target):
        score[ei] = float(side[source, ei])
    return score


def expected_pair_scores(
    tree: SpectralTree,
    pairs: Sequence[tuple[int, int]],
    *,
    side_table: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    side = source_side_table(tree) if side_table is None else side_table
    rows = np.stack(
        [pair_score_vector(tree, pair, side_table=side) for pair in pairs],
        axis=0,
    )
    return rows, np.mean(rows, axis=0)


def cubic_budget_oracle(
    coefficient: Sequence[float],
    total_budget: float,
) -> np.ndarray:
    """Exact optimum of sum C_e/g_e^2 subject to sum g_e=B."""
    c = np.asarray(coefficient, dtype=float)
    if np.any(c <= 0):
        raise ValueError("all coefficients must be positive")
    score = np.cbrt(c)
    return float(total_budget) * score / float(np.sum(score))


def stochastic_mfpt_plasticity(
    n_nodes: int,
    edges,
    task_pairs: Sequence[tuple[int, int]],
    body_pairs: Sequence[tuple[int, int]],
    *,
    seed: int,
    steps: int = 20_000,
    learning_rate: float = 0.0002,
    body_tax: float = 0.05,
    total_budget: float | None = None,
) -> np.ndarray:
    """Online descent on task MFPT + body_tax * background MFPT.

    For one sampled pair, score_e is the number of vertices on the source side
    of an edge if that edge lies on the source->target path, otherwise zero.
    Since pair MFPT is sum score_e/g_e^2, the negative gradient is
    2*score_e/g_e^3. Mean subtraction is the global fixed-budget homeostat.
    """
    n_edges = len(edges)
    budget = float(n_edges if total_budget is None else total_budget)
    topology = SpectralTree(n_nodes, edges, np.ones(n_edges, dtype=float))
    side = source_side_table(topology)
    task_rows, _ = expected_pair_scores(
        topology, task_pairs, side_table=side
    )
    body_rows, _ = expected_pair_scores(
        topology, body_pairs, side_table=side
    )

    rng = np.random.default_rng(int(seed))
    weights = np.full(n_edges, budget / n_edges, dtype=float)

    for _ in range(int(steps)):
        task_score = task_rows[int(rng.integers(0, len(task_rows)))]
        body_score = body_rows[int(rng.integers(0, len(body_rows)))]
        score = task_score + float(body_tax) * body_score
        update = 2.0 * score / (weights ** 3)
        weights += float(learning_rate) * (
            update - float(np.mean(update))
        )
        if np.any(weights <= 0):
            raise RuntimeError("learning rate drove a coupling non-positive")
        weights += (budget - float(np.sum(weights))) / n_edges

    return weights
