from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Edge:
    u: int
    v: int


class SpectralTree:
    """Finite edge-block spectral triple on a connected tree.

    Algebra: A = C(V), represented on H = direct_sum_e C^2 by
        pi(a)|_e = diag(a_u, a_v)

    Dirac block on edge e=(u,v):
        D_e = [[0, g_e], [g_e, 0]]

    Therefore
        ||[D, pi(a)]|| = max_e g_e |a_u-a_v|.

    The Connes distance between pure vertex states is exactly the weighted
    shortest-path metric with edge lengths 1/g_e. On a tree the path is unique.
    """

    def __init__(self, n_nodes: int, edges: Sequence[tuple[int, int] | Edge], couplings: Sequence[float]):
        self.n_nodes = int(n_nodes)
        self.edges = [e if isinstance(e, Edge) else Edge(int(e[0]), int(e[1])) for e in edges]
        self.couplings = np.asarray(couplings, dtype=float)
        if len(self.edges) != len(self.couplings):
            raise ValueError("one positive coupling is required per edge")
        if np.any(self.couplings <= 0):
            raise ValueError("couplings must be positive")
        if len(self.edges) != self.n_nodes - 1:
            raise ValueError("SpectralTree expects a tree with n_nodes-1 edges")
        self._adj: list[list[tuple[int, int]]] = [[] for _ in range(self.n_nodes)]
        for ei, edge in enumerate(self.edges):
            self._adj[edge.u].append((edge.v, ei))
            self._adj[edge.v].append((edge.u, ei))
        self._validate_connected()

    def _validate_connected(self) -> None:
        seen = {0}
        q = deque([0])
        while q:
            u = q.popleft()
            for v, _ in self._adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        if len(seen) != self.n_nodes:
            raise ValueError("tree must be connected")

    def path_edge_indices(self, source: int, target: int) -> list[int]:
        source = int(source)
        target = int(target)
        parent: dict[int, tuple[int, int]] = {source: (-1, -1)}
        q = deque([source])
        while q:
            u = q.popleft()
            if u == target:
                break
            for v, ei in self._adj[u]:
                if v not in parent:
                    parent[v] = (u, ei)
                    q.append(v)
        if target not in parent:
            raise ValueError("disconnected vertices")
        path: list[int] = []
        cur = target
        while cur != source:
            prev, ei = parent[cur]
            path.append(ei)
            cur = prev
        path.reverse()
        return path

    def connes_distance(self, source: int, target: int) -> float:
        return float(sum(1.0 / self.couplings[ei] for ei in self.path_edge_indices(source, target)))

    def distance_witness(self, source: int) -> np.ndarray:
        """A vertex function saturating distance from source to every vertex."""
        out = np.zeros(self.n_nodes, dtype=float)
        q = deque([int(source)])
        seen = {int(source)}
        while q:
            u = q.popleft()
            for v, ei in self._adj[u]:
                if v in seen:
                    continue
                out[v] = out[u] + 1.0 / self.couplings[ei]
                seen.add(v)
                q.append(v)
        return out

    def lipschitz_seminorm(self, observable: Sequence[float]) -> float:
        a = np.asarray(observable, dtype=float)
        if a.shape != (self.n_nodes,):
            raise ValueError("observable must have one value per vertex")
        return float(max(
            self.couplings[ei] * abs(a[edge.u] - a[edge.v])
            for ei, edge in enumerate(self.edges)
        ))

    def dirac_matrix(self) -> np.ndarray:
        """Explicit block-diagonal D, useful only for auditing small examples."""
        dim = 2 * len(self.edges)
        D = np.zeros((dim, dim), dtype=float)
        for ei, g in enumerate(self.couplings):
            k = 2 * ei
            D[k, k + 1] = g
            D[k + 1, k] = g
        return D

    def representation(self, observable: Sequence[float]) -> np.ndarray:
        a = np.asarray(observable, dtype=float)
        dim = 2 * len(self.edges)
        A = np.zeros((dim, dim), dtype=float)
        for ei, edge in enumerate(self.edges):
            k = 2 * ei
            A[k, k] = a[edge.u]
            A[k + 1, k + 1] = a[edge.v]
        return A

    def explicit_commutator_norm(self, observable: Sequence[float]) -> float:
        D = self.dirac_matrix()
        A = self.representation(observable)
        comm = D @ A - A @ D
        return float(np.linalg.norm(comm, ord=2))


def complete_binary_tree(depth: int = 4) -> tuple[int, list[tuple[int, int]]]:
    n_nodes = 2 ** (depth + 1) - 1
    edges: list[tuple[int, int]] = []
    for i in range(2 ** depth - 1):
        edges.append((i, 2 * i + 1))
        edges.append((i, 2 * i + 2))
    return n_nodes, edges


def edge_flow(tree: SpectralTree, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    flow = np.zeros(len(tree.edges), dtype=float)
    for source, target in pairs:
        for ei in tree.path_edge_indices(source, target):
            flow[ei] += 1.0 / len(pairs)
    return flow


def mean_distance(tree: SpectralTree, pairs: Sequence[tuple[int, int]]) -> float:
    return float(np.mean([tree.connes_distance(s, t) for s, t in pairs]))


def regularized_metric_objective(
    tree: SpectralTree,
    task_flow: Sequence[float],
    standing_tax: float,
) -> float:
    f = np.asarray(task_flow, dtype=float)
    return float(np.sum((f + float(standing_tax)) / tree.couplings))


def analytic_budget_oracle(
    task_flow: Sequence[float],
    standing_tax: float,
    total_budget: float,
) -> np.ndarray:
    """Exact minimizer of sum_e (F_e+lambda)/g_e subject to sum g_e=B."""
    f = np.asarray(task_flow, dtype=float)
    score = np.sqrt(f + float(standing_tax))
    return float(total_budget) * score / float(np.sum(score))


def traffic_proportional_allocation(
    task_flow: Sequence[float],
    standing_tax: float,
    total_budget: float,
) -> np.ndarray:
    f = np.asarray(task_flow, dtype=float)
    score = f + float(standing_tax)
    return float(total_budget) * score / float(np.sum(score))


def stochastic_metric_plasticity(
    n_nodes: int,
    edges: Sequence[tuple[int, int] | Edge],
    train_pairs: Sequence[tuple[int, int]],
    *,
    seed: int,
    steps: int = 10_000,
    learning_rate: float = 0.002,
    standing_tax: float = 0.05,
    total_budget: float | None = None,
    shuffle_traffic: bool = False,
) -> np.ndarray:
    """Online projected descent on expected Connes distance + standing-body tax.

    Current task contributes q_e=1 on each edge of its unique path and zero
    elsewhere. The stochastic negative gradient of (q_e+lambda)/g_e is
    +(q_e+lambda)/g_e^2. Subtracting the mean update is the global homeostat that
    keeps the total coupling budget fixed.
    """
    n_edges = len(edges)
    budget = float(n_edges if total_budget is None else total_budget)
    weights = np.full(n_edges, budget / n_edges, dtype=float)
    rng = np.random.default_rng(int(seed))

    topology = SpectralTree(n_nodes, edges, np.ones(n_edges))
    pair_paths = [topology.path_edge_indices(s, t) for s, t in train_pairs]

    for _ in range(int(steps)):
        path = pair_paths[int(rng.integers(0, len(pair_paths)))]
        q = np.zeros(n_edges, dtype=float)
        q[path] = 1.0
        if shuffle_traffic:
            q = q[rng.permutation(n_edges)]
        update = (q + float(standing_tax)) / (weights * weights)
        weights += float(learning_rate) * (update - float(np.mean(update)))
        if np.any(weights <= 0):
            raise RuntimeError("learning rate drove a coupling non-positive")
        weights += (budget - float(np.sum(weights))) / n_edges

    return weights
