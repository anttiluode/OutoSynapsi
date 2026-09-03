#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from outosynapsi import SpectralTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/GATE0.json")
    args = parser.parse_args()
    rng = np.random.default_rng(240903)
    max_commutator_error = 0.0
    max_distance_witness_error = 0.0
    max_two_point_error = 0.0

    for g in np.geomspace(0.1, 10.0, 41):
        tree = SpectralTree(2, [(0, 1)], [g])
        max_two_point_error = max(max_two_point_error, abs(tree.connes_distance(0, 1) - 1.0 / g))

    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (5, 6)]
    tree = SpectralTree(7, edges, rng.uniform(0.25, 2.5, size=len(edges)))
    for _ in range(128):
        a = rng.normal(size=7)
        max_commutator_error = max(max_commutator_error, abs(tree.lipschitz_seminorm(a) - tree.explicit_commutator_norm(a)))

    for source in range(7):
        witness = tree.distance_witness(source)
        if tree.lipschitz_seminorm(witness) > 1.0 + 1e-10:
            raise AssertionError("distance witness left Lipschitz ball")
        for target in range(7):
            max_distance_witness_error = max(max_distance_witness_error, abs((witness[target] - witness[source]) - tree.connes_distance(source, target)))

    passed = max_two_point_error < 1e-12 and max_commutator_error < 1e-12 and max_distance_witness_error < 1e-12
    result = {
        "gate": 0,
        "classification": "EDGE_COUPLING_IS_EXACT_INVERSE_SPECTRAL_LENGTH_IN_THIS_TRIPLE" if passed else "EXACT_METRIC_AUDIT_FAILED",
        "passed": passed,
        "construction": "A=C(V), H=direct_sum_edges C^2, pi(a)|e=diag(a_u,a_v), D_e=[[0,g_e],[g_e,0]]",
        "identity": "||[D,pi(a)]|| = max_e g_e |a_u-a_v|",
        "distance": "d(i,j) = shortest_path sum_e 1/g_e; unique path on a tree",
        "max_two_point_absolute_error": float(max_two_point_error),
        "max_commutator_norm_absolute_error": float(max_commutator_error),
        "max_distance_witness_absolute_error": float(max_distance_witness_error),
        "scope": "Exact for the chosen finite edge-block spectral triple; not every graph Dirac operator gives this distance."
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)

if __name__ == "__main__":
    main()
