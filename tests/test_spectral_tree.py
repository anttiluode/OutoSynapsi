import unittest

import numpy as np

from outosynapsi import SpectralTree, complete_binary_tree, stochastic_metric_plasticity
from outosynapsi.dynamics import directional_volume_resistance, mfpt_matrix, source_side_table


class SpectralTreeTests(unittest.TestCase):
    def test_two_point_distance_is_inverse_coupling(self):
        for g in (0.25, 0.5, 1.0, 2.0, 7.0):
            tree = SpectralTree(2, [(0, 1)], [g])
            self.assertAlmostEqual(tree.connes_distance(0, 1), 1.0 / g, places=12)

    def test_explicit_commutator_matches_edge_formula(self):
        tree = SpectralTree(4, [(0, 1), (1, 2), (1, 3)], [0.7, 1.4, 2.1])
        a = np.array([0.2, -0.3, 1.1, 0.7])
        self.assertAlmostEqual(
            tree.lipschitz_seminorm(a),
            tree.explicit_commutator_norm(a),
            places=12,
        )

    def test_distance_witness_saturates_lipschitz_ball(self):
        tree = SpectralTree(5, [(0, 1), (1, 2), (1, 3), (3, 4)], [1.0, 2.0, 0.5, 4.0])
        witness = tree.distance_witness(0)
        self.assertLessEqual(tree.lipschitz_seminorm(witness), 1.0 + 1e-12)
        self.assertAlmostEqual(witness[4] - witness[0], tree.connes_distance(0, 4), places=12)

    def test_local_diffusion_mfpt_matches_directional_tree_formula(self):
        tree = SpectralTree(
            5,
            [(0, 1), (1, 2), (1, 3), (3, 4)],
            [0.8, 1.7, 0.55, 2.2],
        )
        mfpt = mfpt_matrix(tree)
        side = source_side_table(tree)
        for source in range(tree.n_nodes):
            for target in range(tree.n_nodes):
                if source == target:
                    continue
                predicted = directional_volume_resistance(
                    tree, source, target, side_table=side
                )
                self.assertAlmostEqual(
                    float(mfpt[source, target]), predicted, places=10
                )

    def test_projected_plasticity_preserves_budget(self):
        n, edges = complete_binary_tree(3)
        weights = stochastic_metric_plasticity(
            n,
            edges,
            [(7, 13), (8, 14)],
            seed=1,
            steps=200,
        )
        self.assertAlmostEqual(float(weights.sum()), float(len(edges)), places=10)
        self.assertTrue(np.all(weights > 0))


if __name__ == "__main__":
    unittest.main()
