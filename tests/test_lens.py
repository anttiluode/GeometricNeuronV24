from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnv24.lens import (  # noqa: E402
    Address,
    adjoint_gradient,
    area_overlap_1d,
    lens_mask,
    lens_matrix,
    numerical_rank,
    pair_rank_prediction,
    stack_lenses,
)


class LensTests(unittest.TestCase):
    def test_every_address_is_an_average(self) -> None:
        for n in (16, 32):
            for scale in (1, 3, 5, 8, 15):
                h = lens_matrix(n, scale)
                self.assertTrue(np.allclose(h.sum(axis=1), 1.0, atol=2e-14))
                self.assertTrue(np.all(h >= 0.0))

    def test_pixel_scale_is_identity(self) -> None:
        h = lens_matrix(16, 16)
        self.assertTrue(np.allclose(h, np.eye(16 * 16), atol=2e-14))

    def test_address_matches_matrix_row(self) -> None:
        address = Address(scale=7, row=3, col=5)
        mask = lens_mask(32, address).reshape(-1)
        h = lens_matrix(32, 7)
        self.assertTrue(np.allclose(mask, h[3 * 7 + 5]))

    def test_nested_dyadic_lenses_add_no_rank(self) -> None:
        self.assertEqual(numerical_rank(stack_lenses(16, [4, 8])), 8**2)

    def test_pair_formula_for_selected_partitions(self) -> None:
        for a, b in ((4, 8), (5, 8), (7, 8), (6, 10)):
            observed = numerical_rank(stack_lenses(16, [a, b]))
            self.assertEqual(observed, pair_rank_prediction(a, b, n=16))

    def test_nondivisor_overlap_still_conserves_constant(self) -> None:
        p = area_overlap_1d(32, 15)
        self.assertTrue(np.allclose(p @ np.ones(32), 1.0, atol=2e-14))

    def test_transpose_is_loss_gradient(self) -> None:
        rng = np.random.default_rng(3)
        h = stack_lenses(12, [5, 7])
        estimate = rng.normal(size=12 * 12)
        observed = rng.normal(size=h.shape[0])
        direction = rng.normal(size=12 * 12)
        direction /= np.linalg.norm(direction)
        gradient = adjoint_gradient(h, estimate, observed)

        def loss(candidate: np.ndarray) -> float:
            residual = h @ candidate - observed
            return 0.5 * float(residual @ residual)

        eps = 1e-6
        fd = (loss(estimate + eps * direction) - loss(estimate - eps * direction)) / (2 * eps)
        self.assertAlmostEqual(fd, float(gradient @ direction), places=7)


if __name__ == "__main__":
    unittest.main()
