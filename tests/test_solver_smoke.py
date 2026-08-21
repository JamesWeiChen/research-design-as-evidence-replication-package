"""Smoke tests for the public NHB solver surface."""

from __future__ import annotations

import math
import unittest

import numpy as np

import solver
from solver import DesignMixture, Model, NormalPrior, Params


class SolverSmokeTests(unittest.TestCase):
    """Check core identities, numerical behavior, and public API choices."""

    @staticmethod
    def benchmark_model() -> Model:
        return Model(
            NormalPrior(0.0, 1.0),
            Params(
                alpha=1.0,
                beta=1.0,
                c_a=200.0,
                c_p=4.0,
                c_d=4.0 / 25.0,
                delta0=15.0,
                V=1.0,
                r=15.0 / 7.0,
            ),
        )

    def test_parameter_identity(self) -> None:
        params = Params()
        design = 1.25
        self.assertGreater(params.omega(design), 0.0)
        self.assertLess(params.omega(design), 1.0)
        self.assertAlmostEqual(
            params.C_min(design, params.Y(design)),
            params.V,
            places=13,
        )

    def test_exact_normal_moments(self) -> None:
        prior = NormalPrior(0.0, 1.0)
        moments = prior.window_moments(
            np.asarray([-math.inf]),
            np.asarray([math.inf]),
            center=0.0,
        )[:, 0]
        np.testing.assert_allclose(moments, [1.0, 0.0, 1.0], atol=1e-14)

    def test_cutoff_residual(self) -> None:
        model = self.benchmark_model()
        result = model.cutoff_result(2.0)
        self.assertTrue(math.isfinite(result.value))
        self.assertLess(abs(model.G(result.value, 2.0)), 1e-9)
        self.assertEqual(result.status, "finite_boundary")

    def test_bracket_error_reports_provenance(self) -> None:
        model = self.benchmark_model()
        with self.assertRaisesRegex(ValueError, "caller-supplied"):
            model.cutoff_result(1.0, lo=2.0, hi=1.0)

    def test_design_mixture_sorts_and_normalizes(self) -> None:
        mixture = DesignMixture(
            support=(2.0, 0.5),
            weights=(3.0, 1.0),
        )
        self.assertEqual(mixture.support, (0.5, 2.0))
        self.assertEqual(mixture.weights, (0.25, 0.75))

    def test_selected_public_surface_has_no_compatibility_aliases(self) -> None:
        model = self.benchmark_model()
        self.assertFalse(hasattr(model, "comp"))
        self.assertFalse(hasattr(model, "_U_hat"))
        removed_completion = "uniform_" + "completion"
        self.assertFalse(hasattr(solver, removed_completion))


if __name__ == "__main__":
    unittest.main()
