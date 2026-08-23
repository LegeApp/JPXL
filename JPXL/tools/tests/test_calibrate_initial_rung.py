#!/usr/bin/env python3
"""Unit tests for the initial-rung calibration helpers (stdlib unittest)."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import calibrate_initial_rung as cal


class TestInterpolation(unittest.TestCase):
    def test_exact_bracketed_crossing(self):
        # Two points straddling a target; the crossing is log-linear in
        # (ln scale, ln(100-score)). Build a curve whose loss halves per scale
        # doubling and check a target between the two rungs.
        points = [(1000, 90.0), (2000, 95.0)]
        # loss at 1000 = ln(10), at 2000 = ln(5); target 92.something between.
        scale, sat = cal.crossing_scale(points, 92.5)
        self.assertFalse(sat)
        self.assertTrue(1000 < scale < 2000, scale)

    def test_saturated_when_finest_misses(self):
        points = [(400, 20.0), (20000, 60.0)]
        scale, sat = cal.crossing_scale(points, 90.0)
        self.assertTrue(sat)
        self.assertLessEqual(scale, cal.SCALE_MAX)

    def test_low_target_extrapolates_below_and_clamps(self):
        # Even the coarsest rung already beats target 30 -> crossing below it,
        # extrapolated and clamped to >=1.
        points = [(400, 80.0), (800, 90.0)]
        scale, sat = cal.crossing_scale(points, 30.0)
        self.assertFalse(sat)
        self.assertGreaterEqual(scale, cal.SCALE_MIN)
        self.assertLess(scale, 400)

    def test_monotone_drops_reversals(self):
        curve = cal.monotone_curve([(400, 50.0), (800, 45.0), (1600, 70.0)])
        scores = [s for _, s in curve]
        self.assertEqual(scores, [50.0, 50.0, 70.0])


class TestBucketing(unittest.TestCase):
    def test_flat_bucket_edges(self):
        self.assertEqual(cal.flat_bucket(0.0), 0)
        self.assertEqual(cal.flat_bucket(0.19), 0)
        self.assertEqual(cal.flat_bucket(0.2), 1)
        self.assertEqual(cal.flat_bucket(0.5), 1)
        self.assertEqual(cal.flat_bucket(0.6), 2)
        self.assertEqual(cal.flat_bucket(1.0), 2)

    def test_luma_bucket_indices(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        edges = cal.luma_bucket_edges(values)
        self.assertEqual(len(edges), 4)
        self.assertTrue(all(edges[i] < edges[i + 1] for i in range(3)))
        self.assertEqual(cal.bucket_index(edges[0] - 0.01, edges), 0)
        self.assertEqual(cal.bucket_index(edges[-1] + 100.0, edges), 4)

    def test_edges_strictly_increasing_on_clustered_sample(self):
        edges = cal.luma_bucket_edges([5.0] * 10)
        self.assertTrue(all(edges[i] < edges[i + 1] for i in range(3)))

    def test_percentile_matches_endpoints(self):
        s = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(cal.percentile(s, 0.0), 1.0)
        self.assertEqual(cal.percentile(s, 1.0), 4.0)
        self.assertAlmostEqual(cal.percentile(s, 0.5), 2.5)


class TestOls(unittest.TestCase):
    def test_recovers_known_coefficients(self):
        # y = 2 + 3*x1 - 1*x2 exactly; OLS must recover [2, 3, -1].
        rows = [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            [1.0, -1.0, 4.0],
        ]
        true = [2.0, 3.0, -1.0]
        y = [sum(c * v for c, v in zip(true, r)) for r in rows]
        coef = cal.ols_fit(rows, y)
        for got, want in zip(coef, true):
            self.assertAlmostEqual(got, want, places=6)

    def test_solve_linear_identity(self):
        a = [[2.0, 1.0], [1.0, 3.0]]
        b = [3.0, 5.0]
        x = cal.solve_linear(a, b)
        # 2x+y=3, x+3y=5 -> x=0.8, y=1.4
        self.assertAlmostEqual(x[0], 0.8, places=6)
        self.assertAlmostEqual(x[1], 1.4, places=6)

    def test_fallback_predict_roundtrips_a_fit(self):
        # A perfectly linear ln(scale) relation is recovered and predicted.
        rows = []
        y = []
        for target in (50.0, 80.0, 95.0):
            for luma in (1e-4, 1e-2):
                for flat in (0.0, 0.5):
                    row = cal.fallback_design_row(target, luma, flat)
                    val = 5.0 + 0.7 * row[1] + 0.2 * row[2] + 1.1 * row[3]
                    rows.append(row)
                    y.append(val)
        coef = cal.ols_fit(rows, y)
        pred = cal.fallback_predict(coef, 80.0, 1e-3, 0.25)
        expect = math.exp(5.0 + 0.7 * math.log(20.0) + 0.2 * math.log(1e-3 + 1e-6) + 1.1 * 0.25)
        self.assertAlmostEqual(pred, cal._clamp_scale(expect), delta=2)


class TestGeomean(unittest.TestCase):
    def test_geomean(self):
        self.assertAlmostEqual(cal.geomean([1.0, 100.0]), 10.0)
        self.assertEqual(cal.geomean([]), 0.0)


if __name__ == "__main__":
    unittest.main()
