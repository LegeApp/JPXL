"""Unit tests for the pure fitting helpers of quality_predictor_v2.py."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import quality_predictor_v2 as qpv2


class SolveTest(unittest.TestCase):
    def test_solve_linear_recovers_exact_solution(self):
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        rhs = [5.0, 10.0]
        x = qpv2.solve_linear(matrix, rhs)
        self.assertAlmostEqual(x[0], 1.0, places=9)
        self.assertAlmostEqual(x[1], 3.0, places=9)


class OlsTest(unittest.TestCase):
    def test_ols_recovers_a_noiseless_line(self):
        xs = [[float(i)] for i in range(10)]
        ys = [3.0 + 2.0 * i for i in range(10)]
        w = qpv2.fit_ols(xs, ys, [1.0] * 10, ridge=0.0)
        self.assertAlmostEqual(w[0], 3.0, places=6)
        self.assertAlmostEqual(w[1], 2.0, places=6)


class QuantileTest(unittest.TestCase):
    def test_median_fit_splits_asymmetric_noise(self):
        # y = 5 + x with one gross positive outlier; the median fit should
        # stay near the line while OLS is dragged upward.
        xs = [[float(i)] for i in range(20)]
        ys = [5.0 + i for i in range(20)]
        ys[10] += 50.0
        w = qpv2.fit_quantile(xs, ys, [1.0] * 20, tau=0.5)
        predicted = qpv2.predict(w, [10.0])
        self.assertLess(abs(predicted - 15.0), 1.0)

    def test_higher_tau_sits_above_lower_tau(self):
        # Constant-feature rows with spread: quantiles must order.
        xs = [[0.0]] * 50
        ys = [float(i) for i in range(50)]
        w50 = qpv2.fit_quantile(xs, ys, [1.0] * 50, tau=0.5)
        w90 = qpv2.fit_quantile(xs, ys, [1.0] * 50, tau=0.9)
        self.assertGreater(qpv2.predict(w90, [0.0]), qpv2.predict(w50, [0.0]) + 5.0)


class LogisticTest(unittest.TestCase):
    def test_degenerate_single_class_saturates(self):
        w = qpv2.fit_logistic([[0.0]] * 5, [0.0] * 5, [1.0] * 5)
        self.assertLess(w[0], -5.0)
        w = qpv2.fit_logistic([[0.0]] * 5, [1.0] * 5, [1.0] * 5)
        self.assertGreater(w[0], 5.0)

    def test_separable_classes_order_by_feature(self):
        xs = [[-1.0]] * 10 + [[1.0]] * 10
        ys = [0.0] * 10 + [1.0] * 10
        w = qpv2.fit_logistic(xs, ys, [1.0] * 20)
        self.assertGreater(w[1], 0.5)


class StandardizerTest(unittest.TestCase):
    def test_median_and_mad(self):
        rows = [[float(v)] for v in (1, 2, 3, 4, 100)]
        centers, scales = qpv2.robust_standardizer(rows)
        self.assertEqual(centers[0], 3.0)
        # MAD of [2,1,0,1,97] sorted -> 1; scale = 1.4826.
        self.assertAlmostEqual(scales[0], 1.4826, places=4)


class BytesTest(unittest.TestCase):
    ROW = {
        "label_bytes": 1000,
        "neighbor_bytes": [
            {"effective_scale": 500, "bytes": 800},
            {"effective_scale": 1000, "bytes": 1000},
            {"effective_scale": 2000, "bytes": 1300},
        ],
    }

    def test_interpolates_between_priced_points(self):
        ratio = qpv2.simulate_bytes_ratio(self.ROW, 1000.0)
        self.assertAlmostEqual(ratio, 1.0, places=6)
        finer = qpv2.simulate_bytes_ratio(self.ROW, 2000.0)
        self.assertAlmostEqual(finer, 1.3, places=6)

    def test_none_without_enough_pricing(self):
        row = {"label_bytes": 1000, "neighbor_bytes": [{"effective_scale": 1000, "bytes": 1000}]}
        self.assertIsNone(qpv2.simulate_bytes_ratio(row, 1000.0))


class FamilyWeightTest(unittest.TestCase):
    def test_each_family_carries_equal_total_weight(self):
        rows = [
            {"family_id": "a"},
            {"family_id": "a"},
            {"family_id": "b"},
        ]
        weights = qpv2.family_weights(rows)
        self.assertAlmostEqual(weights[0] + weights[1], 1.0)
        self.assertAlmostEqual(weights[2], 1.0)


class GateTest(unittest.TestCase):
    def test_gate_verdicts(self):
        metrics = {
            "median_abs_ln_error": 0.08,
            "p90_abs_ln_error": 0.25,
            "p99_abs_ln_error": 0.5,
            "first_plan_success": 0.9,
            "simulated_byte_geomean": 1.01,
        }
        verdicts = qpv2.gate_verdicts(metrics)
        self.assertTrue(verdicts["quick_screen"]["passed"])
        self.assertTrue(verdicts["production"]["passed"])
        metrics["p90_abs_ln_error"] = 0.5
        verdicts = qpv2.gate_verdicts(metrics)
        self.assertFalse(verdicts["quick_screen"]["passed"])
        self.assertFalse(verdicts["production"]["passed"])


class EndToEndTest(unittest.TestCase):
    def synthetic_rows(self) -> list[dict]:
        # Crossing law: ln(scale) = 6 + 1.5*ln(loss ratio) + 0.8*ln(q50) style —
        # exactly linear in the trainer's feature space, so LOFO must nail it.
        rows = []
        for fam in range(8):
            q50 = 10.0 ** (-6 + fam * 0.4)
            flat = 0.1 * fam / 8.0
            sf = {
                "width": 640,
                "height": 480,
                "grayscale": False,
                "luma_variance_q10": q50 * 0.5,
                "luma_variance_q50": q50,
                "luma_variance_q90": q50 * 2.0,
                "chroma_variance_q50": q50 * 0.7,
                "flat_fraction": flat,
                "edge_proxy": q50,
            }
            for t_idx, target in enumerate(qpv2.TARGET_KNOTS):
                # Linear in ln(loss) — the pooled model's family — plus a
                # deterministic residual spread so the quantile fits have
                # something to separate on.
                y = (
                    6.0
                    - 0.35 * math.log(qpv2.EPS + q50)
                    - 0.9 * (math.log(100.0 - target) - qpv2.X_T_CENTER)
                    + 0.15 * math.sin(2.7 * (fam * 7 + t_idx))
                )
                scale = math.exp(y)
                rows.append(
                    {
                        "image_id": f"img{fam}",
                        "family_id": f"fam{fam}",
                        "split": "calibration",
                        "class": "photo",
                        "target": target,
                        "state": "crossed",
                        "crossing_scale": scale,
                        "label_rung": int(scale),
                        "beta": 0.8,
                        "label_bytes": 1000,
                        "neighbor_bytes": [
                            {"effective_scale": scale / 1.3, "bytes": 900},
                            {"effective_scale": scale, "bytes": 1000},
                            {"effective_scale": scale * 1.3, "bytes": 1100},
                        ],
                        "source_features": sf,
                    }
                )
        return rows

    def test_lofo_recovers_a_linear_law(self):
        rows = self.synthetic_rows()
        metrics = qpv2.evaluate_lofo(rows, "source")
        self.assertIsNotNone(metrics["median_abs_ln_error"])
        # The law is representable; the deterministic +-0.15 spread bounds
        # the median error from above, and the tau=0.9 candidate should
        # clear most of that spread.
        self.assertLess(metrics["median_abs_ln_error"], 0.2)
        self.assertGreaterEqual(metrics["first_plan_success"], 0.75)

    def test_candidate_quantile_sits_above_the_median(self):
        model = qpv2.train_all(self.synthetic_rows(), "source")
        for target, knot in model["knots"].items():
            self.assertGreater(
                knot["candidate"][0],
                knot["median"][0],
                f"tau=0.9 must sit above tau=0.5 at target {target}",
            )


if __name__ == "__main__":
    unittest.main()
