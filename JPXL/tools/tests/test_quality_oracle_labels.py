"""Unit tests for the pure helpers of quality_oracle_labels.py."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import quality_oracle_labels as qol


class GridTest(unittest.TestCase):
    def test_geometric_grid_includes_endpoints_strictly_increasing(self):
        grid = qol.geometric_grid(1, 73728, 1.6)
        self.assertEqual(grid[0], 1)
        self.assertEqual(grid[-1], 73728)
        self.assertTrue(all(a < b for a, b in zip(grid, grid[1:])))

    def test_geometric_grid_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            qol.geometric_grid(10, 5, 1.6)
        with self.assertRaises(ValueError):
            qol.geometric_grid(1, 10, 1.0)


class CrossingTest(unittest.TestCase):
    POINTS = [(100, 20.0), (200, 40.0), (400, 60.0), (800, 80.0), (1600, 95.0)]

    def test_crossed_picks_the_coarsest_meeting_point(self):
        state = qol.crossing_state(self.POINTS, 70.0)
        self.assertEqual(state["state"], "crossed")
        self.assertEqual(state["below"], (400, 60.0))
        self.assertEqual(state["above"], (800, 80.0))

    def test_local_reversal_cannot_hide_a_coarser_feasible_point(self):
        # 400 meets 55 but 800 dips below it again: the label is still 400.
        points = [(100, 20.0), (400, 60.0), (800, 50.0), (1600, 95.0)]
        state = qol.crossing_state(points, 55.0)
        self.assertEqual(state["above"], (400, 60.0))

    def test_censored_when_nothing_meets(self):
        self.assertEqual(qol.crossing_state(self.POINTS, 99.0)["state"], "censored")

    def test_floor_when_the_first_point_meets(self):
        state = qol.crossing_state(self.POINTS, 10.0)
        self.assertEqual(state["state"], "floor")
        self.assertEqual(state["above"], (100, 20.0))


class RefineTest(unittest.TestCase):
    def test_refine_scales_are_strictly_inside_and_geometric(self):
        inner = qol.refine_scales(1000, 8000, 3)
        self.assertTrue(all(1000 < s < 8000 for s in inner))
        self.assertEqual(inner, sorted(set(inner)))
        # Geometric spacing: successive ratios are near-equal.
        ratios = [b / a for a, b in zip([1000] + inner, inner + [8000])]
        self.assertLess(max(ratios) / min(ratios), 1.05)

    def test_refine_scales_empty_for_adjacent_bracket(self):
        self.assertEqual(qol.refine_scales(1000, 1001, 3), [])


class InterpTest(unittest.TestCase):
    def test_interp_matches_exact_power_law(self):
        # loss = 60 * (scale/1000)^-0.8, the curve the encoder tests use.
        def score(scale: float) -> float:
            return 100.0 - 60.0 * (scale / 1000.0) ** -0.8

        target = 85.0
        exact = 1000.0 * (60.0 / (100.0 - target)) ** (1.0 / 0.8)
        below = (2000, score(2000))
        above = (40000, score(40000))
        got = qol.interp_crossing(below, above, target)
        self.assertLess(abs(math.log(got / exact)), 1e-6)

    def test_interp_falls_back_to_midpoint_on_unordered_loss(self):
        got = qol.interp_crossing((1000, 90.0), (2000, 80.0), 85.0)
        self.assertLess(abs(math.log(got / math.sqrt(1000 * 2000))), 1e-9)


class BetaTest(unittest.TestCase):
    def test_beta_recovers_the_power_law_exponent(self):
        points = [
            (s, 100.0 - 60.0 * (s / 1000.0) ** -0.8)
            for s in (2000, 4000, 8000, 16000, 32000)
        ]
        beta = qol.local_beta(points, 8000.0)
        self.assertIsNotNone(beta)
        self.assertLess(abs(beta - 0.8), 1e-6)

    def test_beta_is_none_for_flat_or_rising_loss(self):
        self.assertIsNone(qol.local_beta([(1000, 50.0), (2000, 40.0)], 1500.0))
        self.assertIsNone(qol.local_beta([(1000, 50.0)], 1000.0))


class MergeTest(unittest.TestCase):
    def test_merge_prefers_priced_records_and_sorts_by_rung(self):
        first = [{"rung": 10, "effective_scale": 11, "score": 50.0, "bytes": None}]
        second = [
            {"rung": 10, "effective_scale": 11, "score": 50.0, "bytes": 123},
            {"rung": 5, "effective_scale": 6, "score": 30.0, "bytes": None},
        ]
        merged = qol.merge_points([first, second])
        self.assertEqual([r["rung"] for r in merged], [5, 10])
        self.assertEqual(merged[1]["bytes"], 123)

    def test_merge_never_downgrades_a_priced_record(self):
        first = [{"rung": 10, "effective_scale": 11, "score": 50.0, "bytes": 123}]
        second = [{"rung": 10, "effective_scale": 11, "score": 50.0, "bytes": None}]
        merged = qol.merge_points([first, second])
        self.assertEqual(merged[0]["bytes"], 123)


class LabelsTest(unittest.TestCase):
    RAW = {
        "schema": qol.RAW_SCHEMA,
        "targets": [50.0, 99.0],
        "image": {
            "id": "img",
            "family_id": "fam",
            "split": "calibration",
            "class": "photo",
            "variant_id": "img",
            "source_capture_id": None,
        },
        "points": [
            {"rung": 99, "effective_scale": 100, "score": 20.0, "bytes": None},
            {"rung": 399, "effective_scale": 400, "score": 45.0, "bytes": 900},
            {"rung": 799, "effective_scale": 800, "score": 60.0, "bytes": 1400},
            {"rung": 1599, "effective_scale": 1600, "score": 95.0, "bytes": None},
        ],
    }

    def test_crossed_and_censored_rows(self):
        rows = qol.labels_for_raw(self.RAW)
        crossed = rows[0]
        self.assertEqual(crossed["state"], "crossed")
        self.assertEqual(crossed["label_rung"], 799)
        self.assertEqual(crossed["label_bytes"], 1400)
        self.assertTrue(400 < crossed["crossing_scale"] < 800)
        self.assertEqual(
            [n["bytes"] for n in crossed["neighbor_bytes"]], [900, 1400]
        )
        censored = rows[1]
        self.assertEqual(censored["state"], "censored")
        self.assertIsNone(censored["label_rung"])
        self.assertEqual(censored["top_score"], 95.0)


if __name__ == "__main__":
    unittest.main()
