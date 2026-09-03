"""Tests for quality_corpus_extend.py: burst folding, sampling, splits."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import quality_corpus_extend as qce  # noqa: E402


def files(*names: str) -> list[tuple[str, str, int]]:
    return [(".", n, 1) for n in names]


class BurstFoldTest(unittest.TestCase):
    def test_frames_seconds_apart_share_the_first_frames_family(self) -> None:
        fam = qce.fold_bursts(
            files("20240501_125726.jpg", "20240501_125729.heic", "20240501_125735.jpg"), 120
        )
        self.assertEqual(len(set(fam.values())), 1)
        self.assertEqual(fam[(".", "20240501_125735.jpg")], qce.family_key(".", "20240501_125726.jpg"))

    def test_window_chains_and_breaks(self) -> None:
        fam = qce.fold_bursts(
            files("20240501_100000.jpg", "20240501_100100.jpg", "20240501_100200.jpg", "20240501_100500.jpg"), 90
        )
        self.assertEqual(fam[(".", "20240501_100200.jpg")], fam[(".", "20240501_100000.jpg")])
        self.assertNotEqual(fam[(".", "20240501_100500.jpg")], fam[(".", "20240501_100000.jpg")])

    def test_unstamped_and_disabled_keep_stem_families(self) -> None:
        fam = qce.fold_bursts(files("a.jpg", "20240501_100000.jpg", "20240501_100001.jpg"), 0)
        self.assertEqual(len(set(fam.values())), 3)
        self.assertEqual(fam[(".", "a.jpg")], qce.family_key(".", "a.jpg"))


class SampleTest(unittest.TestCase):
    def test_burst_yields_one_pick_and_excluded_stems_are_skipped(self) -> None:
        images = files("20240501_100000.jpg", "20240501_100005.jpg", "20240502_100000.jpg", "x.png")
        fam = qce.fold_bursts(images, 60)
        picked = qce.sample(images, 10, 10, fam, frozenset(["x"]))
        names = sorted(n for _, n in picked)
        self.assertEqual(len(names), 2)
        self.assertNotIn("x.png", names)
        self.assertIn("20240502_100000.jpg", names)

    def test_day_grouping_caps_per_day(self) -> None:
        images = files(*[f"20240503_1000{i:02d}.jpg" for i in range(0, 50, 5)] + ["20240504_120000.jpg"])
        fam = qce.fold_bursts(images, 0)
        picked = qce.sample(images, 100, 2, fam, frozenset(), group_by_stamp_date=True)
        self.assertEqual(len(picked), 3)
        self.assertIn((".", "20240504_120000.jpg"), picked)

    def test_split_is_deterministic_and_in_range(self) -> None:
        self.assertEqual(qce.split_for_family("f/a", 0.2, 0.3), qce.split_for_family("f/a", 0.2, 0.3))
        self.assertIn(qce.split_for_family("f/a", 0.2, 0.3), {"calibration", "development", "ext-holdout"})


if __name__ == "__main__":
    unittest.main()
