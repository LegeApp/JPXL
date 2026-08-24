"""Unit tests for make-quality-guard-fixtures.py.

Stdlib-only (unittest), matching test_codec_compare.py. These exercise the pure
encoders, colour math, and the fixture registry's split/family/size invariants
without materialising the large photo PPMs, so the suite stays fast.
"""

import hashlib
import importlib.util
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "make-quality-guard-fixtures.py"
SPEC = importlib.util.spec_from_file_location("mqgf", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mqgf = importlib.util.module_from_spec(SPEC)
# Register before exec so the @dataclass in the module can resolve its own
# string annotations (PEP 563) against a real sys.modules entry.
sys.modules["mqgf"] = mqgf
SPEC.loader.exec_module(mqgf)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class EncoderTests(unittest.TestCase):
    def test_ppm_p6_8bit_header_and_body(self):
        arr = np.arange(2 * 1 * 3, dtype=np.uint8).reshape(1, 2, 3)
        out = mqgf.ppm_p6(arr)
        self.assertTrue(out.startswith(b"P6\n2 1\n255\n"))
        self.assertEqual(out[len(b"P6\n2 1\n255\n"):], arr.tobytes())

    def test_ppm_p6_16bit_is_big_endian_maxval_65535(self):
        arr = np.array([[[0x0102, 0x0304, 0x0506]]], dtype=np.uint16)
        out = mqgf.ppm_p6(arr)
        self.assertTrue(out.startswith(b"P6\n1 1\n65535\n"))
        self.assertEqual(out[len(b"P6\n1 1\n65535\n"):], bytes((1, 2, 3, 4, 5, 6)))

    def test_png16_is_valid_16bit_truecolor(self):
        # Pillow truncates 16-bit RGB PNGs to 8-bit on read, so validate the
        # file structurally: IHDR must declare bit depth 16 / colour type 2, and
        # the decompressed raster (filter byte 0 + big-endian samples) must match.
        import struct
        import zlib

        arr = (np.linspace(0, 65535, 4 * 3 * 3).astype(np.uint16)).reshape(3, 4, 3)
        png = mqgf.png_encode(arr)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        # First chunk after signature is IHDR.
        length = struct.unpack(">I", png[8:12])[0]
        self.assertEqual(png[12:16], b"IHDR")
        width, height, depth, colour = struct.unpack(">IIBB", png[16:16 + 10])
        self.assertEqual((width, height, depth, colour), (4, 3, 16, 2))
        # Concatenate IDAT payloads and inflate.
        offset = 8
        idat = b""
        while offset < len(png):
            clen = struct.unpack(">I", png[offset:offset + 4])[0]
            ctype = png[offset + 4:offset + 8]
            if ctype == b"IDAT":
                idat += png[offset + 8:offset + 8 + clen]
            offset += 12 + clen
        raw = zlib.decompress(idat)
        line = width * 3 * 2
        for y in range(height):
            self.assertEqual(raw[y * (line + 1)], 0)  # filter type None
            row = raw[y * (line + 1) + 1: y * (line + 1) + 1 + line]
            self.assertEqual(row, arr[y].astype(">u2").tobytes())
        _ = length  # header length field is present and read

    def test_srgb_linear_roundtrip(self):
        x = np.linspace(0.0, 1.0, 257)
        back = mqgf.linear_to_srgb(mqgf.srgb_to_linear(x))
        self.assertTrue(np.allclose(back, x, atol=1e-6))


class BuilderDeterminismTests(unittest.TestCase):
    def test_synthetic_gradient_is_deterministic(self):
        a = mqgf.build_gradient(64, 32, "sky", seed=777)
        b = mqgf.build_gradient(64, 32, "sky", seed=777)
        self.assertEqual(sha(mqgf.ppm_p6(a)), sha(mqgf.ppm_p6(b)))

    def test_text_render_is_deterministic(self):
        a = mqgf.build_text(320, 240, seed=5, dark=False)
        b = mqgf.build_text(320, 240, seed=5, dark=False)
        self.assertEqual(sha(mqgf.ppm_p6(a)), sha(mqgf.ppm_p6(b)))

    def test_grayscale_channels_are_equal(self):
        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        g = mqgf.build_grayscale(rgb)
        self.assertTrue(np.array_equal(g[..., 0], g[..., 1]))
        self.assertTrue(np.array_equal(g[..., 1], g[..., 2]))


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = mqgf.registry()

    def test_ids_are_unique(self):
        ids = [f.fixture_id for f in self.fixtures]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_manifest_entry_has_required_fields(self):
        for fx in self.fixtures:
            # Uses the fixture's declared class/split; strata/provenance are the
            # fields load_manifest enforces.
            self.assertTrue(fx.klass)
            self.assertIn(fx.split, {"calibration", "development", "holdout"})
            self.assertTrue(fx.provenance)
            self.assertTrue(fx.license)

    def test_no_source_family_spans_two_splits(self):
        """Real-image families (parent stem) must live in exactly one split."""
        family_splits: dict[str, set] = defaultdict(set)
        for fx in self.fixtures:
            if fx.parent is None:
                continue
            stem = Path(fx.parent["path"]).stem
            # Normalise 4mp/6mp/result variants of one capture to one family key.
            for suffix in ("_4mp", "_6mp", "_result"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
            family_splits[stem].add(fx.split)
        offenders = {k: v for k, v in family_splits.items() if len(v) > 1}
        self.assertEqual(offenders, {}, f"families in multiple splits: {offenders}")

    def test_holdout_and_calibration_photo_families(self):
        splits_by_stem: dict[str, str] = {}
        for fx in self.fixtures:
            if fx.parent:
                splits_by_stem.setdefault(fx.parent["path"], fx.split)
        # 201839 capture is the locked holdout; 203230 is a calibration anchor.
        self.assertEqual(
            splits_by_stem["test-set/test-set-4mp/20260606_201839_4mp.png"], "holdout")
        self.assertEqual(
            splits_by_stem["test-set/test-set-4mp/20260606_203230_4mp.png"], "calibration")

    def test_non_tiny_fixtures_meet_min_size(self):
        for fx in self.fixtures:
            if fx.klass == "tiny":
                continue
            arr = fx.build()
            h, w = arr.shape[0], arr.shape[1]
            self.assertGreaterEqual(min(w, h), 256, f"{fx.fixture_id} is {w}x{h}")

    def test_each_synthetic_class_has_all_three_splits(self):
        by_class_split = defaultdict(set)
        for fx in self.fixtures:
            if fx.kind == "synthetic":
                by_class_split[fx.klass].add(fx.split)
        for klass in ("text-screenshot", "line-art", "gradient", "saturated"):
            self.assertEqual(
                by_class_split[klass],
                {"calibration", "development", "holdout"},
                f"{klass} missing a split: {by_class_split[klass]}",
            )

    def test_split_and_class_counts_are_stable(self):
        # 2026-08-24: the one-shot program's coverage expansion added nine
        # calibration/development fixtures (noise-lowlight and grayscale from
        # more captures; second text/UI/hatch/sky families). The locked
        # holdout is frozen at 13.
        by_split = Counter(f.split for f in self.fixtures)
        self.assertEqual(by_split["calibration"], 24)
        self.assertEqual(by_split["development"], 19)
        self.assertEqual(by_split["holdout"], 13)

    def test_families_never_cross_splits(self):
        by_family = defaultdict(set)
        for fx in self.fixtures:
            by_family[mqgf.family_fields(fx)["family_id"]].add(fx.split)
        leaking = {fam: s for fam, s in by_family.items() if len(s) > 1}
        self.assertEqual(leaking, {}, "image families must stay in one split")


if __name__ == "__main__":
    unittest.main()
