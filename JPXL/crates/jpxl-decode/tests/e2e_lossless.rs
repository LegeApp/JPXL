//! End-to-end lossless modular decoding, checked against the deterministic
//! source formulas (18181-1 Annexes A, F, G, H — `docs/PLAN.md` slice 7).
//!
//! # Why the sources are recomputed rather than read
//!
//! Every fixture in `tests/fixtures/handmade/` was produced by `cjxl -d 0`
//! from a synthetic image whose formula is recorded in its `.txt` provenance
//! sidecar. Recomputing that formula here makes the assertion *bit-exact
//! against the original samples* while needing neither `djxl` nor the
//! generated sources on disk — so these tests are the real conformance gate
//! and they run in any checkout. [`matches_djxl_output`] adds the live
//! oracle comparison on top, and skips when no oracle is installed.
//!
//! # Bit-exactness
//!
//! `docs/PLAN.md` puts modular lossless in the **bit-exact** regime: the
//! arithmetic is integer throughout, so "close" is a bug. Nothing here is a
//! tolerance comparison.
//!
//! # The H.5.2 clamp
//!
//! Fixtures 05, 09 and 10 were `#[ignore]`d through slice 7 with a divergence
//! blamed on H.5.2's `max_error` walk. That diagnosis was wrong and is
//! corrected here: the walk is the clause as written, and the `true_err`
//! values fed into it were wrong because H.5.2's clamp is not the single
//! symmetric clamp the clause prints. Fixtures 20 and 21 are the minimised
//! probes that pin the two halves of the clamp down in opposite directions;
//! see [`EXPERIMENT_CLAMP_SYMMETRIC`] and
//! `docs/experiments/2026-08-03-h52-clamp-asymmetry.md`.
//!
//! [`EXPERIMENT_CLAMP_SYMMETRIC`]: jpxl_decode::modular::weighted::EXPERIMENT_CLAMP_SYMMETRIC

#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

use std::path::{Path, PathBuf};

use jpxl_core::limits::Limits;
use jpxl_decode::decode;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("manifest dir has two ancestors")
        .join("tests")
        .join("fixtures")
        .join("handmade")
}

fn read_fixture(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// A fixture plus the formula its source image was generated from.
struct Case {
    name: &'static str,
    width: u32,
    height: u32,
    channels: usize,
    bits: u32,
    /// `(x, y, channel) -> sample`, transcribed from the `.txt` sidecar.
    sample: fn(u32, u32, usize) -> i32,
}

/// Decodes `case` and asserts every sample equals its source formula.
fn check(case: &Case) {
    let bytes = read_fixture(case.name);
    let image = decode(&bytes, &Limits::default())
        .unwrap_or_else(|e| panic!("{}: decode failed: {e}", case.name));

    assert_eq!(
        (image.width, image.height),
        (case.width, case.height),
        "{}: dimensions",
        case.name
    );
    assert_eq!(
        image.num_colour_channels, case.channels,
        "{}: colour channel count",
        case.name
    );
    assert_eq!(
        image.colour_bits_per_sample(),
        case.bits,
        "{}: bit depth",
        case.name
    );

    for c in 0..case.channels {
        let plane = &image.planes[c];
        for y in 0..case.height {
            for x in 0..case.width {
                let got = plane.get(x, y);
                let want = (case.sample)(x, y, c);
                assert_eq!(
                    got, want,
                    "{}: channel {c} sample ({x}, {y}) is {got}, source says {want}",
                    case.name
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Source formulas, transcribed from the `.txt` provenance sidecars.
// ---------------------------------------------------------------------------

/// The RGB gradient shared by fixtures 03, 05, 09, 11 and 13:
/// `R = x * 255 / (w-1)`, `G = y * 255 / (h-1)`, `B = (x + y) mod 256`.
const fn rgb_gradient(x: u32, y: u32, c: usize, w: u32, h: u32) -> i32 {
    match c {
        0 => ((x as i64 * 255) / (w as i64 - 1)) as i32,
        1 => ((y as i64 * 255) / (h as i64 - 1)) as i32,
        _ => ((x + y) % 256) as i32,
    }
}

/// The diagonal grey ramp of fixtures 07, 08 and 12: `(x + y) * max / span`.
const fn grey_ramp(x: u32, y: u32, max: i64, span: i64) -> i32 {
    (((x as i64 + y as i64) * max) / span) as i32
}

// ---------------------------------------------------------------------------
// Bit-exact fixtures
// ---------------------------------------------------------------------------

#[test]
fn gradient_8x8_rgb() {
    check(&Case {
        name: "03_gradient_8x8_lossless.jxl",
        width: 8,
        height: 8,
        channels: 3,
        bits: 8,
        sample: |x, y, c| rgb_gradient(x, y, c, 8, 8),
    });
}

/// The minimal modular stream: one 8x8 grey channel, `-e 1`.
///
/// This is the fixture that proved the G.1.3 pipeline end to end, and the one
/// whose 15-colour palette (the ramp `(x+y)*255/14` has exactly 15 distinct
/// values) confirmed the `ModularHeader` parse independently of the samples.
#[test]
fn modular_grey_8x8() {
    check(&Case {
        name: "07_modular_gray_8x8_lossless.jxl",
        width: 8,
        height: 8,
        channels: 1,
        bits: 8,
        sample: |x, y, _| grey_ramp(x, y, 255, 14),
    });
}

/// 16-bit samples, and the only container-wrapped fixture — so this also
/// covers `container::extract_codestream`.
#[test]
fn modular_grey16_32x32_from_a_container() {
    let bytes = read_fixture("08_modular_gray16_32x32_lossless.jxl");
    assert_eq!(
        &bytes[..4],
        &[0x00, 0x00, 0x00, 0x0C],
        "fixture 08 should still be a container"
    );
    check(&Case {
        name: "08_modular_gray16_32x32_lossless.jxl",
        width: 32,
        height: 32,
        channels: 1,
        bits: 16,
        sample: |x, y, _| grey_ramp(x, y, 65535, 62),
    });
}

/// 256x256 at `-e 7`: exactly one group at the default group size, and the
/// largest fixture that decodes bit-exactly (196 608 samples).
#[test]
fn modular_gradient_256x256() {
    check(&Case {
        name: "11_modular_gradient_256x256_lossless.jxl",
        width: 256,
        height: 256,
        channels: 3,
        bits: 8,
        sample: |x, y, c| rgb_gradient(x, y, c, 256, 256),
    });
}

/// 300x200 at `-e 7`. The encoder chose `group_size_shift = 2`, i.e.
/// `group_dim = 512`, so this is a single group despite the dimensions — the
/// partial-group paths of G.2.3/G.4.2 are therefore *not* exercised by it.
#[test]
fn modular_grey_300x200() {
    check(&Case {
        name: "12_modular_gray_300x200_lossless.jxl",
        width: 300,
        height: 200,
        channels: 1,
        bits: 8,
        sample: |x, y, _| grey_ramp(x, y, 255, 498),
    });
}

#[test]
fn modular_rgb_16x16() {
    check(&Case {
        name: "13_modular_rgb_16x16_lossless.jxl",
        width: 16,
        height: 16,
        channels: 3,
        bits: 8,
        sample: |x, y, c| rgb_gradient(x, y, c, 16, 16),
    });
}

// ---------------------------------------------------------------------------
// The fixtures the H.5.2 clamp correction unblocked (slice 8)
// ---------------------------------------------------------------------------

/// 128x128, four flat quadrants: a `Palette(num_c 3, nb_colours 4)` whose
/// 4x3 meta-channel is where the slice-7 divergence began.
///
/// What it proves: the MA tree here is a single branch on property 15
/// (`max_error > -255`), so every one of the meta-channel's twelve samples is
/// a direct test of the H.5.1 `max_error` output — and, through `true_err`,
/// of the H.5.2 clamp. Slice 7 read the divergence as a broken `max_error`
/// walk; it is really the clamp, and at `(2, 0)` of the meta-channel the
/// literal symmetric clamp turns `true_err = -2242` into `-1896`, which is
/// exactly the amount needed to flip the property-15 comparison at `(1, 1)`.
#[test]
fn modular_palette_128x128() {
    // Four 64x64 blocks: white, blue / red, black — see the `.txt` sidecar.
    check(&Case {
        name: "10_modular_palette_128x128_lossless.jxl",
        width: 128,
        height: 128,
        channels: 3,
        bits: 8,
        sample: |x, y, c| {
            let rgb = match (x < 64, y < 64) {
                (true, true) => [0, 0, 0],
                (false, true) => [255, 255, 255],
                (true, false) => [237, 28, 36],
                (false, false) => [0, 114, 255],
            };
            rgb[c]
        },
    });
}

/// 64x64 RGB at `-e 3`: `Palette`, `Palette`, `RCT(type 6)`.
///
/// What it proves: the most property-15-sensitive fixture in the set — 34
/// leaves, and every one of the 33 decision nodes tests property 15, with
/// thresholds 0, ±3, ±7, ±31, 47, 95, 191, 392, … A single wrong `true_err`
/// anywhere in 12 416 samples desynchronises it, so this is the fixture that
/// says the corrected clamp is right sample after sample rather than on
/// average.
#[test]
fn modular_rgb_64x64() {
    check(&Case {
        name: "09_modular_rgb_64x64_lossless.jxl",
        width: 64,
        height: 64,
        channels: 3,
        bits: 8,
        sample: |x, y, c| rgb_gradient(x, y, c, 64, 64),
    });
}

/// 300x200 RGB at `-e 7`: `Palette(200 colours)` then `RCT(type 10)`.
///
/// What it proves: the largest fixture in the set (180 200 coded samples over
/// four channels) and the one whose divergence survived every other H.5
/// reading. Fixture 21 is its 260x10 minimisation.
#[test]
fn gradient_300x200_rgb() {
    check(&Case {
        name: "05_gradient_300x200_lossless.jxl",
        width: 300,
        height: 200,
        channels: 3,
        bits: 8,
        sample: |x, y, c| rgb_gradient(x, y, c, 300, 200),
    });
}

// ---------------------------------------------------------------------------
// H.5.2 clamp probes (slice 8) — see docs/experiments/
// ---------------------------------------------------------------------------

/// 24x24, six flat colour bands: `Palette(num_c 3, nb_colours 6)`.
///
/// What it proves: the 6x3 meta-channel decodes with the `SelfCorrecting`
/// predictor at *every* sample and its MA tree branches only on property 0
/// (the channel index), so the decoded palette is a direct readout of the
/// weighted prediction — nothing else can absorb an error. At `(3, 0)` the
/// three neighbouring errors are `(W +864, N 0, NW 0)` and
/// `W3 = N3 = NE3 = 576`, so H.5.2's printed symmetric clamp would force the
/// prediction to 576; the stream requires the unclamped 400. That is the
/// counter-example to the lower half of the clamp.
#[test]
fn modular_palette_bands_24x24() {
    // Colour i is (36i, 77i, 151i) mod 256, in 4-pixel bands — see the sidecar.
    check(&Case {
        name: "20_modular_palette_bands_24x24_lossless.jxl",
        width: 24,
        height: 24,
        channels: 3,
        bits: 8,
        sample: |x, _y, c| {
            let i = ((x / 4) % 6) as i32;
            match c {
                0 => (36 * i) % 256,
                1 => (77 * i) % 256,
                _ => (151 * i) % 256,
            }
        },
    });
}

/// 260x10, the minimisation of fixture 05: same `Palette` + `RCT(type 10)`
/// chain, 7 810 coded samples instead of 180 200.
///
/// What it proves: the opposite half of the clamp from fixture 20. At coded
/// channel 2 `(251, 5)` — where the diagonal sawtooth wraps — every
/// neighbouring error is zero, `min(W3, N3, NE3) = -32` and the unclamped
/// prediction is `-40`; the stream requires `-32`, so the lower bound *is*
/// applied there. Fixtures 20 and 21 cannot both be satisfied by one guard in
/// front of one symmetric clamp.
#[test]
fn gradient_260x10_rgb() {
    check(&Case {
        name: "21_gradient_260x10_lossless.jxl",
        width: 260,
        height: 10,
        channels: 3,
        bits: 8,
        sample: |x, y, c| rgb_gradient(x, y, c, 260, 10),
    });
}

// ---------------------------------------------------------------------------
// Live oracle comparison
// ---------------------------------------------------------------------------

/// Decodes every bit-exact fixture with `djxl` and compares sample for sample.
///
/// Skips silently when no oracle is installed, per the project rule that CI
/// must be green on a machine with no libjxl. Install with
/// `tools/setup-oracles.sh`.
#[test]
fn matches_djxl_output() {
    use jpxl_conformance::{Image, OracleKind, oracle};

    let Some(djxl) = oracle::find(OracleKind::Djxl) else {
        println!("skipping: no djxl oracle installed (run tools/setup-oracles.sh)");
        return;
    };

    let temp = std::env::temp_dir().join("jpxl-e2e-oracle");
    std::fs::create_dir_all(&temp).expect("temp dir");

    let mut compared = 0usize;
    for name in BIT_EXACT_FIXTURES {
        let input = fixture_dir().join(name);
        let output = temp.join(format!("{name}.ppm"));
        if djxl.decode_to_ppm(&input, &output).is_err() {
            println!("skipping {name}: djxl declined to decode it");
            continue;
        }
        let reference = Image::from_ppm(&std::fs::read(&output).expect("djxl output"))
            .unwrap_or_else(|e| panic!("{name}: parsing djxl PPM: {e:?}"));

        let bytes = read_fixture(name);
        let ours = decode(&bytes, &Limits::default())
            .unwrap_or_else(|e| panic!("{name}: our decode failed: {e}"));

        assert_eq!(
            (reference.w, reference.h),
            (ours.width, ours.height),
            "{name}: dimensions differ from djxl"
        );
        // djxl always writes a PPM, so a greyscale image comes back as three
        // identical channels. Fold that away rather than weakening the check.
        let ours_samples = ours.interleaved_colour();
        let reference_samples: Vec<u16> =
            if reference.channels == 3 && ours.num_colour_channels == 1 {
                for triple in reference.samples.chunks_exact(3) {
                    assert!(
                        triple[0] == triple[1] && triple[1] == triple[2],
                        "{name}: djxl produced a non-grey pixel for a greyscale image"
                    );
                }
                reference.samples.iter().step_by(3).copied().collect()
            } else {
                reference.samples.clone()
            };
        assert_eq!(
            reference_samples.len(),
            ours_samples.len(),
            "{name}: sample count differs from djxl"
        );
        for (i, (&r, &o)) in reference_samples
            .iter()
            .zip(ours_samples.iter())
            .enumerate()
        {
            assert_eq!(o, r, "{name}: sample {i} is {o}, djxl says {r}");
        }
        compared += 1;
    }
    println!("compared {compared} fixture(s) against djxl byte for byte");
}

/// Every fixture the lossless modular path decodes bit-exactly.
const BIT_EXACT_FIXTURES: &[&str] = &[
    "03_gradient_8x8_lossless.jxl",
    "05_gradient_300x200_lossless.jxl",
    "07_modular_gray_8x8_lossless.jxl",
    "08_modular_gray16_32x32_lossless.jxl",
    "09_modular_rgb_64x64_lossless.jxl",
    "10_modular_palette_128x128_lossless.jxl",
    "11_modular_gradient_256x256_lossless.jxl",
    "12_modular_gray_300x200_lossless.jxl",
    "13_modular_rgb_16x16_lossless.jxl",
    "20_modular_palette_bands_24x24_lossless.jxl",
    "21_gradient_260x10_lossless.jxl",
];

// ---------------------------------------------------------------------------
// Rejection of what this slice does not implement
// ---------------------------------------------------------------------------

/// A VarDCT frame must be a typed `Unsupported`, never wrong pixels.
#[test]
fn lossy_fixtures_report_vardct_as_unsupported() {
    for name in ["04_gradient_8x8_lossy.jxl", "06_gradient_300x200_lossy.jxl"] {
        let bytes = read_fixture(name);
        let err = decode(&bytes, &Limits::default())
            .err()
            .unwrap_or_else(|| panic!("{name}: a VarDCT frame must not decode"));
        let text = err.to_string();
        assert!(
            text.contains("Annex I") || text.contains("L.2"),
            "{name}: {text}"
        );
    }
}

#[test]
fn a_truncated_fixture_errors_rather_than_panicking() {
    for name in BIT_EXACT_FIXTURES {
        let bytes = read_fixture(name);
        // Every 7th prefix, so the sweep stays quick on the larger fixtures.
        for cut in (0..bytes.len()).step_by(7) {
            let _ = decode(&bytes[..cut], &Limits::default());
        }
    }
}

#[test]
fn a_tiny_allocation_budget_is_refused_not_ignored() {
    let bytes = read_fixture("11_modular_gradient_256x256_lossless.jxl");
    let limits = Limits {
        max_alloc_bytes: 4096,
        ..Limits::default()
    };
    let err = decode(&bytes, &limits).expect_err("196 608 samples do not fit in 4 KiB");
    assert!(err.to_string().contains("limit exceeded"), "{err}");
}
