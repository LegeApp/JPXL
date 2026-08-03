//! End-to-end ICC profile decoding (18181-1 E.4 — `docs/PLAN.md` slice 4).
//!
//! # What these tests prove
//!
//! Each fixture `3N_icc_*.jxl` was produced by `cjxl -x icc_pathname=P` for a
//! profile `P` that `tools/make-icc-fixtures.sh` builds byte by byte, and the
//! reference profile is checked in beside it as `3N_icc_*.icc`. The encoder
//! therefore had to set `want_icc` (E.2) and store `P` in the compressed
//! representation of E.4, and the fixture-generation script has already
//! verified with `djxl --orig_icc_out` that `P` survives the round trip.
//!
//! So [`decodes_every_profile_byte_for_byte`] is the real gate: it needs no
//! oracle and asserts our decoder reproduces `P` exactly.
//! [`matches_djxl_orig_icc_out`] repeats the comparison against a live `djxl`
//! and skips when none is installed.
//!
//! `docs/PLAN.md` puts ICC bytes in the **bit-exact** class. There is no
//! tolerance here: one wrong byte is a failure.
//!
//! # Why the pixel assertions are here too
//!
//! Table A.1 puts the ICC profile *between* the image headers and the first
//! frame, in the same bit stream. A decoder that mis-reads its length leaves
//! the reader on the wrong bit and every frame after it is garbage — so
//! [`consuming_the_profile_leaves_the_frame_readable`] decodes the pixels of
//! the same fixtures and checks them against the source formulas. That is the
//! only test here that would catch an ICC decoder which happened to produce
//! the right bytes from the wrong number of symbols.

#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

use std::path::{Path, PathBuf};

use jpxl_core::limits::Limits;
use jpxl_decode::{decode, extract_icc_profile};

/// Every ICC fixture and the profile it must decode to.
const CASES: &[(&str, &str)] = &[
    (
        "30_icc_gray_32x32_lossless.jxl",
        "30_icc_gray_32x32_lossless.icc",
    ),
    (
        "31_icc_rgb_32x32_lossless.jxl",
        "31_icc_rgb_32x32_lossless.icc",
    ),
    (
        "32_icc_rgb_appl_32x32_lossless.jxl",
        "32_icc_rgb_appl_32x32_lossless.icc",
    ),
    (
        "33_icc_rgb_private_32x32_lossless.jxl",
        "33_icc_rgb_private_32x32_lossless.icc",
    ),
    (
        "34_icc_gray_table_32x32_lossless.jxl",
        "34_icc_gray_table_32x32_lossless.icc",
    ),
    (
        "35_icc_rgb_curves_300x200_lossless.jxl",
        "35_icc_rgb_curves_300x200_lossless.icc",
    ),
    (
        "36_icc_rgb_v4_para_32x32_lossless.jxl",
        "36_icc_rgb_v4_para_32x32_lossless.icc",
    ),
];

/// `(x, y, channel) -> sample`, transcribed from a fixture's `.txt` sidecar.
type SampleFn = fn(u32, u32, usize) -> i32;

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

/// Reports the first differing byte, which localises a broken stage far better
/// than "the vectors are not equal".
fn assert_identical(name: &str, ours: &[u8], reference: &[u8]) {
    if ours == reference {
        return;
    }
    let first = ours
        .iter()
        .zip(reference.iter())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| ours.len().min(reference.len()));
    panic!(
        "{name}: decoded ICC profile differs from the reference.\n  \
         our length {}, reference length {}\n  \
         first difference at byte {first}: ours {:?}, reference {:?}",
        ours.len(),
        reference.len(),
        ours.get(first),
        reference.get(first),
    );
}

/// The gate for this slice: every fixture's profile decodes byte for byte.
#[test]
fn decodes_every_profile_byte_for_byte() {
    for (fixture, expected) in CASES {
        let bytes = read_fixture(fixture);
        let profile = extract_icc_profile(&bytes, &Limits::default())
            .unwrap_or_else(|e| panic!("{fixture}: ICC decode failed: {e}"))
            .unwrap_or_else(|| panic!("{fixture}: want_icc was not set"));
        let reference = read_fixture(expected);
        assert_identical(fixture, &profile, &reference);
    }
}

/// Proves the decoded bytes really are an ICC profile and not merely equal to a
/// file we also produced: the size field in the header must match the length,
/// and the ICC file signature must be at byte 36. Both come out of E.4.3's
/// prediction stage, so this fails independently if that stage is wrong.
#[test]
fn every_decoded_profile_is_self_consistent() {
    for (fixture, _) in CASES {
        let bytes = read_fixture(fixture);
        let profile = extract_icc_profile(&bytes, &Limits::default())
            .expect("ICC decode")
            .expect("want_icc");
        assert!(
            profile.len() >= 132,
            "{fixture}: profile is impossibly short"
        );
        let declared = u32::from_be_bytes([profile[0], profile[1], profile[2], profile[3]]);
        assert_eq!(
            declared as usize,
            profile.len(),
            "{fixture}: header size field disagrees with the decoded length"
        );
        assert_eq!(&profile[36..40], b"acsp", "{fixture}: no 'acsp' signature");
        let num_tags =
            u32::from_be_bytes([profile[128], profile[129], profile[130], profile[131]]) as usize;
        assert!(
            132 + num_tags * 12 <= profile.len(),
            "{fixture}: the tag table does not fit in the profile"
        );
    }
}

/// Proves the profile ends up on [`jpxl_decode::DecodedImage`] and, crucially,
/// that consuming it leaves the reader on the right bit: the frame after it
/// still decodes to the source formula.
#[test]
fn consuming_the_profile_leaves_the_frame_readable() {
    // (fixture, width, height, channels, sample formula from the sidecar)
    let cases: &[(&str, u32, u32, usize, SampleFn)] = &[
        ("30_icc_gray_32x32_lossless.jxl", 32, 32, 1, |x, y, _| {
            ((x * 255 / 31 + y * 255 / 31) / 2) as i32
        }),
        (
            "34_icc_gray_table_32x32_lossless.jxl",
            32,
            32,
            1,
            |x, y, _| ((x * 255 / 31 + y * 255 / 31) / 2) as i32,
        ),
        (
            "31_icc_rgb_32x32_lossless.jxl",
            32,
            32,
            3,
            |x, y, c| match c {
                0 => (x * 255 / 31) as i32,
                1 => (y * 255 / 31) as i32,
                _ => (((x + y) * 4) & 0xFF) as i32,
            },
        ),
        (
            "35_icc_rgb_curves_300x200_lossless.jxl",
            300,
            200,
            3,
            |x, y, c| match c {
                0 => (x * 255 / 299) as i32,
                1 => (y * 255 / 199) as i32,
                _ => ((x + y) % 256) as i32,
            },
        ),
    ];

    for &(name, width, height, channels, sample) in cases {
        let bytes = read_fixture(name);
        let image = decode(&bytes, &Limits::default())
            .unwrap_or_else(|e| panic!("{name}: decode failed: {e}"));
        assert_eq!((image.width, image.height), (width, height), "{name}: size");
        assert_eq!(image.num_colour_channels, channels, "{name}: channels");
        assert!(
            image.icc_profile.is_some(),
            "{name}: decode() did not attach the profile"
        );

        for c in 0..channels {
            let plane = &image.planes[c];
            for y in 0..height {
                for x in 0..width {
                    assert_eq!(
                        plane.get(x, y),
                        sample(x, y, c),
                        "{name}: channel {c} sample ({x}, {y})"
                    );
                }
            }
        }
    }
}

/// Proves a codestream with an enumerated colour space reports no profile
/// rather than trying to decode one — the `want_icc == false` branch of E.2.
#[test]
fn a_codestream_without_a_profile_reports_none() {
    let bytes = read_fixture("09_modular_rgb_64x64_lossless.jxl");
    let profile = extract_icc_profile(&bytes, &Limits::default()).expect("headers decode");
    assert!(profile.is_none());
    let image = decode(&bytes, &Limits::default()).expect("decode");
    assert!(image.icc_profile.is_none());
}

/// Proves a truncated codestream is an error, never a panic, at every cut
/// point inside the ICC payload.
#[test]
fn truncation_inside_the_profile_never_panics() {
    let full = read_fixture("31_icc_rgb_32x32_lossless.jxl");
    for cut in 0..full.len() {
        let _ = extract_icc_profile(&full[..cut], &Limits::default());
    }
}

// ---------------------------------------------------------------------------
// Live oracle comparison
// ---------------------------------------------------------------------------

/// Re-extracts each profile with `djxl --orig_icc_out` and compares byte for
/// byte, so the checked-in references cannot drift from what the oracle says.
///
/// Skips silently when no oracle is installed, per the project rule that CI
/// without `tools/setup-oracles.sh` must still be green.
#[test]
fn matches_djxl_orig_icc_out() {
    use jpxl_conformance::oracle::{self, OracleKind};

    let Some(djxl) = oracle::find(OracleKind::Djxl) else {
        println!("skipping: no djxl oracle installed (run tools/setup-oracles.sh)");
        return;
    };

    let temp = std::env::temp_dir().join("jpxl-e2e-icc");
    std::fs::create_dir_all(&temp).expect("temp dir");

    let mut compared = 0usize;
    for (fixture, _) in CASES {
        let input = fixture_dir().join(fixture);
        let icc_out = temp.join(format!("{fixture}.icc"));
        let pixels_out = temp.join(format!("{fixture}.ppm"));
        let _ = std::fs::remove_file(&icc_out);

        // `--orig_icc_out` is djxl's own name for "the profile as stored in the
        // codestream", i.e. before any --color_space conversion.
        let status = std::process::Command::new(&djxl.path)
            .arg(&input)
            .arg(&pixels_out)
            .arg(format!("--orig_icc_out={}", icc_out.display()))
            .arg("--quiet")
            .status();
        match status {
            Ok(s) if s.success() => {}
            _ => {
                println!("skipping {fixture}: djxl declined to decode it");
                continue;
            }
        }
        let Ok(reference) = std::fs::read(&icc_out) else {
            println!("skipping {fixture}: djxl wrote no ICC profile");
            continue;
        };

        let bytes = read_fixture(fixture);
        let ours = extract_icc_profile(&bytes, &Limits::default())
            .unwrap_or_else(|e| panic!("{fixture}: our ICC decode failed: {e}"))
            .unwrap_or_else(|| panic!("{fixture}: want_icc was not set"));
        assert_identical(fixture, &ours, &reference);
        compared += 1;
    }
    println!("compared {compared} ICC fixture(s) against djxl byte for byte");
}
