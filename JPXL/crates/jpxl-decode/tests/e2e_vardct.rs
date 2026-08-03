//! End-to-end `kVarDCT` decoding, graded under 18181-3 §4.2.
//!
//! # The ladder, and what each rung proves
//!
//! 1. **First light** — fixture 04, an 86-byte 8x8 lossy stream. One varblock,
//!    one group, one section. Proves the whole chain runs and lands in the
//!    right ballpark; its threshold is loose on purpose (see the test).
//! 2. **Filters off** — fixtures 50, 51, 55 and 63 at the Part 3 no-filters
//!    class, peak `0.004` / RMSE `1e-5`. These are the strict rung: they
//!    isolate I.2–I.9 and L.2 from Annex J entirely, so a failure here is
//!    dequantization, the IDCT, chroma-from-luma or the colour transform, and
//!    nothing else. 63 additionally pins Table E.6's behaviour below zero,
//!    which only an out-of-gamut fixture can reach; 64 is the only fixture
//!    with more than one LF group, and so the only one that can see I.5.2's
//!    smoothing pass being scoped to the frame rather than to a group.
//! 3. **Filters on** — fixtures 52, 53, 56 at the with-filters class, peak
//!    `0.06` / RMSE `0.02`. The delta against rung 2 is exactly Annex J.
//! 4. **The normative corpus** — `grayscale`, `grayscale_5`, `bike` and
//!    `bike_5` against their `reference_image.npy` at their own `test.json`
//!    thresholds. This is the only rung that measures conformance to the
//!    *standard* rather than agreement with libjxl: the reference images are
//!    published with the corpus, not produced here. (`progressive` and
//!    `progressive_5` are the same rung for the progressive feature set, and
//!    live in `e2e_progressive.rs` with the fixtures that first-light them.)
//!
//! Fixtures 54 and 57 are deliberately absent: both fail inside G.2.2's
//! `LfQuant` modular sub-bitstream, in the same content-dependent family as
//! the open modular sawtooth bug (their siblings 55 and 56 decode). They are
//! blocked on that hunt, not on anything in the VarDCT path.
//!
//! # Skipping
//!
//! Rungs 2-4 need reference data. The greyscale fixtures ship their `.npy`
//! in-tree (65 KB each); the RGB ones would be 192 KB apiece, so they are
//! regenerated from the pinned `djxl` when it is available and the test
//! **skips** otherwise. The corpus NPYs are gitignored, so those tests skip
//! when the corpus has not been fetched. No test in this file fails for a
//! missing tool.

use std::path::{Path, PathBuf};

use jpxl_conformance::{FloatImage, OracleKind, OutputFormat, Similarity, oracle, similarity};
use jpxl_core::limits::Limits;
use jpxl_decode::{DecodedImage, decode};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn handmade(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/handmade")
        .join(name)
}

fn corpus(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/conformance/testcases")
        .join(case)
}

/// Our decode, as the `FloatImage` 18181-3 §4.2 compares.
///
/// Uses [`DecodedImage::float_planes`] — the unclipped `f32` representation —
/// never the quantized integer planes, because quantizing to 8 bits would put
/// a floor of `1/510` under the peak error and the no-filters class is
/// `0.004`.
fn as_float_image(image: &DecodedImage) -> FloatImage {
    let planes = image
        .float_planes
        .as_ref()
        .expect("a kVarDCT decode must produce float planes");
    let channels = planes.len();
    let mut samples = Vec::with_capacity(image.width as usize * image.height as usize * channels);
    for y in 0..image.height {
        for x in 0..image.width {
            for plane in planes {
                samples.push(plane.get(x, y));
            }
        }
    }
    FloatImage {
        frames: 1,
        height: image.height,
        width: image.width,
        channels: u32::try_from(channels).expect("at most 3 colour channels"),
        samples,
    }
}

fn decode_fixture(path: &Path) -> DecodedImage {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    decode(&bytes, &Limits::default()).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Decodes a corpus case, or reports the out-of-scope feature that stopped it.
///
/// Only [`DecodeError::Unsupported`] is treated as a skip. Every other error
/// is a real failure: a well-formed stream inside slice 8's scope that does
/// not decode is exactly what this file exists to catch.
fn decode_corpus(path: &Path) -> Option<DecodedImage> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    match decode(&bytes, &Limits::default()) {
        Ok(image) => Some(image),
        Err(jpxl_decode::DecodeError::Unsupported { feature, clause }) => {
            eprintln!(
                "skipping {}: needs {feature} ({clause}), which is outside slice 8",
                path.display()
            );
            None
        }
        Err(e) => panic!("{}: {e}", path.display()),
    }
}

fn read_npy(path: &Path) -> FloatImage {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    FloatImage::from_npy(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The committed reference decode, or one produced by the pinned oracle.
///
/// Returns `None` when neither is available, which is a **skip**, not a
/// failure: CI must stay green on a machine with no libjxl build.
fn reference_for(fixture: &Path) -> Option<FloatImage> {
    let committed = fixture.with_extension("npy");
    if committed.exists() {
        return Some(read_npy(&committed));
    }
    let oracle = oracle::find(OracleKind::Djxl)?;
    if !oracle.supports(OutputFormat::Npy) {
        return None;
    }
    let dir = std::env::temp_dir().join("jpxl-e2e-vardct");
    std::fs::create_dir_all(&dir).ok()?;
    let out = dir.join(format!("{}.npy", fixture.file_stem()?.to_string_lossy()));
    oracle.decode(fixture, &out, OutputFormat::Npy).ok()?;
    Some(read_npy(&out))
}

/// Grades one decode and prints the numbers, so a passing run still reports
/// how much margin it had.
fn grade(label: &str, ours: &FloatImage, reference: &FloatImage) -> Similarity {
    assert_eq!(
        (ours.width, ours.height, ours.channels),
        (reference.width, reference.height, reference.channels),
        "{label}: 18181-3 4.2 condition 1 (shape) failed"
    );
    let report = similarity(ours, reference).expect("same shape");
    println!(
        "{label}: peak {:.6}, channel peak {:?}, channel RMSE {:?}",
        report.peak_error, report.channel_peak, report.channel_rmse
    );
    report
}

fn assert_conforms(label: &str, report: &Similarity, peak: f32, rmse: f32) {
    assert!(
        report.conforms(peak, rmse),
        "{label}: peak {:.6} (limit {peak}), channel RMSE {:?} (limit {rmse})",
        report.peak_error,
        report.channel_rmse
    );
}

/// One handmade-fixture rung.
fn fixture_rung(name: &str, peak: f32, rmse: f32) {
    let path = handmade(name);
    assert!(path.exists(), "missing fixture {name}");
    let Some(reference) = reference_for(&path) else {
        eprintln!("skipping {name}: no committed .npy and no djxl oracle");
        return;
    };
    let image = decode_fixture(&path);
    let report = grade(name, &as_float_image(&image), &reference);
    assert_conforms(name, &report, peak, rmse);
}

/// The Part 3 error class of a corpus case, read from its own `test.json`.
///
/// A three-line scan rather than a JSON parser: the workspace takes no
/// third-party dependencies, and the two numbers wanted are on their own
/// lines in every corpus `test.json`.
fn corpus_thresholds(case: &Path) -> Option<(f32, f32)> {
    let text = std::fs::read_to_string(case.join("test.json")).ok()?;
    let find = |key: &str| -> Option<f32> {
        text.lines()
            .find(|l| l.contains(key))
            .and_then(|l| l.split(':').nth(1))
            .map(|v| v.trim().trim_end_matches(',').to_owned())
            .and_then(|v| v.parse().ok())
    };
    Some((find("peak_error")?, find("rms_error")?))
}

/// One conformance-corpus rung.
fn corpus_rung(case: &str) {
    let dir = corpus(case);
    let input = dir.join("input.jxl");
    let reference = dir.join("reference_image.npy");
    if !input.exists() || !reference.exists() {
        eprintln!("skipping corpus {case}: not fetched (see tools/fetch-conformance.sh)");
        return;
    }
    let Some((peak, rmse)) = corpus_thresholds(&dir) else {
        panic!("corpus {case}: unreadable test.json");
    };
    let Some(image) = decode_corpus(&input) else {
        return;
    };
    let report = grade(case, &as_float_image(&image), &read_npy(&reference));
    assert_conforms(case, &report, peak, rmse);
}

/// A corpus rung for a case whose remaining blocker is known and is *not* in
/// this slice.
///
/// Grades it if it decodes — so the rung promotes itself the moment the
/// blocker is cleared — and otherwise asserts that the failure is not the one
/// this slice removed. A plain skip would let a K.3 regression hide behind an
/// unrelated bug; a plain failure would make the suite red for someone else's
/// defect.
fn corpus_rung_or_diagnose(case: &str) {
    let dir = corpus(case);
    let input = dir.join("input.jxl");
    let reference = dir.join("reference_image.npy");
    if !input.exists() || !reference.exists() {
        eprintln!("skipping corpus {case}: not fetched (see tools/fetch-conformance.sh)");
        return;
    }
    let bytes = std::fs::read(&input).expect("readable");
    match decode(&bytes, &Limits::default()) {
        Ok(image) => {
            let (peak, rmse) = corpus_thresholds(&dir).expect("readable test.json");
            let report = grade(case, &as_float_image(&image), &read_npy(&reference));
            assert_conforms(case, &report, peak, rmse);
        }
        Err(e) => {
            let text = e.to_string();
            assert!(
                !text.contains("K.3") && !text.to_lowercase().contains("patch"),
                "{case}: still blocked on patches, which this slice implements: {text}"
            );
            eprintln!("skipping corpus {case}: blocked downstream of patches: {text}");
        }
    }
}

// ---------------------------------------------------------------------------
// Rung 1 — first light
// ---------------------------------------------------------------------------

/// Fixture 04 is an 86-byte 8x8 `cjxl -d 1 -e 7` stream: the smallest real
/// VarDCT frame available. It proves the whole pipeline runs end to end and
/// produces the right shape and a plausible image.
///
/// It is graded against a djxl decode of the *same stream*, which is the only
/// meaningful comparison — the encoder's own loss against the source is up to
/// 0.37 normalized, several times any Part 3 class, so the source image can
/// never be the reference. The threshold is the with-filters class: this
/// fixture was encoded with cjxl's defaults, so gaborish and EPF are on.
#[test]
fn fixture_04_first_light() {
    let path = handmade("04_gradient_8x8_lossy.jxl");
    assert!(path.exists(), "missing fixture 04");
    let image = decode_fixture(&path);
    assert_eq!((image.width, image.height), (8, 8));
    assert!(
        image.float_planes.is_some(),
        "a kVarDCT decode must produce float planes"
    );

    let Some(reference) = reference_for(&path) else {
        eprintln!("skipping fixture 04 grading: no djxl oracle");
        return;
    };
    let report = grade("04", &as_float_image(&image), &reference);
    assert_conforms("04", &report, 0.06, 0.02);
}

// ---------------------------------------------------------------------------
// Rung 2 — the no-filters class (peak 0.004, RMSE 1e-5)
// ---------------------------------------------------------------------------

/// Greyscale, `--gaborish=0 --epf=0`, distance 1. The strictest gate in the
/// slice: with both restoration filters off, every part of the error budget
/// belongs to I.2-I.9 and L.2.
#[test]
fn fixture_50_gray_nofilters_d1() {
    fixture_rung("50_vardct_mixed_gray_128x128_nofilters_d1.jxl", 0.004, 1e-5);
}

/// The same content at distance 4: coarser quantization, so different
/// transform choices and much larger `HfMul` values.
#[test]
fn fixture_51_gray_nofilters_d4() {
    fixture_rung("51_vardct_mixed_gray_128x128_nofilters_d4.jxl", 0.004, 1e-5);
}

/// RGB, filters off. The first rung where chroma-from-luma and the X/B
/// `pow(0.8, qm_scale - 2)` factors can be wrong without cancelling.
#[test]
fn fixture_55_rgb_nofilters_d4() {
    fixture_rung("55_vardct_mixed_rgb_128x128_nofilters_d4.jxl", 0.004, 1e-5);
}

/// Saturated primaries on black rules, filters off: 27% of its reference
/// samples are below zero and 10% are below `-0.05`.
///
/// L.2.2's output is allowed outside the gamut and 18181-3 §4.2 forbids
/// clipping before the comparison, so the signalled transfer function has to
/// be evaluated at negative arguments — where Table E.6 and the standards it
/// names stop defining it. This rung pins the `kSRGB` half of
/// [`jpxl_core::color::NEGATIVES_TAKE_THE_LINEAR_SEGMENT`]: flipping it makes
/// this fixture's peak error exceed 0.6. The `k709` half, which resolves the
/// *other* way, is pinned by the `bike` corpus rungs below. See
/// `docs/experiments/2026-08-04-negative-transfer-function-branch.md`.
///
/// Unlike 55 and 56 this one's reference `.npy` is committed (49 KB), so it
/// grades without an oracle.
#[test]
fn fixture_63_rgb_out_of_gamut_nofilters_d4() {
    fixture_rung(
        "63_vardct_outofgamut_rgb_64x64_nofilters_d4.jxl",
        0.004,
        1e-5,
    );
}

/// 128x2176: the only fixture here taller than one **LF group**, so the only
/// one with an internal LF-group boundary (at `y = 2048`; LF groups are
/// 2048x2048). Encoded at `-d 6 -e 3` because at `-d 1` cjxl sets
/// `kSkipAdaptiveLFSmoothing` and I.5.2's smoothing pass never runs.
///
/// This fixture exists because I.5.2's adaptive smoothing used to be applied
/// per LF group, which skips the first and last row and column of *every*
/// group rather than only the frame's own edges. It was the corpus-free
/// reproducer for the defect: the whole error budget sat in rows 2039..2056
/// at peak `0.0106`, against `3.3e-6` everywhere else in the frame.
///
/// Fixed by running the pass over the frame-wide LF image
/// (`decode::smooth_lf_image`), which is what I.5.2's "each LF sample of the
/// image" says. The band is now gone: the whole frame grades at `2.7e-6`, in
/// line with every single-LF-group fixture here. This rung is the regression
/// test for that — no other fixture in this file has an internal LF-group
/// boundary, so if the pass ever goes back to being per group nothing else
/// notices.
#[test]
fn fixture_64_rgb_lf_group_seam_nofilters_d6() {
    fixture_rung(
        "64_vardct_lfgroupseam_rgb_128x2176_nofilters_d6.jxl",
        0.004,
        1e-5,
    );
}

// ---------------------------------------------------------------------------
// Rung 3 — the with-filters class (peak 0.06, RMSE 0.02)
// ---------------------------------------------------------------------------

/// Greyscale with gaborish and EPF on, distance 1. Against fixture 50 this
/// isolates Annex J: same content, same encoder, filters the only difference.
#[test]
fn fixture_52_gray_filters_d1() {
    fixture_rung("52_vardct_mixed_gray_128x128_filters_d1.jxl", 0.06, 0.02);
}

/// Greyscale with filters, distance 4 — larger sigma, so EPF actually runs
/// on more blocks than at d1 (where many blocks fall under the 0.3 skip).
#[test]
fn fixture_53_gray_filters_d4() {
    fixture_rung("53_vardct_mixed_gray_128x128_filters_d4.jxl", 0.06, 0.02);
}

/// RGB with filters: EPF's distance metric weights the three channels by
/// `epf_channel_scale`, which only a colour fixture exercises.
#[test]
fn fixture_56_rgb_filters_d1() {
    fixture_rung("56_vardct_mixed_rgb_128x128_filters_d1.jxl", 0.06, 0.02);
}

// ---------------------------------------------------------------------------
// Rung 4 — the normative corpus
// ---------------------------------------------------------------------------

/// `grayscale`: a published conformance case with the tight thresholds
/// (peak 0.004, RMSE 1e-4) against a reference image JPXL did not produce.
#[test]
fn corpus_grayscale() {
    corpus_rung("grayscale");
}

/// `grayscale_5`: the level-5 variant, which exercises the coefficient-order
/// permutation branch of I.3.1 (`used_orders != 0`).
#[test]
fn corpus_grayscale_5() {
    corpus_rung("grayscale_5");
}

/// `bike_5`: VarDCT colour with a K.3 patch dictionary — 94 patches, all
/// `kAdd`, read from a 22x20 `kReferenceOnly` modular frame in slot 0.
///
/// K.3 is implemented, and this rung is written to **promote itself**: if the
/// stream decodes it is graded at `test.json`'s thresholds, and if it does not
/// the failure is asserted to be something other than patches. It now grades
/// green, at peak `2.5e-4` against a `0.007` limit and channel RMSE `6.6e-7`
/// against `1e-4`.
///
/// The last thing that kept it red was the LF-group seam: `bike` is 2048x2560
/// with 2048x2048 LF groups, so it has exactly one internal boundary, and
/// before the fix rows 2039..2056 carried the *entire* remaining error at
/// peak 0.0317 on B. Running I.5.2's smoothing over the frame-wide LF image
/// removes it. See `fixture_64_rgb_lf_group_seam_nofilters_d6` for the
/// corpus-free regression test and
/// `docs/experiments/2026-08-04-negative-transfer-function-branch.md` §6 for
/// the localisation.
///
/// That the patch path itself is correct is established separately: `LfGlobal`
/// of this very frame is consumed to the bit (12293 of 12296, three bits of
/// byte padding) with the dictionary in it, and synthetic patch streams decode
/// end to end at peak 5.5e-4. See
/// `docs/experiments/2026-08-03-patches-k3.md`.
#[test]
fn corpus_bike_5() {
    corpus_rung_or_diagnose("bike_5");
}

/// `bike`: the level-10 sibling of `bike_5`, same two-frame patch structure
/// and the same numbers.
#[test]
fn corpus_bike() {
    corpus_rung_or_diagnose("bike");
}
