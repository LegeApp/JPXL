//! End-to-end lossless modular decoding through a **multi-section** TOC
//! (18181-1 F.3, Annex G — separate `LfGlobal` / `LfGroup` / `HfGlobal` /
//! per-group `PassGroup` sections located via the TOC), as opposed to
//! `tests/e2e_lossless.rs`, whose fixtures all collapse to `num_sections ==
//! 1` (F.3.1: that happens whenever `num_groups == 1 && num_passes == 1`).
//!
//! # What this proves that `e2e_lossless.rs` does not
//!
//! Every fixture 07-13 fits in one group at the `group_dim` cjxl chose for
//! it (12_modular_gray_300x200 is 300x200 but the encoder picked `group_dim
//! = 512`, so it is still one group — see that fixture's `.txt` sidecar and
//! `HANDOFF.md`, 2026-08-02). The multi-section decode path — reading
//! `LfGlobal`/`ModularLfGroup`/`Modular group data` as *separate* TOC
//! sections at *separate* byte offsets, sharing one channel list and one set
//! of clustered distributions across all of them (G.1.3/G.2.3/G.4.2) — is
//! implemented per spec but was, before this file, never exercised by an
//! actual multi-section stream. `fixture_has_multiple_sections` below parses
//! each fixture's real TOC with this crate's own `FrameGeometry`/`read_toc`
//! and asserts `num_sections > 1`, so this file FAILS rather than silently
//! proving nothing if a future oracle rebuild changes cjxl's group_dim
//! heuristic and regenerates 14-16 single-section (14/15's sidecars call
//! this out explicitly).
//!
//! # Fixtures
//!
//! * `14_modular_gray_600x520_multisection_lossless.jxl` — 600x520
//!   grayscale, exceeds 512px in both axes, `group_dim = 256`, 3x3 = 9
//!   groups, `num_sections = 12`. Single channel, isolates multi-section
//!   handling from RCT/palette transform selection.
//! * `15_modular_rgb_600x520_multisection_lossless.jxl` — same canvas as 14,
//!   RGB variant (same gradient formula as 09/11/13): multi-section
//!   *together with* the RCT and per-group entropy state across 3 channels.
//! * `16_modular_gray_511x8_multisection_lossless.jxl` — a 121-byte smoke
//!   fixture: `group_dim = 256` at 511px wide gives 2 groups (one partial
//!   right edge), `num_sections = 5`. Does not meet AGENTS.md's ">= 256x256"
//!   multi-group rule by itself (14/15 do); it is a fast regression
//!   companion for the same TOC/group-copy path.
//!
//! See each fixture's `.txt` sidecar in `tests/fixtures/handmade/` for the
//! full provenance and the black-box probing that found these dimensions
//! (no cjxl flag forces a smaller `group_dim` in this oracle build — checked
//! with `cjxl -v -v --help`, same conclusion 13's sidecar reached for a
//! predictor-selection flag).
//!
//! # Bit-exactness
//!
//! Same regime as `e2e_lossless.rs`: modular lossless is bit-exact
//! (`docs/PLAN.md`), checked here against the deterministic source formula
//! *and* against live `djxl` output when the oracle is installed.

#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

use std::path::{Path, PathBuf};

use jpxl_bitstream::BitReader;
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_decode::decode;
use jpxl_decode::frame::{FrameGeometry, read_frame_header, read_toc};
use jpxl_decode::headers::decode_image_headers_metered;

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

/// Parses just enough of a fixture (image headers, the first frame's header
/// and geometry, then its TOC) to report `num_sections` — the same sequence
/// `jpxl_decode::decode::decode` runs internally, stopped one step earlier so
/// the test can inspect the TOC directly rather than only its downstream
/// effect on pixels. All three fixtures here are naked codestreams (`jpxl
/// info` confirms "naked codestream" for each), so container extraction is
/// not needed, unlike fixture 08 in `e2e_lossless.rs`.
fn parsed_num_sections(bytes: &[u8]) -> u64 {
    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    let mut reader = BitReader::new(bytes);
    let headers = decode_image_headers_metered(&mut reader, &limits, &mut guard)
        .expect("image headers parse");
    reader
        .zero_pad_to_byte()
        .expect("byte-align before the frame");
    let header = read_frame_header(
        &mut reader,
        &headers.metadata,
        headers.width(),
        headers.height(),
        &limits,
        &mut guard,
    )
    .expect("frame header parses");
    let geometry = FrameGeometry::from_header(
        &header,
        headers.width(),
        headers.height(),
        &limits,
        &mut guard,
    )
    .expect("geometry derives from the header");
    let toc =
        read_toc(&mut reader, geometry.num_sections(), &limits, &mut guard).expect("TOC parses");
    assert_eq!(
        u64::try_from(toc.len()).expect("section count fits in u64"),
        geometry.num_sections(),
        "TOC entry count must match the geometry's num_sections"
    );
    geometry.num_sections()
}

// ---------------------------------------------------------------------------
// The headline assertion: these fixtures really are multi-section
// ---------------------------------------------------------------------------

/// Fails loudly — rather than silently proving nothing — if a future oracle
/// rebuild regenerates 14/15/16 as single-section streams (a different
/// cjxl `group_dim` heuristic could do that; see the fixtures' `.txt`
/// sidecars). This is the guard the whole file exists to provide.
#[test]
fn fixture_has_multiple_sections() {
    let cases: &[(&str, u64)] = &[
        (
            "14_modular_gray_600x520_multisection_lossless.jxl",
            12, // 2 header sections + 1 LF group + 9 pass groups
        ),
        (
            "15_modular_rgb_600x520_multisection_lossless.jxl",
            12, // same canvas as 14
        ),
        (
            "16_modular_gray_511x8_multisection_lossless.jxl",
            5, // 2 header sections + 1 LF group + 2 pass groups
        ),
    ];
    for (name, expected) in cases {
        let bytes = read_fixture(name);
        let num_sections = parsed_num_sections(&bytes);
        assert!(
            num_sections > 1,
            "{name}: expected a multi-section TOC (num_sections > 1), got {num_sections} \
             — this fixture no longer proves the multi-section path; see its .txt sidecar"
        );
        assert_eq!(
            num_sections, *expected,
            "{name}: num_sections changed from the value recorded in its .txt sidecar"
        );
    }
}

// ---------------------------------------------------------------------------
// Pixel bit-exactness against the deterministic source formula
// ---------------------------------------------------------------------------

/// The diagonal grey ramp of fixtures 14 and 16 (same family as 07/08/12):
/// `(x + y) * 255 / (w + h - 2)`.
const fn grey_ramp(x: u32, y: u32, w: u32, h: u32) -> i32 {
    (((x as i64 + y as i64) * 255) / (w as i64 + h as i64 - 2)) as i32
}

/// The RGB gradient of fixture 15 (same family as 09/11/13):
/// `R = x * 255 / (w-1)`, `G = y * 255 / (h-1)`, `B = (x + y) mod 256`.
const fn rgb_gradient(x: u32, y: u32, c: usize, w: u32, h: u32) -> i32 {
    match c {
        0 => ((x as i64 * 255) / (w as i64 - 1)) as i32,
        1 => ((y as i64 * 255) / (h as i64 - 1)) as i32,
        _ => ((x + y) % 256) as i32,
    }
}

struct Case {
    name: &'static str,
    width: u32,
    height: u32,
    channels: usize,
    sample: fn(u32, u32, usize) -> i32,
}

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

#[test]
fn modular_grey_600x520_multisection() {
    check(&Case {
        name: "14_modular_gray_600x520_multisection_lossless.jxl",
        width: 600,
        height: 520,
        channels: 1,
        sample: |x, y, _| grey_ramp(x, y, 600, 520),
    });
}

#[test]
fn modular_rgb_600x520_multisection() {
    check(&Case {
        name: "15_modular_rgb_600x520_multisection_lossless.jxl",
        width: 600,
        height: 520,
        channels: 3,
        sample: |x, y, c| rgb_gradient(x, y, c, 600, 520),
    });
}

#[test]
fn modular_grey_511x8_multisection_smoke() {
    check(&Case {
        name: "16_modular_gray_511x8_multisection_lossless.jxl",
        width: 511,
        height: 8,
        channels: 1,
        sample: |x, y, _| grey_ramp(x, y, 511, 8),
    });
}

// ---------------------------------------------------------------------------
// Live oracle comparison
// ---------------------------------------------------------------------------

/// The fixtures this file decodes bit-exactly.
const MULTISECTION_FIXTURES: &[&str] = &[
    "14_modular_gray_600x520_multisection_lossless.jxl",
    "15_modular_rgb_600x520_multisection_lossless.jxl",
    "16_modular_gray_511x8_multisection_lossless.jxl",
];

/// Decodes every multi-section fixture with `djxl` and compares sample for
/// sample, exactly as `e2e_lossless.rs::matches_djxl_output` does for the
/// single-section fixtures. Skips silently when no oracle is installed
/// (`tools/setup-oracles.sh`), per the project rule that CI must be green on
/// a machine with no libjxl.
#[test]
fn matches_djxl_output() {
    use jpxl_conformance::{Image, OracleKind, oracle};

    let Some(djxl) = oracle::find(OracleKind::Djxl) else {
        println!("skipping: no djxl oracle installed (run tools/setup-oracles.sh)");
        return;
    };

    let temp = std::env::temp_dir().join("jpxl-e2e-multisection-oracle");
    std::fs::create_dir_all(&temp).expect("temp dir");

    let mut compared = 0usize;
    for name in MULTISECTION_FIXTURES {
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
    println!("compared {compared} multi-section fixture(s) against djxl byte for byte");
}
