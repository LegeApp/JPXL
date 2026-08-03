//! End-to-end smoke tests over the committed handmade fixtures.
//!
//! These read real files off disk rather than hex literals, so they also assert
//! that the fixtures themselves are still the bytes their sidecars describe.
//!
//! Oracle-backed tests run by default but **skip silently** (printing why) when
//! no reference decoder is installed — CI must be green on a machine with no
//! libjxl. Install oracles with `tools/setup-oracles.sh`.

use std::path::{Path, PathBuf};

use jpxl_conformance::{
    Image, OracleError, OracleKind, OutputFormat, StreamKind, max_abs_error, oracle,
    peak_error_per_channel, sniff,
};

/// `JPXL/tests/fixtures/handmade`.
fn handmade_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<workspace>/crates/jpxl-conformance`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("manifest dir has two ancestors")
        .join("tests")
        .join("fixtures")
        .join("handmade")
}

fn read_fixture(name: &str) -> Vec<u8> {
    let path = handmade_dir().join(name);
    std::fs::read(&path).unwrap_or_else(|err| panic!("reading {}: {err}", path.display()))
}

#[test]
fn every_handmade_fixture_has_a_provenance_sidecar() {
    let dir = handmade_dir();
    let entries =
        std::fs::read_dir(&dir).unwrap_or_else(|err| panic!("listing {}: {err}", dir.display()));
    let mut fixtures = 0_usize;
    for entry in entries {
        let path = entry.expect("readable directory entry").path();
        if path.extension().is_some_and(|ext| ext == "txt") {
            continue;
        }
        // A committed reference `.npy` (VarDCT fixtures 50+) is provenance
        // for its *sibling* `.jxl`, not a fixture in its own right, and is
        // documented in that sibling's sidecar (digest, shape, regeneration
        // command) rather than one of its own.
        if path.extension().is_some_and(|ext| ext == "npy") {
            continue;
        }
        fixtures += 1;
        let mut sidecar = path.clone().into_os_string();
        sidecar.push(".txt");
        assert!(
            Path::new(&sidecar).is_file(),
            "{} has no .txt provenance sidecar",
            path.display()
        );
    }
    assert!(fixtures >= 3, "expected at least three handmade fixtures");
}

#[test]
fn naked_codestream_fixture_sniffs_as_a_codestream() {
    let bytes = read_fixture("00_signature_naked.bin");
    assert_eq!(bytes.len(), 6, "fixture drifted from its sidecar");
    assert_eq!(sniff(&bytes), StreamKind::NakedCodestream);
}

#[test]
fn container_fixture_sniffs_as_a_container() {
    let bytes = read_fixture("01_signature_container.bin");
    assert_eq!(bytes.len(), 20, "fixture drifted from its sidecar");
    assert_eq!(sniff(&bytes), StreamKind::Container);
}

#[test]
fn png_fixture_sniffs_as_unknown() {
    let bytes = read_fixture("02_not_jxl.bin");
    assert_eq!(sniff(&bytes), StreamKind::Unknown);
    assert!(!sniff(&bytes).is_recognized());
}

#[test]
fn every_prefix_of_a_fixture_classifies_without_panicking() {
    let bytes = read_fixture("01_signature_container.bin");
    for len in 0..=bytes.len() {
        let prefix = bytes.get(..len).expect("len is within the fixture");
        let kind = sniff(prefix);
        if len < 12 {
            assert_eq!(kind, StreamKind::Unknown, "{len}-byte prefix");
        } else {
            assert_eq!(kind, StreamKind::Container, "{len}-byte prefix");
        }
    }
}

#[test]
fn oracle_discovery_is_infallible() {
    // Never asserts that an oracle exists; only that looking is safe and that
    // whatever is reported has the file name it claims.
    for found in oracle::discover() {
        let name = found
            .path
            .file_name()
            .expect("oracle path has a file name")
            .to_string_lossy()
            .into_owned();
        assert!(
            name.starts_with(found.kind.binary_name()),
            "{} does not look like {}",
            found.path.display(),
            found.kind
        );
    }
}

/// The four cjxl-produced fixtures are naked codestreams, not containers.
///
/// `cjxl` emits a bare codestream when no metadata forces a container, so a
/// change here means either the encoder revision changed or someone swapped a
/// fixture without updating its sidecar.
#[test]
fn encoded_fixtures_sniff_as_naked_codestreams() {
    for name in ENCODED_FIXTURES {
        let bytes = read_fixture(name);
        assert_eq!(
            sniff(&bytes),
            StreamKind::NakedCodestream,
            "{name} should be a naked codestream"
        );
    }
}

/// Every encoder-produced fixture in `handmade/`, smallest first.
const ENCODED_FIXTURES: &[&str] = &[
    "03_gradient_8x8_lossless.jxl",
    "04_gradient_8x8_lossy.jxl",
    "05_gradient_300x200_lossless.jxl",
    "06_gradient_300x200_lossy.jxl",
];

/// Rebuild the synthetic source the fixtures were encoded from.
///
/// The recipe lives in `tools/make-handmade-fixtures.sh` and is restated in
/// each sidecar; recomputing it here rather than reading
/// `generated/src_*.ppm` keeps these tests working from a clean checkout,
/// where `generated/` does not exist.
fn expected_gradient(w: u32, h: u32) -> Image {
    let mut samples = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for y in 0..h {
        for x in 0..w {
            samples.push(u16::try_from(x * 255 / w.saturating_sub(1).max(1)).unwrap_or(u16::MAX));
            samples.push(u16::try_from(y * 255 / h.saturating_sub(1).max(1)).unwrap_or(u16::MAX));
            samples.push(u16::try_from((x + y) % 256).unwrap_or(u16::MAX));
        }
    }
    Image {
        w,
        h,
        channels: 3,
        max_value: 255,
        samples,
    }
}

/// Decode `fixture` with `djxl` into a PPM, or `None` if no oracle is present.
///
/// Skipping is deliberate and silent-ish: it prints why and returns `None` so
/// the caller can `return` without failing. CI without libjxl stays green.
fn djxl_decode(fixture: &str, tag: &str) -> Option<Image> {
    let djxl = oracle::find(OracleKind::Djxl).or_else(|| {
        eprintln!("skipping: no djxl oracle found (run tools/setup-oracles.sh)");
        None
    })?;
    eprintln!("using {} at {}", djxl.kind, djxl.path.display());

    let output = std::env::temp_dir().join(format!("jpxl-smoke-{tag}.ppm"));
    let _ = std::fs::remove_file(&output);

    match djxl.decode_to_ppm(&handmade_dir().join(fixture), &output) {
        Ok(()) => {}
        Err(err) if err.is_unavailable() => {
            eprintln!("skipping: {err}");
            return None;
        }
        Err(err) => panic!("djxl failed on {fixture}: {err}"),
    }

    let bytes = std::fs::read(&output)
        .unwrap_or_else(|err| panic!("djxl reported success but wrote nothing: {err}"));
    let _ = std::fs::remove_file(&output);
    Some(Image::from_ppm(&bytes).expect("djxl emitted a valid PPM"))
}

/// The multi-group lossless fixture must decode **bit-exactly**.
///
/// This is the anchor test for the whole harness: it proves the oracle is
/// wired up, that the fixture is the image its sidecar claims, and that
/// [`max_abs_error`] agrees with a byte comparison. 300x200 spans more than
/// one 256x256 group in x and is a partial group in both axes.
#[test]
fn djxl_decodes_the_multigroup_lossless_fixture_bit_exactly() {
    let Some(decoded) = djxl_decode("05_gradient_300x200_lossless.jxl", "lossless") else {
        return;
    };

    assert_eq!((decoded.w, decoded.h, decoded.channels), (300, 200, 3));
    assert_eq!(decoded.max_value, 255);
    assert_eq!(decoded.len(), 300 * 200 * 3);

    let expected = expected_gradient(300, 200);
    assert_eq!(
        max_abs_error(&decoded, &expected),
        Some(0),
        "a lossless fixture must round-trip exactly"
    );
}

/// The 8x8 lossless fixture, same contract at the smallest possible size.
#[test]
fn djxl_decodes_the_tiny_lossless_fixture_bit_exactly() {
    let Some(decoded) = djxl_decode("03_gradient_8x8_lossless.jxl", "tiny") else {
        return;
    };
    assert_eq!((decoded.w, decoded.h), (8, 8));
    assert_eq!(max_abs_error(&decoded, &expected_gradient(8, 8)), Some(0));
}

/// The lossy fixture decodes to something close to, but not equal to, the
/// source.
///
/// The budget is loose on purpose. This measures **cjxl's** loss at `-d 1`,
/// not any decoder's accuracy, and a gradient is nearly all edge, so the peak
/// is large (53/28/94 per channel when this was written). The test exists to
/// catch "lossy decode produced garbage", not to police the encoder.
#[test]
fn djxl_decodes_the_multigroup_lossy_fixture_within_budget() {
    let Some(decoded) = djxl_decode("06_gradient_300x200_lossy.jxl", "lossy") else {
        return;
    };
    assert_eq!((decoded.w, decoded.h, decoded.channels), (300, 200, 3));

    let expected = expected_gradient(300, 200);
    let peaks = peak_error_per_channel(&decoded, &expected).expect("same shape");
    let worst = max_abs_error(&decoded, &expected).expect("same shape");
    eprintln!("lossy peak error per channel: {peaks:?} (worst {worst})");

    assert!(worst > 0, "a -d 1 encode should not be bit-exact");
    assert!(
        worst <= 128,
        "lossy decode is further from the source than any plausible -d 1 loss: {peaks:?}"
    );
}

/// `jxl-oxide` refuses PPM up front and really does write PNG.
///
/// Pins the surprise documented on [`jpxl_conformance::Oracle::decode`]:
/// jxl-oxide ignores the output extension, so asking it for `out.ppm` would
/// otherwise succeed and leave PNG bytes in a file named `.ppm`.
#[test]
fn jxl_oxide_writes_png_and_rejects_ppm() {
    let Some(oxide) = oracle::find(OracleKind::JxlOxide) else {
        eprintln!("skipping: no jxl-oxide oracle found (cargo install jxl-oxide-cli)");
        return;
    };

    let input = handmade_dir().join("05_gradient_300x200_lossless.jxl");

    let err = oxide
        .decode_to_ppm(&input, Path::new("unused.ppm"))
        .expect_err("jxl-oxide has no PNM writer");
    assert!(
        matches!(err, OracleError::UnsupportedFormat { .. }),
        "{err}"
    );

    let output = std::env::temp_dir().join("jpxl-smoke-oxide.png");
    let _ = std::fs::remove_file(&output);
    match oxide.decode(&input, &output, OutputFormat::Png) {
        Ok(()) => {}
        Err(err) if err.is_unavailable() => {
            eprintln!("skipping: {err}");
            return;
        }
        Err(err) => panic!("jxl-oxide failed: {err}"),
    }

    let bytes = std::fs::read(&output).expect("jxl-oxide reported success");
    let _ = std::fs::remove_file(&output);
    // Only the magic is checked: this crate has no PNG reader and is not
    // getting one just for a smoke test. See `Oracle::decode` for the npy
    // route if pixel comparison against jxl-oxide is ever needed.
    assert_eq!(
        bytes.get(..8),
        Some([0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A].as_slice()),
        "jxl-oxide should have written a PNG"
    );
}
