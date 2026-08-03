//! Official conformance corpus wiring (18181-3 Annex A / §4.3).
//!
//! `tests/fixtures/conformance/` is a clone of the official corpus at a
//! pinned commit, but its big reference files (`reference_image.npy`,
//! `*.icc`) are gitignored and fetched separately by the corpus's own
//! `scripts/download_and_symlink_using_curl.sh`. This file proves that,
//! once that script has been run, the result is actually usable by this
//! crate's own [`FloatImage::from_npy`] reader and that `test.json`'s
//! per-frame thresholds are readable text -- not that JPXL's decoder passes
//! any test case (VarDCT decode does not exist yet; that grading is wave
//! 3's job once it does).
//!
//! Skips (does not fail) when the corpus clone or its downloaded references
//! are missing, printing exactly what to run.

use std::path::{Path, PathBuf};

use jpxl_conformance::FloatImage;

/// `JPXL/tests/fixtures/conformance`.
fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("manifest dir has two ancestors")
        .join("tests")
        .join("fixtures")
        .join("conformance")
}

/// Pull a numeric value out of `"key": <number>` in a small hand-written
/// JSON file. Not a JSON parser -- just enough to read the two thresholds
/// `test.json`'s `frames[0]` entry carries, without a dependency.
fn json_number_after(text: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{key}\"");
    let after_key = text.split_once(&pattern)?.1;
    let after_colon = after_key.trim_start().strip_prefix(':')?.trim_start();
    let end = after_colon
        .find([',', '\n', '}'])
        .unwrap_or(after_colon.len());
    after_colon.get(..end)?.trim().parse::<f64>().ok()
}

#[test]
fn bike_5_reference_npy_and_thresholds_are_readable() {
    let case_dir = corpus_dir().join("testcases").join("bike_5");
    if !case_dir.is_dir() {
        eprintln!(
            "skipping: no corpus clone at {} (run tools/fetch-conformance.sh)",
            case_dir.display()
        );
        return;
    }

    let npy_path = case_dir.join("reference_image.npy");
    let npy_bytes = match std::fs::read(&npy_path) {
        Ok(bytes) if !bytes.is_empty() => bytes,
        _ => {
            eprintln!(
                "skipping: {} is missing or empty (run \
                 tests/fixtures/conformance/scripts/download_and_symlink_using_curl.sh \
                 from within that directory to fetch the gitignored corpus references)",
                npy_path.display()
            );
            return;
        }
    };

    let reference = FloatImage::from_npy(&npy_bytes).expect(
        "bike_5's reference_image.npy should parse as the NPY subset 18181-3 4.1.2 defines",
    );
    assert_eq!(reference.frames, 1, "bike_5 is a single still frame");
    assert_eq!(reference.channels, 3, "bike_5 is RGB, no extra channels");
    assert!(reference.height > 0 && reference.width > 0);
    assert_eq!(
        reference.samples.len(),
        (reference.frames as usize)
            * (reference.height as usize)
            * (reference.width as usize)
            * (reference.channels as usize),
        "sample count must match frames * height * width * channels"
    );

    let test_json_path = case_dir.join("test.json");
    let test_json = std::fs::read_to_string(&test_json_path)
        .unwrap_or_else(|err| panic!("reading {}: {err}", test_json_path.display()));
    let peak_error = json_number_after(&test_json, "peak_error")
        .expect("test.json should have a parseable frames[0].peak_error");
    let rms_error = json_number_after(&test_json, "rms_error")
        .expect("test.json should have a parseable frames[0].rms_error");
    assert!(peak_error > 0.0, "peak_error should be a positive bound");
    assert!(rms_error > 0.0, "rms_error should be a positive bound");
    eprintln!(
        "bike_5: reference {}x{}x{}x{} f32, thresholds peak<={peak_error} rms<={rms_error}",
        reference.frames, reference.height, reference.width, reference.channels
    );
}
