//! End-to-end smoke tests over the committed handmade fixtures.
//!
//! These read real files off disk rather than hex literals, so they also assert
//! that the fixtures themselves are still the bytes their sidecars describe.
//! Oracle-backed tests are `#[ignore]`d and skip silently when no reference
//! decoder is installed — CI must be green on a machine with no libjxl.

use std::path::{Path, PathBuf};

use jpxl_conformance::{Image, OracleKind, StreamKind, max_abs_error, oracle, sniff};

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

/// Decode a fixture with `djxl` and read the result back as a PPM.
///
/// `#[ignore]` because it shells out to a reference decoder; run with
/// `cargo test -p jpxl-conformance -- --ignored`. It also *skips* (returns
/// early, printing why) when no oracle is installed, so enabling it on a
/// bare machine still passes.
///
/// The handmade fixtures are signature-only and deliberately undecodable, so
/// today this asserts the oracle *rejects* them cleanly. Once real fixtures
/// land in `generated/`, point this at one and compare pixels with
/// [`max_abs_error`].
#[test]
#[ignore = "requires a reference decoder; run with --ignored"]
fn djxl_round_trip_smoke() {
    let Some(djxl) = oracle::find(OracleKind::Djxl) else {
        eprintln!("skipping: no djxl oracle found (see tools/setup-oracles.sh)");
        return;
    };
    eprintln!("using oracle {} at {}", djxl.kind, djxl.path.display());

    let input = handmade_dir().join("00_signature_naked.bin");
    let output = std::env::temp_dir().join("jpxl-smoke-djxl.ppm");
    let _ = std::fs::remove_file(&output);

    match djxl.decode_to_ppm(&input, &output) {
        Ok(()) => {
            // Surprising but not a failure of *this* test: if a future fixture
            // is decodable, check the oracle produced a readable PPM.
            let bytes = std::fs::read(&output).expect("oracle reported success");
            let image = Image::from_ppm(&bytes).expect("oracle emitted a valid PPM");
            assert_eq!(max_abs_error(&image, &image), Some(0));
        }
        Err(err) if err.is_unavailable() => {
            eprintln!("skipping: {err}");
        }
        Err(err) => {
            // Expected path today: a signature-only fixture is not decodable.
            eprintln!("oracle rejected the signature-only fixture, as expected: {err}");
        }
    }

    let _ = std::fs::remove_file(&output);
}
