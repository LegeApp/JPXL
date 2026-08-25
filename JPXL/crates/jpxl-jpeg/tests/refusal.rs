//! Graceful, typed refusal of out-of-scope JPEG modes, and non-panicking
//! rejection of malformed input.
//!
//! Phase A supports Huffman-coded 8-bit baseline/progressive DCT only. Every
//! other well-formed JPEG mode must be rejected with [`JpegError::Unsupported`]
//! — never mis-parsed into silent garbage — and every malformed byte string
//! must return an `Err`, never panic (decode paths are attacker-facing).

#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "tests index their own just-loaded data; a panic here is a failing \
              test, which is the intended signal"
)]

use jpxl_jpeg::{JpegError, parse};

/// Builds a minimal `SOI` + one marker stream (enough to reach the code path
/// that classifies the marker).
fn soi_then(marker_code: u8, tail: &[u8]) -> Vec<u8> {
    let mut v = vec![0xFF, 0xD8, 0xFF, marker_code];
    v.extend_from_slice(tail);
    v
}

#[test]
fn refuses_arithmetic_coded_jpeg_fixture() {
    let data = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/refuse_arithmetic.jpg"
    ))
    .expect("arithmetic fixture");
    match parse(&data) {
        Err(JpegError::Unsupported(_)) => {}
        other => panic!("expected Unsupported for arithmetic JPEG, got {other:?}"),
    }
}

#[test]
fn refuses_lossless_sof3() {
    // SOF3 is classified before its body is read.
    assert!(matches!(
        parse(&soi_then(0xC3, &[])),
        Err(JpegError::Unsupported(_))
    ));
}

#[test]
fn refuses_hierarchical_sof5() {
    assert!(matches!(
        parse(&soi_then(0xC5, &[])),
        Err(JpegError::Unsupported(_))
    ));
}

#[test]
fn refuses_arithmetic_sof9() {
    assert!(matches!(
        parse(&soi_then(0xC9, &[])),
        Err(JpegError::Unsupported(_))
    ));
}

#[test]
fn refuses_arithmetic_conditioning_dac() {
    assert!(matches!(
        parse(&soi_then(0xCC, &[])),
        Err(JpegError::Unsupported(_))
    ));
}

#[test]
fn refuses_hierarchical_dhp() {
    assert!(matches!(
        parse(&soi_then(0xDE, &[])),
        Err(JpegError::Unsupported(_))
    ));
}

#[test]
fn refuses_twelve_bit_precision() {
    // A well-formed baseline SOF0 header claiming 12-bit samples.
    let sof0_12bit = soi_then(
        0xC0,
        &[
            0x00, 0x11, // Lf = 17
            0x0C, // P  = 12-bit precision
            0x00, 0x10, // Y  = 16
            0x00, 0x10, // X  = 16
            0x03, // Nf = 3
            0x01, 0x11, 0x00, // component 1
            0x02, 0x11, 0x00, // component 2
            0x03, 0x11, 0x00, // component 3
        ],
    );
    assert!(matches!(parse(&sof0_12bit), Err(JpegError::Unsupported(_))));
}

// ---- Malformed input must error, not panic --------------------------------

#[test]
fn rejects_non_jpeg_bytes() {
    assert!(parse(b"not a jpeg at all").is_err());
    assert!(parse(&[]).is_err());
    assert!(parse(&[0xFF]).is_err());
    assert!(parse(&[0xFF, 0xD8]).is_err()); // SOI then nothing
}

#[test]
fn rejects_truncated_baseline_without_panic() {
    // Take a real baseline JPEG and truncate it at many offsets; none may panic.
    let data = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/base_444.jpg"
    ))
    .unwrap();
    for cut in (2..data.len()).step_by(97) {
        // Must return Ok or Err, but never unwind.
        let _ = parse(&data[..cut]);
    }
}

#[test]
fn rejects_corrupted_entropy_without_panic() {
    let mut data = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/prog_420.jpg"
    ))
    .unwrap();
    // Flip bytes across the stream; parsing must stay panic-free.
    for i in (0..data.len()).step_by(31) {
        let saved = data[i];
        data[i] ^= 0xA5;
        let _ = parse(&data);
        data[i] = saved;
    }
}
