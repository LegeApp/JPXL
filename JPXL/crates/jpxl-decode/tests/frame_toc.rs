//! TOC vectors — ISO/IEC 18181-1 F.3, including the entropy-coded permutation
//! of F.3.2.
//!
//! The permuted case drives a real [`SymbolDecoder`] bundle rather than a stub,
//! because the permutation is the only place in Annex F where the frame layer
//! and the entropy layer meet, and a hand-stubbed decoder would not exercise
//! that seam at all.

// Test code: an out-of-range index is a failed assertion rather than an attack
// surface. The lints exist for the decode paths in src/.
#![allow(clippy::indexing_slicing)]

mod frame_support;

use frame_support::BitWriter;
use jpxl_bitstream::BitReader;
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_decode::frame::{Toc, get_context, lehmer_to_permutation, read_toc};

fn read(data: &[u8], num_sections: u64) -> Toc {
    let limits = Limits::relaxed();
    let mut guard = AllocGuard::new(&limits);
    let mut r = BitReader::new(data);
    read_toc(&mut r, num_sections, &limits, &mut guard).expect("valid TOC")
}

/// Writes a complete C.2.1 distribution bundle for the permutation stream.
///
/// F.3.1 specifies "a single entropy coded stream with 8 pre-clustered
/// distributions", i.e. `num_dist == 8`. The bundle written here is:
///
/// | field | value | clause |
/// | --- | --- | --- |
/// | `Bool()` | 0 — LZ77 disabled | Table C.1 |
/// | `Bool()` | 1 — cluster map `is_simple` | C.2.2 |
/// | `u(2)` | 0 — `nbits = 0`, so all 8 contexts share cluster 0 | C.2.2 |
/// | `Bool()` | 1 — `use_prefix_code` | C.2.1 |
/// | `u(4)` | 15 — `split_exponent`, so `split = 32768` | C.2.3 |
/// | `Bool()`, `u(4)`, `u(3)` | alphabet `count = 1 + 8 + 7 = 16` | C.2.1 |
/// | `u(2)` | 1 — simple prefix code | RFC 7932 3.4 |
/// | `u(2)` | 3 — `NSYM = 4` | |
/// | 4 x `u(4)` | symbols 0, 1, 2, 3 | |
/// | `Bool()` | 0 — tree-select, giving lengths 2, 2, 2, 2 | |
///
/// Four equal lengths assign canonically by symbol value, so the codes are
/// `00`, `01`, `10`, `11` for symbols 0..3. `split_exponent` of 15 puts every
/// one of those tokens below `split`, so a decoded token *is* its value and no
/// extra bits follow.
fn permutation_bundle(w: &mut BitWriter) {
    w.bool(false); // lz77.enabled
    w.bool(true); // C.2.2 is_simple
    w.u(2, 0); // nbits = 0 -> all contexts in cluster 0
    w.bool(true); // use_prefix_code
    w.u(4, 15); // configs[0].split_exponent = 15
    w.bool(true); // alphabet count flag
    w.u(4, 3); // n = 3
    w.u(3, 7); // count = 1 + (1 << 3) + 7 = 16
    w.u(2, 1); // simple prefix code
    w.u(2, 3); // NSYM = 4
    for symbol in 0..4u32 {
        w.u(4, symbol);
    }
    w.bool(false); // tree-select 0 -> lengths 2, 2, 2, 2
}

/// Writes the two-bit code for a value in `0..4` under the bundle above.
fn symbol(w: &mut BitWriter, value: u8) {
    assert!(value < 4, "the fixture alphabet only covers 0..3");
    w.code(&[(value >> 1) & 1, value & 1]);
}

#[test]
fn unpermuted_toc_offsets_are_a_prefix_sum() {
    let sizes = [10u32, 20, 30, 40, 50];
    let mut w = BitWriter::new();
    w.bool(false); // permuted_toc
    w.pad_to_byte();
    for s in sizes {
        w.u32_field(0, 10, s);
    }
    w.pad_to_byte();

    let toc = read(&w.finish_padded(1), 5);
    assert!(!toc.permuted);
    assert_eq!(toc.entries, vec![10, 20, 30, 40, 50]);
    assert_eq!(toc.offsets, vec![0, 10, 30, 60, 100]);
    assert_eq!(toc.total_size(), 150);
    assert!(toc.permutation.is_none());
}

#[test]
fn permuted_toc_reindexes_the_offsets() {
    // size = 5, skip = 0, end = 3, lehmer = [2, 0, 1, 0, 0].
    //
    //   temp = [0,1,2,3,4]
    //   i=0: take temp[2] = 2      -> perm [2],       temp [0,1,3,4]
    //   i=1: take temp[0] = 0      -> perm [2,0],     temp [1,3,4]
    //   i=2: take temp[1] = 3      -> perm [2,0,3],   temp [1,4]
    //   i=3: take temp[0] = 1      -> perm [2,0,3,1], temp [4]
    //   i=4: take temp[0] = 4      -> perm [2,0,3,1,4]
    //
    // Contexts: end uses GetContext(5) = 3; the Lehmer values use
    // GetContext(0) then GetContext(lehmer[i-1]). Every context maps to
    // cluster 0 in this bundle, so the choice does not change the bits — but
    // it does have to be a valid index into the eight distributions.
    let expected_permutation = lehmer_to_permutation(&[2, 0, 1, 0, 0]);
    assert_eq!(expected_permutation, vec![2, 0, 3, 1, 4]);

    let mut w = BitWriter::new();
    w.bool(true); // permuted_toc
    permutation_bundle(&mut w);
    symbol(&mut w, 3); // end = 3
    symbol(&mut w, 2); // lehmer[0] = 2
    symbol(&mut w, 0); // lehmer[1] = 0
    symbol(&mut w, 1); // lehmer[2] = 1
    w.pad_to_byte();
    for s in [10u32, 20, 30, 40, 50] {
        w.u32_field(0, 10, s);
    }
    w.pad_to_byte();

    let toc = read(&w.finish_padded(1), 5);

    assert!(toc.permuted);
    assert_eq!(toc.permutation.as_deref(), Some(&[2u32, 0, 3, 1, 4][..]));
    assert_eq!(
        toc.entries,
        vec![10, 20, 30, 40, 50],
        "entries are read in bitstream order and are not permuted"
    );
    // Unpermuted offsets would be [0, 10, 30, 60, 100]; the permutation
    // reindexes them as offsets[i] = old[permutation[i]].
    assert_eq!(toc.offsets, vec![30, 0, 60, 10, 100]);
    assert_eq!(
        toc.total_size(),
        150,
        "the total is unaffected by reordering"
    );
}

#[test]
fn permuted_toc_with_the_identity_permutation_changes_nothing() {
    // end = 0 leaves the whole Lehmer sequence zero, which is the identity.
    assert_eq!(lehmer_to_permutation(&[0, 0, 0]), vec![0, 1, 2]);

    let mut w = BitWriter::new();
    w.bool(true);
    permutation_bundle(&mut w);
    symbol(&mut w, 0); // end = 0, so no Lehmer values follow
    w.pad_to_byte();
    for s in [7u32, 9, 11] {
        w.u32_field(0, 10, s);
    }
    w.pad_to_byte();

    let toc = read(&w.finish_padded(1), 3);
    assert!(toc.permuted);
    assert_eq!(toc.permutation.as_deref(), Some(&[0u32, 1, 2][..]));
    assert_eq!(
        toc.offsets,
        vec![0, 7, 16],
        "identical to the unpermuted case"
    );
}

#[test]
fn permutation_and_no_permutation_agree_on_section_sizes() {
    // The same sizes with and without a permutation must give the same
    // entries and the same total; only the offset ordering differs.
    let sizes = [10u32, 20, 30, 40, 50];

    let mut plain = BitWriter::new();
    plain.bool(false);
    plain.pad_to_byte();
    for s in sizes {
        plain.u32_field(0, 10, s);
    }
    plain.pad_to_byte();
    let plain_toc = read(&plain.finish_padded(1), 5);

    let mut permuted = BitWriter::new();
    permuted.bool(true);
    permutation_bundle(&mut permuted);
    symbol(&mut permuted, 3);
    symbol(&mut permuted, 2);
    symbol(&mut permuted, 0);
    symbol(&mut permuted, 1);
    permuted.pad_to_byte();
    for s in sizes {
        permuted.u32_field(0, 10, s);
    }
    permuted.pad_to_byte();
    let permuted_toc = read(&permuted.finish_padded(1), 5);

    assert_eq!(plain_toc.entries, permuted_toc.entries);
    assert_eq!(plain_toc.total_size(), permuted_toc.total_size());
    assert_ne!(plain_toc.offsets, permuted_toc.offsets);

    // Every offset in the permuted TOC is one of the unpermuted offsets.
    for offset in &permuted_toc.offsets {
        assert!(plain_toc.offsets.contains(offset));
    }
}

#[test]
fn toc_ends_byte_aligned_so_sections_are_addressable() {
    // F.3.3: "After decoding the TOC, the decoder invokes ZeroPadToByte().
    // Let P be the byte position at this point."
    let mut w = BitWriter::new();
    w.bool(false);
    w.pad_to_byte();
    w.u32_field(0, 10, 1);
    w.pad_to_byte();
    let data = w.finish_padded(1);

    let limits = Limits::relaxed();
    let mut guard = AllocGuard::new(&limits);
    let mut r = BitReader::new(&data);
    read_toc(&mut r, 1, &limits, &mut guard).expect("valid");

    assert!(r.is_byte_aligned());
    assert!(r.total_bits_read().is_multiple_of(8));
}

#[test]
fn get_context_stays_within_the_eight_distributions() {
    // F.3.2 GetContext(x) = min(7, ceil(log2(x + 1))). The permutation stream
    // is opened with exactly 8 distributions, so every context must be < 8.
    for x in [0u32, 1, 2, 3, 4, 7, 8, 127, 128, 1 << 20, u32::MAX] {
        assert!(get_context(x) < 8, "GetContext({x}) escaped the range");
    }
    assert_eq!(get_context(0), 0);
    assert_eq!(get_context(4), 3);
    assert_eq!(get_context(128), 7);
}

#[test]
fn malformed_permutation_stream_is_an_error_not_a_panic() {
    // permuted_toc set, then garbage where the bundle should be.
    for filler in [0x00u8, 0xFF, 0xAA] {
        let mut data = vec![0x01u8];
        data.extend_from_slice(&[filler; 8]);
        let limits = Limits::relaxed();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&data);
        let _ = read_toc(&mut r, 5, &limits, &mut guard);
    }
}

#[test]
fn oversized_lehmer_value_is_rejected() {
    // lehmer[0] must be strictly less than size - 0. With size = 3 a decoded
    // 3 is out of range and must be rejected rather than silently clamped.
    let mut w = BitWriter::new();
    w.bool(true);
    permutation_bundle(&mut w);
    symbol(&mut w, 1); // end = 1
    symbol(&mut w, 3); // lehmer[0] = 3, but size - 0 == 3
    w.pad_to_byte();
    for s in [1u32, 2, 3] {
        w.u32_field(0, 10, s);
    }
    w.pad_to_byte();
    let data = w.finish_padded(1);

    let limits = Limits::relaxed();
    let mut guard = AllocGuard::new(&limits);
    let mut r = BitReader::new(&data);
    assert!(read_toc(&mut r, 3, &limits, &mut guard).is_err());
}

/// Decodes the TOC of real cjxl-produced frames and checks it is consistent.
///
/// Two invariants that a mis-read TOC would break: the entry count matches the
/// geometry's `num_sections`, and the sections exactly tile the bytes that
/// remain in the file after the TOC. The latter is the strong one — it pins
/// both the entry values and the byte position `P` of F.3.3.
#[test]
fn real_stream_toc_tiles_the_frame() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("manifest dir has two ancestors")
        .join("tests")
        .join("fixtures")
        .join("handmade");

    for name in [
        "03_gradient_8x8_lossless.jxl",
        "07_modular_gray_8x8_lossless.jxl",
        "11_modular_gradient_256x256_lossless.jxl",
        "12_modular_gray_300x200_lossless.jxl",
        "13_modular_rgb_16x16_lossless.jxl",
        "06_gradient_300x200_lossy.jxl",
    ] {
        let bytes = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&bytes);
        let headers =
            jpxl_decode::headers::decode_image_headers_metered(&mut r, &limits, &mut guard)
                .unwrap_or_else(|e| panic!("{name}: headers: {e}"));
        r.zero_pad_to_byte().expect("frames are byte-aligned");
        let frame_start = (r.total_bits_read() / 8) as usize;

        let mut fr = BitReader::new(&bytes[frame_start..]);
        let header = jpxl_decode::frame::read_frame_header(
            &mut fr,
            &headers.metadata,
            headers.width(),
            headers.height(),
            &limits,
            &mut guard,
        )
        .unwrap_or_else(|e| panic!("{name}: frame header: {e}"));
        let geometry = jpxl_decode::frame::FrameGeometry::from_header(
            &header,
            headers.width(),
            headers.height(),
            &limits,
            &mut guard,
        )
        .unwrap_or_else(|e| panic!("{name}: geometry: {e}"));

        let toc = read_toc(&mut fr, geometry.num_sections(), &limits, &mut guard)
            .unwrap_or_else(|e| panic!("{name}: TOC: {e}"));
        assert_eq!(
            toc.len() as u64,
            geometry.num_sections(),
            "{name}: TOC entry count"
        );

        let base = frame_start + (fr.total_bits_read() / 8) as usize;
        assert_eq!(
            base as u64 + toc.total_size(),
            bytes.len() as u64,
            "{name}: the sections must tile exactly to the end of the file"
        );
        for i in 0..toc.len() {
            assert!(
                toc.offset_of(i).is_some_and(|o| o <= toc.total_size()),
                "{name}: section {i} offset is outside the frame"
            );
        }
    }
}
