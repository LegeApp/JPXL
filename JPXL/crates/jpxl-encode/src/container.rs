//! The minimal JPEG XL container (ISO/IEC 18181-2 clauses 8 and 9).
//!
//! A container is a sequence of boxes: a big-endian `u32` length that counts
//! the header, a four-byte type, then the payload (18181-2 clause 8). The
//! smallest legal file this encoder can produce is three boxes:
//!
//! | Box | Clause | Content |
//! |---|---|---|
//! | signature | 9.1 | the twelve fixed bytes |
//! | `ftyp` | 9.2 | the twenty fixed bytes |
//! | `jxlc` | 9.9 | the whole naked codestream |
//!
//! and a fourth, `jxll` (9.3), when the codestream needs level 10. 9.3 says an
//! absent level box means level 5, and Annex M of Part 1 lists
//! `modular_16bit_buffers = true` as a level-5 requirement — which a 16-bit
//! image cannot honour. Writing the level box is therefore not decoration: a
//! 16-bit file with no `jxll` claims a level it does not meet.
//!
//! [`wrap_fragmented`] produces the alternative 9.10 form instead: the same
//! codestream split across `jxlp` boxes. A file written in one pass gains
//! nothing from fragmentation — 9.10 exists so a reader can get the header
//! early — but the writer is worth having, because it is what lets the
//! decoder's reassembly be tested against a byte-identical `jxlc` reference
//! rather than against itself.

/// The 12-byte JPEG XL signature box (18181-2 9.1).
const SIGNATURE_BOX: [u8; 12] = [
    0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
];

/// The 20-byte file type box (18181-2 9.2).
const FTYP_BOX: [u8; 20] = [
    0x00, 0x00, 0x00, 0x14, b'f', b't', b'y', b'p', b'j', b'x', b'l', b' ', 0x00, 0x00, 0x00, 0x00,
    b'j', b'x', b'l', b' ',
];

/// The default level when no level box is present (18181-2 9.3).
pub const DEFAULT_LEVEL: u8 = 5;

/// The level a codestream outside the 18181-1 Annex M level-5 limits needs.
pub const EXTENDED_LEVEL: u8 = 10;

/// The high bit of a `jxlp` index, marking the final fragment (18181-2 9.10).
const JXLP_LAST_FLAG: u32 = 0x8000_0000;

/// Wraps `codestream` in a container, declaring `level` when it is not the
/// default.
///
/// The `jxlc` box is written with an explicit length rather than the
/// run-to-end-of-file form, so appending a box later stays possible.
#[must_use]
pub fn wrap(codestream: &[u8], level: u8) -> Vec<u8> {
    let mut out = preamble(codestream.len(), level);
    // 8.1: a length of 0 means "to the end of the file"; used only if the
    // explicit length would not fit in the 32-bit field.
    let explicit = u32::try_from(codestream.len() + 8).ok();
    out.extend_from_slice(&explicit.unwrap_or(0).to_be_bytes());
    out.extend_from_slice(b"jxlc");
    out.extend_from_slice(codestream);
    out
}

/// Wraps `codestream` in a container, split across `jxlp` boxes of at most
/// `fragment_size` payload bytes each (18181-2 9.10).
///
/// The index of fragment `i` is `i`, and the last one additionally carries
/// `2^31`. A zero `fragment_size` is treated as one — the alternative is a
/// silent infinite loop — and an empty codestream still produces one (empty)
/// fragment, which 9.10 NOTE 2 permits and which keeps "at least one `jxlp`"
/// true.
#[must_use]
pub fn wrap_fragmented(codestream: &[u8], level: u8, fragment_size: usize) -> Vec<u8> {
    let fragment_size = fragment_size.max(1);
    let chunks: Vec<&[u8]> = if codestream.is_empty() {
        vec![&[]]
    } else {
        codestream.chunks(fragment_size).collect()
    };

    let mut out = preamble(codestream.len() + 12 * chunks.len(), level);
    for (i, chunk) in chunks.iter().enumerate() {
        let mut index = u32::try_from(i).unwrap_or(u32::MAX - 1);
        if i + 1 == chunks.len() {
            index |= JXLP_LAST_FLAG;
        }
        // LBox counts the header, the u32 index and the payload.
        let explicit = u32::try_from(chunk.len() + 12).ok();
        out.extend_from_slice(&explicit.unwrap_or(0).to_be_bytes());
        out.extend_from_slice(b"jxlp");
        out.extend_from_slice(&index.to_be_bytes());
        out.extend_from_slice(chunk);
    }
    out
}

/// Appends an `Exif` box (18181-2 9.5) to an already-wrapped container.
///
/// The payload must be the raw Exif block as JEITA CP-3451E / CP-3461B
/// define it — beginning with the TIFF header — and the box's
/// `tiff_header_offset` field is therefore written as zero. Clause 5 leaves
/// box order free after the file type box, so appending after the codestream
/// boxes is conforming; [`wrap`] writes the `jxlc` box with an explicit
/// length precisely so a box can follow it.
///
/// Per 9.5, codestream fields (orientation, dimensions) take precedence over
/// Exif equivalents at decode time; the caller is responsible for not
/// contradicting them.
pub fn append_exif(container: &mut Vec<u8>, exif_payload: &[u8]) {
    // LBox counts the 8-byte header and the u32 tiff_header_offset.
    let explicit = u32::try_from(exif_payload.len() + 12).ok();
    container.extend_from_slice(&explicit.unwrap_or(0).to_be_bytes());
    container.extend_from_slice(b"Exif");
    container.extend_from_slice(&0u32.to_be_bytes()); // tiff_header_offset
    container.extend_from_slice(exif_payload);
}

/// The signature box, the file type box and — when the level is not the
/// default — the level box, i.e. everything before the codestream boxes.
fn preamble(payload_hint: usize, level: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload_hint + 64);
    out.extend_from_slice(&SIGNATURE_BOX);
    out.extend_from_slice(&FTYP_BOX);
    if level != DEFAULT_LEVEL {
        // 9.3: the level box is the third box, and its content is one u8.
        out.extend_from_slice(&9u32.to_be_bytes());
        out.extend_from_slice(b"jxll");
        out.push(level);
    }
    out
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    reason = "the hand-written box walk below reads the file it just built, \
              and an out-of-range index there is exactly the failure the test \
              is looking for"
)]
mod tests {
    use super::*;
    use jpxl_core::limits::{AllocGuard, Limits};

    #[test]
    fn the_codestream_comes_back_out_byte_identically() {
        let codestream: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        for level in [DEFAULT_LEVEL, EXTENDED_LEVEL] {
            let file = wrap(&codestream, level);
            assert!(jpxl_decode::container::is_container(&file));
            let mut guard = AllocGuard::new(&Limits::relaxed());
            let extracted =
                jpxl_decode::container::extract_codestream(&file, &mut guard).expect("jxlc");
            assert_eq!(extracted, codestream, "level {level}");
        }
    }

    #[test]
    fn the_box_order_is_the_one_clause_9_requires() {
        let file = wrap(&[0xFF, 0x0A], EXTENDED_LEVEL);
        assert_eq!(file.get(..12), Some(&SIGNATURE_BOX[..]));
        assert_eq!(file.get(12..32), Some(&FTYP_BOX[..]));
        assert_eq!(
            file.get(32..40),
            Some(&[0, 0, 0, 9, b'j', b'x', b'l', b'l'][..])
        );
        assert_eq!(file.get(40), Some(&EXTENDED_LEVEL));
        assert_eq!(
            file.get(41..49),
            Some(&[0, 0, 0, 10, b'j', b'x', b'l', b'c'][..])
        );
        assert_eq!(file.get(49..), Some(&[0xFFu8, 0x0A][..]));
    }

    #[test]
    fn a_level_five_file_has_no_level_box() {
        let file = wrap(&[0xFF, 0x0A], DEFAULT_LEVEL);
        assert_eq!(file.len(), 12 + 20 + 8 + 2);
    }

    /// The claim slice 9 needs: a fragmented file and a whole one carry the
    /// same codestream, so the decoder's 9.10 reassembly can be checked
    /// against a `jxlc` reference rather than against itself.
    #[test]
    fn fragmenting_changes_the_boxes_and_not_the_codestream() {
        let codestream: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        for size in [1usize, 7, 999, 1000, 1001, 4096] {
            let file = wrap_fragmented(&codestream, DEFAULT_LEVEL, size);
            let mut guard = AllocGuard::new(&Limits::relaxed());
            let tree = jpxl_decode::container::BoxTree::parse(&file, &mut guard).expect("parses");
            tree.validate().expect("a conforming 9.10 file");
            assert_eq!(
                tree.codestream(&mut guard).expect("reassembles"),
                codestream,
                "fragment size {size}"
            );
            let expected = codestream.len().div_ceil(size);
            let fragments = tree
                .boxes()
                .iter()
                .filter(|b| b.kind == jpxl_decode::container::BoxKind::PartialCodestream)
                .count();
            assert_eq!(fragments, expected, "fragment size {size}");
        }
    }

    #[test]
    fn a_degenerate_fragment_size_still_produces_a_conforming_file() {
        // Zero would divide by zero or loop forever; one empty fragment is
        // what 9.10 NOTE 2 permits for an empty codestream.
        for (codestream, size) in [(&[0xFFu8, 0x0A][..], 0usize), (&[][..], 16)] {
            let file = wrap_fragmented(codestream, DEFAULT_LEVEL, size);
            let mut guard = AllocGuard::new(&Limits::relaxed());
            let tree = jpxl_decode::container::BoxTree::parse(&file, &mut guard).expect("parses");
            tree.validate().expect("conforming");
            assert_eq!(
                tree.codestream(&mut guard).expect("reassembles"),
                codestream
            );
        }
    }

    /// The claim the archive pipeline needs: an appended `Exif` box survives
    /// the decoder's box walk with its payload intact, and does not disturb
    /// codestream extraction — for a whole `jxlc` file and a fragmented one.
    #[test]
    fn an_appended_exif_box_round_trips_and_leaves_the_codestream_alone() {
        let codestream: Vec<u8> = (0..500u32).map(|i| (i % 251) as u8).collect();
        // A plausible payload head: little-endian TIFF header, then filler.
        let mut exif = vec![0x49, 0x49, 0x2A, 0x00];
        exif.extend((0..64u32).map(|i| (i % 7) as u8));

        for mut file in [
            wrap(&codestream, DEFAULT_LEVEL),
            wrap_fragmented(&codestream, EXTENDED_LEVEL, 128),
        ] {
            append_exif(&mut file, &exif);
            let mut guard = AllocGuard::new(&Limits::relaxed());
            let tree = jpxl_decode::container::BoxTree::parse(&file, &mut guard).expect("parses");
            tree.validate().expect("conforming with an Exif box");
            assert_eq!(tree.codestream(&mut guard).expect("codestream"), codestream);
            let parsed = tree.exif().expect("well-formed Exif boxes");
            assert_eq!(parsed.len(), 1);
            assert_eq!(parsed[0].tiff_header_offset, 0);
            assert_eq!(parsed[0].payload, &exif[..]);
        }
    }

    #[test]
    fn the_last_fragment_is_the_only_one_carrying_the_final_marker() {
        // Written out by hand rather than trusting the decoder, because the
        // decoder's check and this writer would otherwise agree by sharing a
        // single wrong idea of where the marker goes.
        let file = wrap_fragmented(&[1, 2, 3, 4, 5], DEFAULT_LEVEL, 2);
        let mut pos = 12 + 20;
        let mut indices = Vec::new();
        while pos < file.len() {
            let len = u32::from_be_bytes([file[pos], file[pos + 1], file[pos + 2], file[pos + 3]])
                as usize;
            assert_eq!(&file[pos + 4..pos + 8], b"jxlp");
            indices.push(u32::from_be_bytes([
                file[pos + 8],
                file[pos + 9],
                file[pos + 10],
                file[pos + 11],
            ]));
            pos += len;
        }
        assert_eq!(pos, file.len(), "the boxes must tile the file");
        assert_eq!(indices, vec![0, 1, 2 | JXLP_LAST_FLAG]);
    }
}
