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
//! `jxlp` (9.10) fragmentation is not produced. It exists for progressive
//! delivery, buys nothing for a file written in one go, and the concatenation
//! rules are one more thing to get wrong.

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

/// Wraps `codestream` in a container, declaring `level` when it is not the
/// default.
///
/// The `jxlc` box is written with an explicit length rather than the
/// run-to-end-of-file form, so appending a box later stays possible.
#[must_use]
pub fn wrap(codestream: &[u8], level: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(codestream.len() + 64);
    out.extend_from_slice(&SIGNATURE_BOX);
    out.extend_from_slice(&FTYP_BOX);
    if level != DEFAULT_LEVEL {
        // 9.3: the level box is the third box, and its content is one u8.
        out.extend_from_slice(&9u32.to_be_bytes());
        out.extend_from_slice(b"jxll");
        out.push(level);
    }
    // 8.1: a length of 0 means "to the end of the file"; used only if the
    // explicit length would not fit in the 32-bit field.
    let explicit = u32::try_from(codestream.len() + 8).ok();
    out.extend_from_slice(&explicit.unwrap_or(0).to_be_bytes());
    out.extend_from_slice(b"jxlc");
    out.extend_from_slice(codestream);
    out
}

#[cfg(test)]
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
}
