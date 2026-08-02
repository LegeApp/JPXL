//! Minimal read-only codestream extraction from a JPEG XL container
//! (ISO/IEC 18181-2 clauses 8 and 9).
//!
//! Slice 9 owns container support properly — box trees, `Exif`, `xml `,
//! `brob`, `jbrd`, the whole of Part 2. What lives here is the one thing an
//! end-to-end *codestream* decoder cannot do without: finding the codestream
//! inside a container so a container-wrapped fixture is testable at all.
//!
//! # Box format (18181-2 clause 8)
//!
//! Every box is a big-endian `u32` length, a four-byte type, then the payload.
//! A length of 0 means "to the end of the file"; a length of 1 means the real
//! length is the next big-endian `u64` and the payload starts after it. The
//! length counts the header, so the payload is `length - 8` (or `length - 16`
//! for the extended form).
//!
//! # What is extracted
//!
//! * `jxlc` (9.9) — the whole codestream in one box.
//! * `jxlp` (9.10) — codestream fragments. Each payload starts with a
//!   big-endian `u32` index whose top bit marks the final fragment; the
//!   remaining bytes concatenate **in index order** to form the codestream.
//!
//! Everything else is skipped. This is deliberately not a box parser: it does
//! not validate the signature box, `ftyp`, or ordering, and it reports what it
//! cannot handle rather than guessing.

use jpxl_core::limits::AllocGuard;

use crate::error::{DecodeError, Result};

/// The 12-byte JPEG XL container signature box (18181-2 clause 9.1).
pub const CONTAINER_SIGNATURE: [u8; 12] = [
    0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
];

/// Whether `data` starts with the container signature box.
#[must_use]
pub fn is_container(data: &[u8]) -> bool {
    data.starts_with(&CONTAINER_SIGNATURE)
}

/// Extracts the naked codestream from a container (18181-2 9.9 and 9.10).
///
/// # Errors
///
/// [`DecodeError::Unsupported`] if the container holds no codestream box, and
/// [`DecodeError::Core`] if the reassembled codestream exceeds the guard.
pub fn extract_codestream(data: &[u8], guard: &mut AllocGuard) -> Result<Vec<u8>> {
    let mut fragments: Vec<(u32, &[u8])> = Vec::new();
    let mut pos = 0usize;

    while pos + 8 <= data.len() {
        let header = data.get(pos..pos + 8).unwrap_or_default();
        let raw_len = be_u32(header.get(..4).unwrap_or_default());
        let kind = header.get(4..8).unwrap_or_default();
        let (payload_start, payload_end) = match raw_len {
            // 8.1: length 0 means the box runs to the end of the file.
            0 => (pos + 8, data.len()),
            // 8.1: length 1 means a 64-bit length follows the type.
            1 => {
                let ext = data
                    .get(pos + 8..pos + 16)
                    .ok_or_else(|| unsupported("a truncated extended-length box header"))?;
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(ext);
                let len = u64::from_be_bytes(bytes);
                let end = usize::try_from(len)
                    .ok()
                    .and_then(|l| pos.checked_add(l))
                    .filter(|&e| e <= data.len() && e >= pos + 16)
                    .ok_or_else(|| unsupported("an extended-length box that overruns the file"))?;
                (pos + 16, end)
            }
            len => {
                let end = usize::try_from(len)
                    .ok()
                    .and_then(|l| pos.checked_add(l))
                    .filter(|&e| e <= data.len() && e >= pos + 8)
                    .ok_or_else(|| unsupported("a box length that overruns the file"))?;
                (pos + 8, end)
            }
        };
        let payload = data.get(payload_start..payload_end).unwrap_or_default();

        match kind {
            b"jxlc" => fragments.push((0, payload)),
            b"jxlp" => {
                if payload.len() < 4 {
                    return Err(unsupported("a jxlp box with no fragment index"));
                }
                let index = be_u32(payload.get(..4).unwrap_or_default());
                // 9.10: the top bit marks the last fragment; the rest is the
                // ordering index.
                fragments.push((index & 0x7FFF_FFFF, payload.get(4..).unwrap_or_default()));
            }
            _ => {}
        }

        if payload_end <= pos {
            break;
        }
        pos = payload_end;
    }

    if fragments.is_empty() {
        return Err(unsupported(
            "a container with no jxlc or jxlp codestream box",
        ));
    }
    fragments.sort_by_key(|&(index, _)| index);
    let total: usize = fragments.iter().map(|(_, bytes)| bytes.len()).sum();
    guard.charge(total as u64)?;
    let mut out = Vec::with_capacity(total);
    for (_, bytes) in fragments {
        out.extend_from_slice(bytes);
    }
    Ok(out)
}

/// Reads a big-endian `u32` from the first four bytes of `bytes`, or 0.
fn be_u32(bytes: &[u8]) -> u32 {
    let mut buf = [0u8; 4];
    let n = bytes.len().min(4);
    if let (Some(dst), Some(src)) = (buf.get_mut(..n), bytes.get(..n)) {
        dst.copy_from_slice(src);
    }
    u32::from_be_bytes(buf)
}

const fn unsupported(what: &'static str) -> DecodeError {
    DecodeError::Unsupported {
        feature: what,
        clause: "18181-2 clause 8",
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
mod tests {
    use jpxl_core::limits::Limits;

    use super::*;

    fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&((payload.len() + 8) as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        out
    }

    fn guard() -> AllocGuard {
        AllocGuard::new(&Limits::relaxed())
    }

    #[test]
    fn extracts_a_single_jxlc_box() {
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&boxed(b"ftyp", b"jxl \0\0\0\0jxl "));
        file.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A, 1, 2, 3]));
        assert_eq!(
            extract_codestream(&file, &mut guard()).expect("jxlc"),
            vec![0xFF, 0x0A, 1, 2, 3]
        );
    }

    #[test]
    fn concatenates_jxlp_fragments_in_index_order() {
        // Emitted out of order to prove the sort is doing the work; the last
        // fragment carries the 0x8000_0000 marker bit, which must not affect
        // its position.
        let mut file = CONTAINER_SIGNATURE.to_vec();
        let mut second = 1u32.to_be_bytes().to_vec();
        second.extend_from_slice(&[3, 4]);
        let mut first = 0u32.to_be_bytes().to_vec();
        first.extend_from_slice(&[0xFF, 0x0A]);
        let mut third = (2u32 | 0x8000_0000).to_be_bytes().to_vec();
        third.extend_from_slice(&[5]);
        file.extend_from_slice(&boxed(b"jxlp", &second));
        file.extend_from_slice(&boxed(b"jxlp", &first));
        file.extend_from_slice(&boxed(b"jxlp", &third));

        assert_eq!(
            extract_codestream(&file, &mut guard()).expect("jxlp"),
            vec![0xFF, 0x0A, 3, 4, 5]
        );
    }

    #[test]
    fn a_zero_length_box_runs_to_the_end_of_the_file() {
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(b"jxlc");
        file.extend_from_slice(&[0xFF, 0x0A, 9]);
        assert_eq!(
            extract_codestream(&file, &mut guard()).expect("jxlc"),
            vec![0xFF, 0x0A, 9]
        );
    }

    #[test]
    fn a_container_without_a_codestream_box_is_reported() {
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&boxed(b"Exif", b"whatever"));
        let err = extract_codestream(&file, &mut guard()).expect_err("no codestream");
        assert!(err.to_string().contains("18181-2"));
    }

    #[test]
    fn overrunning_and_truncated_boxes_error_rather_than_panic() {
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&0xFFFF_FFFFu32.to_be_bytes());
        file.extend_from_slice(b"jxlc");
        assert!(extract_codestream(&file, &mut guard()).is_err());

        // Every prefix of a well-formed container must be handled.
        let mut good = CONTAINER_SIGNATURE.to_vec();
        good.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A, 1, 2, 3]));
        for cut in 0..good.len() {
            let _ = extract_codestream(&good[..cut], &mut guard());
        }
    }

    #[test]
    fn recognises_the_signature_box() {
        assert!(is_container(&CONTAINER_SIGNATURE));
        assert!(!is_container(&[0xFF, 0x0A]));
        assert!(!is_container(&[]));
    }
}
