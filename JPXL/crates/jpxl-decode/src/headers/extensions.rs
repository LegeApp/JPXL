//! The `Extensions` bundle (18181-1 B.3).
//!
//! Table B.3:
//!
//! ```text
//! condition          type     default   name
//!                    U64()              extensions
//! extensions != 0    U64()    0         extension_bits[NumExt]
//! ```
//!
//! `extensions` is a bit array: bit `i` (least-significant is 0) says extension
//! `ext_id == i` is present. `NumExt` is its population count, and
//! `extension_bits[i]` is the payload length in bits of the `i`-th present
//! extension, counted from just after `extension_bits` has been read.
//!
//! No extension is defined for `ImageMetadata` yet (Annex N), so the payload is
//! skipped rather than interpreted. Skipping is what makes an unknown extension
//! forward-compatible instead of fatal, which is the entire point of the bundle.

use jpxl_bitstream::{BitReader, read_u64, trace_field};

use crate::error::{DecodeError, Result};

/// A parsed `Extensions` bundle (18181-1 B.3).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Extensions {
    /// Presence bit array; bit `i` marks `ext_id == i`.
    pub extensions: u64,
    /// Payload bit lengths, in ascending `ext_id` order. Length is `NumExt`.
    pub extension_bits: Vec<u64>,
}

impl Extensions {
    /// Number of extensions present: the population count of `extensions`.
    #[must_use]
    pub const fn num_ext(&self) -> u32 {
        self.extensions.count_ones()
    }

    /// Whether the extension with `ext_id` is present.
    #[must_use]
    pub const fn has(&self, ext_id: u32) -> bool {
        ext_id < 64 && (self.extensions >> ext_id) & 1 == 1
    }

    /// Total payload bits across all present extensions.
    ///
    /// # Errors
    ///
    /// [`DecodeError::FieldOutOfRange`] if the lengths sum past `u64`.
    pub fn total_payload_bits(&self) -> Result<u64> {
        let mut total: u64 = 0;
        for bits in &self.extension_bits {
            total = total
                .checked_add(*bits)
                .ok_or_else(|| DecodeError::out_of_range("extension_bits", "B.3", u64::MAX))?;
        }
        Ok(total)
    }
}

/// Reads an `Extensions` bundle and skips every present extension's payload
/// (18181-1 B.3).
///
/// # Errors
///
/// Bitstream errors if the payloads run past the end of the input, or
/// [`DecodeError::FieldOutOfRange`] if the payload lengths overflow.
pub fn read_extensions(reader: &mut BitReader<'_>) -> Result<Extensions> {
    let extensions = trace_field!(reader, "extensions", read_u64(reader))?;

    let mut extension_bits = Vec::new();
    if extensions != 0 {
        // NumExt entries, one per set bit; at most 64, so no allocation cap is
        // needed beyond what the bit array itself bounds.
        let num_ext = extensions.count_ones();
        extension_bits.reserve(num_ext as usize);
        for _ in 0..num_ext {
            extension_bits.push(trace_field!(reader, "extension_bits", read_u64(reader))?);
        }
    }

    let parsed = Extensions {
        extensions,
        extension_bits,
    };

    // B.3: "The decoder reads all these bits for all extensions which are
    // present." Nothing in Annex N applies to ImageMetadata yet, so consume
    // and discard.
    let skip = parsed.total_payload_bits()?;
    if skip != 0 {
        trace_field!(reader, "extension_payloads", reader.skip_bits(skip))?;
    }

    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;

    #[test]
    fn absent_extensions_cost_two_bits() {
        // U64() selector 0 => value 0.
        let mut w = BitWriter::new();
        w.u64_field(0);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let ext = read_extensions(&mut r).expect("valid");

        assert_eq!(ext.extensions, 0);
        assert_eq!(ext.num_ext(), 0);
        assert!(ext.extension_bits.is_empty());
        assert_eq!(r.total_bits_read(), 2, "U64() selector 0 is two bits");
    }

    #[test]
    fn one_extension_skips_its_payload() {
        // extensions = 1 (only ext_id 0 present), extension_bits[0] = 3.
        // Both are U64() selector 1 => 1 + u(4), so six bits each; then the
        // three payload bits.
        let mut w = BitWriter::new();
        w.u64_field(1).u64_field(3).u(3, 0b111);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let ext = read_extensions(&mut r).expect("valid");

        assert_eq!(ext.extensions, 1);
        assert_eq!(ext.num_ext(), 1);
        assert!(ext.has(0));
        assert!(!ext.has(1));
        assert_eq!(ext.extension_bits, vec![3]);
        assert_eq!(
            r.total_bits_read(),
            15,
            "6 + 6 + 3 bits, including the skipped payload"
        );
    }

    #[test]
    fn two_extensions_read_lengths_in_ascending_ext_id_order() {
        // extensions = 0b101 marks ext_id 0 and 2 present, so NumExt is 2 and
        // the two lengths belong to ext_id 0 and 2 in that order.
        let mut w = BitWriter::new();
        w.u64_field(0b101).u64_field(2).u64_field(5);
        w.u(2, 0).u(5, 0); // the two payloads
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let ext = read_extensions(&mut r).expect("valid");

        assert_eq!(ext.extensions, 0b101);
        assert_eq!(ext.num_ext(), 2);
        assert!(ext.has(0) && ext.has(2));
        assert!(!ext.has(1));
        assert_eq!(ext.extension_bits, vec![2, 5]);
        assert_eq!(ext.total_payload_bits().expect("no overflow"), 7);
        assert_eq!(r.total_bits_read(), 25);
    }

    #[test]
    fn payload_spans_many_bytes() {
        // A 100-bit payload crossing a dozen byte boundaries.
        let mut w = BitWriter::new();
        w.u64_field(1).u64_field(100);
        for _ in 0..100 {
            w.bit(true);
        }
        let expected = w.bit_len();
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let ext = read_extensions(&mut r).expect("valid");

        assert_eq!(ext.extension_bits, vec![100]);
        assert_eq!(r.total_bits_read(), expected);
    }

    #[test]
    fn truncated_payload_is_an_error() {
        // Claims 200 payload bits that the buffer does not contain.
        let mut w = BitWriter::new();
        w.u64_field(1).u64_field(200);
        let data = w.finish_padded(2);
        let mut r = BitReader::new(&data);
        assert!(read_extensions(&mut r).is_err());
    }

    #[test]
    fn has_rejects_out_of_range_ext_id() {
        let ext = Extensions {
            extensions: u64::MAX,
            extension_bits: Vec::new(),
        };
        assert!(ext.has(63));
        assert!(!ext.has(64), "ext_id is bounded by the 64-bit array");
    }
}
