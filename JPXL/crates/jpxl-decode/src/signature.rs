//! The codestream signature (18181-1 D.1).

use jpxl_bitstream::{BitReader, trace_field};

use crate::error::{DecodeError, Result};

/// 18181-1 D.1: the signature as a `u(16)` value.
///
/// The clause states the signature is "the fixed two-byte sequence `0xFF0A`,
/// i.e. it is equal to the value 2815". Those two statements agree only under
/// the B.2.1 bit order: reading `u(16)` LSB-first over the bytes `FF 0A`
/// assembles the little-endian value `0x0AFF`, and `0x0AFF == 2815`.
pub const CODESTREAM_SIGNATURE: u32 = 0x0AFF;

/// The signature as it appears in the byte stream, in stream order.
pub const SIGNATURE_BYTES: [u8; 2] = [0xFF, 0x0A];

/// Reads and validates the 16-bit codestream signature (18181-1 D.1).
///
/// # Errors
///
/// [`DecodeError::InvalidSignature`] if the value is not
/// [`CODESTREAM_SIGNATURE`], or a bitstream error if fewer than 16 bits remain.
pub fn read_signature(reader: &mut BitReader<'_>) -> Result<()> {
    let found = trace_field!(reader, "signature", reader.read_bits(16))?;
    if found == CODESTREAM_SIGNATURE {
        Ok(())
    } else {
        Err(DecodeError::InvalidSignature { found })
    }
}

/// Cheap check for the two signature bytes at the start of `data`.
///
/// Does not validate anything past the signature. Use this to sniff a buffer
/// before committing to a full parse; a naked codestream starts with these
/// bytes, whereas a Part 2 container starts with a box header.
#[must_use]
pub fn starts_with_signature(data: &[u8]) -> bool {
    data.starts_with(&SIGNATURE_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_the_spec_byte_sequence() {
        // 18181-1 D.1: bytes FF 0A, read as u(16) LSB-first => 0x0AFF == 2815.
        let mut r = BitReader::new(&SIGNATURE_BYTES);
        assert!(read_signature(&mut r).is_ok());
        assert_eq!(r.total_bits_read(), 16);
        assert_eq!(CODESTREAM_SIGNATURE, 2815);
    }

    #[test]
    fn rejects_byte_swapped_signature() {
        // 0A FF is the same two bytes in the wrong order.
        let mut r = BitReader::new(&[0x0A, 0xFF]);
        let err = read_signature(&mut r).expect_err("must reject");
        assert!(matches!(
            err,
            DecodeError::InvalidSignature { found: 0xFF0A }
        ));
    }

    #[test]
    fn rejects_truncated_input() {
        let mut r = BitReader::new(&[0xFF]);
        assert!(read_signature(&mut r).is_err());
    }

    #[test]
    fn sniffing() {
        assert!(starts_with_signature(&[0xFF, 0x0A, 0x00]));
        assert!(!starts_with_signature(&[0xFF]));
        assert!(!starts_with_signature(&[0x00, 0x00, 0x00, 0x0C]));
    }
}
