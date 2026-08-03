//! The ICC header stage (18181-1 E.4.3).
//!
//! The first `min(128, output_size)` bytes of the profile are coded as
//! residuals against a fixed prediction: for byte `i` the decoder computes a
//! prediction `p` from the position and the header bytes already produced,
//! reads one residual `e` from the *data* stream, and outputs `(p + e) & 255`.
//! The command stream is not touched at all in this subclause.
//!
//! The prediction table is the ICC header of a typical display profile: the
//! profile size in the first four bytes, `mntrRGB XYZ ` at 12, `acsp` at 36,
//! the four platform signatures at 40, and the D50 PCS illuminant at 68.
//! Nothing here is a heuristic — the constants are exactly the ones E.4.3
//! lists, and any other profile simply codes larger residuals.

use super::error::{Result, malformed};
use super::stream::ByteStream;

/// The largest ICC header E.4.3 predicts, in bytes.
pub const ICC_HEADER_SIZE: usize = 128;

/// `"mntrRGB XYZ "`, predicted for header bytes 12..=23 (18181-1 E.4.3).
const PROFILE_CLASS_AND_SPACES: &[u8; 12] = b"mntrRGB XYZ ";

/// `"acsp"`, the ICC profile file signature, predicted for bytes 36..=39.
const FILE_SIGNATURE: &[u8; 4] = b"acsp";

/// The prediction for header byte `i`, given the residuals decoded so far.
///
/// `header` holds the header bytes already appended to the result, so
/// `header[j]` is only ever consulted for `j < i`.
fn predict(i: usize, output_size: u32, header: &[u8]) -> u8 {
    let at = |j: usize| header.get(j).copied().unwrap_or(0);
    let size = output_size.to_be_bytes();

    match i {
        // Bytes 0..=3 are the profile size, which the decoder already knows.
        0..=3 => size.get(i).copied().unwrap_or(0),
        // Byte 8 is the major profile version; 4 is the ICC v4 value.
        8 => 4,
        12..=23 => PROFILE_CLASS_AND_SPACES.get(i - 12).copied().unwrap_or(0),
        36..=39 => FILE_SIGNATURE.get(i - 36).copied().unwrap_or(0),
        // Bytes 40..=43 are the primary platform signature. Byte 40 is not
        // predicted; the rest are completed from it, and byte 41 only when 40
        // already determines the string ("APPL", "MSFT").
        41 | 42 if at(40) == b'A' => b'P',
        43 if at(40) == b'A' => b'L',
        41 if at(40) == b'M' => b'S',
        42 if at(40) == b'M' => b'F',
        43 if at(40) == b'M' => b'T',
        42 if at(40) == b'S' && at(41) == b'G' => b'I',
        43 if at(40) == b'S' && at(41) == b'G' => 32,
        42 if at(40) == b'S' && at(41) == b'U' => b'N',
        43 if at(40) == b'S' && at(41) == b'U' => b'W',
        // Bytes 68..=79 are the PCS illuminant, predicted as D50 in s15Fixed16:
        // 0x0000F6D6, 0x00010000, 0x0000D32D. Only the nonzero bytes appear.
        70 => 246,
        71 => 214,
        73 => 1,
        78 => 211,
        79 => 45,
        // Bytes 80..=83 (the profile creator) are predicted from bytes 4..=7
        // (the preferred CMM), which are commonly the same signature.
        80..=83 => at(4 + i - 80),
        _ => 0,
    }
}

/// Decodes the ICC header (18181-1 E.4.3) and appends it to `out`.
///
/// Returns `true` when `output_size <= 128`, in which case the header *is* the
/// whole profile and E.4.4 and E.4.5 are skipped.
///
/// # Errors
///
/// [`IccError::Malformed`](super::IccError::Malformed) if the data stream ends
/// before `min(128, output_size)` residuals have been read.
pub fn decode_header(
    data: &mut ByteStream<'_>,
    output_size: u64,
    out: &mut Vec<u8>,
) -> Result<bool> {
    let header_size = usize::try_from(output_size)
        .unwrap_or(usize::MAX)
        .min(ICC_HEADER_SIZE);
    // `output_size` is capped well below 2^32 by the caller (Table M.1), so the
    // truncation the big-endian prediction needs cannot lose information.
    let size32 = u32::try_from(output_size)
        .map_err(|_| malformed!("E.4.3: output_size {output_size} does not fit 32 bits"))?;

    let base = out.len();
    for i in 0..header_size {
        let residual = data.u8()?;
        let predicted = predict(i, size32, out.get(base..).unwrap_or_default());
        out.push(predicted.wrapping_add(residual));
    }
    Ok(output_size <= ICC_HEADER_SIZE as u64)
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "hand-written spec vectors read better with direct indexing; a panic in a test is \
              a failing test"
)]
mod tests {
    use super::*;

    /// Proves the all-zero residual case reproduces the exact header E.4.3
    /// predicts, which is the ICC header of a canonical display profile.
    #[test]
    fn zero_residuals_reproduce_the_predicted_header() {
        let residuals = [0u8; 128];
        let mut data = ByteStream::new(&residuals, "data");
        let mut out = Vec::new();
        let finished = decode_header(&mut data, 200, &mut out).expect("valid");
        assert!(!finished, "output_size > 128 means the profile continues");
        assert_eq!(out.len(), 128);

        assert_eq!(&out[0..4], &200u32.to_be_bytes());
        assert_eq!(out[8], 4);
        assert_eq!(&out[12..24], b"mntrRGB XYZ ");
        assert_eq!(&out[36..40], b"acsp");
        // Platform byte 40 has no prediction, so a zero residual leaves it 0
        // and none of the platform completions fire.
        assert_eq!(&out[40..44], &[0, 0, 0, 0]);
        assert_eq!(&out[68..72], &[0, 0, 246, 214]);
        assert_eq!(&out[72..76], &[0, 1, 0, 0]);
        assert_eq!(&out[76..80], &[0, 0, 211, 45]);
        assert_eq!(&out[80..84], &out[4..8]);
    }

    /// Proves the platform-signature completions are driven by the *decoded*
    /// byte 40, i.e. the prediction is sequential and self-referential.
    #[test]
    fn platform_signature_completes_from_byte_forty() {
        for (first, rest) in [(b'A', b"PPL"), (b'M', b"SFT")] {
            let mut residuals = [0u8; 128];
            residuals[40] = first;
            let mut data = ByteStream::new(&residuals, "data");
            let mut out = Vec::new();
            decode_header(&mut data, 512, &mut out).expect("valid");
            assert_eq!(&out[40..44], &[first, rest[0], rest[1], rest[2]]);
        }
    }

    /// Proves the two-byte-deep `"SGI "` and `"SUNW"` completions, which are the
    /// only place E.4.3 consults header[41] as well as header[40].
    #[test]
    fn sgi_and_sunw_need_two_decoded_bytes() {
        for (b40, b41, expected) in [(b'S', b'G', b"SGI "), (b'S', b'U', b"SUNW")] {
            let mut residuals = [0u8; 128];
            residuals[40] = b40;
            // Byte 41 has no prediction under 'S', so its residual is literal.
            residuals[41] = b41;
            let mut data = ByteStream::new(&residuals, "data");
            let mut out = Vec::new();
            decode_header(&mut data, 512, &mut out).expect("valid");
            assert_eq!(&out[40..44], expected);
        }
    }

    /// Proves the residual is added modulo 256 rather than saturating.
    #[test]
    fn residuals_wrap_around() {
        let mut residuals = [0u8; 128];
        residuals[8] = 253; // prediction 4, so the output is (4 + 253) & 255.
        let mut data = ByteStream::new(&residuals, "data");
        let mut out = Vec::new();
        decode_header(&mut data, 512, &mut out).expect("valid");
        assert_eq!(out[8], 1);
    }

    /// Proves a profile of 128 bytes or fewer terminates the whole decode, and
    /// that only `output_size` residuals are consumed.
    #[test]
    fn a_short_profile_is_header_only() {
        let residuals = [0u8; 128];
        let mut data = ByteStream::new(&residuals, "data");
        let mut out = Vec::new();
        let finished = decode_header(&mut data, 40, &mut out).expect("valid");
        assert!(finished);
        assert_eq!(out.len(), 40);
        assert_eq!(data.position(), 40);
    }

    #[test]
    fn a_truncated_data_stream_is_an_error() {
        let residuals = [0u8; 10];
        let mut data = ByteStream::new(&residuals, "data");
        let mut out = Vec::new();
        assert!(decode_header(&mut data, 512, &mut out).is_err());
    }
}
