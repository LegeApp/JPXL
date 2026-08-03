//! Embedded ICC profile decoding — ISO/IEC 18181-1 E.4.
//!
//! When `ColourEncoding.want_icc` is set (E.2), the colour space is described
//! by an ICC profile carried in the codestream rather than by the enumerated
//! fields. Table A.1 places that profile immediately after the image headers
//! and before the first frame, and E.4 specifies how to decode it.
//!
//! # The three layers
//!
//! ```text
//! codestream bits ──E.4.1──▶ encoded ICC stream ──E.4.2──▶ command + data
//!                                                            streams
//!                                                              │
//!                             E.4.3 header ─┐                  │
//!                             E.4.4 tag list ├── concatenated ─┘
//!                             E.4.5 main content ─┘  = ICC profile
//! ```
//!
//! 1. **E.4.1** is pure Annex C: a `U64()` length, 41 pre-clustered
//!    distributions, then that many integers, each in `[0, 255]`, with the
//!    context chosen by [`icc_context`] from the byte position and the two
//!    previous bytes. The result is a byte string, the *encoded ICC stream*.
//! 2. **E.4.2** splits that byte string into a command stream and a data
//!    stream after two `Varint()` headers.
//! 3. **E.4.3**, **E.4.4** and **E.4.5** consume both streams in turn and each
//!    append one part of the profile: a prediction-coded 128-byte ICC header, a
//!    dictionary-coded tag table, and a small command language for everything
//!    else.
//!
//! Nothing in this module interprets the profile. A JPEG XL decoder does not
//! need to understand ICC to decode pixels (the note in F.2 says as much); it
//! needs to reproduce the profile's bytes exactly and hand them to whatever
//! does.
//!
//! # Bit-exactness
//!
//! `docs/PLAN.md` puts ICC bytes in the bit-exact class: the decoded profile is
//! byte-identical to the one the encoder was given. `tests/e2e_icc.rs` proves
//! it against `djxl --orig_icc_out`.

pub mod content;
pub mod context;
pub mod error;
pub mod header;
pub mod stream;
pub mod tags;

use jpxl_bitstream::{BitReader, read_u64, trace_field};
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_entropy::SymbolDecoder;

use crate::container;
use crate::headers::decode_image_headers_metered;

pub use context::{NUM_ICC_CONTEXTS, icc_context};
pub use error::{IccError, Result};

use error::malformed;
use stream::ByteStream;

/// Largest ICC profile this decoder will reconstruct, in bytes.
///
/// Table M.1 caps `output_size` (E.4.2) at `1 << 22` for level 5 and `1 << 28`
/// for level 10. JPXL does not yet parse the level signalling of Annex N, so it
/// applies the permissive level-10 bound; the allocation is metered through the
/// caller's [`AllocGuard`] regardless, which is what actually protects memory.
pub const MAX_ICC_OUTPUT_SIZE: u64 = 1 << 28;

/// Largest *encoded* ICC stream this decoder will accept, in bytes.
///
/// E.4 gives no explicit bound on `enc_size`; a stream that codes more bytes
/// than the largest legal profile cannot be describing a legal profile, since
/// every byte of the encoded stream is consumed (E.4.2) and the commands can
/// only shrink or hold the size, never expand it beyond `output_size`.
pub const MAX_ICC_ENCODED_SIZE: u64 = MAX_ICC_OUTPUT_SIZE;

/// Reads and decodes the ICC profile at the reader's current position
/// (18181-1 E.4).
///
/// The reader must be positioned immediately after `ImageMetadata`, as Table
/// A.1 requires, and is left immediately after the entropy-coded ICC stream —
/// generally not on a byte boundary. `Frame` alignment (F.1) is the caller's
/// job, exactly as it is when no profile is present.
///
/// # Errors
///
/// [`IccError`] for a stream that violates E.4, or a limit error if the profile
/// would allocate more than `guard` permits.
pub fn read_icc_profile(reader: &mut BitReader<'_>, guard: &mut AllocGuard) -> Result<Vec<u8>> {
    let encoded = read_encoded_stream(reader, guard)?;
    reconstruct(&encoded, guard)
}

/// E.4.1: entropy-decodes the encoded ICC stream from the codestream.
fn read_encoded_stream(reader: &mut BitReader<'_>, guard: &mut AllocGuard) -> Result<Vec<u8>> {
    let enc_size = trace_field!(reader, "icc.enc_size", read_u64(reader))?;
    if enc_size > MAX_ICC_ENCODED_SIZE {
        return Err(malformed!(
            "E.4.1: enc_size {enc_size} exceeds the {MAX_ICC_ENCODED_SIZE}-byte bound implied by \
             Table M.1"
        ));
    }
    guard.charge(enc_size)?;
    let capacity = usize::try_from(enc_size)
        .map_err(|_| malformed!("E.4.1: enc_size {enc_size} does not fit in memory"))?;

    let mut decoder = SymbolDecoder::open(reader, NUM_ICC_CONTEXTS, guard)?;
    let mut out: Vec<u8> = Vec::with_capacity(capacity);
    for index in 0..enc_size {
        // "prev_byte and prev_prev_byte are ... 0 if they do not exist yet".
        let len = out.len();
        let b1 = len
            .checked_sub(1)
            .and_then(|i| out.get(i))
            .copied()
            .unwrap_or(0);
        let b2 = len
            .checked_sub(2)
            .and_then(|i| out.get(i))
            .copied()
            .unwrap_or(0);
        let value = decoder.read_uint(reader, icc_context(index, b1, b2))?;
        let byte = u8::try_from(value).map_err(|_| {
            malformed!("E.4.1: decoded value {value} at index {index} is outside [0, 255]")
        })?;
        out.push(byte);
    }
    decoder.finish()?;
    Ok(out)
}

/// E.4.2 through E.4.5: turns an encoded ICC stream into the ICC profile.
///
/// Exposed separately from [`read_icc_profile`] so the byte-level stages can be
/// tested without an entropy-coded fixture.
///
/// # Errors
///
/// [`IccError::Malformed`] for a stream that violates E.4.2..E.4.5, or a limit
/// error if `output_size` exceeds what `guard` permits.
pub fn reconstruct(encoded: &[u8], guard: &mut AllocGuard) -> Result<Vec<u8>> {
    let mut head = ByteStream::new(encoded, "encoded ICC");
    let output_size = head.varint()?;
    let commands_size = head.varint()?;

    if output_size > MAX_ICC_OUTPUT_SIZE {
        return Err(malformed!(
            "E.4.2: output_size {output_size} exceeds the Table M.1 maximum of \
             {MAX_ICC_OUTPUT_SIZE}"
        ));
    }
    let capacity = usize::try_from(output_size)
        .map_err(|_| malformed!("E.4.2: output_size {output_size} does not fit in memory"))?;
    guard.charge(output_size)?;

    // "the commands stream does not extend beyond the end of the encoded ICC
    // stream": the split is validated, not clamped.
    let split = usize::try_from(commands_size)
        .ok()
        .filter(|&n| n <= head.remaining())
        .ok_or_else(|| {
            malformed!(
                "E.4.2: commands_size {commands_size} overruns the {} bytes left in the encoded \
                 ICC stream",
                head.remaining()
            )
        })?;
    let rest = encoded.get(head.position()..).unwrap_or_default();
    let (command_bytes, data_bytes) = rest.split_at(split);
    let mut commands = ByteStream::new(command_bytes, "command");
    let mut data = ByteStream::new(data_bytes, "data");

    let mut out = Vec::with_capacity(capacity);
    // E.4.3 — the header, always present, never reading the command stream.
    if header::decode_header(&mut data, output_size, &mut out)? {
        return finish(out, output_size);
    }
    // E.4.4 — the tag table, which may declare the decode complete.
    if tags::decode_tag_list(&mut commands, &mut data, output_size, &mut out)? {
        return finish(out, output_size);
    }
    // E.4.5 — everything else, running until the command stream is exhausted.
    content::decode_main_content(&mut commands, &mut data, output_size, &mut out)?;
    finish(out, output_size)
}

/// E.4.2: "the size of the resulting ICC profile is exactly output_size bytes".
fn finish(out: Vec<u8>, output_size: u64) -> Result<Vec<u8>> {
    if out.len() as u64 != output_size {
        return Err(malformed!(
            "E.4.2: the decoded profile is {} bytes but output_size is {output_size}",
            out.len()
        ));
    }
    Ok(out)
}

/// Decodes only the ICC profile of a JPEG XL file or naked codestream.
///
/// Returns `None` when the image signals an enumerated colour space rather than
/// an embedded profile (`want_icc == false`, E.2). No frame is decoded, so this
/// works on codestreams whose frame data JPXL cannot yet handle.
///
/// # Errors
///
/// Any header error, or an [`IccError`] wrapped as
/// [`DecodeError::Icc`](crate::DecodeError::Icc).
pub fn extract_icc_profile(data: &[u8], limits: &Limits) -> crate::Result<Option<Vec<u8>>> {
    let mut guard = AllocGuard::new(limits);
    let extracted;
    let codestream = if container::is_container(data) {
        extracted = container::extract_codestream(data, &mut guard)?;
        extracted.as_slice()
    } else {
        data
    };

    let mut reader = BitReader::new(codestream);
    let headers = decode_image_headers_metered(&mut reader, limits, &mut guard)?;
    if !headers.metadata.colour_encoding.want_icc {
        return Ok(None);
    }
    Ok(Some(read_icc_profile(&mut reader, &mut guard)?))
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "hand-written spec vectors read better with direct indexing; a panic in a test is \
              a failing test"
)]
mod tests {
    use super::*;

    /// Builds an encoded ICC stream from its Table E.10 parts.
    fn encoded(output_size: u64, commands: &[u8], data: &[u8]) -> Vec<u8> {
        fn varint(mut v: u64, out: &mut Vec<u8>) {
            loop {
                // Masked to seven bits, so the narrowing is exact.
                let byte = u8::try_from(v & 127).unwrap_or(0);
                v >>= 7;
                if v == 0 {
                    out.push(byte);
                    return;
                }
                out.push(byte | 128);
            }
        }
        let mut out = Vec::new();
        varint(output_size, &mut out);
        varint(commands.len() as u64, &mut out);
        out.extend_from_slice(commands);
        out.extend_from_slice(data);
        out
    }

    /// Proves the three stages concatenate in the order E.4.2 states, using a
    /// profile short enough to be header-only.
    #[test]
    fn a_header_only_profile_skips_the_later_stages() {
        let stream = encoded(32, &[], &[0u8; 32]);
        let mut guard = AllocGuard::new(&Limits::relaxed());
        let profile = reconstruct(&stream, &mut guard).expect("valid");
        assert_eq!(profile.len(), 32);
        assert_eq!(&profile[0..4], &32u32.to_be_bytes());
        assert_eq!(&profile[12..24], b"mntrRGB XYZ "[..12].as_ref());
    }

    /// Proves a full three-stage profile assembles end to end: a 128-byte
    /// header, a one-tag table, and a `curv` payload from the main content.
    #[test]
    fn header_tag_list_and_main_content_concatenate() {
        // Tag list: num_tags = 1 (v = 2), tagcode 16 ("desc"), terminator.
        // Main content: command 21 emits "curv" plus four zero bytes.
        let commands = [0x02u8, 16, 0x00, 21];
        let data = [0u8; 128];
        let stream = encoded(128 + 4 + 12 + 8, &commands, &data);
        let mut guard = AllocGuard::new(&Limits::relaxed());
        let profile = reconstruct(&stream, &mut guard).expect("valid");

        assert_eq!(profile.len(), 152);
        assert_eq!(&profile[128..132], &1u32.to_be_bytes());
        assert_eq!(&profile[132..136], b"desc");
        assert_eq!(&profile[144..148], b"curv");
    }

    /// Proves the declared size is enforced: a stream whose commands stop short
    /// of `output_size` is malformed rather than silently truncated.
    #[test]
    fn a_short_result_is_rejected() {
        let stream = encoded(200, &[], &[0u8; 128]);
        let mut guard = AllocGuard::new(&Limits::relaxed());
        assert!(reconstruct(&stream, &mut guard).is_err());
    }

    /// Proves the command/data split is validated against the stream length.
    #[test]
    fn an_overlong_command_stream_is_rejected() {
        let mut stream = encoded(32, &[], &[0u8; 32]);
        stream[1] = 100; // commands_size far past the end
        let mut guard = AllocGuard::new(&Limits::relaxed());
        assert!(reconstruct(&stream, &mut guard).is_err());
    }

    /// Proves `output_size` is charged to the guard before the buffer exists,
    /// so a profile claiming to be huge is refused rather than allocated.
    #[test]
    fn output_size_is_metered_before_allocation() {
        let limits = Limits {
            max_alloc_bytes: 1024,
            ..Limits::default()
        };
        let mut guard = AllocGuard::new(&limits);
        let stream = encoded(1 << 20, &[], &[0u8; 8]);
        assert!(reconstruct(&stream, &mut guard).is_err());
        assert!(guard.charged() <= 1024);
    }

    /// Proves the Table M.1 bound is applied even under relaxed limits.
    #[test]
    fn an_absurd_output_size_is_rejected_outright() {
        let stream = encoded(MAX_ICC_OUTPUT_SIZE + 1, &[], &[]);
        let mut guard = AllocGuard::new(&Limits::relaxed());
        assert!(reconstruct(&stream, &mut guard).is_err());
    }

    /// Proves malformed encoded streams never panic, whatever the byte soup.
    #[test]
    fn garbage_streams_error_without_panicking() {
        let mut seed = 0x9E37_79B9u32;
        for _ in 0..2000 {
            let mut bytes = Vec::new();
            for _ in 0..48 {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                #[expect(clippy::cast_possible_truncation, reason = "byte soup by design")]
                bytes.push((seed >> 15) as u8);
            }
            let mut guard = AllocGuard::new(&Limits::default());
            let _ = reconstruct(&bytes, &mut guard);
        }
    }
}
