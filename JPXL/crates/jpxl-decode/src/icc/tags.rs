//! The ICC tag list stage (18181-1 E.4.4).
//!
//! An ICC tag table is a count followed by twelve bytes per tag: a four-byte
//! signature, an offset and a size. In a real profile those offsets and sizes
//! are almost always consecutive, so E.4.4 codes each tag as a single command
//! byte — a code selecting the signature out of a fixed dictionary, plus two
//! flag bits saying whether the offset and size differ from the running
//! prediction (previous start + previous size, and previous size).
//!
//! Two codes stand for whole groups: `rTRC` (code 2) also emits `gTRC` and
//! `bTRC` sharing its offset and size, and `rXYZ` (code 3) also emits `gXYZ`
//! and `bXYZ` at consecutive offsets. That is what makes a matrix-shaper
//! profile's twelve-entry tag table collapse to a handful of command bytes.

use super::error::{Result, malformed};
use super::stream::ByteStream;

/// The tag signatures E.4.4 selects with `tagcode` 4..=20.
const TAG_DICTIONARY: [&[u8; 4]; 17] = [
    b"cprt", b"wtpt", b"bkpt", b"rXYZ", b"gXYZ", b"bXYZ", b"kXYZ", b"rTRC", b"gTRC", b"bTRC",
    b"kTRC", b"chad", b"desc", b"chrm", b"dmnd", b"dmda", b"lumi",
];

/// Tags whose payload is a 20-byte `XYZType` or `XYZ`-shaped record, for which
/// E.4.4 predicts `tagsize = 20` instead of the previous tag's size.
const TWENTY_BYTE_TAGS: [&[u8; 4]; 7] = [
    b"rXYZ", b"gXYZ", b"bXYZ", b"kXYZ", b"wtpt", b"bkpt", b"lumi",
];

/// Bytes an ICC tag table spends per entry: signature, offset, size.
const BYTES_PER_TAG_ENTRY: u32 = 12;

/// Where the first tag payload starts once the tag table is accounted for:
/// the 128-byte header, the 4-byte count, and 12 bytes per tag — expressed by
/// E.4.4 as `num_tags * 12 + 128`, the count's four bytes being folded in.
const FIRST_TAG_START_BASE: u32 = 128;

/// Decodes the ICC tag list (18181-1 E.4.4), appending it to `out`.
///
/// Returns `true` when the decoder is *finished* — the command stream ran out
/// before the tag count could be read — in which case E.4.5 is skipped.
///
/// `output_size` bounds the output for the same reason it does in E.4.5: one
/// command byte can emit up to 36 bytes of tag table, so an unbounded loop here
/// would let a short command stream inflate the result far past what the
/// caller metered.
///
/// # Errors
///
/// [`IccError::Malformed`](super::IccError::Malformed) for a `tagcode` E.4.4
/// says is unreachable, when either sub-stream ends mid-tag, or when the tag
/// table would grow the profile past `output_size`.
pub fn decode_tag_list(
    commands: &mut ByteStream<'_>,
    data: &mut ByteStream<'_>,
    output_size: u64,
    out: &mut Vec<u8>,
) -> Result<bool> {
    // "If the end of the command stream is reached, the decoder is finished."
    if commands.at_end() {
        return Ok(true);
    }

    let v = commands.varint()?;
    if v == 0 {
        // num_tags == -1: output nothing and fall through to the main content.
        return Ok(false);
    }
    let num_tags = u32::try_from(v - 1)
        .map_err(|_| malformed!("E.4.4: num_tags {} does not fit 32 bits", v - 1))?;
    append(out, &num_tags.to_be_bytes(), output_size)?;

    let mut previous_tagstart = num_tags
        .wrapping_mul(BYTES_PER_TAG_ENTRY)
        .wrapping_add(FIRST_TAG_START_BASE);
    let mut previous_tagsize = 0u32;

    while !commands.at_end() {
        let command = commands.u8()?;
        let tagcode = command & 63;
        if tagcode == 0 {
            break;
        }

        let mut custom = [0u8; 4];
        let tag: &[u8; 4] = match tagcode {
            1 => {
                custom.copy_from_slice(data.take(4)?);
                &custom
            }
            2 => b"rTRC",
            3 => b"rXYZ",
            4..=20 => TAG_DICTIONARY
                .get(tagcode as usize - 4)
                .ok_or_else(|| malformed!("E.4.4: tagcode {tagcode} has no dictionary entry"))?,
            _ => {
                return Err(malformed!(
                    "E.4.4: tagcode {tagcode} is in the branch the clause states is not reached"
                ));
            }
        };

        let mut tagstart = previous_tagstart.wrapping_add(previous_tagsize);
        if command & 64 != 0 {
            tagstart = narrow(commands.varint()?, "tagstart")?;
        }
        let mut tagsize = previous_tagsize;
        if TWENTY_BYTE_TAGS.contains(&tag) {
            tagsize = 20;
        }
        if command & 128 != 0 {
            tagsize = narrow(commands.varint()?, "tagsize")?;
        }
        previous_tagstart = tagstart;
        previous_tagsize = tagsize;

        push_entry(out, tag, tagstart, tagsize, output_size)?;
        match tagcode {
            2 => {
                push_entry(out, b"gTRC", tagstart, tagsize, output_size)?;
                push_entry(out, b"bTRC", tagstart, tagsize, output_size)?;
            }
            3 => {
                push_entry(
                    out,
                    b"gXYZ",
                    tagstart.wrapping_add(tagsize),
                    tagsize,
                    output_size,
                )?;
                push_entry(
                    out,
                    b"bXYZ",
                    tagstart.wrapping_add(tagsize.wrapping_mul(2)),
                    tagsize,
                    output_size,
                )?;
            }
            _ => {}
        }
    }
    Ok(false)
}

/// Appends one 12-byte tag table entry: signature, offset, size.
fn push_entry(
    out: &mut Vec<u8>,
    tag: &[u8; 4],
    tagstart: u32,
    tagsize: u32,
    output_size: u64,
) -> Result<()> {
    append(out, tag, output_size)?;
    append(out, &tagstart.to_be_bytes(), output_size)?;
    append(out, &tagsize.to_be_bytes(), output_size)
}

/// Appends `bytes`, refusing to grow the output past `output_size`.
///
/// E.4.2 states the decoded profile is exactly `output_size` bytes, so this is
/// the clause's own invariant enforced as it is built rather than checked once
/// at the end -- which also keeps the allocation inside what the caller
/// metered.
fn append(out: &mut Vec<u8>, bytes: &[u8], output_size: u64) -> Result<()> {
    if out.len() as u64 + bytes.len() as u64 > output_size {
        return Err(malformed!(
            "E.4.2: the tag list grew the profile past the declared output_size of {output_size}"
        ));
    }
    out.extend_from_slice(bytes);
    Ok(())
}

/// Narrows a `Varint()` to the 32 bits E.4.4 appends it as.
fn narrow(value: u64, field: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| malformed!("E.4.4: {field} {value} does not fit 32 bits"))
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "hand-written spec vectors read better with direct indexing; a panic in a test is \
              a failing test"
)]
mod tests {
    use super::*;

    fn run(commands: &[u8], data: &[u8]) -> (Vec<u8>, bool) {
        let mut c = ByteStream::new(commands, "command");
        let mut d = ByteStream::new(data, "data");
        let mut out = Vec::new();
        let finished = decode_tag_list(&mut c, &mut d, 1 << 20, &mut out).expect("valid");
        (out, finished)
    }

    /// Proves an exhausted command stream ends the whole decode without
    /// emitting a tag count — the "profile is header-only" path of E.4.4.
    #[test]
    fn an_empty_command_stream_finishes_the_decode() {
        let (out, finished) = run(&[], &[]);
        assert!(finished);
        assert!(out.is_empty());
    }

    /// Proves `v == 0` (num_tags == -1) emits nothing but still lets the main
    /// content run — distinct from the finished case above.
    #[test]
    fn a_zero_varint_skips_the_tag_list_without_finishing() {
        let (out, finished) = run(&[0x00], &[]);
        assert!(!finished);
        assert!(out.is_empty());
    }

    /// Proves the default prediction: with no flag bits the first tag starts at
    /// `num_tags * 12 + 128` and, for a non-XYZ tag, inherits size 0.
    #[test]
    fn the_first_tag_start_is_predicted_from_the_tag_count() {
        // v = 2 (num_tags = 1), then tagcode 16 ("desc"), then terminator 0.
        let (out, _) = run(&[0x02, 16, 0x00], &[]);
        assert_eq!(&out[0..4], &1u32.to_be_bytes());
        assert_eq!(&out[4..8], b"desc");
        // num_tags * 12 + 128 with num_tags == 1.
        assert_eq!(&out[8..12], &140u32.to_be_bytes());
        assert_eq!(&out[12..16], &0u32.to_be_bytes());
    }

    /// Proves the XYZ-shaped signatures get the implicit size of 20, and that
    /// the running offset prediction chains from tag to tag.
    #[test]
    fn xyz_tags_predict_a_size_of_twenty_and_chain() {
        // num_tags = 3; wtpt (tagcode 5), bkpt (tagcode 6), terminator.
        let (out, _) = run(&[0x04, 5, 6, 0x00], &[]);
        let start = 3u32 * 12 + 128;
        assert_eq!(&out[4..8], b"wtpt");
        assert_eq!(&out[8..12], &start.to_be_bytes());
        assert_eq!(&out[12..16], &20u32.to_be_bytes());
        assert_eq!(&out[16..20], b"bkpt");
        assert_eq!(&out[20..24], &(start + 20).to_be_bytes());
        assert_eq!(&out[24..28], &20u32.to_be_bytes());
    }

    /// Proves tagcode 2 expands to the three TRC curves sharing one payload,
    /// which is how a grey or matrix-shaper profile with identical curves is
    /// coded.
    #[test]
    fn tagcode_two_expands_to_the_three_trc_tags() {
        // num_tags = 3, then tagcode 2 with an explicit size of 14.
        let (out, _) = run(&[0x04, 2 | 128, 14, 0x00], &[]);
        let start = 3u32 * 12 + 128;
        for (index, sig) in [b"rTRC", b"gTRC", b"bTRC"].into_iter().enumerate() {
            let base = 4 + index * 12;
            assert_eq!(&out[base..base + 4], sig);
            assert_eq!(&out[base + 4..base + 8], &start.to_be_bytes());
            assert_eq!(&out[base + 8..base + 12], &14u32.to_be_bytes());
        }
    }

    /// Proves tagcode 3 expands to three consecutive 20-byte XYZ payloads, the
    /// colourant matrix of a matrix-shaper profile.
    #[test]
    fn tagcode_three_expands_to_three_consecutive_colourants() {
        let (out, _) = run(&[0x04, 3, 0x00], &[]);
        let start = 3u32 * 12 + 128;
        for (index, sig) in [b"rXYZ", b"gXYZ", b"bXYZ"].into_iter().enumerate() {
            let base = 4 + index * 12;
            assert_eq!(&out[base..base + 4], sig);
            #[expect(clippy::cast_possible_truncation, reason = "index is 0..3")]
            let expected = start + 20 * index as u32;
            assert_eq!(&out[base + 4..base + 8], &expected.to_be_bytes());
            assert_eq!(&out[base + 8..base + 12], &20u32.to_be_bytes());
        }
    }

    /// Proves the two flag bits read explicit values from the *command* stream
    /// while a custom signature comes from the *data* stream, i.e. the two
    /// cursors advance independently.
    #[test]
    fn flags_take_explicit_start_and_size_and_a_custom_signature() {
        // num_tags = 1; tagcode 1 with both flags; tagstart 300, tagsize 7.
        let (out, _) = run(&[0x02, 1 | 64 | 128, 0xAC, 0x02, 7, 0x00], b"prvt");
        assert_eq!(&out[4..8], b"prvt");
        assert_eq!(&out[8..12], &300u32.to_be_bytes());
        assert_eq!(&out[12..16], &7u32.to_be_bytes());
    }

    /// Proves a tagcode the clause declares unreachable is rejected rather than
    /// silently producing an empty signature.
    #[test]
    fn an_unreachable_tagcode_is_rejected() {
        let mut c = ByteStream::new(&[0x02, 21], "command");
        let mut d = ByteStream::new(&[], "data");
        let mut out = Vec::new();
        assert!(decode_tag_list(&mut c, &mut d, 1 << 20, &mut out).is_err());
    }

    /// Proves the declared output size bounds the tag table too: one command
    /// byte emits up to 36 bytes, so an unbounded loop here would be an
    /// amplification bug.
    #[test]
    fn the_declared_output_size_bounds_the_tag_table() {
        let commands = [0x08u8, 3, 3, 3, 0x00];
        let mut c = ByteStream::new(&commands, "command");
        let mut d = ByteStream::new(&[], "data");
        let mut out = Vec::new();
        assert!(decode_tag_list(&mut c, &mut d, 16, &mut out).is_err());
    }
}
