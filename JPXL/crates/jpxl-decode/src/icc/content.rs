//! The ICC main content stage (18181-1 E.4.5).
//!
//! Everything after the tag table is coded as a short program: a command byte
//! from the command stream, sometimes a length and flags, and payload bytes
//! from the data stream. Five commands exist.
//!
//! * `1` — copy `num` bytes through unchanged.
//! * `2`, `3` — copy `num` bytes after de-interleaving them into 2 or 4 rows
//!   ([`shuffle`]). ICC payloads are arrays of fixed-width numbers, so putting
//!   all the high bytes together makes the entropy coder's job much easier.
//! * `4` — the same de-interleave, then an Nth-order predictor over the values
//!   already in the output, `stride` bytes back. This is what codes a tone
//!   curve: a smooth ramp becomes near-zero residuals.
//! * `10` — an `XYZ ` tag payload: the type signature, four reserved zeros, and
//!   twelve literal bytes.
//! * `16`..=`23` — a bare type signature from a dictionary plus four reserved
//!   zeros, which is the eight-byte preamble every ICC tag payload begins with.

use super::error::{Result, malformed};
use super::stream::ByteStream;

/// Type signatures E.4.5 emits for commands 16..=23.
const TYPE_DICTIONARY: [&[u8; 4]; 8] = [
    b"XYZ ", b"desc", b"text", b"mluc", b"para", b"curv", b"sf32", b"gbd ",
];

/// `Shuffle(bytes, width)` (18181-1 E.4.5).
///
/// The bytes are written row by row into a matrix of `width` rows and
/// `ceil(len / width)` columns, the short rows being the last ones, and read
/// back column by column. `Shuffle((1..=7), 2)` is `(1, 5, 2, 6, 3, 7, 4)`,
/// the example the clause gives.
///
/// This is a transpose, so it is its own inverse only when `len` is a multiple
/// of `width`; the clause defines the direction used here and no other.
#[must_use]
pub fn shuffle(bytes: &[u8], width: usize) -> Vec<u8> {
    if width <= 1 || bytes.len() <= 1 {
        return bytes.to_vec();
    }
    let len = bytes.len();
    let base = len / width;
    let remainder = len % width;
    // Row `r` holds `base + 1` bytes while `r < remainder`, `base` after that,
    // so the missing cells sit at the bottom of the last column.
    let row_len = |r: usize| base + usize::from(r < remainder);
    let mut starts = Vec::with_capacity(width);
    let mut acc = 0usize;
    for r in 0..width {
        starts.push(acc);
        acc += row_len(r);
    }

    let columns = base + usize::from(remainder > 0);
    let mut out = Vec::with_capacity(len);
    for c in 0..columns {
        for r in 0..width {
            if c < row_len(r)
                && let Some(&byte) = starts.get(r).and_then(|&s| bytes.get(s + c))
            {
                out.push(byte);
            }
        }
    }
    out
}

/// Decodes the main content (18181-1 E.4.5), appending it to `out`.
///
/// `output_size` is the declared final profile size; the output is never grown
/// past it, since E.4.2 states the result is exactly that many bytes.
///
/// # Errors
///
/// [`IccError::Malformed`](super::IccError::Malformed) for a command byte
/// E.4.5 says is unreachable, a predictor whose history is not yet in the
/// output, or either sub-stream ending mid-command.
pub fn decode_main_content(
    commands: &mut ByteStream<'_>,
    data: &mut ByteStream<'_>,
    output_size: u64,
    out: &mut Vec<u8>,
) -> Result<()> {
    while !commands.at_end() {
        let command = commands.u8()?;
        match command {
            1 => {
                let num = read_num(commands, output_size)?;
                let bytes = data.take(num)?;
                append(out, bytes, output_size)?;
            }
            2 | 3 => {
                let num = read_num(commands, output_size)?;
                let width = if command == 2 { 2 } else { 4 };
                let bytes = shuffle(data.take(num)?, width);
                append(out, &bytes, output_size)?;
            }
            4 => decode_predicted(commands, data, output_size, out)?,
            10 => {
                append(out, b"XYZ ", output_size)?;
                append(out, &[0, 0, 0, 0], output_size)?;
                let bytes = data.take(12)?;
                append(out, bytes, output_size)?;
            }
            16..=23 => {
                let signature = TYPE_DICTIONARY
                    .get(command as usize - 16)
                    .ok_or_else(|| malformed!("E.4.5: command {command} has no type signature"))?;
                append(out, *signature, output_size)?;
                append(out, &[0, 0, 0, 0], output_size)?;
            }
            _ => {
                return Err(malformed!(
                    "E.4.5: command {command} is in the branch the clause states is not reached"
                ));
            }
        }
    }
    Ok(())
}

/// Command 4: de-interleave, then run an Nth-order predictor over the output.
fn decode_predicted(
    commands: &mut ByteStream<'_>,
    data: &mut ByteStream<'_>,
    output_size: u64,
    out: &mut Vec<u8>,
) -> Result<()> {
    let flags = commands.u8()?;
    let width = usize::from(flags & 3) + 1;
    if width == 3 {
        return Err(malformed!("E.4.5: command 4 flags encode width 3"));
    }
    let order = usize::from(flags & 12) >> 2;
    if order == 3 {
        return Err(malformed!("E.4.5: command 4 flags encode order 3"));
    }
    let mut stride = width;
    if flags & 16 != 0 {
        stride = usize::try_from(commands.varint()?)
            .map_err(|_| malformed!("E.4.5: stride does not fit in memory"))?;
    }
    if stride < width {
        return Err(malformed!(
            "E.4.5: stride {stride} is smaller than width {width}"
        ));
    }

    let num = read_num(commands, output_size)?;
    let raw = data.take(num)?;
    let bytes = if width == 2 || width == 4 {
        shuffle(raw, width)
    } else {
        raw.to_vec()
    };

    let mask: u32 = if width >= 4 {
        u32::MAX
    } else {
        (1u32 << (8 * width)) - 1
    };
    let order_terms = order + 1;

    let mut i = 0usize;
    while i < num {
        // The history is read afresh each round, from the output as it now
        // stands: the previous round's bytes are part of it.
        let mut prev = [0u32; 3];
        for (j, slot) in prev.iter_mut().take(order_terms).enumerate() {
            let back = stride
                .checked_mul(j + 1)
                .ok_or_else(|| malformed!("E.4.5: predictor history offset overflows"))?;
            let start = out.len().checked_sub(back).ok_or_else(|| {
                malformed!(
                    "E.4.5: predictor reads {back} bytes before the start of a {}-byte output",
                    out.len()
                )
            })?;
            let window = out
                .get(start..start + width)
                .ok_or_else(|| malformed!("E.4.5: predictor history runs past the output"))?;
            let mut value = 0u32;
            for &byte in window {
                value = (value << 8) | u32::from(byte);
            }
            *slot = value;
        }

        let (p0, p1, p2) = (prev[0], prev[1], prev[2]);
        let predicted = match order {
            0 => p0,
            1 => p0.wrapping_mul(2).wrapping_sub(p1),
            _ => p0
                .wrapping_mul(3)
                .wrapping_sub(p1.wrapping_mul(3))
                .wrapping_add(p2),
        } & mask;

        for j in 0..width {
            if i + j >= num {
                break;
            }
            let residual = bytes
                .get(i + j)
                .copied()
                .ok_or_else(|| malformed!("E.4.5: predictor ran past its payload"))?;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the mask keeps only the low 8 bits, which is what & 255 means"
            )]
            let byte = (residual.wrapping_add((predicted >> (8 * (width - 1 - j))) as u8)) as u8;
            append(out, &[byte], output_size)?;
        }
        i += width;
    }
    Ok(())
}

/// Reads a `num` operand, rejecting the zero E.4.5 excludes and anything that
/// could not fit in the declared profile.
fn read_num(commands: &mut ByteStream<'_>, output_size: u64) -> Result<usize> {
    let num = commands.varint()?;
    if num == 0 {
        return Err(malformed!("E.4.5: a command length of 0 is not permitted"));
    }
    if num > output_size {
        return Err(malformed!(
            "E.4.5: command length {num} exceeds the declared profile size {output_size}"
        ));
    }
    usize::try_from(num).map_err(|_| malformed!("E.4.5: command length {num} does not fit memory"))
}

/// Appends `bytes`, refusing to grow the output past `output_size`.
fn append(out: &mut Vec<u8>, bytes: &[u8], output_size: u64) -> Result<()> {
    if out.len() as u64 + bytes.len() as u64 > output_size {
        return Err(malformed!(
            "E.4.2: the decoded profile grew past the declared output_size of {output_size}"
        ));
    }
    out.extend_from_slice(bytes);
    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "hand-written spec vectors read better with direct indexing; a panic in a test is \
              a failing test"
)]
mod tests {
    use super::*;

    fn run(commands: &[u8], data: &[u8], seed: &[u8]) -> Vec<u8> {
        let mut c = ByteStream::new(commands, "command");
        let mut d = ByteStream::new(data, "data");
        let mut out = seed.to_vec();
        decode_main_content(&mut c, &mut d, 1 << 20, &mut out).expect("valid");
        out
    }

    /// The worked example printed in E.4.5.
    #[test]
    fn shuffle_matches_the_clause_example() {
        assert_eq!(
            shuffle(&[1, 2, 3, 4, 5, 6, 7], 2),
            vec![1, 5, 2, 6, 3, 7, 4]
        );
    }

    /// Proves the transpose for an exact multiple of the width, where the
    /// matrix has no missing cells.
    #[test]
    fn shuffle_transposes_an_exact_multiple() {
        assert_eq!(
            shuffle(&[1, 2, 3, 4, 5, 6, 7, 8], 4),
            vec![1, 3, 5, 7, 2, 4, 6, 8]
        );
    }

    /// Proves the missing cells are taken off the *bottom of the last column*
    /// and not off the end of the input, which is the one part of the clause's
    /// wording a naive transpose gets wrong.
    #[test]
    fn shuffle_puts_short_rows_last() {
        // 4 rows of lengths 2, 2, 1, 1 for 6 bytes.
        assert_eq!(shuffle(&[1, 2, 3, 4, 5, 6], 4), vec![1, 3, 5, 6, 2, 4]);
        assert_eq!(shuffle(&[1], 4), vec![1]);
        assert_eq!(shuffle(&[], 2), Vec::<u8>::new());
    }

    /// Proves command 1 is a plain copy from the data stream.
    #[test]
    fn command_one_copies_literally() {
        assert_eq!(run(&[1, 3], &[9, 8, 7], &[]), vec![9, 8, 7]);
    }

    /// Proves commands 2 and 3 shuffle with widths 2 and 4 respectively.
    #[test]
    fn commands_two_and_three_shuffle() {
        assert_eq!(run(&[2, 4], &[1, 2, 3, 4], &[]), vec![1, 3, 2, 4]);
        assert_eq!(
            run(&[3, 8], &[1, 2, 3, 4, 5, 6, 7, 8], &[]),
            vec![1, 3, 5, 7, 2, 4, 6, 8]
        );
    }

    /// Proves command 10 emits the `XYZ ` type preamble plus twelve literals.
    #[test]
    fn command_ten_emits_an_xyz_payload() {
        let out = run(&[10], &[1; 12], &[]);
        assert_eq!(&out[0..4], b"XYZ ");
        assert_eq!(&out[4..8], &[0, 0, 0, 0]);
        assert_eq!(&out[8..20], &[1u8; 12]);
        assert_eq!(out.len(), 20);
    }

    /// Proves the type-signature dictionary of commands 16..=23.
    #[test]
    fn commands_sixteen_to_twentythree_emit_type_signatures() {
        let out = run(&[16, 21], &[], &[]);
        assert_eq!(&out[0..4], b"XYZ ");
        assert_eq!(&out[8..12], b"curv");
        assert_eq!(&out[12..16], &[0, 0, 0, 0]);
    }

    /// Proves the order-0 predictor with width 1 is a running delta against the
    /// byte `stride` back, and that the history includes bytes produced by this
    /// very command.
    #[test]
    fn order_zero_width_one_is_a_running_delta() {
        // flags: width 1 (0), order 0 (0), no explicit stride.
        let out = run(&[4, 0b0000_0000, 4], &[1, 1, 1, 1], &[10]);
        assert_eq!(out, vec![10, 11, 12, 13, 14]);
    }

    /// Proves the order-1 predictor extrapolates linearly, which is what makes a
    /// uniformly increasing tone curve code as zeros.
    #[test]
    fn order_one_extrapolates_a_linear_ramp() {
        // width 1, order 1 (bits 2..3 = 01), stride 1 -> p = 2*prev0 - prev1.
        let out = run(&[4, 0b0000_0100, 3], &[0, 0, 0], &[10, 12]);
        assert_eq!(out, vec![10, 12, 14, 16, 18]);
    }

    /// Proves the order-2 predictor extrapolates a quadratic, and that
    /// arithmetic happens on `width`-byte integers rather than per byte: with
    /// width 2 the low byte carries into the high one.
    #[test]
    fn order_two_works_on_width_two_integers() {
        // Seed 0x00FE, 0x00FF, 0x0100 (a ramp crossing a byte boundary).
        // width 2 (bits 0..1 = 01), order 2 (bits 2..3 = 10) -> flags 0b1001.
        // Payload of 2 zero residuals after shuffling; predicted value is
        // 3*0x0100 - 3*0x00FF + 0x00FE = 0x0101.
        let seed = [0x00, 0xFE, 0x00, 0xFF, 0x01, 0x00];
        let out = run(&[4, 0b0000_1001, 2], &[0, 0], &seed);
        assert_eq!(&out[6..8], &[0x01, 0x01]);
    }

    /// Proves an explicit stride is read from the command stream and used as
    /// the history distance, which is how interleaved records (an ICC LUT row,
    /// say) are predicted against the previous record.
    #[test]
    fn an_explicit_stride_selects_the_history_distance() {
        // width 1, order 0, flags bit 4 set -> stride follows as a varint.
        let out = run(&[4, 0b0001_0000, 3, 2], &[0, 0], &[5, 7, 9]);
        // Each output byte repeats the byte 3 positions back.
        assert_eq!(out, vec![5, 7, 9, 5, 7]);
    }

    /// Proves the predictor refuses to read history that does not exist rather
    /// than reading zeros or wrapping around the buffer.
    #[test]
    fn a_predictor_without_history_is_rejected() {
        let mut c = ByteStream::new(&[4, 0b0000_0000, 2], "command");
        let mut d = ByteStream::new(&[0, 0], "data");
        let mut out = Vec::new();
        assert!(decode_main_content(&mut c, &mut d, 1 << 20, &mut out).is_err());
    }

    /// Proves the two flag encodings the clause excludes are rejected.
    #[test]
    fn width_three_and_order_three_are_rejected() {
        for flags in [0b0000_0010u8, 0b0000_1100] {
            let program = [4, flags, 1];
            let mut c = ByteStream::new(&program, "command");
            let mut d = ByteStream::new(&[0], "data");
            let mut out = vec![0u8; 64];
            assert!(decode_main_content(&mut c, &mut d, 1 << 20, &mut out).is_err());
        }
    }

    /// Proves an unreachable command byte is an error, not a silent skip.
    #[test]
    fn an_unreachable_command_is_rejected() {
        let mut c = ByteStream::new(&[7], "command");
        let mut d = ByteStream::new(&[], "data");
        let mut out = Vec::new();
        assert!(decode_main_content(&mut c, &mut d, 1 << 20, &mut out).is_err());
    }

    /// Proves the declared output size is a hard bound on what the commands may
    /// produce, so a malicious command stream cannot inflate the allocation
    /// past what was metered.
    #[test]
    fn the_declared_output_size_bounds_the_result() {
        let mut c = ByteStream::new(&[1, 8], "command");
        let mut d = ByteStream::new(&[0u8; 8], "data");
        let mut out = Vec::new();
        assert!(decode_main_content(&mut c, &mut d, 4, &mut out).is_err());
    }
}
