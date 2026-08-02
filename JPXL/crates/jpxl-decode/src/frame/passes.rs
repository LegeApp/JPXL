//! The `Passes` bundle (18181-1 F.2, Table F.6).
//!
//! ```text
//! condition          type                  default  name
//!                    U32(1, 2, 3, 4+u(3))  1        num_passes
//! num_passes != 1    U32(0, 1, 2, 3+u(1))  0        num_ds
//! num_passes != 1    u(2)                  0        shift[num_passes - 1]
//! num_passes != 1    U32(1, 2, 4, 8)       1        downsample[num_ds]
//! num_passes != 1    U32(0, 1, 2, u(3))    0        last_pass[num_ds]
//! ```
//!
//! A progressive frame is split into passes; `(downsample, last_pass)` pairs
//! say which prefix of the passes suffices to display the image at a given
//! downsampling factor. F.2 states the constraints that make the pairs usable:
//! `num_ds < num_passes`, `downsample` strictly decreasing, `last_pass`
//! strictly increasing, and every `last_pass[i] <= num_passes - 1`.
//!
//! The decoder also behaves as if a final implicit pair `(1, num_passes - 1)`
//! were present; [`Passes::pairs_with_implicit_final`] materialises it.

use jpxl_bitstream::{BitReader, U32Dist, U32Spec, read_u32, trace_field};

use crate::frame::error::{FrameError, Result};

/// 18181-1 F.6: `U32(1, 2, 3, 4 + u(3))`.
const NUM_PASSES_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(1),
    U32Dist::Val(2),
    U32Dist::Val(3),
    U32Dist::BitsOffset { bits: 3, offset: 4 },
]);

/// 18181-1 F.6: `U32(0, 1, 2, 3 + u(1))`.
const NUM_DS_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::Val(2),
    U32Dist::BitsOffset { bits: 1, offset: 3 },
]);

/// 18181-1 F.6: `U32(1, 2, 4, 8)`.
const DOWNSAMPLE_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(1),
    U32Dist::Val(2),
    U32Dist::Val(4),
    U32Dist::Val(8),
]);

/// 18181-1 F.6: `U32(0, 1, 2, u(3))`.
const LAST_PASS_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::Val(2),
    U32Dist::bits(3),
]);

/// Largest `num_ds` the clause permits: "between 0 and 4, inclusive".
pub const MAX_NUM_DS: u32 = 4;

/// A decoded `Passes` bundle (18181-1 Table F.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Passes {
    /// Number of passes the frame is partitioned into; at least 1.
    pub num_passes: u32,
    /// Left-shift applied to each pass's HF coefficients after entropy
    /// decoding. Length is `num_passes - 1`; the last pass shifts by 0.
    pub shift: Vec<u32>,
    /// Downsampling factors, strictly decreasing. Length is `num_ds`.
    pub downsample: Vec<u32>,
    /// Final pass index per downsampling factor, strictly increasing.
    pub last_pass: Vec<u32>,
}

impl Default for Passes {
    /// The Table F.6 defaults: one pass, nothing else stored.
    fn default() -> Self {
        Self {
            num_passes: 1,
            shift: Vec::new(),
            downsample: Vec::new(),
            last_pass: Vec::new(),
        }
    }
}

impl Passes {
    /// Number of `(downsample, last_pass)` pairs stored in the codestream.
    #[must_use]
    pub fn num_ds(&self) -> usize {
        self.downsample.len()
    }

    /// The stored pairs plus the implicit final `(1, num_passes - 1)` pair
    /// that F.2 says the decoder behaves as if it had read.
    #[must_use]
    pub fn pairs_with_implicit_final(&self) -> Vec<(u32, u32)> {
        let mut pairs: Vec<(u32, u32)> = self
            .downsample
            .iter()
            .copied()
            .zip(self.last_pass.iter().copied())
            .collect();
        pairs.push((1, self.num_passes - 1));
        pairs
    }

    /// The shift applied to pass `index`, treating the final pass as 0.
    #[must_use]
    pub fn shift_for(&self, index: u32) -> u32 {
        usize::try_from(index)
            .ok()
            .and_then(|i| self.shift.get(i).copied())
            .unwrap_or(0)
    }
}

/// Reads a `Passes` bundle (18181-1 Table F.6).
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] if `num_ds` is above [`MAX_NUM_DS`] or not
/// strictly less than `num_passes`, if a `last_pass` exceeds `num_passes - 1`,
/// or if the monotonicity constraints on `downsample`/`last_pass` are broken.
pub fn read_passes(reader: &mut BitReader<'_>) -> Result<Passes> {
    let num_passes = trace_field!(
        reader,
        "passes.num_passes",
        read_u32(reader, &NUM_PASSES_SPEC)
    )?;
    if num_passes == 0 {
        return Err(FrameError::out_of_range("num_passes", "F.6", 0));
    }
    if num_passes == 1 {
        return Ok(Passes::default());
    }

    let num_ds = trace_field!(reader, "passes.num_ds", read_u32(reader, &NUM_DS_SPEC))?;
    if num_ds > MAX_NUM_DS {
        return Err(FrameError::out_of_range("num_ds", "F.6", u64::from(num_ds)));
    }
    // "It is strictly smaller than num_passes."
    if num_ds >= num_passes {
        return Err(FrameError::out_of_range("num_ds", "F.6", u64::from(num_ds)));
    }

    // Both loop counts are bounded by the clause (num_passes <= 11, num_ds
    // <= 4), so no allocation metering is needed.
    let mut shift = Vec::with_capacity((num_passes - 1) as usize);
    for _ in 0..(num_passes - 1) {
        shift.push(trace_field!(reader, "passes.shift", reader.read_bits(2))?);
    }

    let mut downsample = Vec::with_capacity(num_ds as usize);
    for _ in 0..num_ds {
        downsample.push(trace_field!(
            reader,
            "passes.downsample",
            read_u32(reader, &DOWNSAMPLE_SPEC)
        )?);
    }

    let mut last_pass = Vec::with_capacity(num_ds as usize);
    for _ in 0..num_ds {
        last_pass.push(trace_field!(
            reader,
            "passes.last_pass",
            read_u32(reader, &LAST_PASS_SPEC)
        )?);
    }

    let passes = Passes {
        num_passes,
        shift,
        downsample,
        last_pass,
    };
    validate(&passes)?;
    Ok(passes)
}

/// Enforces the F.2 constraints on the decoded pairs.
fn validate(passes: &Passes) -> Result<()> {
    for (i, value) in passes.last_pass.iter().enumerate() {
        if *value > passes.num_passes - 1 {
            return Err(FrameError::out_of_range(
                "last_pass",
                "F.2",
                u64::from(*value),
            ));
        }
        if i > 0 {
            // "The sequence last_pass is strictly increasing."
            let previous = passes.last_pass.get(i - 1).copied().unwrap_or(0);
            if *value <= previous {
                return Err(FrameError::out_of_range(
                    "last_pass",
                    "F.2",
                    u64::from(*value),
                ));
            }
        }
    }
    for (i, value) in passes.downsample.iter().enumerate() {
        if i > 0 {
            // "The sequence downsample is strictly decreasing."
            let previous = passes.downsample.get(i - 1).copied().unwrap_or(0);
            if *value >= previous {
                return Err(FrameError::out_of_range(
                    "downsample",
                    "F.2",
                    u64::from(*value),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;

    fn read(bytes: &[u8]) -> Result<Passes> {
        let mut r = BitReader::new(bytes);
        read_passes(&mut r)
    }

    #[test]
    fn single_pass_is_two_bits() {
        // num_passes = U32 selector 0 => 1, and nothing else is read.
        let mut w = BitWriter::new();
        w.u32_field(0, 0, 0);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let passes = read_passes(&mut r).expect("valid");

        assert_eq!(r.total_bits_read(), 2);
        assert_eq!(passes, Passes::default());
        assert_eq!(passes.num_passes, 1);
        assert_eq!(passes.shift_for(0), 0);
    }

    #[test]
    fn two_passes_no_downsampling_pairs() {
        // num_passes = 2 (selector 1), num_ds = 0 (selector 0),
        // shift[0] = u(2) = 1.
        let mut w = BitWriter::new();
        w.u32_field(1, 0, 0).u32_field(0, 0, 0).u(2, 1);
        let passes = read(&w.finish_padded(1)).expect("valid");

        assert_eq!(passes.num_passes, 2);
        assert_eq!(passes.num_ds(), 0);
        assert_eq!(passes.shift, vec![1]);
        assert_eq!(passes.shift_for(0), 1);
        assert_eq!(passes.shift_for(1), 0, "the last pass shifts by 0");
    }

    #[test]
    fn escape_distribution_reaches_eleven_passes() {
        // num_passes selector 3 => 4 + u(3); payload 7 => 11, the maximum.
        let mut w = BitWriter::new();
        w.u32_field(3, 3, 7).u32_field(0, 0, 0);
        for _ in 0..10 {
            w.u(2, 0);
        }
        let passes = read(&w.finish_padded(1)).expect("valid");
        assert_eq!(passes.num_passes, 11);
        assert_eq!(passes.shift.len(), 10, "num_passes - 1 shift entries");
    }

    #[test]
    fn downsample_and_last_pass_pairs() {
        // num_passes = 3, num_ds = 2, shifts, then two pairs:
        // downsample = [8, 2] (strictly decreasing),
        // last_pass  = [0, 1] (strictly increasing).
        let mut w = BitWriter::new();
        w.u32_field(2, 0, 0) // num_passes = 3
            .u32_field(2, 0, 0) // num_ds = 2
            .u(2, 0)
            .u(2, 1) // shift[0..2]
            .u32_field(3, 0, 0) // downsample[0] = 8
            .u32_field(1, 0, 0) // downsample[1] = 2
            .u32_field(0, 0, 0) // last_pass[0] = 0
            .u32_field(1, 0, 0); // last_pass[1] = 1
        let passes = read(&w.finish_padded(1)).expect("valid");

        assert_eq!(passes.num_passes, 3);
        assert_eq!(passes.downsample, vec![8, 2]);
        assert_eq!(passes.last_pass, vec![0, 1]);
        assert_eq!(
            passes.pairs_with_implicit_final(),
            vec![(8, 0), (2, 1), (1, 2)],
            "F.2 adds an implicit final pair (1, num_passes - 1)"
        );
    }

    #[test]
    fn num_ds_must_be_below_num_passes() {
        // num_passes = 2 but num_ds = 2 violates "strictly smaller".
        let mut w = BitWriter::new();
        w.u32_field(1, 0, 0).u32_field(2, 0, 0).u(2, 0);
        assert!(read(&w.finish_padded(1)).is_err());
    }

    #[test]
    fn last_pass_above_num_passes_minus_one_rejected() {
        // num_passes = 2, num_ds = 1, last_pass[0] = u(3) = 5 > 1.
        let mut w = BitWriter::new();
        w.u32_field(1, 0, 0)
            .u32_field(1, 0, 0)
            .u(2, 0)
            .u32_field(0, 0, 0) // downsample[0] = 1
            .u32_field(3, 3, 5); // last_pass[0] = 5
        assert!(read(&w.finish_padded(1)).is_err());
    }

    #[test]
    fn non_monotonic_sequences_rejected() {
        // downsample = [2, 8] is increasing, which the clause forbids.
        let mut w = BitWriter::new();
        w.u32_field(2, 0, 0) // num_passes = 3
            .u32_field(2, 0, 0) // num_ds = 2
            .u(2, 0)
            .u(2, 0)
            .u32_field(1, 0, 0) // downsample[0] = 2
            .u32_field(3, 0, 0) // downsample[1] = 8  <- not decreasing
            .u32_field(0, 0, 0)
            .u32_field(1, 0, 0);
        assert!(read(&w.finish_padded(1)).is_err());

        // last_pass = [1, 0] is decreasing, which the clause forbids.
        let mut w = BitWriter::new();
        w.u32_field(2, 0, 0)
            .u32_field(2, 0, 0)
            .u(2, 0)
            .u(2, 0)
            .u32_field(3, 0, 0)
            .u32_field(1, 0, 0)
            .u32_field(1, 0, 0) // last_pass[0] = 1
            .u32_field(0, 0, 0); // last_pass[1] = 0 <- not increasing
        assert!(read(&w.finish_padded(1)).is_err());
    }

    #[test]
    fn num_ds_above_four_rejected() {
        // num_ds selector 3 => 3 + u(1); payload 1 => 4, which is allowed,
        // but only when num_passes exceeds it.
        let mut w = BitWriter::new();
        w.u32_field(3, 3, 7) // num_passes = 11
            .u32_field(3, 1, 1); // num_ds = 4
        for _ in 0..10 {
            w.u(2, 0);
        }
        for v in [3u32, 2, 1, 0] {
            w.u32_field(v, 0, 0); // downsample 8, 4, 2, 1
        }
        for v in [0u32, 1, 2, 3] {
            w.u32_field(if v == 3 { 3 } else { v }, if v == 3 { 3 } else { 0 }, v);
        }
        let passes = read(&w.finish_padded(2)).expect("num_ds = 4 is the maximum");
        assert_eq!(passes.num_ds(), 4);
        assert_eq!(passes.downsample, vec![8, 4, 2, 1]);
    }

    #[test]
    fn truncated_bundle_errors() {
        assert!(read(&[]).is_err());
    }
}
