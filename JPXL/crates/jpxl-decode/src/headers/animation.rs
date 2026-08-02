//! The `AnimationHeader` bundle (18181-1 D.3.4).
//!
//! ```text
//! Table D.6 — AnimationHeader bundle
//! condition   type                                 default   name
//!             U32(100, 1000, 1 + u(10), 1 + u(30)) 0         tps_numerator
//!             U32(1, 1001, 1 + u(8), 1 + u(10))    0         tps_denominator
//!             U32(0, u(3), u(16), u(32))           0         num_loops
//!             Bool()                               false     have_timecodes
//! ```
//!
//! Present only when `metadata.have_animation`. The two `tps` fields give ticks
//! per second as a rational; the shortcut values (100/1, 1000/1001) cover the
//! common cinema and NTSC-derived rates without spending bits.

use jpxl_bitstream::{BitReader, U32Dist, U32Spec, read_bool, read_u32, trace_field};

use crate::error::{DecodeError, Result};

/// 18181-1 D.6: `U32(100, 1000, 1 + u(10), 1 + u(30))`.
const TPS_NUMERATOR_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(100),
    U32Dist::Val(1000),
    U32Dist::BitsOffset {
        bits: 10,
        offset: 1,
    },
    U32Dist::BitsOffset {
        bits: 30,
        offset: 1,
    },
]);

/// 18181-1 D.6: `U32(1, 1001, 1 + u(8), 1 + u(10))`.
const TPS_DENOMINATOR_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(1),
    U32Dist::Val(1001),
    U32Dist::BitsOffset { bits: 8, offset: 1 },
    U32Dist::BitsOffset {
        bits: 10,
        offset: 1,
    },
]);

/// 18181-1 D.6: `U32(0, u(3), u(16), u(32))`.
const NUM_LOOPS_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::bits(3),
    U32Dist::bits(16),
    U32Dist::bits(32),
]);

/// A decoded `AnimationHeader` (18181-1 D.3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AnimationHeader {
    /// Numerator of the ticks-per-second rate.
    pub tps_numerator: u32,
    /// Denominator of the ticks-per-second rate.
    pub tps_denominator: u32,
    /// Times to repeat the animation; 0 means loop forever.
    pub num_loops: u32,
    /// Whether frame headers carry time codes.
    pub have_timecodes: bool,
}

impl AnimationHeader {
    /// Whether the animation loops indefinitely (`num_loops == 0`).
    #[must_use]
    pub const fn loops_forever(&self) -> bool {
        self.num_loops == 0
    }

    /// Ticks per second as an `f64`, or `None` if the denominator is zero.
    #[must_use]
    pub fn ticks_per_second(&self) -> Option<f64> {
        if self.tps_denominator == 0 {
            return None;
        }
        Some(f64::from(self.tps_numerator) / f64::from(self.tps_denominator))
    }
}

/// Reads an `AnimationHeader` bundle (18181-1 D.3.4).
///
/// # Errors
///
/// [`DecodeError::FieldOutOfRange`] if `tps_denominator` is zero, which would
/// make the tick rate undefined, or a bitstream error on truncation.
pub fn read_animation_header(reader: &mut BitReader<'_>) -> Result<AnimationHeader> {
    let tps_numerator = trace_field!(
        reader,
        "animation.tps_numerator",
        read_u32(reader, &TPS_NUMERATOR_SPEC)
    )?;
    let tps_denominator = trace_field!(
        reader,
        "animation.tps_denominator",
        read_u32(reader, &TPS_DENOMINATOR_SPEC)
    )?;
    let num_loops = trace_field!(
        reader,
        "animation.num_loops",
        read_u32(reader, &NUM_LOOPS_SPEC)
    )?;
    let have_timecodes = trace_field!(reader, "animation.have_timecodes", read_bool(reader))?;

    // D.3.4 defines the tick rate as tps_numerator / tps_denominator. Every
    // distribution for the denominator has a minimum of 1, so a zero here can
    // only come from a malformed stream; rejecting it keeps every consumer of
    // the rate free of a division guard.
    if tps_denominator == 0 {
        return Err(DecodeError::out_of_range("tps_denominator", "D.3.4", 0));
    }

    Ok(AnimationHeader {
        tps_numerator,
        tps_denominator,
        num_loops,
        have_timecodes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;

    fn read(bytes: &[u8]) -> Result<AnimationHeader> {
        let mut r = BitReader::new(bytes);
        read_animation_header(&mut r)
    }

    #[test]
    fn shortcut_values() {
        // tps 100/1, num_loops 0, no timecodes: all selector 0. 7 bits total.
        let mut w = BitWriter::new();
        w.u32_field(0, 0, 0)
            .u32_field(0, 0, 0)
            .u32_field(0, 0, 0)
            .bool(false);
        assert_eq!(w.bit_len(), 7);

        let anim = read(&w.finish_padded(1)).expect("valid");
        assert_eq!(anim.tps_numerator, 100);
        assert_eq!(anim.tps_denominator, 1);
        assert_eq!(anim.num_loops, 0);
        assert!(!anim.have_timecodes);
        assert!(anim.loops_forever());
        assert_eq!(anim.ticks_per_second(), Some(100.0));
    }

    #[test]
    fn ntsc_style_rate() {
        // 1000 / 1001 is the second shortcut of each field.
        let mut w = BitWriter::new();
        w.u32_field(1, 0, 0)
            .u32_field(1, 0, 0)
            .u32_field(0, 0, 0)
            .bool(true);
        let anim = read(&w.finish_padded(1)).expect("valid");
        assert_eq!(anim.tps_numerator, 1000);
        assert_eq!(anim.tps_denominator, 1001);
        assert!(anim.have_timecodes);
    }

    #[test]
    fn escape_distributions_and_loop_count() {
        // tps_numerator selector 2 => 1 + u(10) payload 23 => 24.
        // tps_denominator selector 2 => 1 + u(8) payload 0 => 1.
        // num_loops selector 1 => u(3) payload 5 => 5.
        let mut w = BitWriter::new();
        w.u32_field(2, 10, 23)
            .u32_field(2, 8, 0)
            .u32_field(1, 3, 5)
            .bool(false);
        let anim = read(&w.finish_padded(1)).expect("valid");
        assert_eq!(anim.tps_numerator, 24);
        assert_eq!(anim.tps_denominator, 1);
        assert_eq!(anim.num_loops, 5);
        assert!(!anim.loops_forever());
        assert_eq!(anim.ticks_per_second(), Some(24.0));
    }

    #[test]
    fn num_loops_full_width() {
        // num_loops selector 3 => u(32), spanning five bytes from a bit offset.
        let mut w = BitWriter::new();
        w.u32_field(0, 0, 0)
            .u32_field(0, 0, 0)
            .u32_field(3, 32, u32::MAX)
            .bool(false);
        let anim = read(&w.finish_padded(2)).expect("valid");
        assert_eq!(anim.num_loops, u32::MAX);
    }

    #[test]
    fn truncated_header_errors() {
        assert!(read(&[]).is_err());
    }
}
