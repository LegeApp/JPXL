//! LF dequantization weights, the quantizer, and the LF correlation factors
//! (18181-1 G.1.2, I.2.1 and I.2.3).
//!
//! These are the three scalar bundles of `LfGlobal` that describe how the LF
//! plane is scaled and decorrelated. They are grouped in one module because
//! they are read consecutively and because I.2.1's multipliers cannot be
//! computed without G.1.2's weights.
//!
//! ```text
//! Table G.2 — LfChannelDequantization bundle
//! condition      type    default   name
//!                Bool()  true      all_default
//! !all_default   F16()   1/32      m_x_lf
//! !all_default   F16()   1/4       m_y_lf
//! !all_default   F16()   1/2       m_b_lf
//!
//! Table I.2 — Quantizer bundle
//! U32(1 + u(11), 2049 + u(11), 4097 + u(12), 8193 + u(16))   global_scale
//! U32(16, 1 + u(5), 1 + u(8), 1 + u(16))                     quant_lf
//!
//! Table I.3 — LfChannelCorrelation bundle
//! condition      type                                default  name
//!                Bool()                              true     all_default
//! !all_default   U32(84, 256, 2 + u(8), 258 + u(16)) 84       colour_factor
//! !all_default   F16()                               0.0      base_correlation_x
//! !all_default   F16()                               1.0      base_correlation_b
//! !all_default   u(8)                                128      x_factor_lf
//! !all_default   u(8)                                128      b_factor_lf
//! ```
//!
//! # Why the multipliers are a newtype
//!
//! I.2.1 defines `mXDC`, `mYDC` and `mBDC` as the values a *quantized* LF
//! coefficient is multiplied by (I.5.2: `dX = mXDC * qx / (1 << extra_precision)`).
//! They are dequantization multipliers, not quantization weights, and the two
//! are reciprocals of each other. [`LfDequantMultipliers`] names which one it
//! is so that the confusion cannot survive a type check.

use jpxl_bitstream::trace_field;
use jpxl_bitstream::{BitReader, U32Dist, U32Spec, read_bool, read_f16_as_f32, read_u32};
use jpxl_core::limits::AllocGuard;

use crate::error::{DecodeError, Result};

use super::block_ctx::{HfBlockContext, read_hf_block_context};

// ---------------------------------------------------------------------------
// G.1.2 — LF dequantization weights
// ---------------------------------------------------------------------------

/// Table G.2 default `m_x_lf`.
pub const DEFAULT_M_X_LF: f32 = 1.0 / 32.0;
/// Table G.2 default `m_y_lf`.
pub const DEFAULT_M_Y_LF: f32 = 1.0 / 4.0;
/// Table G.2 default `m_b_lf`.
pub const DEFAULT_M_B_LF: f32 = 1.0 / 2.0;

/// The divisor G.1.2 applies to each weight to obtain its `_unscaled` form.
pub const LF_WEIGHT_SCALE: f32 = 128.0;

/// A decoded `LfChannelDequantization` bundle (18181-1 G.1.2).
///
/// The fields are the weights exactly as they appear in Table G.2; G.1.2's
/// `m_*_lf_unscaled` values are [`LfChannelDequantization::unscaled`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LfChannelDequantization {
    /// `m_x_lf`.
    pub m_x_lf: f32,
    /// `m_y_lf`.
    pub m_y_lf: f32,
    /// `m_b_lf`.
    pub m_b_lf: f32,
}

impl Default for LfChannelDequantization {
    fn default() -> Self {
        Self {
            m_x_lf: DEFAULT_M_X_LF,
            m_y_lf: DEFAULT_M_Y_LF,
            m_b_lf: DEFAULT_M_B_LF,
        }
    }
}

impl LfChannelDequantization {
    /// G.1.2's `m_x_lf_unscaled`, `m_y_lf_unscaled`, `m_b_lf_unscaled`:
    /// each weight divided by 128, in X, Y, B order.
    #[must_use]
    pub fn unscaled(&self) -> [f32; 3] {
        [
            self.m_x_lf / LF_WEIGHT_SCALE,
            self.m_y_lf / LF_WEIGHT_SCALE,
            self.m_b_lf / LF_WEIGHT_SCALE,
        ]
    }
}

/// Reads an `LfChannelDequantization` bundle (18181-1 G.1.2).
///
/// This row of Table G.1 has **no condition**: it is present for modular
/// frames too, where the weights are then unused. `decode.rs` currently reads
/// and discards the same four fields inline; this function is the typed
/// replacement for that read.
///
/// # Errors
///
/// [`DecodeError::Bitstream`] on truncation or an invalid `F16()`.
pub fn read_lf_channel_dequantization(
    reader: &mut BitReader<'_>,
) -> Result<LfChannelDequantization> {
    let all_default = trace_field!(reader, "lf_dequant.all_default", read_bool(reader))?;
    if all_default {
        return Ok(LfChannelDequantization::default());
    }
    let m_x_lf = trace_field!(reader, "lf_dequant.m_x_lf", read_f16_as_f32(reader))?;
    let m_y_lf = trace_field!(reader, "lf_dequant.m_y_lf", read_f16_as_f32(reader))?;
    let m_b_lf = trace_field!(reader, "lf_dequant.m_b_lf", read_f16_as_f32(reader))?;
    Ok(LfChannelDequantization {
        m_x_lf,
        m_y_lf,
        m_b_lf,
    })
}

// ---------------------------------------------------------------------------
// I.2.1 — Quantizer
// ---------------------------------------------------------------------------

/// `U32(1 + u(11), 2049 + u(11), 4097 + u(12), 8193 + u(16))` (Table I.2).
const GLOBAL_SCALE_SPEC: U32Spec = U32Spec::new([
    U32Dist::BitsOffset {
        bits: 11,
        offset: 1,
    },
    U32Dist::BitsOffset {
        bits: 11,
        offset: 2049,
    },
    U32Dist::BitsOffset {
        bits: 12,
        offset: 4097,
    },
    U32Dist::BitsOffset {
        bits: 16,
        offset: 8193,
    },
]);

/// `U32(16, 1 + u(5), 1 + u(8), 1 + u(16))` (Table I.2).
const QUANT_LF_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(16),
    U32Dist::BitsOffset { bits: 5, offset: 1 },
    U32Dist::BitsOffset { bits: 8, offset: 1 },
    U32Dist::BitsOffset {
        bits: 16,
        offset: 1,
    },
]);

/// The numerator `1 << 16` shared by I.2.1's and I.5.3's multipliers.
const QUANT_NUMERATOR: f64 = 65536.0;

/// A decoded `Quantizer` bundle (18181-1 I.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quantizer {
    /// `global_scale`, at least 1 by construction of Table I.2.
    pub global_scale: u32,
    /// `quant_lf`, at least 1 by construction of Table I.2.
    pub quant_lf: u32,
}

/// I.2.1's `mXDC`, `mYDC` and `mBDC`, in X, Y, B order.
///
/// Multiply a **quantized** LF coefficient by these to dequantize it (I.5.2).
/// They are not quantization weights; see the module documentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LfDequantMultipliers([f32; 3]);

impl LfDequantMultipliers {
    /// `mXDC`.
    #[must_use]
    pub const fn x(&self) -> f32 {
        self.0[0]
    }

    /// `mYDC`.
    #[must_use]
    pub const fn y(&self) -> f32 {
        self.0[1]
    }

    /// `mBDC`.
    #[must_use]
    pub const fn b(&self) -> f32 {
        self.0[2]
    }

    /// The multiplier for channel `c` in Table I.1's X, Y, B numbering.
    ///
    /// # Errors
    ///
    /// [`DecodeError::FieldOutOfRange`] if `c > 2`.
    pub fn channel(&self, c: usize) -> Result<f32> {
        self.0.get(c).copied().ok_or_else(|| {
            DecodeError::out_of_range("channel", "I.2.1", u64::try_from(c).unwrap_or(u64::MAX))
        })
    }

    /// All three multipliers in X, Y, B order.
    #[must_use]
    pub const fn as_array(&self) -> [f32; 3] {
        self.0
    }
}

impl Quantizer {
    /// I.2.1's LF dequantization multipliers.
    ///
    /// `mXDC = (1 << 16) * m_x_lf_unscaled / (global_scale * quant_lf)`, and
    /// likewise for Y and B.
    ///
    /// Both denominators are at least 1 for every selector of Table I.2 —
    /// `global_scale` starts at `1 + u(11)` and `quant_lf` at the constant 16
    /// or `1 + u(k)` — so this cannot divide by zero. The arithmetic is done in
    /// `f64` because `global_scale * quant_lf` reaches `73728 * 65536`, which
    /// is exact in `f64` and not in `f32`.
    // The only cast is the deliberate f64 -> f32 narrowing at the end of the
    // computation: the multipliers are consumed by the f32 sample pipeline, and
    // the intermediate is f64 precisely so the narrowing happens once.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn lf_multipliers(&self, weights: &LfChannelDequantization) -> LfDequantMultipliers {
        let denom = f64::from(self.global_scale) * f64::from(self.quant_lf);
        let unscaled = weights.unscaled();
        let mut out = [0.0f32; 3];
        for (slot, w) in out.iter_mut().zip(unscaled) {
            *slot = (QUANT_NUMERATOR * f64::from(w) / denom) as f32;
        }
        LfDequantMultipliers(out)
    }
}

/// Reads a `Quantizer` bundle (18181-1 I.2.1).
///
/// # Errors
///
/// [`DecodeError::Bitstream`] on truncation.
pub fn read_quantizer(reader: &mut BitReader<'_>) -> Result<Quantizer> {
    let global_scale = trace_field!(
        reader,
        "quantizer.global_scale",
        read_u32(reader, &GLOBAL_SCALE_SPEC)
    )?;
    let quant_lf = trace_field!(
        reader,
        "quantizer.quant_lf",
        read_u32(reader, &QUANT_LF_SPEC)
    )?;
    Ok(Quantizer {
        global_scale,
        quant_lf,
    })
}

// ---------------------------------------------------------------------------
// I.2.3 — LF channel correlation factors
// ---------------------------------------------------------------------------

/// `U32(84, 256, 2 + u(8), 258 + u(16))` (Table I.3).
const COLOUR_FACTOR_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(84),
    U32Dist::Val(256),
    U32Dist::BitsOffset { bits: 8, offset: 2 },
    U32Dist::BitsOffset {
        bits: 16,
        offset: 258,
    },
]);

/// Table I.3 default `colour_factor`.
pub const DEFAULT_COLOUR_FACTOR: u32 = 84;
/// Table I.3 default `base_correlation_x`.
pub const DEFAULT_BASE_CORRELATION_X: f32 = 0.0;
/// Table I.3 default `base_correlation_b`.
pub const DEFAULT_BASE_CORRELATION_B: f32 = 1.0;
/// Table I.3 default `x_factor_lf` and `b_factor_lf`.
pub const DEFAULT_FACTOR_LF: u8 = 128;

/// A decoded `LfChannelCorrelation` bundle (18181-1 I.2.3).
///
/// I.6 turns these into the chroma-from-luma factors
/// `kX = base_correlation_x + x_factor / colour_factor` (and likewise for B),
/// where the LF `x_factor` is `x_factor_lf - 128`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LfChannelCorrelation {
    /// `colour_factor`, the divisor of I.6. At least 2 for every selector.
    pub colour_factor: u32,
    /// `base_correlation_x`.
    pub base_correlation_x: f32,
    /// `base_correlation_b`.
    pub base_correlation_b: f32,
    /// `x_factor_lf`, biased by 128.
    pub x_factor_lf: u8,
    /// `b_factor_lf`, biased by 128.
    pub b_factor_lf: u8,
}

impl Default for LfChannelCorrelation {
    fn default() -> Self {
        Self {
            colour_factor: DEFAULT_COLOUR_FACTOR,
            base_correlation_x: DEFAULT_BASE_CORRELATION_X,
            base_correlation_b: DEFAULT_BASE_CORRELATION_B,
            x_factor_lf: DEFAULT_FACTOR_LF,
            b_factor_lf: DEFAULT_FACTOR_LF,
        }
    }
}

impl LfChannelCorrelation {
    /// I.6's LF `x_factor`: `x_factor_lf - 128`.
    #[must_use]
    pub const fn x_factor(&self) -> i32 {
        self.x_factor_lf as i32 - 128
    }

    /// I.6's LF `b_factor`: `b_factor_lf - 128`.
    #[must_use]
    pub const fn b_factor(&self) -> i32 {
        self.b_factor_lf as i32 - 128
    }

    /// Reports whether every field equals its Table I.3 default.
    #[must_use]
    pub fn is_all_default(&self) -> bool {
        *self == Self::default()
    }
}

/// Reads an `LfChannelCorrelation` bundle (18181-1 I.2.3).
///
/// # Errors
///
/// [`DecodeError::Bitstream`] on truncation or an invalid `F16()`.
pub fn read_lf_channel_correlation(reader: &mut BitReader<'_>) -> Result<LfChannelCorrelation> {
    let all_default = trace_field!(reader, "lf_chan_corr.all_default", read_bool(reader))?;
    if all_default {
        return Ok(LfChannelCorrelation::default());
    }
    let colour_factor = trace_field!(
        reader,
        "lf_chan_corr.colour_factor",
        read_u32(reader, &COLOUR_FACTOR_SPEC)
    )?;
    let base_correlation_x = trace_field!(
        reader,
        "lf_chan_corr.base_correlation_x",
        read_f16_as_f32(reader)
    )?;
    let base_correlation_b = trace_field!(
        reader,
        "lf_chan_corr.base_correlation_b",
        read_f16_as_f32(reader)
    )?;
    let x_factor_lf = trace_field!(reader, "lf_chan_corr.x_factor_lf", reader.read_bits(8))?;
    let b_factor_lf = trace_field!(reader, "lf_chan_corr.b_factor_lf", reader.read_bits(8))?;

    // `u(8)` cannot exceed 255, but narrow through `try_from` rather than a
    // cast so a future change of field width cannot silently truncate.
    let narrow = |v: u32, field: &'static str| {
        u8::try_from(v).map_err(|_| DecodeError::out_of_range(field, "I.2.3", u64::from(v)))
    };

    Ok(LfChannelCorrelation {
        colour_factor,
        base_correlation_x,
        base_correlation_b,
        x_factor_lf: narrow(x_factor_lf, "x_factor_lf")?,
        b_factor_lf: narrow(b_factor_lf, "b_factor_lf")?,
    })
}

// ---------------------------------------------------------------------------
// The kVarDCT rows of Table G.1, as one bundle
// ---------------------------------------------------------------------------

/// The three `kVarDCT`-only rows of `LfGlobal` (Table G.1).
///
/// `lf_dequant` is deliberately **not** part of this struct: its row of
/// Table G.1 has no condition, so it is read for modular frames too and belongs
/// to the caller's unconditional path.
#[derive(Debug, Clone)]
pub struct LfGlobalVarDct {
    /// I.2.1 `Quantizer`.
    pub quantizer: Quantizer,
    /// I.2.2 HF block context model.
    pub hf_block_ctx: HfBlockContext,
    /// I.2.3 `LfChannelCorrelation`.
    pub lf_chan_corr: LfChannelCorrelation,
}

/// Reads the `kVarDCT`-only rows of `LfGlobal` in Table G.1 order.
///
/// Call this immediately after [`read_lf_channel_dequantization`] and
/// immediately before `GlobalModular` (G.1.3).
///
/// # Errors
///
/// [`DecodeError::Bitstream`] on truncation, [`DecodeError::Entropy`] if the
/// I.2.2 clustering map is invalid, or [`DecodeError::FieldOutOfRange`] if the
/// block context model violates I.2.2's bounds.
pub fn read_lf_global_vardct(
    reader: &mut BitReader<'_>,
    guard: &mut AllocGuard,
) -> Result<LfGlobalVarDct> {
    let quantizer = read_quantizer(reader)?;
    let hf_block_ctx = read_hf_block_context(reader, guard)?;
    let lf_chan_corr = read_lf_channel_correlation(reader)?;
    Ok(LfGlobalVarDct {
        quantizer,
        hf_block_ctx,
        lf_chan_corr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;

    #[test]
    fn lf_dequant_all_default_is_one_bit() {
        // Proves the G.1.2 fast path consumes exactly the Bool() and nothing
        // else, which is what makes the modular reader's inline skip correct.
        let mut w = BitWriter::new();
        w.bool(true);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let v = read_lf_channel_dequantization(&mut r).expect("valid");
        assert_eq!(r.total_bits_read(), 1);
        assert_eq!(v, LfChannelDequantization::default());
        assert_eq!(v.m_x_lf, 1.0 / 32.0);
        assert_eq!(v.m_y_lf, 1.0 / 4.0);
        assert_eq!(v.m_b_lf, 1.0 / 2.0);
    }

    #[test]
    fn lf_dequant_explicit_is_one_bit_plus_three_f16() {
        // Proves the field count and order of Table G.2.
        let mut w = BitWriter::new();
        w.bool(false);
        w.f16_bits(0x3C00); // 1.0
        w.f16_bits(0x4000); // 2.0
        w.f16_bits(0x4200); // 3.0
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let v = read_lf_channel_dequantization(&mut r).expect("valid");
        assert_eq!(r.total_bits_read(), 1 + 3 * 16);
        assert_eq!(v.m_x_lf, 1.0);
        assert_eq!(v.m_y_lf, 2.0);
        assert_eq!(v.m_b_lf, 3.0);
        // G.1.2's division by 128.
        assert_eq!(v.unscaled(), [1.0 / 128.0, 2.0 / 128.0, 3.0 / 128.0]);
    }

    #[test]
    fn lf_dequant_truncated_errors() {
        let mut w = BitWriter::new();
        w.bool(false).f16_bits(0x3C00);
        let data = w.finish();
        let mut r = BitReader::new(&data);
        assert!(read_lf_channel_dequantization(&mut r).is_err());
    }

    #[test]
    fn quantizer_selector_zero_is_two_plus_eleven_plus_two_bits() {
        // global_scale selector 0 = 1 + u(11); quant_lf selector 0 = the
        // constant 16 with no payload. Proves Table I.2's first alternatives.
        let mut w = BitWriter::new();
        w.u32_field(0, 11, 2047);
        w.u32_field(0, 0, 0);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let q = read_quantizer(&mut r).expect("valid");
        assert_eq!(r.total_bits_read(), (2 + 11) + 2);
        // The largest u(11) payload is 2047, so selector 0 tops out at 2048 —
        // exactly where selector 1's 2049 + u(11) takes over.
        assert_eq!(q.global_scale, 2048);
        assert_eq!(q.quant_lf, 16);
    }

    #[test]
    fn quantizer_covers_every_selector() {
        // Proves each of the eight Table I.2 alternatives decodes to its
        // documented offset, and that no alternative can yield zero — the
        // invariant that makes lf_multipliers division-safe.
        let cases: [(u32, u32, u32, u32); 4] = [
            (0, 11, 0, 1),
            (1, 11, 0, 2049),
            (2, 12, 0, 4097),
            (3, 16, 0, 8193),
        ];
        for (sel, bits, payload, expected) in cases {
            let mut w = BitWriter::new();
            w.u32_field(sel, bits, payload);
            w.u32_field(0, 0, 0);
            let data = w.finish_padded(2);
            let mut r = BitReader::new(&data);
            let q = read_quantizer(&mut r).expect("valid");
            assert_eq!(q.global_scale, expected, "global_scale selector {sel}");
            assert!(q.global_scale >= 1);
        }

        let cases: [(u32, u32, u32, u32); 4] =
            [(0, 0, 0, 16), (1, 5, 0, 1), (2, 8, 0, 1), (3, 16, 0, 1)];
        for (sel, bits, payload, expected) in cases {
            let mut w = BitWriter::new();
            w.u32_field(0, 11, 0);
            w.u32_field(sel, bits, payload);
            let data = w.finish_padded(2);
            let mut r = BitReader::new(&data);
            let q = read_quantizer(&mut r).expect("valid");
            assert_eq!(q.quant_lf, expected, "quant_lf selector {sel}");
            assert!(q.quant_lf >= 1);
        }
    }

    #[test]
    fn lf_multipliers_match_the_hand_computation() {
        // I.2.1 with the Table G.2 and Table I.2 defaults:
        //   mYDC = 65536 * (1/4/128) / (4096 * 16) = 65536/512/65536 = 1/512.
        let q = Quantizer {
            global_scale: 4096,
            quant_lf: 16,
        };
        let m = q.lf_multipliers(&LfChannelDequantization::default());
        assert!((m.y() - 1.0 / 512.0).abs() < 1e-12, "{}", m.y());
        // m_x_lf is 1/32 = (1/4)/8, so mXDC is mYDC/8; m_b_lf is 1/2 = 2*(1/4).
        assert!((m.x() - (1.0 / 512.0) / 8.0).abs() < 1e-12);
        assert!((m.b() - (1.0 / 512.0) * 2.0).abs() < 1e-12);
        assert_eq!(m.channel(0).expect("x"), m.x());
        assert_eq!(m.channel(2).expect("b"), m.b());
        assert!(m.channel(3).is_err());
    }

    #[test]
    fn lf_chan_corr_all_default_is_one_bit() {
        let mut w = BitWriter::new();
        w.bool(true);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let c = read_lf_channel_correlation(&mut r).expect("valid");
        assert_eq!(r.total_bits_read(), 1);
        assert_eq!(c, LfChannelCorrelation::default());
        assert_eq!(c.colour_factor, 84);
        assert_eq!(c.base_correlation_x, 0.0);
        assert_eq!(c.base_correlation_b, 1.0);
        // The bias means the default correlation offset is exactly zero.
        assert_eq!(c.x_factor(), 0);
        assert_eq!(c.b_factor(), 0);
    }

    #[test]
    fn lf_chan_corr_explicit_bit_count() {
        // 1 + U32(sel 0, no payload) + 16 + 16 + 8 + 8 = 51 bits. Proves the
        // Table I.3 field order and that colour_factor is a U32(), not a u(8).
        let mut w = BitWriter::new();
        w.bool(false);
        w.u32_field(1, 0, 0); // colour_factor = 256
        w.f16_bits(0x3C00); // base_correlation_x = 1.0
        w.f16_bits(0x0000); // base_correlation_b = 0.0
        w.u(8, 200);
        w.u(8, 100);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let c = read_lf_channel_correlation(&mut r).expect("valid");
        assert_eq!(r.total_bits_read(), 1 + 2 + 16 + 16 + 8 + 8);
        assert_eq!(c.colour_factor, 256);
        assert_eq!(c.base_correlation_x, 1.0);
        assert_eq!(c.base_correlation_b, 0.0);
        assert_eq!(c.x_factor(), 72);
        assert_eq!(c.b_factor(), -28);
    }

    #[test]
    fn lf_chan_corr_colour_factor_is_never_zero() {
        // I.6 divides by colour_factor; Table I.3's smallest alternative is
        // 2 + u(8), so a malicious stream cannot make it zero.
        for (sel, bits) in [(0u32, 0u32), (1, 0), (2, 8), (3, 16)] {
            let mut w = BitWriter::new();
            w.bool(false);
            w.u32_field(sel, bits, 0);
            w.f16_bits(0).f16_bits(0).u(8, 0).u(8, 0);
            let data = w.finish_padded(2);
            let mut r = BitReader::new(&data);
            let c = read_lf_channel_correlation(&mut r).expect("valid");
            assert!(c.colour_factor >= 2, "selector {sel}");
        }
    }

    #[test]
    fn lf_chan_corr_truncated_errors() {
        let mut w = BitWriter::new();
        w.bool(false).u32_field(0, 0, 0).f16_bits(0x3C00);
        let data = w.finish();
        let mut r = BitReader::new(&data);
        assert!(read_lf_channel_correlation(&mut r).is_err());
    }

    #[test]
    fn lf_global_vardct_reads_the_three_rows_in_table_order() {
        // quantizer (2 + 2 bits with both selector-0 constants) + block context
        // (1 bit for the default map) + lf_chan_corr (1 bit) = 6 bits. Proves
        // the Table G.1 ordering of the kVarDCT rows.
        let mut w = BitWriter::new();
        w.u32_field(0, 11, 0); // global_scale = 1
        w.u32_field(0, 0, 0); // quant_lf = 16
        w.bool(true); // I.2.2 default block_ctx_map
        w.bool(true); // I.2.3 all_default
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let limits = jpxl_core::limits::Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let v = read_lf_global_vardct(&mut r, &mut guard).expect("valid");
        assert_eq!(r.total_bits_read(), (2 + 11) + 2 + 1 + 1);
        assert_eq!(v.quantizer.global_scale, 1);
        assert_eq!(v.hf_block_ctx.nb_block_ctx(), 15);
        assert_eq!(v.lf_chan_corr, LfChannelCorrelation::default());
    }
}
