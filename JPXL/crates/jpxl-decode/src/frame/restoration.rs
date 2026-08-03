//! The `RestorationFilter` bundle (18181-1 J.1, Table J.1).
//!
//! ```text
//! condition                                     type        default          name
//!                                               Bool()      true             all_default
//! !all_default                                  Bool()      true             gab
//! gab                                           Bool()      false            gab_custom
//! gab_custom                                    F16() x6    .115169525/...   gab_{x,y,b}_weight{1,2}
//! !all_default                                  u(2)        2                epf_iters
//! !all_default and epf_iters and kVarDCT        Bool()      false            epf_sharp_custom
//! epf_sharp_custom                              F16()       {0,1/7,..,1}     epf_sharp_lut[8]
//! !all_default and epf_iters                    Bool()      false            epf_weight_custom
//! epf_weight_custom                             F16()       {40,5,3.5}       epf_channel_scale[3]
//! epf_weight_custom                             u(32)       0                (ignored)
//! !all_default and epf_iters                    Bool()      false            epf_sigma_custom
//! epf_sigma_custom and kVarDCT                  F16()       0.46             epf_quant_mul
//! epf_sigma_custom                              F16()       0.9              epf_pass0_sigma_scale
//! epf_sigma_custom                              F16()       6.5              epf_pass2_sigma_scale
//! epf_sigma_custom                              F16()       2/3              epf_border_sad_mul
//! !all_default and epf_iters and kModular       F16()       1.0              epf_sigma_for_modular
//! !all_default                                  Extensions                   extensions
//! ```
//!
//! Only the parameters are modelled here; J.3 (Gabor-like transform) and J.4
//! (edge-preserving filter) are pixel-stage work for a later slice.
//!
//! # The `gab_custom` condition
//!
//! Both transcriptions render the `gab_custom` row's condition as bare `gab`.
//! Taken literally that is self-contradictory: `gab` *defaults to true*, so an
//! `all_default` bundle would still read a `gab_custom` bit, and `all_default`
//! would not mean "read nothing further".
//!
//! Every sibling row whose guard could be true by default spells the guard out
//! in full — `epf_iters` also defaults to a truthy value (2), and its three
//! dependent rows are all written `!all_default and epf_iters`. That the
//! authors wrote `!all_default and X` exactly where a default would otherwise
//! leak, and that in the OCR the `gab_custom` condition cell is a wrapped
//! multi-line cell whose first line could have been dropped, both point the
//! same way.
//!
//! This implementation therefore reads `gab_custom` only when
//! `!all_default && gab`, preserving the invariant that every other bundle in
//! the standard obeys: `all_default` costs exactly one bit.
//! [`GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT`] names the decision so it can be
//! flipped in one place.
//!
//! UNEXERCISED (negative result), 2026-08-03: [`read_restoration_filter`] used
//! to return immediately on `all_default`, before `gab` or `gab_custom` were
//! ever computed — which made the two readings bit-identical regardless of
//! the constant, since the literal reading's extra bit lives exactly on the
//! branch the early return skipped. That is fixed (the early return now
//! happens after `gab`/`gab_custom` are resolved), so the constant is live.
//! But no `cjxl` v0.13.0 stream tried — twelve synthetic probes (checkerboard,
//! stripes, two-value noise, and existing gradient fixtures, modular and
//! VarDCT, lossless and lossy, effort 1/3/5/7/9, plus `--gaborish=0/1`
//! explicitly) — ever produced `all_default == true` for this bundle: cjxl
//! always writes at least one non-default field, so `all_default` is 0 and
//! both readings agree. See
//! `docs/experiments/2026-08-03-flip-point-fixtures.md`.

use jpxl_bitstream::{BitReader, read_bool, read_f16_as_f32, trace_field};

use crate::frame::error::Result;
use crate::frame::header::Encoding;
use crate::headers::extensions::{Extensions, read_extensions};

/// Whether `gab_custom` is gated on `!all_default` as well as `gab`.
///
/// See the module documentation: the standard's text says only `gab`, but that
/// reading makes `all_default` fail to suppress the field. Setting this to
/// `false` restores the literal reading.
pub const GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT: bool = true;

/// Table J.1 default Gabor weights: `{x,y,b}_weight1` then `{x,y,b}_weight2`.
pub const DEFAULT_GAB_WEIGHT1: f32 = 0.115_169_525;
/// Table J.1 default second-ring Gabor weight.
pub const DEFAULT_GAB_WEIGHT2: f32 = 0.061_248_592;

/// Table J.1 default `epf_iters`.
pub const DEFAULT_EPF_ITERS: u32 = 2;

/// Table J.1 default `epf_channel_scale`.
pub const DEFAULT_EPF_CHANNEL_SCALE: [f32; 3] = [40.0, 5.0, 3.5];

/// Table J.1 default `epf_sharp_lut`: `{0, 1/7, 2/7, ..., 6/7, 1}`.
pub const DEFAULT_EPF_SHARP_LUT: [f32; 8] = [
    0.0,
    1.0 / 7.0,
    2.0 / 7.0,
    3.0 / 7.0,
    4.0 / 7.0,
    5.0 / 7.0,
    6.0 / 7.0,
    1.0,
];

/// Per-channel Gabor-like transform weights (18181-1 J.1, J.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaborWeights {
    /// Weight of the four edge neighbours, per channel `{x, y, b}`.
    pub weight1: [f32; 3],
    /// Weight of the four corner neighbours, per channel `{x, y, b}`.
    pub weight2: [f32; 3],
}

impl Default for GaborWeights {
    fn default() -> Self {
        Self {
            weight1: [DEFAULT_GAB_WEIGHT1; 3],
            weight2: [DEFAULT_GAB_WEIGHT2; 3],
        }
    }
}

/// Edge-preserving-filter parameters (18181-1 J.1, J.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpfParams {
    /// Number of filter iterations; 0 disables the filter.
    pub iters: u32,
    /// Sharpness lookup table.
    pub sharp_lut: [f32; 8],
    /// Per-channel weight scaling.
    pub channel_scale: [f32; 3],
    /// Multiplier tying sigma to the quantizer (VarDCT only).
    pub quant_mul: f32,
    /// Sigma scale for pass 0.
    pub pass0_sigma_scale: f32,
    /// Sigma scale for pass 2.
    pub pass2_sigma_scale: f32,
    /// Border sum-of-absolute-differences multiplier.
    pub border_sad_mul: f32,
    /// Fixed sigma used in Modular mode.
    pub sigma_for_modular: f32,
}

impl Default for EpfParams {
    fn default() -> Self {
        Self {
            iters: DEFAULT_EPF_ITERS,
            sharp_lut: DEFAULT_EPF_SHARP_LUT,
            channel_scale: DEFAULT_EPF_CHANNEL_SCALE,
            quant_mul: 0.46,
            pass0_sigma_scale: 0.9,
            pass2_sigma_scale: 6.5,
            border_sad_mul: 2.0 / 3.0,
            sigma_for_modular: 1.0,
        }
    }
}

/// A decoded `RestorationFilter` bundle (18181-1 Table J.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RestorationFilter {
    /// Whether the bundle took its defaults.
    pub all_default: bool,
    /// Whether the Gabor-like transform is applied (J.3).
    pub gab: bool,
    /// Whether custom Gabor weights were signalled.
    pub gab_custom: bool,
    /// Gabor weights, custom or default.
    pub gab_weights: GaborWeights,
    /// Edge-preserving filter parameters.
    pub epf: EpfParams,
}

impl Default for RestorationFilter {
    fn default() -> Self {
        Self {
            all_default: true,
            gab: true,
            gab_custom: false,
            gab_weights: GaborWeights::default(),
            epf: EpfParams::default(),
        }
    }
}

impl RestorationFilter {
    /// Whether the edge-preserving filter runs at all.
    #[must_use]
    pub const fn epf_enabled(&self) -> bool {
        self.epf.iters != 0
    }
}

/// Reads a `RestorationFilter` bundle (18181-1 Table J.1).
///
/// Several rows are gated on the frame's `encoding`, so it is a parameter.
/// The `Extensions` payload is skipped per B.3.
///
/// # Errors
///
/// A bitstream error on truncation or an invalid `F16()`, or a decode error
/// from the nested `Extensions` bundle.
pub fn read_restoration_filter(
    reader: &mut BitReader<'_>,
    encoding: Encoding,
) -> Result<(RestorationFilter, Option<Extensions>)> {
    let all_default = trace_field!(reader, "restoration.all_default", read_bool(reader))?;

    // Table J.1's `gab` row is guarded by `!all_default` and defaults to
    // `true`, so an all-default bundle has `gab == true` without a bit being
    // read for it. Computed here, before the early return, so the flip point
    // below can be tested: under the literal `GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT
    // = false` reading a `gab_custom` bit is read even when `all_default`,
    // which the early return would otherwise make unreachable.
    let gab = if all_default {
        true
    } else {
        trace_field!(reader, "restoration.gab", read_bool(reader))?
    };

    let read_gab_custom = gab && (!GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT || !all_default);
    let gab_custom = if read_gab_custom {
        trace_field!(reader, "restoration.gab_custom", read_bool(reader))?
    } else {
        false
    };

    let mut gab_weights = GaborWeights::default();
    if gab_custom {
        // Table J.1 order is x1, x2, y1, y2, b1, b2 — interleaved per channel,
        // not grouped by ring.
        for channel in 0..3 {
            let w1 = trace_field!(reader, "restoration.gab_weight1", read_f16_as_f32(reader))?;
            let w2 = trace_field!(reader, "restoration.gab_weight2", read_f16_as_f32(reader))?;
            if let (Some(a), Some(b)) = (
                gab_weights.weight1.get_mut(channel),
                gab_weights.weight2.get_mut(channel),
            ) {
                *a = w1;
                *b = w2;
            }
        }
    }

    if all_default {
        // Every other field defaults; only `gab_custom` (and, if it was read
        // as true under the literal reading, its weights) could have consumed
        // bits above.
        return Ok((
            RestorationFilter {
                gab_custom,
                gab_weights,
                ..RestorationFilter::default()
            },
            None,
        ));
    }

    let mut epf = EpfParams {
        iters: trace_field!(reader, "restoration.epf_iters", reader.read_bits(2))?,
        ..EpfParams::default()
    };

    if epf.iters != 0 {
        if encoding == Encoding::VarDct {
            let sharp_custom =
                trace_field!(reader, "restoration.epf_sharp_custom", read_bool(reader))?;
            if sharp_custom {
                for slot in &mut epf.sharp_lut {
                    *slot =
                        trace_field!(reader, "restoration.epf_sharp_lut", read_f16_as_f32(reader))?;
                }
            }
        }

        let weight_custom =
            trace_field!(reader, "restoration.epf_weight_custom", read_bool(reader))?;
        if weight_custom {
            for slot in &mut epf.channel_scale {
                *slot = trace_field!(
                    reader,
                    "restoration.epf_channel_scale",
                    read_f16_as_f32(reader)
                )?;
            }
            // Table J.1 lists a u(32) field explicitly marked "(ignored)".
            // It must still be consumed or every later field shifts by 32 bits.
            trace_field!(reader, "restoration.epf_ignored", reader.read_bits(32))?;
        }

        let sigma_custom = trace_field!(reader, "restoration.epf_sigma_custom", read_bool(reader))?;
        if sigma_custom {
            if encoding == Encoding::VarDct {
                epf.quant_mul =
                    trace_field!(reader, "restoration.epf_quant_mul", read_f16_as_f32(reader))?;
            }
            epf.pass0_sigma_scale = trace_field!(
                reader,
                "restoration.epf_pass0_sigma_scale",
                read_f16_as_f32(reader)
            )?;
            epf.pass2_sigma_scale = trace_field!(
                reader,
                "restoration.epf_pass2_sigma_scale",
                read_f16_as_f32(reader)
            )?;
            epf.border_sad_mul = trace_field!(
                reader,
                "restoration.epf_border_sad_mul",
                read_f16_as_f32(reader)
            )?;
        }

        if encoding == Encoding::Modular {
            epf.sigma_for_modular = trace_field!(
                reader,
                "restoration.epf_sigma_for_modular",
                read_f16_as_f32(reader)
            )?;
        }
    }

    let extensions = read_extensions(reader)?;

    Ok((
        RestorationFilter {
            all_default: false,
            gab,
            gab_custom,
            gab_weights,
            epf,
        },
        Some(extensions),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;

    fn read(bytes: &[u8], encoding: Encoding) -> Result<(RestorationFilter, Option<Extensions>)> {
        let mut r = BitReader::new(bytes);
        read_restoration_filter(&mut r, encoding)
    }

    #[test]
    fn all_default_costs_one_bit() {
        // This is the assertion that changes if the literal reading of the
        // gab_custom condition wins: it would be two bits.
        let mut w = BitWriter::new();
        w.bool(true);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let (rf, ext) = read_restoration_filter(&mut r, Encoding::VarDct).expect("valid");

        assert_eq!(r.total_bits_read(), 1);
        assert_eq!(rf, RestorationFilter::default());
        assert!(rf.gab, "gab defaults to true");
        assert_eq!(rf.epf.iters, 2);
        assert!(ext.is_none());
    }

    #[test]
    fn explicit_defaults_with_gab_and_no_epf() {
        // all_default = 0, gab = 1, gab_custom = 0, epf_iters = 0,
        // extensions = 0. With epf_iters zero none of the EPF rows are read.
        let mut w = BitWriter::new();
        w.bool(false).bool(true).bool(false).u(2, 0).u64_field(0);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let (rf, ext) = read_restoration_filter(&mut r, Encoding::VarDct).expect("valid");

        assert_eq!(r.total_bits_read(), 1 + 1 + 1 + 2 + 2);
        assert!(!rf.all_default);
        assert!(rf.gab);
        assert!(!rf.epf_enabled());
        assert_eq!(ext.map(|e| e.extensions), Some(0));
    }

    #[test]
    fn gab_disabled_skips_gab_custom() {
        let mut w = BitWriter::new();
        w.bool(false).bool(false).u(2, 0).u64_field(0);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let (rf, _) = read_restoration_filter(&mut r, Encoding::VarDct).expect("valid");

        assert_eq!(r.total_bits_read(), 1 + 1 + 2 + 2);
        assert!(!rf.gab);
        assert!(!rf.gab_custom);
        assert_eq!(
            rf.gab_weights,
            GaborWeights::default(),
            "weights keep their defaults when the filter is off"
        );
    }

    #[test]
    fn custom_gabor_weights_are_interleaved_per_channel() {
        // Table J.1 order: x1, x2, y1, y2, b1, b2.
        let mut w = BitWriter::new();
        w.bool(false).bool(true).bool(true);
        // F16 values 1.0, 2.0, 3.0, 4.0, 0.5, 0.25.
        for bits in [0x3C00u16, 0x4000, 0x4200, 0x4400, 0x3800, 0x3400] {
            w.f16_bits(bits);
        }
        w.u(2, 0).u64_field(0);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let (rf, _) = read_restoration_filter(&mut r, Encoding::VarDct).expect("valid");

        assert!(rf.gab_custom);
        assert_eq!(rf.gab_weights.weight1, [1.0, 3.0, 0.5]);
        assert_eq!(rf.gab_weights.weight2, [2.0, 4.0, 0.25]);
    }

    #[test]
    fn epf_sharp_lut_is_vardct_only() {
        // VarDCT: epf_sharp_custom is read.
        let mut w = BitWriter::new();
        w.bool(false)
            .bool(false) // gab off
            .u(2, 1) // epf_iters = 1
            .bool(true); // epf_sharp_custom
        for bits in [
            0x0000u16, 0x3C00, 0x0000, 0x3C00, 0x0000, 0x3C00, 0x0000, 0x3C00,
        ] {
            w.f16_bits(bits);
        }
        w.bool(false) // epf_weight_custom
            .bool(false) // epf_sigma_custom
            .u64_field(0);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let (rf, _) = read_restoration_filter(&mut r, Encoding::VarDct).expect("valid");

        assert_eq!(rf.epf.iters, 1);
        assert_eq!(rf.epf.sharp_lut, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn modular_skips_sharp_lut_but_reads_sigma_for_modular() {
        // Modular: no epf_sharp_custom row; epf_sigma_for_modular is present.
        let mut w = BitWriter::new();
        w.bool(false)
            .bool(false) // gab off
            .u(2, 3) // epf_iters = 3
            .bool(false) // epf_weight_custom
            .bool(false) // epf_sigma_custom
            .f16_bits(0x4000) // epf_sigma_for_modular = 2.0
            .u64_field(0);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let (rf, _) = read_restoration_filter(&mut r, Encoding::Modular).expect("valid");

        assert_eq!(rf.epf.iters, 3);
        assert_eq!(rf.epf.sigma_for_modular, 2.0);
        assert_eq!(
            rf.epf.sharp_lut, DEFAULT_EPF_SHARP_LUT,
            "the sharp LUT row is VarDCT-only"
        );
    }

    #[test]
    fn weight_custom_consumes_the_ignored_u32() {
        // The u(32) "(ignored)" field must still be read, or the following
        // epf_sigma_custom bit lands 32 bits early.
        let mut w = BitWriter::new();
        w.bool(false)
            .bool(false)
            .u(2, 1)
            .bool(false) // epf_sharp_custom
            .bool(true); // epf_weight_custom
        for bits in [0x3C00u16, 0x4000, 0x4200] {
            w.f16_bits(bits);
        }
        w.u(32, 0xDEAD_BEEF) // ignored
            .bool(false) // epf_sigma_custom
            .u64_field(0);
        let expected = w.bit_len();
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let (rf, _) = read_restoration_filter(&mut r, Encoding::VarDct).expect("valid");

        assert_eq!(r.total_bits_read(), expected);
        assert_eq!(rf.epf.channel_scale, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn sigma_custom_skips_quant_mul_in_modular() {
        // VarDCT reads four F16 values, Modular only three.
        let mut vardct = BitWriter::new();
        vardct
            .bool(false)
            .bool(false)
            .u(2, 1)
            .bool(false) // sharp
            .bool(false) // weight
            .bool(true); // sigma_custom
        for bits in [0x3C00u16, 0x4000, 0x4200, 0x4400] {
            vardct.f16_bits(bits);
        }
        vardct.u64_field(0);
        let data = vardct.finish_padded(1);
        let mut r = BitReader::new(&data);
        let (rf, _) = read_restoration_filter(&mut r, Encoding::VarDct).expect("valid");
        assert_eq!(rf.epf.quant_mul, 1.0);
        assert_eq!(rf.epf.pass0_sigma_scale, 2.0);
        assert_eq!(rf.epf.pass2_sigma_scale, 3.0);
        assert_eq!(rf.epf.border_sad_mul, 4.0);

        let mut modular = BitWriter::new();
        modular
            .bool(false)
            .bool(false)
            .u(2, 1)
            .bool(false) // weight
            .bool(true); // sigma_custom
        for bits in [0x4000u16, 0x4200, 0x4400] {
            modular.f16_bits(bits);
        }
        modular.f16_bits(0x3C00).u64_field(0); // sigma_for_modular
        let data = modular.finish_padded(1);
        let mut r = BitReader::new(&data);
        let (rf, _) = read_restoration_filter(&mut r, Encoding::Modular).expect("valid");
        assert_eq!(
            rf.epf.quant_mul, 0.46,
            "quant_mul keeps its default in Modular"
        );
        assert_eq!(rf.epf.pass0_sigma_scale, 2.0);
        assert_eq!(rf.epf.border_sad_mul, 4.0);
    }

    #[test]
    fn default_sharp_lut_is_sevenths() {
        let lut = DEFAULT_EPF_SHARP_LUT;
        assert_eq!(lut.first().copied(), Some(0.0));
        assert_eq!(lut.last().copied(), Some(1.0));
        assert!((lut.get(1).copied().unwrap_or(0.0) - 1.0 / 7.0).abs() < 1e-7);
    }

    #[test]
    fn truncated_bundle_errors() {
        assert!(read(&[], Encoding::VarDct).is_err());
    }
}
