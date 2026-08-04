//! A modular sub-bitstream writer for VarDCT's *control* images (Annex H).
//!
//! VarDCT is not only DCT coefficients. G.2.2's quantized LF planes and
//! G.2.4's `XFromY`/`BFromY`/`BlockInfo`/`Sharpness` planes are all carried as
//! ordinary modular sub-bitstreams — but with **per-channel dimensions**, which
//! is what [`crate::modular`] cannot express: that module encodes one frame's
//! worth of equally sized planes and knows about group rectangles, RCT and the
//! `LfGlobal`/pass-group split. None of that applies here. A control image is a
//! standalone list of small channels of unrelated shapes, written whole.
//!
//! # Coding choices
//!
//! The same deliberately degenerate ones the lossless track uses, for the same
//! reason (`crate::modular`): every choice below is legal, none of them is a
//! search, and the point of the slice is a stream other decoders accept.
//!
//! | Field | Value | Clause |
//! |---|---|---|
//! | `use_global_tree` | false | H.2 |
//! | `wp_params` | all-default | H.5.1 |
//! | `nb_transforms` | 0 | H.2 |
//! | MA tree | one leaf, Gradient, offset 0, multiplier 1 | H.4.2, Table H.3 |
//! | entropy | one context, flat prefix code, no LZ77 | C.2 |
//!
//! A one-leaf tree evaluates no property, so the channel list's shapes never
//! reach the context model and a channel of two rows costs no more machinery
//! than one of two hundred.
//!
//! # The empty case is not a special case, it is the clause
//!
//! H.1 says a sub-bitstream over zero channels is not read at all — not an
//! empty header, *nothing*. Both places this module is called from can hit it:
//! a kVarDCT frame's `GlobalModular` (G.1.3) has `num_channels == num_extra`,
//! which is zero without extra channels, and so do `ModularLfGroup` (G.2.3)
//! and the modular half of a `PassGroup` (G.4.2). Writing a header there would
//! desynchronise every field after it, so [`write_modular_stream`] returns
//! without emitting a bit.

use jpxl_bitstream::{BitWriter, U32Dist, U32Spec};

use crate::entropy::{FlatCode, TREE_CODE, pack_signed};
use crate::error::{EncodeError, Result};

/// 18181-1 H.2: `U32(0, 1, 2 + u(4), 18 + u(8))` for `nb_transforms`.
const NB_TRANSFORMS_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::BitsOffset { bits: 4, offset: 2 },
    U32Dist::BitsOffset {
        bits: 8,
        offset: 18,
    },
]);

/// Number of pre-clustered contexts the MA-tree stream uses (18181-1 H.4.2).
const TREE_NUM_CONTEXTS: usize = 6;

/// Table H.3 row 5: `clamp(W + N - NW, min(W, N), max(W, N))`.
const PREDICTOR_GRADIENT: u32 = 5;

/// One channel of a control image: `width x height` samples in raster order.
#[derive(Debug, Clone, Copy)]
pub struct OutChannel<'a> {
    /// Columns.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// `width * height` samples, row-major.
    pub samples: &'a [i32],
}

impl OutChannel<'_> {
    /// H.2's "skipping any channels having width or height zero".
    fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    fn at(&self, x: u32, y: u32) -> i64 {
        let index = usize::try_from(u64::from(y) * u64::from(self.width) + u64::from(x));
        index
            .ok()
            .and_then(|i| self.samples.get(i))
            .map_or(0, |&v| i64::from(v))
    }
}

/// Writes a whole modular sub-bitstream carrying `channels` (18181-1 H.2).
///
/// Emits nothing at all when `channels` is empty or every channel is empty —
/// see the module documentation.
///
/// # Errors
///
/// [`EncodeError::SampleCountMismatch`] if a channel's sample slice is not
/// `width * height` long, [`EncodeError::ValueOutOfRange`] if a residual is
/// outside what the entropy code can carry, or a bit writer error.
pub fn write_modular_stream(w: &mut BitWriter, channels: &[OutChannel<'_>]) -> Result<()> {
    if channels.is_empty() {
        return Ok(());
    }
    for channel in channels {
        let expected = u64::from(channel.width) * u64::from(channel.height);
        let found = u64::try_from(channel.samples.len()).unwrap_or(u64::MAX);
        if found != expected {
            return Err(EncodeError::SampleCountMismatch { expected, found });
        }
    }

    // The residuals have to exist before the code that carries them can be
    // sized, and the code has to be written before the residuals. Computing
    // them once and keeping them is the same trade the `SectionStore` makes
    // for the TOC.
    let mut residuals: Vec<Vec<u32>> = Vec::with_capacity(channels.len());
    let mut max_packed = 0u32;
    for channel in channels {
        if channel.is_empty() {
            residuals.push(Vec::new());
            continue;
        }
        let mut packed = Vec::with_capacity(channel.samples.len());
        for y in 0..channel.height {
            for x in 0..channel.width {
                let prediction = gradient_prediction(channel, x, y);
                let residual = channel.at(x, y) - prediction;
                let residual =
                    i32::try_from(residual).map_err(|_| EncodeError::ValueOutOfRange {
                        what: "modular residual",
                        value: residual,
                    })?;
                let value = pack_signed(residual);
                max_packed = max_packed.max(value);
                packed.push(value);
            }
        }
        residuals.push(packed);
    }
    let code = FlatCode::for_max_value(max_packed)?;

    write_modular_header(w)?;
    write_single_leaf_tree(w)?;
    code.write_bundle(w, 1)?;
    for packed in &residuals {
        for &value in packed {
            code.write_uint(w, value)?;
        }
    }
    Ok(())
}

/// Writes Table H.1 with no transforms.
fn write_modular_header(w: &mut BitWriter) -> Result<()> {
    w.write_bool(false); // use_global_tree
    w.write_bool(true); // WPHeader: default_wp
    w.write_u32(&NB_TRANSFORMS_SPEC, 0)?;
    Ok(())
}

/// Writes the MA tree of H.4.2: its own six-context entropy stream carrying a
/// single leaf node.
fn write_single_leaf_tree(w: &mut BitWriter) -> Result<()> {
    TREE_CODE.write_bundle(w, TREE_NUM_CONTEXTS)?;
    TREE_CODE.write_uint(w, 0)?; // ctx 1: property + 1 == 0 marks a leaf
    TREE_CODE.write_uint(w, PREDICTOR_GRADIENT)?; // ctx 2: predictor
    TREE_CODE.write_uint(w, pack_signed(0))?; // ctx 3: offset
    TREE_CODE.write_uint(w, 0)?; // ctx 4: mul_log
    TREE_CODE.write_uint(w, 0)?; // ctx 5: mul_bits, so multiplier = 1
    Ok(())
}

/// The Table H.3 row 5 prediction at `(x, y)`, with the H.3 edge substitutions.
///
/// Deliberately a second implementation of the same rule `crate::modular`
/// spells out for the lossless track: that one predicts inside a group
/// rectangle of a wider plane, this one inside a standalone channel, and
/// sharing the code would mean sharing a coordinate convention that is not the
/// same on both sides.
fn gradient_prediction(channel: &OutChannel<'_>, x: u32, y: u32) -> i64 {
    let w = if x > 0 {
        channel.at(x - 1, y)
    } else if y > 0 {
        channel.at(x, y - 1)
    } else {
        0
    };
    let n = if y > 0 { channel.at(x, y - 1) } else { w };
    let nw = if x > 0 && y > 0 {
        channel.at(x - 1, y - 1)
    } else {
        w
    };
    (w + n - nw).clamp(w.min(n), w.max(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_channels_emit_no_bits_at_all() {
        let mut w = BitWriter::new();
        write_modular_stream(&mut w, &[]).expect("writes");
        assert_eq!(
            w.bit_len(),
            0,
            "H.1: N == 0 means the decoder reads nothing"
        );
    }

    #[test]
    fn a_mismatched_channel_is_rejected() {
        let samples = [0i32; 5];
        let mut w = BitWriter::new();
        assert!(matches!(
            write_modular_stream(
                &mut w,
                &[OutChannel {
                    width: 3,
                    height: 2,
                    samples: &samples,
                }],
            ),
            Err(EncodeError::SampleCountMismatch {
                expected: 6,
                found: 5
            })
        ));
    }

    #[test]
    fn the_gradient_matches_the_clause_at_the_edges() {
        let samples = [10i32, 20, 30, 40, 50, 60];
        let channel = OutChannel {
            width: 3,
            height: 2,
            samples: &samples,
        };
        assert_eq!(gradient_prediction(&channel, 0, 0), 0);
        assert_eq!(gradient_prediction(&channel, 1, 0), 10);
        assert_eq!(gradient_prediction(&channel, 0, 1), 10);
        assert_eq!(gradient_prediction(&channel, 1, 1), 40);
    }
}
