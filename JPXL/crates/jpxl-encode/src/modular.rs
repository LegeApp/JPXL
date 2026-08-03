//! The single frame section: `LfGlobal` and the modular sub-bitstream it
//! carries (18181-1 G.1, Annex H).
//!
//! For a one-group, one-pass modular frame the whole of Table F.1 lives in one
//! section, and everything after `LfGlobal` is empty: G.2.3 selects only
//! channels whose `hshift` and `vshift` are both at least 3, and G.4.2 only
//! channels wider or taller than `group_dim`. With no transforms and a single
//! full-resolution channel that fits in one group, neither selects anything, so
//! `GlobalModular` decodes the entire image and the LF-group and pass-group
//! sub-bitstreams have no channels — which H.1 says are not read at all.
//!
//! # Coding choices
//!
//! | Field | Value | Clause |
//! |---|---|---|
//! | `LfChannelDequantization` | all-default | G.1.2 |
//! | global MA tree | absent | G.1.3 |
//! | `use_global_tree` | false | H.1 |
//! | `wp_params` | all-default | H.5.1 |
//! | `nb_transforms` | 0 | H.2 |
//! | MA tree | one leaf | H.4.2 |
//! | predictor | 5, Gradient | Table H.3 |
//! | `offset` / `multiplier` | 0 / 1 | H.4.2 |
//!
//! A one-leaf tree means one entropy context, so no property is ever
//! evaluated and the MA machinery collapses to "always this leaf". The
//! *predictor* still matters, and Gradient is chosen over Zero because it costs
//! the encoder three neighbours and nothing else while turning a smooth image
//! from twelve bits a sample into two or three.
//!
//! # The residual
//!
//! H.3 reconstructs a sample as
//! `UnpackSigned(token) * multiplier + offset + prediction`. With
//! `multiplier = 1` and `offset = 0` the encoder writes
//! `PackSigned(sample - prediction)`, where `prediction` is the Table H.3 row 5
//! gradient over the **already written** neighbours. Since the coding is
//! lossless those neighbours are the source samples, so a single forward raster
//! pass suffices and no reconstruction buffer is needed.

use jpxl_bitstream::{BitWriter, U32Dist, U32Spec};

use crate::entropy::{pack_signed, write_bundle, write_uint};
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

/// Writes the single section of a one-group modular greyscale frame.
///
/// `samples` is in raster order and must hold exactly `width * height` values.
/// The returned bytes are the section body, zero-padded to a byte boundary;
/// their length is what the TOC entry records.
///
/// # Errors
///
/// [`EncodeError::SampleCountMismatch`] if `samples` does not match the
/// dimensions, [`EncodeError::ValueOutOfRange`] if a residual is outside the
/// range the entropy configuration can express, or a bit writer error.
pub fn encode_section(width: u32, height: u32, samples: &[i32]) -> Result<Vec<u8>> {
    let expected = u64::from(width) * u64::from(height);
    let found = u64::try_from(samples.len()).unwrap_or(u64::MAX);
    if expected != found {
        return Err(EncodeError::SampleCountMismatch { expected, found });
    }

    let mut w = BitWriter::new();

    // G.1.2: LfChannelDequantization is present whatever the encoding. Modular
    // mode never uses the weights, but the bit is still there.
    w.write_bool(true); // all_default

    // G.1.3: no global MA tree; the sub-bitstream carries its own.
    w.write_bool(false);

    // Table H.1, the ModularHeader.
    w.write_bool(false); // use_global_tree
    w.write_bool(true); // WPHeader: default_wp
    w.write_u32(&NB_TRANSFORMS_SPEC, 0)?;

    write_single_leaf_tree(&mut w)?;

    // H.4.2: the data stream has one pre-clustered distribution per leaf.
    write_bundle(&mut w, 1)?;
    write_samples(&mut w, width, height, samples)?;

    w.zero_pad_to_byte();
    Ok(w.into_bytes())
}

/// Writes the MA tree of H.4.2: its own six-context entropy stream carrying a
/// single leaf node.
fn write_single_leaf_tree(w: &mut BitWriter) -> Result<()> {
    write_bundle(w, TREE_NUM_CONTEXTS)?;
    // The six contexts, in the order H.4.2's `decode_tree` reads them.
    write_uint(w, 0)?; // ctx 1: property + 1 == 0 marks a leaf
    write_uint(w, PREDICTOR_GRADIENT)?; // ctx 2: predictor
    write_uint(w, pack_signed(0))?; // ctx 3: offset
    write_uint(w, 0)?; // ctx 4: mul_log
    write_uint(w, 0)?; // ctx 5: mul_bits, so multiplier = 1
    Ok(())
}

/// Writes every sample of the one channel in raster order (18181-1 H.3).
fn write_samples(w: &mut BitWriter, width: u32, height: u32, samples: &[i32]) -> Result<()> {
    let stride = usize::try_from(width).unwrap_or(usize::MAX);
    let at = |x: u32, y: u32| -> i64 {
        let index = usize::try_from(y)
            .ok()
            .and_then(|y| y.checked_mul(stride))
            .and_then(|row| usize::try_from(x).ok().and_then(|x| row.checked_add(x)));
        index
            .and_then(|i| samples.get(i))
            .map_or(0, |&v| i64::from(v))
    };

    for y in 0..height {
        for x in 0..width {
            let prediction = gradient_prediction(&at, x, y);
            let value = at(x, y);
            let residual = value - prediction;
            let residual = i32::try_from(residual).map_err(|_| EncodeError::ValueOutOfRange {
                what: "modular residual",
                value: residual,
            })?;
            write_uint(w, pack_signed(residual))?;
        }
    }
    Ok(())
}

/// The Table H.3 row 5 prediction at `(x, y)`, with the H.3 edge substitutions.
///
/// The substitutions cascade exactly as the clause writes them: a missing `W`
/// falls back to `N` and then to zero, `N` falls back to `W`, and `NW` falls
/// back to `W`.
fn gradient_prediction(at: &impl Fn(u32, u32) -> i64, x: u32, y: u32) -> i64 {
    let w = if x > 0 {
        at(x - 1, y)
    } else if y > 0 {
        at(x, y - 1)
    } else {
        0
    };
    let n = if y > 0 { at(x, y - 1) } else { w };
    let nw = if x > 0 && y > 0 { at(x - 1, y - 1) } else { w };
    (w + n - nw).clamp(w.min(n), w.max(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_the_clause_at_the_origin_and_the_edges() {
        // A 3x2 image: the first sample has no neighbours at all, so W, N and
        // NW are all zero and the prediction is zero.
        let data = [10i64, 20, 30, 40, 50, 60];
        let at = |x: u32, y: u32| -> i64 {
            let i = usize::try_from(y).unwrap_or(0) * 3 + usize::try_from(x).unwrap_or(0);
            data.get(i).copied().unwrap_or(0)
        };

        assert_eq!(gradient_prediction(&at, 0, 0), 0);
        // Row 0, x > 0: N falls back to W, NW to W, so the gradient is W.
        assert_eq!(gradient_prediction(&at, 1, 0), 10);
        assert_eq!(gradient_prediction(&at, 2, 0), 20);
        // Column 0, y > 0: W falls back to N, so the gradient is N.
        assert_eq!(gradient_prediction(&at, 0, 1), 10);
        // Interior: clamp(W + N - NW, min, max) = clamp(40 + 20 - 10, 20, 40).
        assert_eq!(gradient_prediction(&at, 1, 1), 40);
    }

    #[test]
    fn a_mismatched_sample_count_is_rejected() {
        assert!(matches!(
            encode_section(4, 4, &[0; 15]),
            Err(EncodeError::SampleCountMismatch {
                expected: 16,
                found: 15
            })
        ));
    }

    #[test]
    fn the_section_is_byte_aligned() {
        for (w, h) in [(1u32, 1u32), (13, 7), (16, 16)] {
            let samples = vec![0i32; (w * h) as usize];
            let section = encode_section(w, h, &samples).expect("encodes");
            assert!(!section.is_empty());
        }
    }
}
