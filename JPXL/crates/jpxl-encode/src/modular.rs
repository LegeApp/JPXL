//! The modular sub-bitstreams: `LfGlobal` and the group sections it precedes
//! (18181-1 G.1, G.4, Annex H).
//!
//! # One channel list, several sections
//!
//! Annex G splits one logical modular image across sections. `GlobalModular`
//! (G.1.3) decodes only the meta channels and any channel that fits inside
//! `group_dim`; whatever is left is decoded per group by G.4.2. This encoder
//! therefore has exactly two shapes:
//!
//! * **single section** — the image fits in one group, so `LfGlobal` carries
//!   every sample and the LF-group / `HfGlobal` / pass-group sections do not
//!   exist at all (F.3.1);
//! * **multi-section** — `LfGlobal` carries the transform list, an MA tree and
//!   an entropy bundle but *no samples*, the LF-group and `HfGlobal` sections
//!   are empty, and one pass-group section per group carries that group's
//!   rectangle of every channel.
//!
//! Nothing here ever produces a channel with `hshift >= 3` (only Squeeze
//! does), so the LF-group sections are always empty, and `HfGlobal` is
//! VarDCT-only. Both still occupy a TOC entry.
//!
//! # Coding choices
//!
//! | Field | Value | Clause |
//! |---|---|---|
//! | `LfChannelDequantization` | all-default | G.1.2 |
//! | global MA tree | absent | G.1.3 |
//! | `use_global_tree` | false, in every sub-bitstream | H.2 |
//! | `wp_params` | all-default | H.5.1 |
//! | `nb_transforms` | 0, or 1 `kRCT` in `LfGlobal` | H.2 |
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
//! Every sub-bitstream carries its own tree and its own distribution bundle
//! (`use_global_tree = false`). H.2 allows a global tree shared from
//! `LfGlobal` instead, which would save roughly 110 bits per group — but a
//! shared tree also shares its distributions across sections, and this
//! encoder's sections are worth more as independent units than those bits are
//! worth. It also keeps one code path for both shapes.
//!
//! # The residual
//!
//! H.3 reconstructs a sample as
//! `UnpackSigned(token) * multiplier + offset + prediction`. With
//! `multiplier = 1` and `offset = 0` the encoder writes
//! `PackSigned(sample - prediction)`, where `prediction` is the Table H.3 row 5
//! gradient over the **already written** neighbours. Since the coding is
//! lossless those neighbours are the source samples, so a single forward raster
//! pass suffices and no reconstruction buffer is needed. Group sub-bitstreams
//! predict inside the group rectangle only: G.4.2 decodes a group as a
//! standalone channel and copies it in, so its H.3 edge substitutions are
//! relative to the group, not to the frame.

use jpxl_bitstream::{BitWriter, U32Dist, U32Spec};

use crate::entropy::{FlatCode, TREE_CODE, pack_signed};
use crate::error::{EncodeError, Result};
use crate::frame::Geometry;

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

/// 18181-1 Table H.7: `U32(u(3), 8 + u(6), 72 + u(10), 1096 + u(13))` for
/// `begin_c`.
const BEGIN_C_SPEC: U32Spec = U32Spec::new([
    U32Dist::bits(3),
    U32Dist::BitsOffset { bits: 6, offset: 8 },
    U32Dist::BitsOffset {
        bits: 10,
        offset: 72,
    },
    U32Dist::BitsOffset {
        bits: 13,
        offset: 1096,
    },
]);

/// 18181-1 Table H.7: `U32(6, u(2), 2 + u(4), 10 + u(6))` for `rct_type`.
///
/// The first distribution being the constant 6 is the clause telling you which
/// transform it expects: `rct_type = 6` is `permutation = 0`, `type = 6`, the
/// YCoCg form, and it costs two bits.
const RCT_TYPE_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(6),
    U32Dist::bits(2),
    U32Dist::BitsOffset { bits: 4, offset: 2 },
    U32Dist::BitsOffset {
        bits: 6,
        offset: 10,
    },
]);

/// Table H.6 `TransformId`: `kRCT`.
const TRANSFORM_ID_RCT: u32 = 0;

/// The `rct_type` this encoder writes: `permutation = 0`, `type = 6` (YCoCg).
pub const RCT_TYPE_YCOCG: u32 = 6;

/// Number of pre-clustered contexts the MA-tree stream uses (18181-1 H.4.2).
const TREE_NUM_CONTEXTS: usize = 6;

/// Table H.3 row 5: `clamp(W + N - NW, min(W, N), max(W, N))`.
const PREDICTOR_GRADIENT: u32 = 5;

/// One channel's samples in raster order, as fed to H.3.
pub type Plane = Vec<i32>;

/// The forward reversible colour transform matching `rct_type = 6` (18181-1
/// H.6.3), i.e. the YCoCg-R lifting.
///
/// H.6.3 specifies the **inverse**; this is the map that inverse undoes. For
/// stored `(A, B, C)` the clause computes
/// `tmp = A - (C >> 1); G = C + tmp; Blue = tmp - (B >> 1); Red = Blue + B`,
/// so `A` is luma, `B` is `R - Blue` and `C` is the green-difference. Solving
/// for the stored triple in that order gives the four lifting steps below.
/// `>>` floors, which is what makes the pair exactly reversible over the
/// negative chroma range; Rust's `>>` on a signed integer is the same
/// arithmetic shift.
#[must_use]
pub fn rct_forward(red: i32, green: i32, blue: i32) -> (i32, i32, i32) {
    let r = i64::from(red);
    let g = i64::from(green);
    let b = i64::from(blue);
    let co = r - b;
    let tmp = b + (co >> 1);
    let cg = g - tmp;
    let y = tmp + (cg >> 1);
    // Inputs are at most 16-bit samples, so every intermediate stays far
    // inside i32; the conversions are exact.
    (narrow(y), narrow(co), narrow(cg))
}

/// Saturating i64 -> i32, used only where the value provably fits.
fn narrow(value: i64) -> i32 {
    i32::try_from(value).unwrap_or(if value < 0 { i32::MIN } else { i32::MAX })
}

/// Applies [`rct_forward`] across three equally sized planes in place.
///
/// # Errors
///
/// [`EncodeError::SampleCountMismatch`] if the three planes differ in length.
pub fn apply_rct(planes: &mut [Plane]) -> Result<()> {
    let [first, second, third] = planes else {
        return Err(EncodeError::unsupported(
            "an RCT over a channel count other than three",
            "H.6.3",
        ));
    };
    let n = first.len();
    if second.len() != n || third.len() != n {
        return Err(EncodeError::SampleCountMismatch {
            expected: n as u64,
            found: second.len().min(third.len()) as u64,
        });
    }
    for i in 0..n {
        let (r, g, b) = (
            first.get(i).copied().unwrap_or(0),
            second.get(i).copied().unwrap_or(0),
            third.get(i).copied().unwrap_or(0),
        );
        let (y, co, cg) = rct_forward(r, g, b);
        if let Some(slot) = first.get_mut(i) {
            *slot = y;
        }
        if let Some(slot) = second.get_mut(i) {
            *slot = co;
        }
        if let Some(slot) = third.get_mut(i) {
            *slot = cg;
        }
    }
    Ok(())
}

/// A rectangle of the frame, in samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// Left edge.
    pub x0: u32,
    /// Top edge.
    pub y0: u32,
    /// Width; always nonzero for a rectangle this encoder emits.
    pub width: u32,
    /// Height; always nonzero for a rectangle this encoder emits.
    pub height: u32,
}

/// The image the modular layer encodes: one or three equally sized planes.
#[derive(Debug, Clone)]
pub struct ModularSource<'a> {
    /// Frame width in samples.
    pub width: u32,
    /// Frame height in samples.
    pub height: u32,
    /// Channel data in G.1.3 order, already colour-transformed if `rct` is set.
    pub planes: &'a [Plane],
    /// Whether `LfGlobal` declares the `kRCT` transform over channels 0..3.
    pub rct: bool,
    /// The prefix code every sample stream uses.
    pub code: FlatCode,
}

impl ModularSource<'_> {
    /// Checks that every plane holds exactly `width * height` samples.
    fn validate(&self) -> Result<()> {
        let expected = u64::from(self.width) * u64::from(self.height);
        for plane in self.planes {
            let found = u64::try_from(plane.len()).unwrap_or(u64::MAX);
            if found != expected {
                return Err(EncodeError::SampleCountMismatch { expected, found });
            }
        }
        if self.rct && self.planes.len() != 3 {
            return Err(EncodeError::unsupported(
                "an RCT over a channel count other than three",
                "H.6.3",
            ));
        }
        Ok(())
    }
}

/// Writes the `LfGlobal` section (18181-1 G.1).
///
/// In the single-section shape this carries every sample of every channel; in
/// the multi-section shape it carries the transform list, the MA tree and the
/// entropy bundle, and no samples at all — G.1.3 stops before the first
/// channel larger than `group_dim`, which in that shape is channel zero.
///
/// # Errors
///
/// [`EncodeError::SampleCountMismatch`] if a plane is the wrong length,
/// [`EncodeError::ValueOutOfRange`] if a residual exceeds the code's range, or
/// a bit writer error.
pub fn encode_lf_global(source: &ModularSource<'_>, geometry: &Geometry) -> Result<Vec<u8>> {
    source.validate()?;
    let mut w = BitWriter::new();

    // G.1.2: LfChannelDequantization is present whatever the encoding. Modular
    // mode never uses the weights, but the bit is still there.
    w.write_bool(true); // all_default

    // G.1.3: no global MA tree; every sub-bitstream carries its own.
    w.write_bool(false);

    write_modular_header(&mut w, source.rct)?;
    write_single_leaf_tree(&mut w)?;
    source.code.write_bundle(&mut w, 1)?;

    // G.1.3: only channels no larger than group_dim are decoded here. Every
    // channel has the frame's dimensions, so it is all of them or none.
    let group_dim = geometry.group_dim();
    if source.width <= group_dim && source.height <= group_dim {
        let rect = Rect {
            x0: 0,
            y0: 0,
            width: source.width,
            height: source.height,
        };
        write_planes(&mut w, source, rect)?;
    }

    w.zero_pad_to_byte();
    Ok(w.into_bytes())
}

/// Writes one pass-group section (18181-1 G.4.2).
///
/// # Errors
///
/// As [`encode_lf_global`].
pub fn encode_group(source: &ModularSource<'_>, rect: Rect) -> Result<Vec<u8>> {
    source.validate()?;
    let mut w = BitWriter::new();

    // A group sub-bitstream declares no transforms of its own: the channel
    // list it works on is the one LfGlobal already transformed.
    write_modular_header(&mut w, false)?;
    write_single_leaf_tree(&mut w)?;
    source.code.write_bundle(&mut w, 1)?;
    write_planes(&mut w, source, rect)?;

    w.zero_pad_to_byte();
    Ok(w.into_bytes())
}

/// Writes Table H.1, optionally declaring the one `kRCT` transform.
fn write_modular_header(w: &mut BitWriter, rct: bool) -> Result<()> {
    w.write_bool(false); // use_global_tree
    w.write_bool(true); // WPHeader: default_wp
    w.write_u32(&NB_TRANSFORMS_SPEC, u32::from(rct))?;
    if rct {
        // Table H.7: TransformId, then begin_c, then rct_type.
        w.write_bits(2, TRANSFORM_ID_RCT)?;
        w.write_u32(&BEGIN_C_SPEC, 0)?;
        w.write_u32(&RCT_TYPE_SPEC, RCT_TYPE_YCOCG)?;
    }
    Ok(())
}

/// Writes the MA tree of H.4.2: its own six-context entropy stream carrying a
/// single leaf node.
fn write_single_leaf_tree(w: &mut BitWriter) -> Result<()> {
    TREE_CODE.write_bundle(w, TREE_NUM_CONTEXTS)?;
    // The six contexts, in the order H.4.2's `decode_tree` reads them.
    TREE_CODE.write_uint(w, 0)?; // ctx 1: property + 1 == 0 marks a leaf
    TREE_CODE.write_uint(w, PREDICTOR_GRADIENT)?; // ctx 2: predictor
    TREE_CODE.write_uint(w, pack_signed(0))?; // ctx 3: offset
    TREE_CODE.write_uint(w, 0)?; // ctx 4: mul_log
    TREE_CODE.write_uint(w, 0)?; // ctx 5: mul_bits, so multiplier = 1
    Ok(())
}

/// Writes `rect` of every channel, in ascending channel order (H.2).
fn write_planes(w: &mut BitWriter, source: &ModularSource<'_>, rect: Rect) -> Result<()> {
    for plane in source.planes {
        write_samples(w, plane, source.width, rect, source.code)?;
    }
    Ok(())
}

/// Writes one channel's samples over `rect` in raster order (18181-1 H.3).
fn write_samples(
    w: &mut BitWriter,
    plane: &[i32],
    stride: u32,
    rect: Rect,
    code: FlatCode,
) -> Result<()> {
    let stride = usize::try_from(stride).unwrap_or(usize::MAX);
    // H.3 works inside the sub-bitstream's own channel, which for a group is
    // the group rectangle, so the neighbour lookups are rectangle-relative.
    let at = |x: u32, y: u32| -> i64 {
        let index = usize::try_from(rect.y0 + y)
            .ok()
            .and_then(|row| row.checked_mul(stride))
            .and_then(|row| {
                usize::try_from(rect.x0 + x)
                    .ok()
                    .and_then(|x| row.checked_add(x))
            });
        index
            .and_then(|i| plane.get(i))
            .map_or(0, |&v| i64::from(v))
    };

    for y in 0..rect.height {
        for x in 0..rect.width {
            let prediction = gradient_prediction(&at, x, y);
            let residual = at(x, y) - prediction;
            let residual = i32::try_from(residual).map_err(|_| EncodeError::ValueOutOfRange {
                what: "modular residual",
                value: residual,
            })?;
            code.write_uint(w, pack_signed(residual))?;
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
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
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

    /// The decoder's inverse RCT, restated from H.6.3 so the forward transform
    /// is checked against the clause rather than against one implementation.
    fn inverse_ycocg(y: i32, co: i32, cg: i32) -> (i32, i32, i32) {
        let (a, b, c) = (i64::from(y), i64::from(co), i64::from(cg));
        let tmp = a - (c >> 1);
        let green = c + tmp;
        let blue = tmp - (b >> 1);
        let red = blue + b;
        (narrow(red), narrow(green), narrow(blue))
    }

    #[test]
    fn the_forward_rct_is_the_exact_inverse_of_the_clause() {
        for r in (0..=65535i32).step_by(2731) {
            for g in (0..=65535i32).step_by(4093) {
                for b in (0..=65535i32).step_by(5051) {
                    let (y, co, cg) = rct_forward(r, g, b);
                    assert_eq!(inverse_ycocg(y, co, cg), (r, g, b), "({r}, {g}, {b})");
                }
            }
        }
        // The 8-bit corners in full, including every extreme triple.
        for r in [0i32, 1, 127, 128, 254, 255] {
            for g in [0i32, 1, 127, 128, 254, 255] {
                for b in [0i32, 1, 127, 128, 254, 255] {
                    let (y, co, cg) = rct_forward(r, g, b);
                    assert_eq!(inverse_ycocg(y, co, cg), (r, g, b), "({r}, {g}, {b})");
                }
            }
        }
    }

    #[test]
    fn the_forward_rct_matches_jpxl_decodes_inverse() {
        // Same check, against the *decoder's* H.6.3 rather than a restatement:
        // if the two disagree the round trip in `roundtrip.rs` would fail for
        // reasons no unit test localises.
        for (r, g, b) in [
            (0i32, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (65535, 0, 32768),
            (12345, 54321, 999),
        ] {
            let (y, co, cg) = rct_forward(r, g, b);
            let v = jpxl_decode::modular::rct::inverse_pixel(
                RCT_TYPE_YCOCG,
                i64::from(y),
                i64::from(co),
                i64::from(cg),
            )
            .expect("rct_type 6 is in range");
            assert_eq!(
                v,
                [i64::from(r), i64::from(g), i64::from(b)],
                "({r},{g},{b})"
            );
        }
    }

    #[test]
    fn a_mismatched_sample_count_is_rejected() {
        let planes = vec![vec![0i32; 15]];
        let source = ModularSource {
            width: 4,
            height: 4,
            planes: &planes,
            rct: false,
            code: TREE_CODE,
        };
        let geometry = Geometry::new(4, 4, 2).expect("valid");
        assert!(matches!(
            encode_lf_global(&source, &geometry),
            Err(EncodeError::SampleCountMismatch {
                expected: 16,
                found: 15
            })
        ));
    }

    #[test]
    fn lf_global_carries_no_samples_when_the_image_exceeds_one_group() {
        let planes = vec![vec![7i32; 300 * 300]];
        let source = ModularSource {
            width: 300,
            height: 300,
            planes: &planes,
            rct: false,
            code: TREE_CODE,
        };
        // group_dim 512 fits the image; group_dim 128 does not.
        let big = encode_lf_global(&source, &Geometry::new(300, 300, 2).expect("valid"))
            .expect("encodes");
        let small = encode_lf_global(&source, &Geometry::new(300, 300, 0).expect("valid"))
            .expect("encodes");
        assert!(
            small.len() * 100 < big.len(),
            "the multi-section LfGlobal must be header-only: {} vs {}",
            small.len(),
            big.len()
        );
    }
}
