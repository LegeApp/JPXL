//! The patch dictionary (18181-1 K.3).
//!
//! A patch is a rectangle copied out of a previously decoded **reference
//! frame** and blitted onto the current frame at one or more positions. It is
//! how an encoder handles content that repeats — text glyphs, logos, the same
//! object at several places — without paying for it more than once.
//!
//! ```text
//! Table G.1 — LfGlobal bundle
//! condition   type      name       subclause
//! kPatches    Patches   patches    K.3.1        <- first row, before lf_dequant
//! ```
//!
//! Two clauses, and they sit at opposite ends of the frame pipeline:
//!
//! * **K.3.1 decoding** happens in `LfGlobal`, as the *first* bundle of Table
//!   G.1 — before `LfChannelDequantization`, before the `kVarDCT` bundles,
//!   before `GlobalModular`. Getting that position wrong shifts every field of
//!   the frame after it, which is exactly how this module's absence used to
//!   present: `LfQuant` failing with "use_global_tree is set but no global MA
//!   tree was supplied", hundreds of bytes downstream of the real problem.
//! * **K.3.2 rendering** happens after the restoration filters of Annex J and
//!   before the inverse colour transforms of Annex L — K.3.2 says so twice,
//!   once as "after applying restoration filters (Annex J)" and once as "the
//!   sample values `new_sample` are in the colour space before the inverse
//!   colour transforms from L.2, L.3 and L.4 are applied". For a `kVarDCT`
//!   frame that means the blit happens in **XYB**, on the same float planes
//!   `epf` just produced.
//!
//! # Entropy layout
//!
//! K.3.1 reads ten pre-clustered distributions (C.1) and then every field of
//! every patch from one stream (C.3.3), with a fixed context per field kind.
//! The contexts are not a detail: they are the whole model, and a decoder that
//! swaps two of them still decodes plausible-looking numbers for a while. They
//! are named in [`PatchContext`] rather than left as bare integers.
//!
//! Because the stream is a complete C.3.2 stream, it ends in the terminal ANS
//! state — so a mis-numbered context, a missed field, or a wrong `+ 1` is
//! caught by [`SymbolDecoder::finish`] with no reference data at all. That is
//! the same free gate that settled the modular and HF-coefficient slices.

use jpxl_bitstream::BitReader;
use jpxl_core::limits::AllocGuard;
use jpxl_entropy::SymbolDecoder;

use crate::frame::error::{FrameError, Result};

/// K.3.1's ten pre-clustered distributions, by what each one codes.
///
/// The numbering is the clause's `DecodeHybridVarLenUint(n)` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum PatchContext {
    /// `num_patches`.
    NumPatches = 0,
    /// `patch[i].ref`, the reference-frame slot.
    Reference = 1,
    /// `width - 1` and `height - 1`.
    Size = 2,
    /// `x0` and `y0`, the patch's origin inside the reference frame.
    Origin = 3,
    /// The first position's `x` and `y`.
    FirstPosition = 4,
    /// A blend mode.
    Mode = 5,
    /// A subsequent position's delta from the previous one, `UnpackSigned`.
    PositionDelta = 6,
    /// `count - 1`, the number of positions this patch is blitted at.
    Count = 7,
    /// The alpha channel index of a blend mode above `kMulAddBelow`.
    AlphaChannel = 8,
    /// The per-position, per-channel `clamp` flag.
    Clamp = 9,
}

/// Number of distributions K.3.1 reads.
pub const NUM_PATCH_CONTEXTS: usize = 10;

/// Largest blend-mode value Table K.1 defines.
pub const MAX_PATCH_BLEND_MODE: u32 = 7;

/// 18181-1 Table K.1 — `PatchBlendMode`.
///
/// The first four rows mirror Table F.7's frame blend modes; the last four
/// come in `Above`/`Below` pairs that differ only in which of `new_sample` and
/// `old_sample` plays which role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PatchBlendMode {
    /// `sample = old_sample`. The patch contributes nothing to this channel.
    #[default]
    None = 0,
    /// `sample = new_sample`.
    Replace = 1,
    /// `sample = old_sample + new_sample`.
    Add = 2,
    /// `sample = old_sample * new_sample`.
    Mul = 3,
    /// Alpha-blend `new_sample` over `old_sample`.
    BlendAbove = 4,
    /// Alpha-blend `new_sample` under `old_sample`.
    BlendBelow = 5,
    /// Table F.7's `kMulAdd`, `new_sample` above.
    MulAddAbove = 6,
    /// Table F.7's `kMulAdd`, roles swapped.
    MulAddBelow = 7,
}

impl PatchBlendMode {
    /// Maps a wire value to a row of Table K.1.
    ///
    /// K.3.1 asserts `mode < 8`, so anything else is a malformed stream.
    #[must_use]
    pub const fn from_value(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::None,
            1 => Self::Replace,
            2 => Self::Add,
            3 => Self::Mul,
            4 => Self::BlendAbove,
            5 => Self::BlendBelow,
            6 => Self::MulAddAbove,
            7 => Self::MulAddBelow,
            _ => return None,
        })
    }

    /// Whether this mode reads an alpha channel, i.e. is above `kMul`.
    ///
    /// K.3.1 gates the `alpha_channel` field on `mode > 3`.
    #[must_use]
    pub const fn uses_alpha(self) -> bool {
        (self as u32) > 3
    }

    /// Whether this mode reads a `clamp` flag, i.e. is above `kAdd`.
    ///
    /// K.3.1 gates the `clamp` field on `mode > 2`. Note the two thresholds
    /// differ by one: `kMul` reads `clamp` but not `alpha_channel`.
    #[must_use]
    pub const fn uses_clamp(self) -> bool {
        (self as u32) > 2
    }
}

/// One channel's blending rule at one position (K.3.1's
/// `blending[j].{mode,alpha_channel,clamp}[c]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PatchBlending {
    /// Table K.1 mode.
    pub mode: PatchBlendMode,
    /// Extra-channel index of the alpha channel, when the mode uses one.
    pub alpha_channel: u32,
    /// Whether alpha is clamped to `[0, 1]` before blending.
    pub clamp: bool,
}

/// One position a patch is blitted at, with its per-channel blending.
///
/// `blending` has `num_extra + 1` entries: index 0 governs all three colour
/// channels together, index `c` governs extra channel `c - 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchPosition {
    /// Frame x of the blit's top-left corner.
    pub x: u32,
    /// Frame y of the blit's top-left corner.
    pub y: u32,
    /// Per-channel-group blending, `num_extra + 1` long.
    pub blending: Vec<PatchBlending>,
}

/// One patch: a rectangle of a reference frame, plus where it goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Patch {
    /// Reference-frame slot the rectangle is read from.
    pub reference: u32,
    /// X of the rectangle's top-left corner in the reference frame.
    pub x0: u32,
    /// Y of the rectangle's top-left corner in the reference frame.
    pub y0: u32,
    /// Rectangle width in samples (at least 1).
    pub width: u32,
    /// Rectangle height in samples (at least 1).
    pub height: u32,
    /// Where the rectangle is blitted, at least one position.
    pub positions: Vec<PatchPosition>,
}

/// A decoded patch dictionary (K.3.1).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PatchDictionary {
    /// The patches, in decode order.
    pub patches: Vec<Patch>,
}

impl PatchDictionary {
    /// Total number of blits this dictionary describes.
    ///
    /// The work K.3.2 does is proportional to this times the patch area, not
    /// to `patches.len()`, which is what makes it the useful number to bound.
    #[must_use]
    pub fn total_positions(&self) -> usize {
        self.patches.iter().map(|p| p.positions.len()).sum()
    }

    /// Whether any patch uses a blend mode that reads an alpha channel.
    #[must_use]
    pub fn uses_alpha(&self) -> bool {
        self.patches.iter().any(|p| {
            p.positions
                .iter()
                .any(|q| q.blending.iter().any(|b| b.mode.uses_alpha()))
        })
    }

    /// Every reference-frame slot this dictionary reads from.
    #[must_use]
    pub fn referenced_slots(&self) -> Vec<u32> {
        let mut slots: Vec<u32> = self.patches.iter().map(|p| p.reference).collect();
        slots.sort_unstable();
        slots.dedup();
        slots
    }
}

/// `UnpackSigned(u)` (18181-1 B.3).
const fn unpack_signed(u: u32) -> i64 {
    if u.is_multiple_of(2) {
        (u / 2) as i64
    } else {
        -((u / 2) as i64) - 1
    }
}

/// Reads the `Patches` bundle of Table G.1 (18181-1 K.3.1).
///
/// `reader` must sit at the very start of `LfGlobal`; this is the first bundle
/// of Table G.1 when the `kPatches` flag is set. On return the reader is
/// positioned at `LfChannelDequantization`.
///
/// `num_extra` is `metadata.num_extra()`; it sets how many blend rules each
/// position carries. `frame_width`/`frame_height` bound the blit positions —
/// K.3.1 states that the rectangle "is fully contained within the frame", and
/// a stream that violates it is rejected rather than clipped.
///
/// `max_patches` is Annex M's "Maximum num_patches (K.3.1)" for the level in
/// force.
///
/// # Errors
///
/// * [`FrameError::Entropy`] if the stream is malformed or does not end in the
///   C.3.2 terminal state.
/// * [`FrameError::FieldOutOfRange`] if `num_patches` exceeds `max_patches`,
///   a blend mode is not one of Table K.1's eight, an alpha channel index is
///   not a valid extra channel, a patch rectangle would fall outside the
///   frame, or a delta-coded position goes negative.
/// * [`FrameError::Core`] if the dictionary exceeds the allocation budget.
pub fn read_patches(
    reader: &mut BitReader<'_>,
    num_extra: usize,
    frame_width: u32,
    frame_height: u32,
    max_patches: u64,
    guard: &mut AllocGuard,
) -> Result<PatchDictionary> {
    let mut decoder = SymbolDecoder::open(reader, NUM_PATCH_CONTEXTS, guard)?;
    let read = |decoder: &mut SymbolDecoder, ctx: PatchContext, r: &mut BitReader<'_>| {
        decoder.read_uint(r, ctx as usize).map_err(FrameError::from)
    };

    let num_patches = read(&mut decoder, PatchContext::NumPatches, reader)?;
    if u64::from(num_patches) > max_patches {
        return Err(FrameError::out_of_range(
            "num_patches",
            "K.3.1",
            u64::from(num_patches),
        ));
    }
    // A patch costs at least its struct plus one position; charge before the
    // Vec is reserved so a huge count cannot allocate first and fail second.
    guard
        .charge(u64::from(num_patches).saturating_mul(64))
        .map_err(FrameError::Core)?;

    let mut patches = Vec::with_capacity(num_patches as usize);
    for _ in 0..num_patches {
        let reference = read(&mut decoder, PatchContext::Reference, reader)?;
        let x0 = read(&mut decoder, PatchContext::Origin, reader)?;
        let y0 = read(&mut decoder, PatchContext::Origin, reader)?;
        let width = read(&mut decoder, PatchContext::Size, reader)?
            .checked_add(1)
            .ok_or_else(|| FrameError::out_of_range("patch width", "K.3.1", u64::MAX))?;
        let height = read(&mut decoder, PatchContext::Size, reader)?
            .checked_add(1)
            .ok_or_else(|| FrameError::out_of_range("patch height", "K.3.1", u64::MAX))?;
        let count = read(&mut decoder, PatchContext::Count, reader)?
            .checked_add(1)
            .ok_or_else(|| FrameError::out_of_range("patch count", "K.3.1", u64::MAX))?;

        // Each position carries num_extra + 1 blend rules; meter the whole
        // patch's positions before reading any of them.
        guard
            .charge(
                u64::from(count)
                    .saturating_mul(num_extra as u64 + 1)
                    .saturating_mul(16),
            )
            .map_err(FrameError::Core)?;

        let mut positions = Vec::with_capacity(count as usize);
        let (mut last_x, mut last_y) = (0i64, 0i64);
        for j in 0..count {
            let (x, y) = if j == 0 {
                (
                    i64::from(read(&mut decoder, PatchContext::FirstPosition, reader)?),
                    i64::from(read(&mut decoder, PatchContext::FirstPosition, reader)?),
                )
            } else {
                let dx = unpack_signed(read(&mut decoder, PatchContext::PositionDelta, reader)?);
                let dy = unpack_signed(read(&mut decoder, PatchContext::PositionDelta, reader)?);
                (last_x + dx, last_y + dy)
            };
            last_x = x;
            last_y = y;

            // K.3.1: "the width x height rectangle with top-left coordinates
            // (x, y) is fully contained within the frame". Rejecting rather
            // than clipping is what keeps a malformed stream from writing
            // outside the canvas — the check is the bounds proof for K.3.2's
            // inner loop, which then needs none.
            let (x, y) = (
                u32::try_from(x)
                    .map_err(|_| FrameError::out_of_range("patch x", "K.3.1", x.unsigned_abs()))?,
                u32::try_from(y)
                    .map_err(|_| FrameError::out_of_range("patch y", "K.3.1", y.unsigned_abs()))?,
            );
            if x.checked_add(width).is_none_or(|e| e > frame_width)
                || y.checked_add(height).is_none_or(|e| e > frame_height)
            {
                return Err(FrameError::out_of_range(
                    "patch rectangle outside the frame",
                    "K.3.1",
                    u64::from(x),
                ));
            }

            let mut blending = Vec::with_capacity(num_extra + 1);
            for _ in 0..num_extra + 1 {
                let raw = read(&mut decoder, PatchContext::Mode, reader)?;
                let mode = PatchBlendMode::from_value(raw).ok_or_else(|| {
                    FrameError::out_of_range("patch blend mode", "K.3.1", u64::from(raw))
                })?;
                // K.3.1 reads alpha_channel only when the mode uses alpha
                // *and* there is more than one alpha channel; with zero or one
                // alpha channel the index is implied.
                // ALPHA_GUARD_COUNTS_EXTRA_CHANNELS: "more than 1 alpha
                // channel" read as "more than one extra channel".
                let alpha_channel =
                    if mode.uses_alpha() && ALPHA_GUARD_COUNTS_EXTRA_CHANNELS && num_extra > 1 {
                        let index = read(&mut decoder, PatchContext::AlphaChannel, reader)?;
                        if index as usize >= num_extra {
                            return Err(FrameError::out_of_range(
                                "patch alpha_channel",
                                "K.3.1",
                                u64::from(index),
                            ));
                        }
                        index
                    } else {
                        0
                    };
                let clamp = if mode.uses_clamp() {
                    read(&mut decoder, PatchContext::Clamp, reader)? != 0
                } else {
                    false
                };
                blending.push(PatchBlending {
                    mode,
                    alpha_channel,
                    clamp,
                });
            }
            positions.push(PatchPosition { x, y, blending });
        }

        patches.push(Patch {
            reference,
            x0,
            y0,
            width,
            height,
            positions,
        });
    }

    // C.3.2: the dictionary is a complete stream, so it must land exactly on
    // the terminal state. This is the gate that proves the context numbering
    // and every `+ 1` without any reference pixels.
    decoder.finish()?;
    Ok(PatchDictionary { patches })
}

/// **Flip point — K.3.1's "more than 1 alpha channel" guard.**
///
/// The clause gates `alpha_channel` on `mode > 3 and /* there is more than 1
/// alpha channel */`. Whether "alpha channel" there counts extra channels of
/// type `kAlpha` specifically, or extra channels generally, changes the *bit
/// count* of a patch dictionary for any image with two or more extra channels
/// of which fewer than two are alpha — and a wrong bit count desynchronises
/// the rest of the dictionary, not just this field.
///
/// `true` (shipped) counts extra channels: the field it guards is a plain
/// extra-channel index, validated as one, and this reading keeps the bit count
/// a function of `num_extra` alone, as every other conditional field in the
/// format is a function of already-decoded header values.
///
/// **Unexercised.** [`read_patches`]'s callers reject `num_extra > 0` before
/// reaching it, and no available stream combines patches with extra channels
/// at all. The alternative reading additionally needs the extra-channel
/// *types* threaded into this function, which is why flipping it is not a
/// one-line change; the constant marks the site, not a working alternative.
pub const ALPHA_GUARD_COUNTS_EXTRA_CHANNELS: bool = true;

/// Annex M's "Maximum num_patches (K.3.1)": `min(1 << 24, fwidth * fheight /
/// 16)`, the same at both levels.
#[must_use]
pub fn max_num_patches(frame_width: u32, frame_height: u32) -> u64 {
    let area = u64::from(frame_width).saturating_mul(u64::from(frame_height));
    (1u64 << 24).min(area / 16)
}

// ---------------------------------------------------------------------------
// K.3.2 — blending one sample
// ---------------------------------------------------------------------------

/// K.3.2 / Table K.1: blends one patch sample over one canvas sample.
///
/// `alpha` is the alpha value the mode reads, already selected from the right
/// extra channel by the caller and already clamped if `clamp` was set; it is
/// ignored by the four modes that do not use alpha.
///
/// The `Below` variants are the `Above` ones with the two samples swapped,
/// which is what Table K.1 says and is why they share one implementation.
#[must_use]
pub fn blend(mode: PatchBlendMode, old_sample: f32, new_sample: f32, alpha: f32) -> f32 {
    match mode {
        PatchBlendMode::None => old_sample,
        PatchBlendMode::Replace => new_sample,
        PatchBlendMode::Add => old_sample + new_sample,
        PatchBlendMode::Mul => old_sample * new_sample,
        // Table F.7 kBlend: alpha-composite the top sample over the bottom.
        PatchBlendMode::BlendAbove => alpha.mul_add(new_sample - old_sample, old_sample),
        PatchBlendMode::BlendBelow => alpha.mul_add(old_sample - new_sample, new_sample),
        // Table F.7 kMulAdd: the bottom sample plus alpha times the top one.
        PatchBlendMode::MulAddAbove => alpha.mul_add(new_sample, old_sample),
        PatchBlendMode::MulAddBelow => alpha.mul_add(old_sample, new_sample),
    }
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "tests index structures they just built; a panic is a failure"
)]
mod tests {
    use super::*;

    #[test]
    fn table_k1_values_round_trip() {
        // Proves the eight rows of Table K.1 map to the wire values the clause
        // prints, and that a ninth value is rejected rather than defaulted.
        for v in 0..=MAX_PATCH_BLEND_MODE {
            let mode = PatchBlendMode::from_value(v).expect("Table K.1 row");
            assert_eq!(mode as u32, v);
        }
        assert!(PatchBlendMode::from_value(8).is_none());
    }

    #[test]
    fn the_two_field_guards_have_different_thresholds() {
        // The trap this test exists for: K.3.1 gates alpha_channel on
        // `mode > 3` and clamp on `mode > 2`, so kMul reads clamp and not
        // alpha. Collapsing the two guards costs or gains one field per
        // channel per position.
        assert!(!PatchBlendMode::Mul.uses_alpha());
        assert!(PatchBlendMode::Mul.uses_clamp());
        assert!(!PatchBlendMode::Add.uses_clamp());
        assert!(PatchBlendMode::BlendAbove.uses_alpha());
        assert!(PatchBlendMode::BlendAbove.uses_clamp());
    }

    #[test]
    fn blend_modes_match_table_k1() {
        // Hand-computed values for every row, with old = 0.25, new = 0.75,
        // alpha = 0.5.
        let (o, n, a) = (0.25f32, 0.75f32, 0.5f32);
        assert_eq!(blend(PatchBlendMode::None, o, n, a), 0.25);
        assert_eq!(blend(PatchBlendMode::Replace, o, n, a), 0.75);
        assert_eq!(blend(PatchBlendMode::Add, o, n, a), 1.0);
        assert_eq!(blend(PatchBlendMode::Mul, o, n, a), 0.1875);
        // 0.25 + 0.5 * (0.75 - 0.25) = 0.5
        assert_eq!(blend(PatchBlendMode::BlendAbove, o, n, a), 0.5);
        // 0.75 + 0.5 * (0.25 - 0.75) = 0.5 -- symmetric at alpha 0.5.
        assert_eq!(blend(PatchBlendMode::BlendBelow, o, n, a), 0.5);
        // 0.25 + 0.5 * 0.75 = 0.625
        assert_eq!(blend(PatchBlendMode::MulAddAbove, o, n, a), 0.625);
        // 0.75 + 0.5 * 0.25 = 0.875
        assert_eq!(blend(PatchBlendMode::MulAddBelow, o, n, a), 0.875);
    }

    #[test]
    fn the_above_below_pairs_are_each_others_mirror() {
        // Table K.1 defines the Below rows purely as the Above rows with the
        // samples swapped; that identity is the whole specification of them,
        // so it is worth asserting directly over a range of inputs.
        for &o in &[-0.5f32, 0.0, 0.25, 1.5] {
            for &n in &[-0.25f32, 0.0, 0.75, 2.0] {
                for &a in &[0.0f32, 0.3, 1.0] {
                    assert_eq!(
                        blend(PatchBlendMode::BlendBelow, o, n, a),
                        blend(PatchBlendMode::BlendAbove, n, o, a),
                    );
                    assert_eq!(
                        blend(PatchBlendMode::MulAddBelow, o, n, a),
                        blend(PatchBlendMode::MulAddAbove, n, o, a),
                    );
                }
            }
        }
    }

    #[test]
    fn alpha_zero_and_one_are_the_endpoints_of_the_blend() {
        // Proves the alpha-blend orientation: at alpha 0 the sample below
        // wins, at alpha 1 the sample above does. Getting this backwards is
        // invisible on a symmetric test case, which is why it is separate.
        let (o, n) = (0.25f32, 0.75f32);
        assert_eq!(blend(PatchBlendMode::BlendAbove, o, n, 0.0), o);
        assert_eq!(blend(PatchBlendMode::BlendAbove, o, n, 1.0), n);
        assert_eq!(blend(PatchBlendMode::BlendBelow, o, n, 0.0), n);
        assert_eq!(blend(PatchBlendMode::BlendBelow, o, n, 1.0), o);
    }

    #[test]
    fn unpack_signed_matches_b3() {
        assert_eq!(unpack_signed(0), 0);
        assert_eq!(unpack_signed(1), -1);
        assert_eq!(unpack_signed(2), 1);
        assert_eq!(unpack_signed(3), -2);
        assert_eq!(unpack_signed(4), 2);
    }

    #[test]
    fn context_numbering_is_the_clauses_own() {
        // The context per field is the entire K.3.1 model; a transposition
        // decodes plausible numbers for a while and then desynchronises.
        assert_eq!(PatchContext::NumPatches as usize, 0);
        assert_eq!(PatchContext::Reference as usize, 1);
        assert_eq!(PatchContext::Size as usize, 2);
        assert_eq!(PatchContext::Origin as usize, 3);
        assert_eq!(PatchContext::FirstPosition as usize, 4);
        assert_eq!(PatchContext::Mode as usize, 5);
        assert_eq!(PatchContext::PositionDelta as usize, 6);
        assert_eq!(PatchContext::Count as usize, 7);
        assert_eq!(PatchContext::AlphaChannel as usize, 8);
        assert_eq!(PatchContext::Clamp as usize, 9);
        assert_eq!(NUM_PATCH_CONTEXTS, 10);
    }

    #[test]
    fn dictionary_summaries() {
        let d = PatchDictionary {
            patches: vec![
                Patch {
                    reference: 2,
                    x0: 0,
                    y0: 0,
                    width: 1,
                    height: 1,
                    positions: vec![
                        PatchPosition {
                            x: 0,
                            y: 0,
                            blending: vec![PatchBlending::default()],
                        },
                        PatchPosition {
                            x: 1,
                            y: 0,
                            blending: vec![PatchBlending {
                                mode: PatchBlendMode::BlendAbove,
                                ..PatchBlending::default()
                            }],
                        },
                    ],
                },
                Patch {
                    reference: 2,
                    x0: 0,
                    y0: 0,
                    width: 1,
                    height: 1,
                    positions: vec![PatchPosition {
                        x: 2,
                        y: 0,
                        blending: vec![PatchBlending::default()],
                    }],
                },
            ],
        };
        assert_eq!(d.total_positions(), 3);
        assert!(d.uses_alpha());
        assert_eq!(d.referenced_slots(), vec![2]);
    }
}
