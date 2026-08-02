//! Transform signalling and the channel-list algebra (18181-1 H.6.1).
//!
//! ```text
//! Table H.7 — TransformInfo bundle
//! condition       type                                              name
//!                 u(2) (TransformId)                                tr
//! tr != kSqueeze  U32(u(3), 8 + u(6), 72 + u(10), 1096 + u(13))     begin_c
//! tr == kRCT      U32(6, u(2), 2 + u(4), 10 + u(6))                 rct_type
//! tr == kPalette  U32(1, 3, 4, 1 + u(13))                           num_c
//! tr == kPalette  U32(u(8), 256 + u(10), 1280 + u(12), 5376+u(16))  nb_colours
//! tr == kPalette  U32(0, 1 + u(8), 257 + u(10), 1281 + u(16))       nb_deltas
//! tr == kPalette  u(4)                                              d_pred
//! tr == kSqueeze  U32(0, 1 + u(4), 9 + u(6), 41 + u(8))             num_sq
//! tr == kSqueeze  SqueezeParams (H.6.2.1)                           sp[num_sq]
//! ```
//!
//! ```text
//! Table H.8 — Transforms
//! kRCT      nb_meta_channels unchanged, channel list unchanged        H.6.3
//! kPalette  nb_meta_channels += 1, or += 2 - num_c if begin_c is a
//!           meta-channel; channels begin_c+1 .. begin_c+num_c-1 go   H.6.4
//! kSqueeze  both depend on the parameters                            H.6.2
//! ```
//!
//! # Two passes over the same list
//!
//! H.2 is explicit that the channel *shapes* are derived from the transform
//! chain before any sample is decoded: the decoder replays the forward channel
//! bookkeeping (never the sample arithmetic) to learn how many channels there
//! are and how big each one is, decodes them, and only then runs the inverse
//! transforms from last to first. [`ChannelLayout::apply_forward`] is the first
//! pass and [`apply_inverse`] is the second.
//!
//! The squeeze default parameters are resolved during the forward pass and
//! stored, because they depend on the channel list *at that point* in the
//! chain, which no longer exists by the time the inverse runs.

use jpxl_bitstream::{BitReader, U32Dist, U32Spec, read_u32, trace_field};
use jpxl_core::limits::AllocGuard;

use super::channel::{Channel, ChannelSpec, SHIFT_UNRELATED};
use super::error::{Result, malformed};
use super::palette::{PaletteContext, PaletteParams};
use super::squeeze::{self, BEGIN_C_SPEC, SqueezeParams};
use super::{palette, rct};

/// H.6.1: `U32(6, u(2), 2 + u(4), 10 + u(6))`.
const RCT_TYPE_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(6),
    U32Dist::BitsOffset { bits: 2, offset: 0 },
    U32Dist::BitsOffset { bits: 4, offset: 2 },
    U32Dist::BitsOffset {
        bits: 6,
        offset: 10,
    },
]);

/// H.6.1: `U32(1, 3, 4, 1 + u(13))`.
const PALETTE_NUM_C_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(1),
    U32Dist::Val(3),
    U32Dist::Val(4),
    U32Dist::BitsOffset {
        bits: 13,
        offset: 1,
    },
]);

/// H.6.1: `U32(u(8), 256 + u(10), 1280 + u(12), 5376 + u(16))`.
const NB_COLOURS_SPEC: U32Spec = U32Spec::new([
    U32Dist::BitsOffset { bits: 8, offset: 0 },
    U32Dist::BitsOffset {
        bits: 10,
        offset: 256,
    },
    U32Dist::BitsOffset {
        bits: 12,
        offset: 1280,
    },
    U32Dist::BitsOffset {
        bits: 16,
        offset: 5376,
    },
]);

/// H.6.1: `U32(0, 1 + u(8), 257 + u(10), 1281 + u(16))`.
const NB_DELTAS_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::BitsOffset { bits: 8, offset: 1 },
    U32Dist::BitsOffset {
        bits: 10,
        offset: 257,
    },
    U32Dist::BitsOffset {
        bits: 16,
        offset: 1281,
    },
]);

/// H.6.1: `U32(0, 1 + u(4), 9 + u(6), 41 + u(8))`.
const NUM_SQ_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::BitsOffset { bits: 4, offset: 1 },
    U32Dist::BitsOffset { bits: 6, offset: 9 },
    U32Dist::BitsOffset {
        bits: 8,
        offset: 41,
    },
]);

/// `TransformId` values of Table H.6.
pub const ID_RCT: u32 = 0;
/// See [`ID_RCT`].
pub const ID_PALETTE: u32 = 1;
/// See [`ID_RCT`].
pub const ID_SQUEEZE: u32 = 2;

/// One signalled transform (Table H.7), with squeeze defaults resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transform {
    /// `kRCT` (H.6.3).
    Rct {
        /// First of the three affected channels.
        begin_c: u32,
        /// Combined permutation and decorrelation selector, below 42.
        rct_type: u32,
    },
    /// `kPalette` (H.6.4).
    Palette(PaletteParams),
    /// `kSqueeze` (H.6.2).
    Squeeze {
        /// The step list. Empty on the wire means "use the H.6.2.1 defaults",
        /// which [`ChannelLayout::apply_forward`] substitutes in place.
        steps: Vec<SqueezeParams>,
    },
}

impl Transform {
    /// Reads one `TransformInfo` bundle (Table H.7).
    ///
    /// # Errors
    ///
    /// [`ModularError::Malformed`](super::ModularError::Malformed) for a
    /// `TransformId` of 3, which Table H.6 does not define, or for an
    /// out-of-range `rct_type`; [`ModularError::Core`](super::ModularError::Core)
    /// if the squeeze step list exceeds the guard.
    pub fn read(reader: &mut BitReader<'_>, guard: &mut AllocGuard) -> Result<Self> {
        let tr = trace_field!(reader, "transform.tr", reader.read_bits(2))?;
        if tr == ID_SQUEEZE {
            let num_sq = trace_field!(reader, "transform.num_sq", read_u32(reader, &NUM_SQ_SPEC))?;
            guard.charge(u64::from(num_sq) * size_of::<SqueezeParams>() as u64)?;
            let mut steps = Vec::with_capacity(num_sq as usize);
            for _ in 0..num_sq {
                steps.push(SqueezeParams::read(reader)?);
            }
            return Ok(Self::Squeeze { steps });
        }

        let begin_c = trace_field!(reader, "transform.begin_c", read_u32(reader, &BEGIN_C_SPEC))?;
        match tr {
            ID_RCT => {
                let rct_type = trace_field!(
                    reader,
                    "transform.rct_type",
                    read_u32(reader, &RCT_TYPE_SPEC)
                )?;
                if rct_type >= rct::MAX_RCT_TYPE {
                    return Err(malformed!(
                        "H.6.3: rct_type = {rct_type} is not below {}",
                        rct::MAX_RCT_TYPE
                    ));
                }
                Ok(Self::Rct { begin_c, rct_type })
            }
            ID_PALETTE => {
                let num_c = trace_field!(
                    reader,
                    "transform.num_c",
                    read_u32(reader, &PALETTE_NUM_C_SPEC)
                )?;
                let nb_colours = trace_field!(
                    reader,
                    "transform.nb_colours",
                    read_u32(reader, &NB_COLOURS_SPEC)
                )?;
                let nb_deltas = trace_field!(
                    reader,
                    "transform.nb_deltas",
                    read_u32(reader, &NB_DELTAS_SPEC)
                )?;
                let d_pred = trace_field!(reader, "transform.d_pred", reader.read_bits(4))?;
                Ok(Self::Palette(PaletteParams {
                    begin_c,
                    num_c,
                    nb_colours,
                    nb_deltas,
                    d_pred,
                }))
            }
            other => Err(malformed!(
                "H.6.1 Table H.6: TransformId {other} is not one of kRCT, kPalette, kSqueeze"
            )),
        }
    }
}

/// The channel list as it exists between transforms, without sample data.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ChannelLayout {
    /// The channels, in index order.
    pub specs: Vec<ChannelSpec>,
    /// How many leading channels are palette meta-channels (H.1).
    pub nb_meta_channels: usize,
}

impl ChannelLayout {
    /// Starts from the initial list of H.1, with no meta-channels.
    #[must_use]
    pub fn new(specs: Vec<ChannelSpec>) -> Self {
        Self {
            specs,
            nb_meta_channels: 0,
        }
    }

    /// Applies one transform's effect on the channel list (Table H.8).
    ///
    /// Mutates `transform` when it is a squeeze with an empty step list, so the
    /// resolved defaults are available to the inverse pass.
    ///
    /// # Errors
    ///
    /// [`ModularError::Malformed`](super::ModularError::Malformed) if the
    /// transform refers to channels that do not exist, or to channels whose
    /// dimensions the transform requires to match and which do not, or if the
    /// result would exceed `max_channels`.
    pub fn apply_forward(&mut self, transform: &mut Transform, max_channels: usize) -> Result<()> {
        match transform {
            Transform::Rct { begin_c, rct_type } => self.forward_rct(*begin_c, *rct_type),
            Transform::Palette(params) => self.forward_palette(params, max_channels),
            Transform::Squeeze { steps } => {
                if steps.is_empty() {
                    *steps = squeeze::default_params(&self.specs, self.nb_meta_channels)?;
                }
                self.forward_squeeze(steps, max_channels)
            }
        }
    }

    /// H.6.3: the RCT changes nothing, but its preconditions are checked here
    /// so a malformed stream fails before any sample is decoded.
    fn forward_rct(&mut self, begin_c: u32, rct_type: u32) -> Result<()> {
        if rct_type >= rct::MAX_RCT_TYPE {
            return Err(malformed!("H.6.3: rct_type = {rct_type} is out of range"));
        }
        let begin = begin_c as usize;
        let end = begin
            .checked_add(3)
            .ok_or_else(|| malformed!("H.6.3: begin_c + 3 overflows"))?;
        if end > self.specs.len() {
            return Err(malformed!(
                "H.6.3: RCT over channels {begin}..{end} but only {} exist",
                self.specs.len()
            ));
        }
        // H.6.3: either the three channels are all outside the meta-channels or
        // all inside them.
        if !(begin >= self.nb_meta_channels || end <= self.nb_meta_channels) {
            return Err(malformed!(
                "H.6.3: RCT straddles the meta-channel boundary at {}",
                self.nb_meta_channels
            ));
        }
        let Some(block) = self.specs.get(begin..end) else {
            return Err(malformed!("H.6.3: RCT channel range vanished"));
        };
        let first = block.first().copied().unwrap_or(ChannelSpec::new(0, 0));
        if block
            .iter()
            .any(|s| s.width != first.width || s.height != first.height)
        {
            return Err(malformed!(
                "H.6.3: the three RCT channels must have the same dimensions"
            ));
        }
        Ok(())
    }

    /// H.6.4 / Table H.8: `num_c` channels collapse to one index channel, and
    /// a palette meta-channel appears at the head of the list.
    fn forward_palette(&mut self, params: &PaletteParams, max_channels: usize) -> Result<()> {
        let num_c = params.num_c as usize;
        if num_c == 0 {
            return Err(malformed!("H.6.4: palette with num_c = 0"));
        }
        let begin = params.begin_c as usize;
        let end = begin
            .checked_add(num_c)
            .ok_or_else(|| malformed!("H.6.4: begin_c + num_c overflows"))?;
        if end > self.specs.len() {
            return Err(malformed!(
                "H.6.4: palette over channels {begin}..{end} but only {} exist",
                self.specs.len()
            ));
        }
        let Some(block) = self.specs.get(begin..end) else {
            return Err(malformed!("H.6.4: palette channel range vanished"));
        };
        let first = block.first().copied().unwrap_or(ChannelSpec::new(0, 0));
        if block.iter().any(|s| *s != first) {
            return Err(malformed!(
                "H.6.4: the palette channels must have identical dimensions and shifts"
            ));
        }

        // nb_meta_channels bookkeeping, H.6.4.
        if begin < self.nb_meta_channels {
            if end > self.nb_meta_channels {
                return Err(malformed!(
                    "H.6.4: a palette starting inside the meta-channels must end inside them"
                ));
            }
            // += 2 - num_c, which is negative for num_c > 2.
            let updated = (self.nb_meta_channels as i64) + 2 - (num_c as i64);
            if updated < 0 {
                return Err(malformed!("H.6.4: nb_meta_channels would go negative"));
            }
            self.nb_meta_channels = usize::try_from(updated)
                .map_err(|_| malformed!("H.6.4: nb_meta_channels overflows"))?;
        } else {
            self.nb_meta_channels += 1;
        }

        // Channels begin+1 .. end-1 are removed, then the meta-channel goes in
        // at index 0.
        self.specs.drain(begin + 1..end);
        self.specs.insert(
            0,
            ChannelSpec::with_shifts(
                params.nb_colours,
                params.num_c,
                SHIFT_UNRELATED,
                SHIFT_UNRELATED,
            ),
        );
        self.check_channel_count(max_channels)
    }

    /// H.6.2: each step halves `num_c` channels and inserts their residuals.
    fn forward_squeeze(&mut self, steps: &[SqueezeParams], max_channels: usize) -> Result<()> {
        for step in steps {
            let begin = step.begin_c as usize;
            let end = step.end_c()? as usize;
            if end >= self.specs.len() {
                return Err(malformed!(
                    "H.6.2: squeeze over channels {begin}..={end} but only {} exist",
                    self.specs.len()
                ));
            }
            let r = if step.in_place {
                end + 1
            } else {
                self.specs.len()
            };
            if begin < self.nb_meta_channels {
                if !step.in_place || end >= self.nb_meta_channels {
                    return Err(malformed!(
                        "H.6.2: a squeeze inside the meta-channels must be in place and stay \
                         inside them"
                    ));
                }
                self.nb_meta_channels += step.num_c as usize;
            }
            for c in begin..=end {
                let Some(spec) = self.specs.get_mut(c) else {
                    return Err(malformed!("H.6.2: squeeze channel {c} vanished"));
                };
                let (w, h) = (spec.width, spec.height);
                if w == 0 || h == 0 {
                    return Err(malformed!(
                        "H.6.2: cannot squeeze the empty channel {c} ({w}x{h})"
                    ));
                }
                let residu = if step.horizontal {
                    spec.width = w.div_ceil(2);
                    if spec.hshift >= 0 {
                        spec.hshift += 1;
                    }
                    ChannelSpec {
                        width: w / 2,
                        ..*spec
                    }
                } else {
                    spec.height = h.div_ceil(2);
                    if spec.vshift >= 0 {
                        spec.vshift += 1;
                    }
                    ChannelSpec {
                        height: h / 2,
                        ..*spec
                    }
                };
                let at = r
                    .checked_add(c - begin)
                    .ok_or_else(|| malformed!("H.6.2: residual index overflows"))?;
                if at > self.specs.len() {
                    return Err(malformed!("H.6.2: cannot insert a residual at {at}"));
                }
                self.specs.insert(at, residu);
            }
            self.check_channel_count(max_channels)?;
        }
        Ok(())
    }

    fn check_channel_count(&self, max_channels: usize) -> Result<()> {
        if self.specs.len() > max_channels {
            return Err(malformed!(
                "Annex M nb_channels_tr: the transform chain produces {} channels, over the \
                 limit of {max_channels}",
                self.specs.len()
            ));
        }
        Ok(())
    }
}

/// Applies the inverse of one transform to decoded channels (H.6).
///
/// Called from last transform to first, per H.2. The `nb_meta_channels`
/// counter is *not* touched here: it is restored from the snapshot the forward
/// pass took, because the Table H.8 update rules are not invertible from the
/// post-transform state alone (`+= 2 - num_c` and `+= 1` can land on the same
/// value).
///
/// # Errors
///
/// Whatever the individual inverse reports; see [`rct::apply_inverse`],
/// [`palette::apply_inverse`] and H.6.2.
pub fn apply_inverse(
    channels: &mut Vec<Channel>,
    transform: &Transform,
    ctx: &PaletteContext,
    guard: &mut AllocGuard,
) -> Result<()> {
    match transform {
        Transform::Rct { begin_c, rct_type } => {
            rct::apply_inverse(channels, *begin_c as usize, *rct_type)
        }
        Transform::Palette(params) => palette::apply_inverse(channels, params, ctx, guard),
        Transform::Squeeze { steps } => inverse_squeeze(channels, steps, guard),
    }
}

/// H.6.2: the squeeze steps are undone in reverse order.
///
/// ```text
/// for (i = sp.size() - 1; i >= 0; i--) {
///   r = sp[i].in_place ? end + 1 : channel.size() + begin - end - 1;
///   for (c = begin; c <= end; c++) {
///     output = channel[c].copy();
///     ... horiz_isqueeze / vert_isqueeze ...
///     channel[c] = output;
///     /* Remove the channel with index r */
///   }
/// }
/// ```
///
/// `r` is computed once per step and stays valid across the inner loop because
/// each iteration removes exactly the channel it just consumed, shifting the
/// next residual down into the same slot.
fn inverse_squeeze(
    channels: &mut Vec<Channel>,
    steps: &[SqueezeParams],
    guard: &mut AllocGuard,
) -> Result<()> {
    for step in steps.iter().rev() {
        let begin = step.begin_c as usize;
        let end = step.end_c()? as usize;
        if end >= channels.len() {
            return Err(malformed!(
                "H.6.2: inverse squeeze over channels {begin}..={end} but only {} exist",
                channels.len()
            ));
        }
        let r = if step.in_place {
            end + 1
        } else {
            channels
                .len()
                .checked_add(begin)
                .and_then(|v| v.checked_sub(end + 1))
                .ok_or_else(|| malformed!("H.6.2: residual index underflows"))?
        };
        for c in begin..=end {
            if r >= channels.len() {
                return Err(malformed!(
                    "H.6.2: residual channel {r} is past the {} that exist",
                    channels.len()
                ));
            }
            if r == c {
                return Err(malformed!(
                    "H.6.2: residual channel {r} coincides with its low-pass channel"
                ));
            }
            let (low, residual) = (
                channels
                    .get(c)
                    .ok_or_else(|| malformed!("H.6.2: low-pass channel {c} vanished"))?,
                channels
                    .get(r)
                    .ok_or_else(|| malformed!("H.6.2: residual channel {r} vanished"))?,
            );
            let output = if step.horizontal {
                squeeze::horiz_isqueeze(low, residual, guard)?
            } else {
                squeeze::vert_isqueeze(low, residual, guard)?
            };
            if let Some(slot) = channels.get_mut(c) {
                *slot = output;
            }
            channels.remove(r);
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    reason = "hand-written spec vectors read better with direct indexing; a panic \
              in a test is a failing test"
)]
mod tests {
    use jpxl_core::limits::Limits;

    use super::*;

    fn guard() -> AllocGuard {
        AllocGuard::new(&Limits::relaxed())
    }

    const MANY: usize = 1 << 16;

    #[test]
    fn rct_leaves_the_layout_alone() {
        let mut layout = ChannelLayout::new(vec![ChannelSpec::new(4, 4); 3]);
        let before = layout.clone();
        let mut t = Transform::Rct {
            begin_c: 0,
            rct_type: 0,
        };
        layout.apply_forward(&mut t, MANY).expect("valid rct");
        assert_eq!(layout, before);
    }

    #[test]
    fn rct_rejects_mismatched_or_missing_channels() {
        let mut layout = ChannelLayout::new(vec![ChannelSpec::new(4, 4); 2]);
        let mut t = Transform::Rct {
            begin_c: 0,
            rct_type: 0,
        };
        assert!(layout.apply_forward(&mut t, MANY).is_err(), "needs three");

        let mut layout = ChannelLayout::new(vec![
            ChannelSpec::new(4, 4),
            ChannelSpec::new(4, 4),
            ChannelSpec::new(2, 4),
        ]);
        assert!(layout.apply_forward(&mut t, MANY).is_err(), "size mismatch");
    }

    #[test]
    fn palette_collapses_channels_and_prepends_a_meta_channel() {
        let mut layout = ChannelLayout::new(vec![ChannelSpec::new(8, 8); 3]);
        let mut t = Transform::Palette(PaletteParams {
            begin_c: 0,
            num_c: 3,
            nb_colours: 17,
            nb_deltas: 0,
            d_pred: 0,
        });
        layout.apply_forward(&mut t, MANY).expect("valid palette");

        assert_eq!(layout.specs.len(), 2, "meta-channel + index channel");
        assert_eq!(layout.nb_meta_channels, 1);
        assert_eq!(
            layout.specs[0],
            ChannelSpec::with_shifts(17, 3, SHIFT_UNRELATED, SHIFT_UNRELATED),
            "the palette is nb_colours x num_c with unrelated shifts"
        );
        assert_eq!(layout.specs[1], ChannelSpec::new(8, 8), "index channel");
    }

    #[test]
    fn palette_inside_the_meta_channels_uses_the_other_counter_rule() {
        // Two existing meta-channels, and the palette covers both of them.
        let mut layout = ChannelLayout {
            specs: vec![ChannelSpec::new(4, 4); 3],
            nb_meta_channels: 2,
        };
        let mut t = Transform::Palette(PaletteParams {
            begin_c: 0,
            num_c: 2,
            nb_colours: 4,
            nb_deltas: 0,
            d_pred: 0,
        });
        layout.apply_forward(&mut t, MANY).expect("valid");
        // nb_meta_channels += 2 - num_c = 0.
        assert_eq!(layout.nb_meta_channels, 2);
    }

    #[test]
    fn horizontal_squeeze_halves_widths_and_inserts_residuals() {
        let mut layout = ChannelLayout::new(vec![ChannelSpec::new(7, 4), ChannelSpec::new(7, 4)]);
        let mut t = Transform::Squeeze {
            steps: vec![SqueezeParams {
                horizontal: true,
                in_place: true,
                begin_c: 0,
                num_c: 2,
            }],
        };
        layout.apply_forward(&mut t, MANY).expect("valid squeeze");

        // Each 7-wide channel becomes a 4-wide low-pass and a 3-wide residual,
        // and in_place puts the residuals right after the pair.
        assert_eq!(layout.specs.len(), 4);
        assert_eq!(layout.specs[0], ChannelSpec::with_shifts(4, 4, 1, 0));
        assert_eq!(layout.specs[1], ChannelSpec::with_shifts(4, 4, 1, 0));
        assert_eq!(layout.specs[2], ChannelSpec::with_shifts(3, 4, 1, 0));
        assert_eq!(layout.specs[3], ChannelSpec::with_shifts(3, 4, 1, 0));
    }

    #[test]
    fn out_of_place_squeeze_appends_residuals() {
        let mut layout = ChannelLayout::new(vec![ChannelSpec::new(4, 4); 3]);
        let mut t = Transform::Squeeze {
            steps: vec![SqueezeParams {
                horizontal: false,
                in_place: false,
                begin_c: 0,
                num_c: 2,
            }],
        };
        layout.apply_forward(&mut t, MANY).expect("valid");
        assert_eq!(layout.specs.len(), 5);
        // Channels 0 and 1 are squeezed, channel 2 untouched, residuals last.
        assert_eq!(layout.specs[0].height, 2);
        assert_eq!(layout.specs[1].height, 2);
        assert_eq!(layout.specs[2], ChannelSpec::new(4, 4));
        assert_eq!(layout.specs[3].height, 2);
        assert_eq!(layout.specs[4].height, 2);
    }

    #[test]
    fn empty_squeeze_step_list_resolves_the_defaults_in_place() {
        let mut layout = ChannelLayout::new(vec![ChannelSpec::new(32, 8)]);
        let mut t = Transform::Squeeze { steps: Vec::new() };
        layout.apply_forward(&mut t, MANY).expect("defaults");
        let Transform::Squeeze { steps } = &t else {
            panic!("still a squeeze");
        };
        assert_eq!(steps.len(), 2, "32x8 needs two horizontal steps");
        assert_eq!(layout.specs.len(), 3, "one low-pass plus two residuals");
    }

    #[test]
    fn channel_count_limit_is_enforced() {
        let mut layout = ChannelLayout::new(vec![ChannelSpec::new(4, 4); 2]);
        let mut t = Transform::Squeeze {
            steps: vec![SqueezeParams {
                horizontal: true,
                in_place: true,
                begin_c: 0,
                num_c: 2,
            }],
        };
        let err = layout
            .apply_forward(&mut t, 3)
            .expect_err("4 channels over a limit of 3");
        assert!(err.to_string().contains("nb_channels_tr"));
    }

    #[test]
    fn squeeze_then_inverse_round_trips_the_layout() {
        // Forward: one 4x4 channel becomes 2x4 + 2x4. Inverse must give 4x4.
        let mut layout = ChannelLayout::new(vec![ChannelSpec::new(4, 4)]);
        let mut t = Transform::Squeeze {
            steps: vec![SqueezeParams {
                horizontal: true,
                in_place: true,
                begin_c: 0,
                num_c: 1,
            }],
        };
        layout.apply_forward(&mut t, MANY).expect("forward");
        assert_eq!(layout.specs.len(), 2);

        let mut channels: Vec<Channel> = layout
            .specs
            .iter()
            .map(|s| Channel::new(*s, &mut guard()).expect("alloc"))
            .collect();
        let ctx = PaletteContext {
            bits_per_sample: 8,
            wp_header: super::super::weighted::WpHeader::default_wp(),
        };
        apply_inverse(&mut channels, &t, &ctx, &mut guard()).expect("inverse");
        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0].width(), 4);
        assert_eq!(channels[0].height(), 4);
        assert_eq!(channels[0].hshift(), 0, "the shift increment is undone");
    }

    #[test]
    fn transform_id_three_is_rejected() {
        // u(2) = 3 has no row in Table H.6.
        let data = [0b0000_0011u8];
        let mut r = BitReader::new(&data);
        let err = Transform::read(&mut r, &mut guard()).expect_err("id 3");
        assert!(err.to_string().contains("TransformId"));
    }

    #[test]
    fn rct_bundle_reads_the_documented_field_order() {
        // tr = 0 (kRCT) as u(2); begin_c via distribution 0 = u(3) -> value 2;
        // rct_type via distribution 0 = the constant 6.
        // Bits, LSB first: 00 | 00 | 010 | 00
        let mut bits: Vec<u8> = vec![0, 0, /* selector */ 0, 0, /* u(3) = 2 */ 0, 1, 0];
        bits.extend_from_slice(&[0, 0]); // rct_type selector 0 -> constant 6
        let mut bytes = vec![0u8; bits.len().div_ceil(8)];
        for (i, b) in bits.iter().enumerate() {
            bytes[i / 8] |= b << (i % 8);
        }
        let mut r = BitReader::new(&bytes);
        let t = Transform::read(&mut r, &mut guard()).expect("rct");
        assert_eq!(
            t,
            Transform::Rct {
                begin_c: 2,
                rct_type: 6
            }
        );
    }
}
