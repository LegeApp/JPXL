//! Modular mode: the modular image sub-bitstream (18181-1 Annex H).
//!
//! A *modular sub-bitstream* encodes an ordered list of integer channels. It
//! carries all pixel data when `frame_header.encoding == kModular`, and it also
//! carries extra channels and auxiliary images inside `kVarDCT` frames.
//!
//! ```text
//! Table H.1 — ModularHeader bundle
//! condition  type                              name
//!            Bool()                            use_global_tree
//!            WPHeader (H.5.1)                  wp_params
//!            U32(0, 1, 2 + u(4), 18 + u(8))    nb_transforms
//!            TransformInfo (H.6.1)             transform[nb_transforms]
//! ```
//!
//! # Scope
//!
//! This module is the sub-bitstream decoder and nothing else. It takes the
//! initial channel list as a typed parameter ([`ChannelSpec`]) rather than
//! deriving it, because H.1 says the dimensions are "signalled or computed
//! elsewhere" — that elsewhere is the frame layer (G.1.3, G.2.3, G.4.2), which
//! is a later slice. Everything from Table H.1 inwards lives here.
//!
//! # Pipeline
//!
//! 1. [`ModularHeader::read`] parses Table H.1 and replays the forward channel
//!    bookkeeping of H.6 to learn the shape of every channel.
//! 2. The MA tree is decoded (H.4.2) from its own six-context entropy stream,
//!    unless `use_global_tree` says the caller supplies it.
//! 3. [`decode_channels`] decodes samples in raster order per H.3, running the
//!    self-correcting predictor of H.5 for every sample.
//! 4. The inverse transforms of H.6 run from last to first.
//!
//! # Bit-exactness
//!
//! Per `docs/PLAN.md` this whole path is **bit-exact**: modular mode is integer
//! arithmetic by construction, so there is no tolerance regime and any
//! divergence from a reference decoder is a bug.
//!
//! # Untrusted input
//!
//! The MA tree, the transform chain and every channel dimension are
//! attacker-controlled. Channel and tree allocations are charged to an
//! [`AllocGuard`] before they happen, tree size and depth are bounded by
//! [`TreeLimits`], the transformed channel count by
//! [`ModularOptions::max_channels`], and `nb_transforms` by
//! [`ModularOptions::max_transforms`]. Nothing here panics on malformed input.
//!
//! # Error type
//!
//! This module defines its own [`ModularError`] rather than extending
//! `crate::error::DecodeError`, which slice 5 does not own. Wiring
//! `From<ModularError> for DecodeError` is slice 7's job; see
//! [`error`](self::error) for the shape it should take.
//!
//! # Specification ambiguities
//!
//! Recorded here because this implementation decides them and slice 7 must
//! confirm each against real `cjxl` streams.
//!
//! 1. **H.2, the global-tree case.** When `use_global_tree` is true the clause
//!    says the global tree "and its clustered distributions" are used, then
//!    that the decoder "starts an entropy-coded stream (C.1)". Those two
//!    statements pull in opposite directions: reusing the global distributions
//!    means the group reads no bundle of its own, while starting a C.1 stream
//!    means it does. [`TreeSource::Global`] takes the second reading — a fresh
//!    bundle with one context per global-tree leaf — because it is what the
//!    text of H.2 literally instructs and it keeps each group independently
//!    decodable. `TODO(slice 7)`: verify against an oracle-produced multi-group
//!    stream, and if the first reading is right, pass the shared bundle in
//!    instead. [`decode_channels`] is public precisely so that change is local.
//! 2. **H.6.2 inverse, the subsampling shifts.** The inverse squeeze pseudocode
//!    copies `channel[c]` and grows one dimension, never restoring the
//!    `hshift`/`vshift` the forward step incremented. Taken as an omission: the
//!    inverse decrements a positive shift, so a reconstructed full-resolution
//!    channel does not claim to be subsampled. Nothing inside Annex H reads the
//!    shifts after the inverse, so this only affects what slice 7 sees.
//! 3. **H.6.4, implicit palette entries.** The formulas divide by 4 with `/`,
//!    which 4.3 defines as exact real division, in a clause where every value
//!    is an integer sample. Integer division is taken; both operands are
//!    non-negative so the two roundings agree.
//! 4. **H.6.4, `if (index & 1 == 0)`.** Table 1 ranks `==` above `&`, which
//!    would make the condition constant-false and the delta table's sign
//!    structure dead. `(index & 1) == 0` is taken.
//!
//! # OCR corruptions detected
//!
//! * Table H.4 rows 4 and 5 read `abs(N)`/`abs(i)` in the LaTeX and
//!   `abs(1)`/`abs(i)` in the markdown; only `abs(N)`/`abs(W)` is consistent
//!   with rows 6 and 7. See [`tree`].
//! * Table H.3 row 13 renders `WW` as `WH` in the LaTeX. The coefficients
//!   summing to 16 is the check; see [`predictor`].
//! * `kDeltaPalette[4]` is `{0, -12, 9}` in the LaTeX and `{0, -12, 0}` in the
//!   markdown; the markdown is right. See [`palette`].
//! * H.5.2's `weight[i] = error2weight(err_sum[i], wp_wi)` is `wp_w[i]`, and
//!   the markdown drops the weight-normalisation block entirely. See
//!   [`weighted`].

pub mod channel;
pub mod error;
pub mod palette;
pub mod predictor;
pub mod rct;
pub mod squeeze;
pub mod transform;
pub mod tree;
pub mod weighted;

use jpxl_bitstream::{BitReader, U32Dist, U32Spec, read_bool, read_u32, trace_field};
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_entropy::SymbolDecoder;

pub use channel::{BYTES_PER_SAMPLE, Channel, ChannelSpec, ModularImage, SHIFT_UNRELATED};
pub use error::{ModularError, Result};
pub use palette::{DELTA_PALETTE, PaletteContext, PaletteParams};
pub use predictor::{Neighbours, Predictor};
pub use squeeze::SqueezeParams;
pub use transform::{ChannelLayout, Transform};
pub use tree::{Leaf, MaTree, PropertyBuilder, TreeLimits, TreeNode};
pub use weighted::{WeightedState, WpHeader};

use error::malformed;

/// H.2: `U32(0, 1, 2 + u(4), 18 + u(8))`.
const NB_TRANSFORMS_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::BitsOffset { bits: 4, offset: 2 },
    U32Dist::BitsOffset {
        bits: 8,
        offset: 18,
    },
]);

/// `UnpackSigned(u)` of 18181-1 4.2.
///
/// > equivalent to `u / 2` if `u` is even and `-(u + 1) / 2` if `u` is odd.
///
/// This is the usual zig-zag mapping: `0, 1, 2, 3, 4, ...` becomes
/// `0, -1, 1, -2, 2, ...`. The extremes matter — `u32::MAX` is odd and maps to
/// `-2^31`, exactly `i32::MIN`, so the result always fits in an `i32`.
#[must_use]
pub const fn unpack_signed(u: u32) -> i32 {
    // Computed in i64 so `u == u32::MAX` does not overflow before the halving.
    let u = u as i64;
    let v = if u % 2 == 0 { u / 2 } else { -(u + 1) / 2 };
    weighted::narrow_to_i32(v)
}

/// Where the MA tree for this sub-bitstream comes from (H.2).
#[derive(Debug, Clone, Copy)]
pub enum TreeSource<'a> {
    /// `use_global_tree` must be false; the tree is read from this stream per
    /// H.4.2.
    Local,
    /// `use_global_tree` must be true; the caller supplies the tree and
    /// distributions decoded from the GlobalModular section (G.1.3).
    ///
    /// See [`GLOBAL_TREE_SHARES_DISTRIBUTIONS`] for what this implies about
    /// the data stream's distributions.
    Global {
        /// The tree and its distribution bundle.
        global: &'a GlobalTree,
        /// Whether to re-seed the per-stream entropy state before decoding.
        ///
        /// H.4.2 reads the `D` bundle "as specified in C.1", and C.1's
        /// initialization ends with the ANS seed of C.3.2. That seed therefore
        /// belongs to the sub-bitstream that immediately follows the bundle —
        /// the `GlobalModular` one — which passes `false`. Every later section
        /// starts a new entropy-coded stream over the same distributions and
        /// passes `true`.
        restart: bool,
    },
}

/// Reads the global MA tree of G.1.3: the tree of H.4.2 plus the `D` bundle.
///
/// The caller has already read G.1.3's leading `Bool()` and found it set.
///
/// # Errors
///
/// Any [`ModularError`] the tree decode reports.
pub fn read_global_tree(
    reader: &mut BitReader<'_>,
    options: &ModularOptions,
    guard: &mut AllocGuard,
) -> Result<GlobalTree> {
    let mut tree_decoder = SymbolDecoder::open(reader, tree::TREE_NUM_CONTEXTS, guard)?;
    let tree = MaTree::decode(reader, &mut tree_decoder, options.tree_limits, guard)?;
    tree_decoder.finish()?;
    // The histograms only: H.2 starts the entropy-coded stream after the
    // ModularHeader of whichever sub-bitstream uses this tree.
    let distributions = SymbolDecoder::open_deferred(reader, tree.num_leaves(), guard)?;
    Ok(GlobalTree {
        tree,
        distributions,
    })
}

/// Caller-supplied bounds and context for a modular sub-bitstream.
///
/// Everything here is either an Annex M level limit that Annex H cannot compute
/// on its own, or a value the referencing clause defines.
#[derive(Debug, Clone, Copy)]
pub struct ModularOptions {
    /// Property 1 of Table H.4, defined per referencing clause in H.4.1.
    pub stream_index: u32,
    /// `metadata.bit_depth.bits_per_sample` (D.3.5), used by H.6.4.
    pub bits_per_sample: u32,
    /// Bounds on MA tree shape; see [`TreeLimits`].
    pub tree_limits: TreeLimits,
    /// Annex M "Maximum nb_transforms (H.2)": 8 at level 5, 512 at level 10.
    pub max_transforms: usize,
    /// Annex M `nb_channels_tr`: 256 at level 5, `1 << 16` at level 10.
    pub max_channels: usize,
}

impl ModularOptions {
    /// The Annex M level-10 bounds, with `stream_index` 0 and 8-bit samples.
    #[must_use]
    pub const fn level10() -> Self {
        Self {
            stream_index: 0,
            bits_per_sample: 8,
            tree_limits: TreeLimits::spec_maximum(),
            max_transforms: 512,
            max_channels: 1 << 16,
        }
    }

    /// The Annex M level-5 bounds.
    #[must_use]
    pub const fn level5() -> Self {
        Self {
            stream_index: 0,
            bits_per_sample: 8,
            tree_limits: TreeLimits::level5(),
            max_transforms: 8,
            max_channels: 256,
        }
    }
}

impl Default for ModularOptions {
    fn default() -> Self {
        Self::level10()
    }
}

/// A decoded `ModularHeader` (Table H.1) plus the channel shapes it implies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularHeader {
    use_global_tree: bool,
    wp_header: WpHeader,
    transforms: Vec<Transform>,
    /// `nb_meta_channels` as it stood *before* each transform, so the inverse
    /// pass can restore it (the Table H.8 updates are not invertible).
    meta_snapshots: Vec<usize>,
    layout: ChannelLayout,
}

impl ModularHeader {
    /// The header of a sub-bitstream with no channels, which H.1 says is not
    /// read at all.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            use_global_tree: false,
            wp_header: WpHeader::default_wp(),
            transforms: Vec::new(),
            meta_snapshots: Vec::new(),
            layout: ChannelLayout::new(Vec::new()),
        }
    }

    /// Reads Table H.1 and derives the transformed channel list.
    ///
    /// # Errors
    ///
    /// [`ModularError::Malformed`](ModularError::Malformed) for a transform
    /// chain that is inconsistent with `initial`, a `TransformId` of 3, or more
    /// than `options.max_transforms` transforms;
    /// [`ModularError::Bitstream`](ModularError::Bitstream) at end of input.
    pub fn read(
        reader: &mut BitReader<'_>,
        initial: &[ChannelSpec],
        options: &ModularOptions,
        guard: &mut AllocGuard,
    ) -> Result<Self> {
        let use_global_tree = trace_field!(reader, "modular.use_global_tree", read_bool(reader))?;
        let wp_header = weighted::read_wp_header(reader)?;
        let nb_transforms = trace_field!(
            reader,
            "modular.nb_transforms",
            read_u32(reader, &NB_TRANSFORMS_SPEC)
        )?;
        if nb_transforms as usize > options.max_transforms {
            return Err(malformed!(
                "Annex M: nb_transforms = {nb_transforms} exceeds the limit of {}",
                options.max_transforms
            ));
        }

        guard.charge(u64::from(nb_transforms) * size_of::<Transform>() as u64)?;
        let mut layout = ChannelLayout::new(initial.to_vec());
        let mut transforms = Vec::with_capacity(nb_transforms as usize);
        let mut meta_snapshots = Vec::with_capacity(nb_transforms as usize);
        for _ in 0..nb_transforms {
            let mut transform = Transform::read(reader, guard)?;
            meta_snapshots.push(layout.nb_meta_channels);
            layout.apply_forward(&mut transform, options.max_channels)?;
            transforms.push(transform);
        }

        Ok(Self {
            use_global_tree,
            wp_header,
            transforms,
            meta_snapshots,
            layout,
        })
    }

    /// Whether the MA tree comes from the GlobalModular section.
    #[must_use]
    pub const fn use_global_tree(&self) -> bool {
        self.use_global_tree
    }

    /// The `wp_params` bundle of Table H.1.
    #[must_use]
    pub const fn wp_header(&self) -> WpHeader {
        self.wp_header
    }

    /// The signalled transforms, in signalled order, with squeeze defaults
    /// already resolved.
    #[must_use]
    pub fn transforms(&self) -> &[Transform] {
        &self.transforms
    }

    /// The channel list after the forward transform chain: exactly the channels
    /// whose samples are coded in this sub-bitstream.
    #[must_use]
    pub const fn layout(&self) -> &ChannelLayout {
        &self.layout
    }

    /// Allocates every channel of the transformed list, zero-filled.
    ///
    /// # Errors
    ///
    /// [`ModularError::Core`](ModularError::Core) if the channels exceed the
    /// guard's budget.
    pub fn allocate_channels(&self, guard: &mut AllocGuard) -> Result<Vec<Channel>> {
        let mut channels = Vec::with_capacity(self.layout.specs.len());
        for spec in &self.layout.specs {
            channels.push(Channel::new(*spec, guard)?);
        }
        Ok(channels)
    }
}

/// The global MA tree of G.1.3 together with the distributions H.4.2 reads
/// after it.
///
/// G.1.3's `GlobalModular` opens with a `Bool()`; when it is set, "an MA tree
/// is decoded as described in H.4.2", and H.4.2's last step is "the decoder
/// reads `(tree.size() + 1) / 2` pre-clustered distributions D". Both halves
/// therefore belong to the global section, which is why they travel together.
#[derive(Debug, Clone)]
pub struct GlobalTree {
    /// The decoded tree.
    pub tree: MaTree,
    /// The `D` bundle read at the end of H.4.2, ready to be
    /// [`restart`](SymbolDecoder::restart)ed for each group.
    pub distributions: SymbolDecoder,
}

/// Whether a `use_global_tree` sub-bitstream reuses the global distributions.
///
/// **Resolved by experiment, slice 7.** H.2 says both that the global tree
/// "and its clustered distributions are used as decoded from the GlobalModular
/// section" and that the decoder then "starts an entropy-coded stream (C.1)".
/// Under `true` (the reading in force) the two are reconciled as: the
/// distribution bundle is shared and only the *per-stream* state of C.1 — the
/// ANS seed and the LZ77 window — is re-initialized, via
/// [`SymbolDecoder::restart`]. Under `false` each sub-bitstream reads a
/// complete fresh bundle.
///
/// Flipping this constant selects the other reading in one place; see the
/// slice-7 report for the evidence.
pub const GLOBAL_TREE_SHARES_DISTRIBUTIONS: bool = true;

/// Which channels a sub-bitstream decodes (18181-1 H.2 and G.1.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelStop {
    /// Every channel of the transformed list, as a group sub-bitstream does.
    All,
    /// G.1.3: the first `nb_meta_channels` channels, then any further channel
    /// whose width *and* height are at most `group_dim`, stopping at the first
    /// one that is not.
    GlobalModular {
        /// `group_dim` of the frame.
        group_dim: u32,
    },
}

/// A modular sub-bitstream whose samples are decoded but whose inverse
/// transforms have not run yet.
///
/// G.1.3 needs exactly this shape: the `GlobalModular` section decodes only a
/// prefix of the channels and explicitly applies "no inverse transforms yet",
/// because the remaining channels arrive later from the LF-group and
/// pass-group sections (G.2.3, G.4.2) and only then does H.6 run over the
/// completed image.
#[derive(Debug, Clone)]
pub struct PartialModular {
    header: ModularHeader,
    channels: Vec<Channel>,
    first_undecoded: usize,
}

impl PartialModular {
    /// The parsed header, including the resolved transform chain.
    #[must_use]
    pub const fn header(&self) -> &ModularHeader {
        &self.header
    }

    /// The full transformed channel list. Channels at or after
    /// [`first_undecoded`](Self::first_undecoded) are still zero-filled.
    #[must_use]
    pub fn channels(&self) -> &[Channel] {
        &self.channels
    }

    /// Mutable access, so G.2.3 and G.4.2 can copy group rectangles in.
    pub fn channels_mut(&mut self) -> &mut [Channel] {
        &mut self.channels
    }

    /// Index of the first channel this sub-bitstream did *not* decode.
    #[must_use]
    pub const fn first_undecoded(&self) -> usize {
        self.first_undecoded
    }

    /// Applies the inverse transforms of H.6, last to first.
    ///
    /// # Errors
    ///
    /// Any [`ModularError`] an inverse transform reports.
    pub fn into_image(
        mut self,
        options: &ModularOptions,
        guard: &mut AllocGuard,
    ) -> Result<ModularImage> {
        let ctx = PaletteContext {
            bits_per_sample: options.bits_per_sample,
            wp_header: self.header.wp_header,
        };
        let mut nb_meta_channels = self.header.layout.nb_meta_channels;
        for (index, transform) in self.header.transforms.iter().enumerate().rev() {
            transform::apply_inverse(&mut self.channels, transform, &ctx, guard)?;
            nb_meta_channels = self
                .header
                .meta_snapshots
                .get(index)
                .copied()
                .unwrap_or(nb_meta_channels);
        }
        Ok(ModularImage::new(self.channels, nb_meta_channels))
    }
}

/// Decodes a complete modular sub-bitstream (18181-1 Annex H).
///
/// `reader` must be positioned at the first bit of the `ModularHeader`. On
/// success it is positioned just past the sub-bitstream's entropy-coded data.
///
/// H.1: when `initial` is empty the decoder takes no action at all — not even
/// the header is read — so this returns an empty image without consuming a bit.
///
/// # Errors
///
/// Any [`ModularError`]: a malformed header, a rejected MA tree, an entropy
/// failure, or a limit rejection.
pub fn decode_sub_bitstream(
    reader: &mut BitReader<'_>,
    initial: &[ChannelSpec],
    options: &ModularOptions,
    tree_source: TreeSource<'_>,
    limits: &Limits,
) -> Result<ModularImage> {
    let mut guard = AllocGuard::new(limits);
    decode_sub_bitstream_with(reader, initial, options, tree_source, &mut guard)
}

/// As [`decode_sub_bitstream`], but sharing an existing [`AllocGuard`].
///
/// Slice 7 decodes many sub-bitstreams per frame and needs one cumulative
/// budget across all of them, not one per group.
///
/// # Errors
///
/// As [`decode_sub_bitstream`].
pub fn decode_sub_bitstream_with(
    reader: &mut BitReader<'_>,
    initial: &[ChannelSpec],
    options: &ModularOptions,
    tree_source: TreeSource<'_>,
    guard: &mut AllocGuard,
) -> Result<ModularImage> {
    // H.1: "In the trivial case where N is zero, the decoder takes no action."
    if initial.is_empty() {
        return Ok(ModularImage::new(Vec::new(), 0));
    }
    let partial = decode_sub_bitstream_partial(
        reader,
        initial,
        options,
        tree_source,
        ChannelStop::All,
        guard,
    )?;
    partial.into_image(options, guard)
}

/// Decodes a modular sub-bitstream up to `stop`, without inverse transforms.
///
/// This is the shape G.1.3 needs; [`decode_sub_bitstream_with`] is this plus
/// [`PartialModular::into_image`].
///
/// # Errors
///
/// As [`decode_sub_bitstream`].
pub fn decode_sub_bitstream_partial(
    reader: &mut BitReader<'_>,
    initial: &[ChannelSpec],
    options: &ModularOptions,
    tree_source: TreeSource<'_>,
    stop: ChannelStop,
    guard: &mut AllocGuard,
) -> Result<PartialModular> {
    // H.1: "In the trivial case where N is zero, the decoder takes no action."
    if initial.is_empty() {
        return Ok(PartialModular {
            header: ModularHeader::empty(),
            channels: Vec::new(),
            first_undecoded: 0,
        });
    }
    let header = ModularHeader::read(reader, initial, options, guard)?;

    // H.2 / H.4.2: the tree, and then the data stream's own distributions.
    let (tree, mut decoder) = match (header.use_global_tree, tree_source) {
        (false, TreeSource::Local) => {
            let mut tree_decoder = SymbolDecoder::open(reader, tree::TREE_NUM_CONTEXTS, guard)?;
            let tree = MaTree::decode(reader, &mut tree_decoder, options.tree_limits, guard)?;
            tree_decoder.finish()?;
            let data_decoder = SymbolDecoder::open(reader, tree.num_leaves(), guard)?;
            (tree, data_decoder)
        }
        (true, TreeSource::Global { global, restart }) => {
            // See `GLOBAL_TREE_SHARES_DISTRIBUTIONS`.
            let decoder = if GLOBAL_TREE_SHARES_DISTRIBUTIONS {
                let mut shared = global.distributions.clone();
                if restart {
                    shared.restart(reader)?;
                }
                shared
            } else {
                SymbolDecoder::open(reader, global.tree.num_leaves(), guard)?
            };
            (global.tree.clone(), decoder)
        }
        (true, TreeSource::Local) => {
            return Err(malformed!(
                "H.2: use_global_tree is set but no global MA tree was supplied"
            ));
        }
        // A sub-bitstream may signal its own tree even when a global one
        // exists: H.2 gates the choice on `use_global_tree` alone, and G.1.3's
        // global tree is optional for the sections that follow it.
        (false, TreeSource::Global { .. }) => {
            let mut tree_decoder = SymbolDecoder::open(reader, tree::TREE_NUM_CONTEXTS, guard)?;
            let tree = MaTree::decode(reader, &mut tree_decoder, options.tree_limits, guard)?;
            tree_decoder.finish()?;
            let data_decoder = SymbolDecoder::open(reader, tree.num_leaves(), guard)?;
            (tree, data_decoder)
        }
    };

    let mut channels = header.allocate_channels(guard)?;
    let first_undecoded = decode_channels(
        reader,
        &mut decoder,
        &tree,
        &header,
        options,
        &mut channels,
        stop,
        guard,
    )?;
    decoder.finish()?;

    Ok(PartialModular {
        header,
        channels,
        first_undecoded,
    })
}

/// Decodes channel samples per H.3, into an already-allocated channel list.
///
/// Returns the index of the first channel that was *not* decoded, which for
/// [`ChannelStop::All`] is `channels.len()`.
///
/// Exposed so slice 7's group wiring can supply its own [`SymbolDecoder`].
///
/// # Errors
///
/// [`ModularError::Entropy`](ModularError::Entropy) if a symbol cannot be
/// decoded, [`ModularError::Malformed`](ModularError::Malformed) if the MA tree
/// selects a context the stream does not have, or
/// [`ModularError::Core`](ModularError::Core) on a limit rejection.
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a distinct piece of decoder state that H.2 \
              names separately; bundling them would only hide the coupling"
)]
pub fn decode_channels(
    reader: &mut BitReader<'_>,
    decoder: &mut SymbolDecoder,
    tree: &MaTree,
    header: &ModularHeader,
    options: &ModularOptions,
    channels: &mut [Channel],
    stop: ChannelStop,
    guard: &mut AllocGuard,
) -> Result<usize> {
    // G.1.3: work out how far this sub-bitstream decodes before computing
    // `dist_multiplier`, which is defined over "all channels that are to be
    // decoded".
    let limit = match stop {
        ChannelStop::All => channels.len(),
        ChannelStop::GlobalModular { group_dim } => {
            let meta = header.layout.nb_meta_channels.min(channels.len());
            let mut end = meta;
            while let Some(c) = channels.get(end) {
                if c.width() > group_dim || c.height() > group_dim {
                    break;
                }
                end += 1;
            }
            end
        }
    };

    // H.3: dist_multiplier is the largest width amongst the channels that are
    // actually decoded, i.e. excluding the zero-sized ones H.2 skips.
    let dist_multiplier = channels
        .get(..limit)
        .unwrap_or(&[])
        .iter()
        .filter(|c| !c.spec().is_empty())
        .map(Channel::width)
        .max()
        .unwrap_or(0);
    decoder.set_dist_multiplier(dist_multiplier);

    for i in 0..limit {
        let spec = channels
            .get(i)
            .map(Channel::spec)
            .unwrap_or_else(|| ChannelSpec::new(0, 0));
        // H.2: "skipping any channels having width or height zero".
        if spec.is_empty() {
            continue;
        }
        let mut properties = PropertyBuilder::new(channels, i, options.stream_index, guard)?;
        let mut wp = WeightedState::new(spec.width, guard)?;

        let (earlier, from_here) = channels.split_at_mut(i);
        let Some(current) = from_here.first_mut() else {
            return Err(malformed!("H.3: channel {i} vanished during decoding"));
        };

        for y in 0..spec.height {
            for x in 0..spec.width {
                let nb = Neighbours::gather(current, x, y);
                // H.5.1: invoked for every sample, whatever the leaf predictor.
                let weighted = wp.predict(&header.wp_header, &nb.into(), x);
                let props = properties.compute(earlier, current, x, y, &nb, weighted.max_error);
                let leaf = tree.traverse(props)?;

                let raw = decoder.read_uint(reader, leaf.ctx)?;
                let diff = i64::from(unpack_signed(raw))
                    .checked_mul(leaf.multiplier)
                    .and_then(|v| v.checked_add(i64::from(leaf.offset)))
                    .ok_or_else(|| {
                        malformed!("H.3: residual * multiplier + offset overflows 64 bits")
                    })?;
                let value = weighted::narrow_to_i32(
                    diff.wrapping_add(nb.predict(leaf.predictor, weighted.prediction)),
                );
                current.set(x, y, value);
                wp.update(x, &weighted, value);
            }
            wp.advance_row();
        }
    }
    Ok(limit)
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    reason = "hand-written spec vectors read better with direct indexing; a panic \
              in a test is a failing test"
)]
mod tests {
    use super::*;

    #[test]
    fn unpack_signed_is_the_zigzag_of_clause_4_2() {
        // The exact table: even u maps to u/2, odd u to -(u+1)/2.
        let expected = [
            (0u32, 0i32),
            (1, -1),
            (2, 1),
            (3, -2),
            (4, 2),
            (5, -3),
            (6, 3),
            (7, -4),
            (8, 4),
            (9, -5),
            (10, 5),
        ];
        for (u, v) in expected {
            assert_eq!(unpack_signed(u), v, "UnpackSigned({u})");
        }
    }

    #[test]
    fn unpack_signed_is_a_bijection_onto_i32_at_the_extremes() {
        // u32::MAX is odd: -(2^32 - 1 + 1) / 2 = -2^31 = i32::MIN.
        assert_eq!(unpack_signed(u32::MAX), i32::MIN);
        // u32::MAX - 1 is even: (2^32 - 2) / 2 = 2^31 - 1 = i32::MAX.
        assert_eq!(unpack_signed(u32::MAX - 1), i32::MAX);
    }

    #[test]
    fn unpack_signed_round_trips_against_the_forward_mapping() {
        // The inverse packing is `v >= 0 ? 2v : -2v - 1`.
        for v in [-70000i32, -3, -1, 0, 1, 2, 65535, i32::MAX] {
            let u = if v >= 0 {
                (v as i64 * 2) as u32
            } else {
                (-(v as i64) * 2 - 1) as u32
            };
            assert_eq!(unpack_signed(u), v, "round trip for {v}");
        }
    }

    #[test]
    fn an_empty_channel_list_consumes_no_bits() {
        let data = [0xFFu8; 4];
        let mut reader = BitReader::new(&data);
        let image = decode_sub_bitstream(
            &mut reader,
            &[],
            &ModularOptions::default(),
            TreeSource::Local,
            &Limits::relaxed(),
        )
        .expect("H.1: N == 0 means no action");
        assert!(image.channels().is_empty());
        assert_eq!(reader.total_bits_read(), 0);
    }

    #[test]
    fn options_carry_the_annex_m_level_bounds() {
        assert_eq!(ModularOptions::level5().max_transforms, 8);
        assert_eq!(ModularOptions::level5().max_channels, 256);
        assert_eq!(ModularOptions::level5().tree_limits.max_depth, 64);
        assert_eq!(ModularOptions::level10().max_transforms, 512);
        assert_eq!(ModularOptions::level10().max_channels, 1 << 16);
    }
}
