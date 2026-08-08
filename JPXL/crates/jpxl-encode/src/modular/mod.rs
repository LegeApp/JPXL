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
//! # Coding choices (slice 19 + palette)
//!
//! | Field | Value | Clause |
//! |---|---|---|
//! | `LfChannelDequantization` | all-default | G.1.2 |
//! | global MA tree | absent | G.1.3 |
//! | `use_global_tree` | false, in every sub-bitstream | H.2 |
//! | `wp_params` | all-default | H.5.1 |
//! | `nb_transforms` | 0, 1 `kRCT`, or 1 `kPalette` in `LfGlobal` | H.2 |
//! | MA tree | learned under depth/leaf caps | H.4.2 |
//! | residual entropy | ANS, trained hybrid-uint; LZ77 when cheaper | C.2 / C.3 |
//!
//! Policy (`lossless::plan_for`) picks transforms, predictors, and MA trees.
//! Residuals use the Annex C ANS path and adopt LZ77 when strictly shorter.
//!
//! Every sub-bitstream carries its own tree and its own distribution bundle
//! (`use_global_tree = false`).
//!
//! # The residual
//!
//! H.3 reconstructs a sample as
//! `UnpackSigned(token) * multiplier + offset + prediction`. With
//! `multiplier = 1` and `offset = 0` the encoder writes
//! `PackSigned(sample - prediction)`. Group sub-bitstreams predict inside the
//! group rectangle only (G.4.2).

pub mod palette;
pub mod squeeze;

pub use palette::{MAX_PALETTE_COLOURS, PaletteForward, PaletteParams, try_exact_palette};
pub use squeeze::{horiz_fsqueeze, tendency, vert_fsqueeze};

use jpxl_bitstream::{BitWriter, U32Dist, U32Spec};
use jpxl_entropy::HybridUintConfig;
use jpxl_entropy::encode::{
    CodingMode, EncoderPlan, EntropyTables, Lz77EncodeParams, SymbolEncoder, TokenCensus,
};

use crate::entropy::{TREE_CODE, pack_signed};
use crate::error::{EncodeError, Result};
use crate::frame::Geometry;

/// Table H.3 predictor indices this track can emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Predictor {
    /// Always 0.
    Zero = 0,
    /// West neighbour.
    West = 1,
    /// North neighbour.
    North = 2,
    /// Average of west and north.
    AverageWestNorth = 3,
    /// Select (edge-aware).
    Select = 4,
    /// Gradient (clamped W+N−NW).
    Gradient = 5,
    /// Self-correcting (H.5), via the shared `jpxl_core::modular_weighted`
    /// state machine. Unlike every other variant, this is stateful and
    /// order-dependent (H.5.1: invoked every sample, in raster order,
    /// unconditionally) -- see [`collect_plane_residuals_weighted`] and
    /// [`MaTree::contains_predictor`].
    Weighted = 6,
}

impl Predictor {
    /// The Table H.3 index written into the MA leaf.
    #[must_use]
    pub const fn index(self) -> u32 {
        self as u32
    }
}

/// One node of an H.4.2 MA tree.
///
/// Decision children are stored left/right where **left** is taken when
/// `property[k] > value` (same convention as the decoder). Leaf `ctx_id`
/// values are assigned in breadth-first leaf encounter order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaNode {
    /// Inner node: `property[k] > value` → left, else right.
    Decision {
        /// Table H.4 property index.
        property: u32,
        /// Decision threshold (signed).
        value: i32,
        /// Child when the test is true.
        left: Box<MaNode>,
        /// Child when the test is false.
        right: Box<MaNode>,
    },
    /// Terminal residual context.
    Leaf {
        /// Table H.3 predictor for this leaf.
        predictor: Predictor,
        /// Residual context id (0..num_leaves).
        ctx_id: usize,
    },
}

/// MA tree this track can emit (H.4.2).
///
/// Supports an arbitrary full binary tree of property splits with a Table H.3
/// predictor per leaf. Policy (`lossless::plan_for`) grows the tree under
/// depth/leaf caps by exact residual+tree bit cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaTree {
    root: MaNode,
    num_leaves: usize,
}

impl MaTree {
    /// Single leaf → one residual context.
    #[must_use]
    pub fn single_leaf(predictor: Predictor) -> Self {
        Self {
            root: MaNode::Leaf {
                predictor,
                ctx_id: 0,
            },
            num_leaves: 1,
        }
    }

    /// Root decision with the same predictor on both leaves (ctx 0 left, 1 right).
    #[must_use]
    pub fn binary_split(property: u32, value: i32, predictor: Predictor) -> Self {
        Self::binary_split_preds(property, value, predictor, predictor)
    }

    /// Root decision with independent predictors on the two leaves.
    #[must_use]
    pub fn binary_split_preds(
        property: u32,
        value: i32,
        left_predictor: Predictor,
        right_predictor: Predictor,
    ) -> Self {
        let mut tree = Self {
            root: MaNode::Decision {
                property,
                value,
                left: Box::new(MaNode::Leaf {
                    predictor: left_predictor,
                    ctx_id: 0,
                }),
                right: Box::new(MaNode::Leaf {
                    predictor: right_predictor,
                    ctx_id: 1,
                }),
            },
            num_leaves: 2,
        };
        tree.reassign_ctx_ids();
        tree
    }

    /// Number of residual contexts (= leaf count).
    #[must_use]
    pub const fn num_contexts(&self) -> usize {
        self.num_leaves
    }

    /// Maximum depth of the tree (root depth 0).
    #[must_use]
    pub fn depth(&self) -> u32 {
        fn depth_of(node: &MaNode) -> u32 {
            match node {
                MaNode::Leaf { .. } => 0,
                MaNode::Decision { left, right, .. } => 1 + depth_of(left).max(depth_of(right)),
            }
        }
        depth_of(&self.root)
    }

    /// Predictor of the first leaf in BFS order (for single-leaf trees: the only one).
    #[must_use]
    pub fn primary_predictor(&self) -> Predictor {
        fn first_leaf(node: &MaNode) -> Predictor {
            match node {
                MaNode::Leaf { predictor, .. } => *predictor,
                MaNode::Decision { left, .. } => first_leaf(left),
            }
        }
        first_leaf(&self.root)
    }

    /// Leaf context for the sample at rectangle-relative `(x, y)`.
    #[must_use]
    pub fn context(&self, at: &impl Fn(u32, u32) -> i64, x: u32, y: u32) -> usize {
        self.leaf_at(at, x, y).0
    }

    /// `(ctx_id, predictor)` for the sample at rectangle-relative `(x, y)`.
    #[must_use]
    pub fn leaf_at(&self, at: &impl Fn(u32, u32) -> i64, x: u32, y: u32) -> (usize, Predictor) {
        fn walk(
            node: &MaNode,
            at: &impl Fn(u32, u32) -> i64,
            x: u32,
            y: u32,
        ) -> (usize, Predictor) {
            match node {
                MaNode::Leaf { predictor, ctx_id } => (*ctx_id, *predictor),
                MaNode::Decision {
                    property,
                    value,
                    left,
                    right,
                } => {
                    let p = property_value(at, x, y, *property);
                    if p > *value {
                        walk(left, at, x, y)
                    } else {
                        walk(right, at, x, y)
                    }
                }
            }
        }
        walk(&self.root, at, x, y)
    }

    /// Breadth-first list of leaf predictors (ctx order).
    #[must_use]
    pub fn leaf_predictors(&self) -> Vec<Predictor> {
        let mut out = vec![Predictor::Zero; self.num_leaves];
        fn collect(node: &MaNode, out: &mut [Predictor]) {
            match node {
                MaNode::Leaf { predictor, ctx_id } => {
                    if let Some(slot) = out.get_mut(*ctx_id) {
                        *slot = *predictor;
                    }
                }
                MaNode::Decision { left, right, .. } => {
                    collect(left, out);
                    collect(right, out);
                }
            }
        }
        collect(&self.root, &mut out);
        out
    }

    /// Whether any leaf uses `predictor`.
    ///
    /// The one case this matters for today is [`Predictor::Weighted`]: unlike
    /// every other predictor, H.5.1 requires its state to be advanced for
    /// **every** sample of the scan, unconditionally, regardless of which
    /// leaf a given sample lands in (see [`collect_plane_residuals_weighted`]).
    /// Residual collection checks this once per tree to decide whether that
    /// full sequential walk is needed, instead of the cheaper order-
    /// independent paths every other predictor allows.
    #[must_use]
    pub fn contains_predictor(&self, predictor: Predictor) -> bool {
        fn any(node: &MaNode, predictor: Predictor) -> bool {
            match node {
                MaNode::Leaf { predictor: p, .. } => *p == predictor,
                MaNode::Decision { left, right, .. } => {
                    any(left, predictor) || any(right, predictor)
                }
            }
        }
        any(&self.root, predictor)
    }

    /// Replaces the leaf with `ctx_id` by a decision whose children are leaves.
    ///
    /// # Errors
    ///
    /// If `ctx_id` is not a leaf of this tree.
    pub fn split_leaf(
        &self,
        ctx_id: usize,
        property: u32,
        value: i32,
        left_predictor: Predictor,
        right_predictor: Predictor,
    ) -> Result<Self> {
        let mut found = false;
        let root = split_leaf_node(
            &self.root,
            ctx_id,
            property,
            value,
            left_predictor,
            right_predictor,
            &mut found,
        )?;
        if !found {
            return Err(EncodeError::unsupported(
                "MA tree leaf ctx_id not found for split",
                "H.4.2",
            ));
        }
        let mut tree = Self {
            root,
            num_leaves: self.num_leaves + 1,
        };
        tree.reassign_ctx_ids();
        Ok(tree)
    }

    /// Sets the predictor of the leaf with `ctx_id`.
    ///
    /// # Errors
    ///
    /// If `ctx_id` is not a leaf of this tree.
    pub fn with_leaf_predictor(&self, ctx_id: usize, predictor: Predictor) -> Result<Self> {
        let mut found = false;
        let root = set_leaf_predictor(&self.root, ctx_id, predictor, &mut found)?;
        if !found {
            return Err(EncodeError::unsupported(
                "MA tree leaf ctx_id not found for predictor change",
                "H.4.2",
            ));
        }
        Ok(Self {
            root,
            num_leaves: self.num_leaves,
        })
    }

    /// Assigns `ctx_id` in breadth-first leaf encounter order (H.4.2).
    fn reassign_ctx_ids(&mut self) {
        let root = std::mem::replace(
            &mut self.root,
            MaNode::Leaf {
                predictor: Predictor::Zero,
                ctx_id: 0,
            },
        );
        let (root, count) = reassign_ctx_ids_bfs(root);
        self.root = root;
        self.num_leaves = count;
    }
}

/// Rebuilds `root` with `ctx_id` assigned in BFS leaf order.
fn reassign_ctx_ids_bfs(root: MaNode) -> (MaNode, usize) {
    #[derive(Clone)]
    enum Flat {
        Decision {
            property: u32,
            value: i32,
            left: usize,
            right: usize,
        },
        Leaf {
            predictor: Predictor,
        },
    }
    let mut flats: Vec<Flat> = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(root);
    while let Some(node) = queue.pop_front() {
        match node {
            MaNode::Leaf { predictor, .. } => {
                flats.push(Flat::Leaf { predictor });
            }
            MaNode::Decision {
                property,
                value,
                left,
                right,
            } => {
                // Children land at the end of the current queue (BFS).
                let left_idx = flats.len() + 1 + queue.len();
                let right_idx = left_idx + 1;
                flats.push(Flat::Decision {
                    property,
                    value,
                    left: left_idx,
                    right: right_idx,
                });
                queue.push_back(*left);
                queue.push_back(*right);
            }
        }
    }
    let mut next = 0usize;
    let mut ctx_of = vec![0usize; flats.len()];
    for (i, flat) in flats.iter().enumerate() {
        if matches!(flat, Flat::Leaf { .. }) {
            if let Some(slot) = ctx_of.get_mut(i) {
                *slot = next;
            }
            next += 1;
        }
    }
    fn rebuild(flats: &[Flat], ctx_of: &[usize], index: usize) -> MaNode {
        match flats.get(index) {
            Some(Flat::Leaf { predictor }) => MaNode::Leaf {
                predictor: *predictor,
                ctx_id: ctx_of.get(index).copied().unwrap_or(0),
            },
            Some(Flat::Decision {
                property,
                value,
                left,
                right,
            }) => MaNode::Decision {
                property: *property,
                value: *value,
                left: Box::new(rebuild(flats, ctx_of, *left)),
                right: Box::new(rebuild(flats, ctx_of, *right)),
            },
            None => MaNode::Leaf {
                predictor: Predictor::Zero,
                ctx_id: 0,
            },
        }
    }
    (rebuild(&flats, &ctx_of, 0), next)
}

fn split_leaf_node(
    node: &MaNode,
    target: usize,
    property: u32,
    value: i32,
    left_predictor: Predictor,
    right_predictor: Predictor,
    found: &mut bool,
) -> Result<MaNode> {
    match node {
        MaNode::Leaf { ctx_id, .. } if *ctx_id == target => {
            *found = true;
            Ok(MaNode::Decision {
                property,
                value,
                left: Box::new(MaNode::Leaf {
                    predictor: left_predictor,
                    ctx_id: 0,
                }),
                right: Box::new(MaNode::Leaf {
                    predictor: right_predictor,
                    ctx_id: 0,
                }),
            })
        }
        MaNode::Leaf { predictor, ctx_id } => Ok(MaNode::Leaf {
            predictor: *predictor,
            ctx_id: *ctx_id,
        }),
        MaNode::Decision {
            property: p,
            value: v,
            left,
            right,
        } => Ok(MaNode::Decision {
            property: *p,
            value: *v,
            left: Box::new(split_leaf_node(
                left,
                target,
                property,
                value,
                left_predictor,
                right_predictor,
                found,
            )?),
            right: Box::new(split_leaf_node(
                right,
                target,
                property,
                value,
                left_predictor,
                right_predictor,
                found,
            )?),
        }),
    }
}

fn set_leaf_predictor(
    node: &MaNode,
    target: usize,
    predictor: Predictor,
    found: &mut bool,
) -> Result<MaNode> {
    match node {
        MaNode::Leaf { ctx_id, .. } if *ctx_id == target => {
            *found = true;
            Ok(MaNode::Leaf {
                predictor,
                ctx_id: *ctx_id,
            })
        }
        MaNode::Leaf {
            predictor: pred,
            ctx_id,
        } => Ok(MaNode::Leaf {
            predictor: *pred,
            ctx_id: *ctx_id,
        }),
        MaNode::Decision {
            property,
            value,
            left,
            right,
        } => Ok(MaNode::Decision {
            property: *property,
            value: *value,
            left: Box::new(set_leaf_predictor(left, target, predictor, found)?),
            right: Box::new(set_leaf_predictor(right, target, predictor, found)?),
        }),
    }
}

/// Static Table H.4 properties this encoder uses for splits.
fn property_value(at: &impl Fn(u32, u32) -> i64, x: u32, y: u32, property: u32) -> i32 {
    let (w, n, nw) = neighbours(at, x, y);
    let raw = match property {
        4 => n.abs(),
        5 => w.abs(),
        6 => n,
        7 => w,
        9 => w + n - nw,
        10 => w - nw,
        11 => nw - n,
        _ => 0,
    };
    i32::try_from(raw).unwrap_or(if raw < 0 { i32::MIN } else { i32::MAX })
}

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
/// Table H.6 `TransformId`: `kPalette`.
const TRANSFORM_ID_PALETTE: u32 = 1;
/// Table H.6 `TransformId`: `kSqueeze`.
const TRANSFORM_ID_SQUEEZE: u32 = 2;

/// H.6.1 `num_sq`: `U32(0, 1 + u(4), 9 + u(6), 41 + u(8))`.
const NUM_SQ_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::BitsOffset { bits: 4, offset: 1 },
    U32Dist::BitsOffset { bits: 6, offset: 9 },
    U32Dist::BitsOffset {
        bits: 8,
        offset: 41,
    },
]);

/// The `rct_type` this encoder writes: `permutation = 0`, `type = 6` (YCoCg).
pub const RCT_TYPE_YCOCG: u32 = 6;

/// H.6.1 palette `num_c`: `U32(1, 3, 4, 1 + u(13))`.
const PALETTE_NUM_C_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(1),
    U32Dist::Val(3),
    U32Dist::Val(4),
    U32Dist::BitsOffset {
        bits: 13,
        offset: 1,
    },
]);

/// H.6.1 `nb_colours`: `U32(u(8), 256 + u(10), 1280 + u(12), 5376 + u(16))`.
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

/// H.6.1 `nb_deltas`: `U32(0, 1 + u(8), 257 + u(10), 1281 + u(16))`.
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

/// Number of pre-clustered contexts the MA-tree stream uses (18181-1 H.4.2).
const TREE_NUM_CONTEXTS: usize = 6;

/// Hybrid-uint configurations tried for a residual stream (same family as
/// the VarDCT policy trainer).
const HYBRID_CANDIDATES: &[(u32, u32, u32)] = &[
    (4, 2, 0),
    (0, 0, 0),
    (1, 0, 0),
    (2, 1, 0),
    (3, 1, 0),
    (3, 2, 0),
    (4, 1, 0),
    (5, 2, 0),
    (6, 2, 0),
];

/// Table C.1 default `min_length`.
const RESIDUAL_LZ77_MIN_LENGTH: u32 = 3;
/// ANS `log_alphabet_size ≤ 8` ⇒ tokens fit in `0..255`. Length triggers start
/// at `min_symbol`, so that value must stay ≤ 224 to leave room for copies.
const RESIDUAL_LZ77_MAX_MIN_SYMBOL: u32 = 224;
/// Lookback bound for the greedy finder (not the full 1 Mi window).
///
/// Full-window search is O(n²) and makes multi-group planning unusable; 256 is
/// enough for long zero runs and short repeated residual patterns.
const RESIDUAL_LZ77_LOOKBACK: usize = 256;

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

/// Shared residual plane storage (Phase-1: plan trials Arc-clone, not deep-copy).
pub type SharedPlane = std::sync::Arc<[i32]>;

/// One residual-coded channel after the transform list (may differ in size).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodedChannel {
    /// Channel width in samples.
    pub width: u32,
    /// Channel height in samples.
    pub height: u32,
    /// Horizontal subsample shift (H.1); −1 = unrelated (palette meta).
    pub hshift: i32,
    /// Vertical subsample shift.
    pub vshift: i32,
    /// Raster-order samples, length `width * height` (shared across plan trials).
    pub data: SharedPlane,
}

impl CodedChannel {
    /// Full-resolution colour/data channel from an owned plane (one deep copy into Arc).
    #[must_use]
    pub fn full(width: u32, height: u32, data: Plane) -> Self {
        let bytes = (data.len() as u64).saturating_mul(4);
        crate::lossless::note_plane_clone_bytes(bytes);
        Self::full_shared(width, height, std::sync::Arc::from(data.into_boxed_slice()))
    }

    /// Full-resolution channel wrapping an already-shared plane (cheap Arc clone).
    #[must_use]
    pub fn full_shared(width: u32, height: u32, data: SharedPlane) -> Self {
        Self {
            width,
            height,
            hshift: 0,
            vshift: 0,
            data,
        }
    }

    /// Palette meta-channel (unrelated shifts).
    #[must_use]
    pub fn meta(width: u32, height: u32, data: Plane) -> Self {
        let bytes = (data.len() as u64).saturating_mul(4);
        crate::lossless::note_plane_clone_bytes(bytes);
        Self {
            width,
            height,
            hshift: -1,
            vshift: -1,
            data: std::sync::Arc::from(data.into_boxed_slice()),
        }
    }

    /// Meta channel from shared storage.
    #[must_use]
    pub fn meta_shared(width: u32, height: u32, data: SharedPlane) -> Self {
        Self {
            width,
            height,
            hshift: -1,
            vshift: -1,
            data,
        }
    }
}

/// G.1.3 / G.2.3 / G.4.2 partition of post-transform channels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelPartition {
    /// Channel indices residual-coded in `LfGlobal` (full channel geometry).
    pub lf_global: Vec<usize>,
    /// Channel indices residual-coded in each `LfGroup` (LF rect, shifted).
    pub lf_group: Vec<usize>,
    /// Channel indices residual-coded in each pass group (group rect).
    pub pass_group: Vec<usize>,
    /// Leading meta-channel count.
    pub nb_meta: usize,
}

/// Partitions channels the way the decoder's G.1.3 / G.2.3 / G.4.2 walks do.
///
/// `GlobalModular`: first `nb_meta` channels, then consecutive channels with
/// both dimensions `≤ group_dim`, stopping at the first oversized one.
/// Remaining channels with `hshift ≥ 3` and `vshift ≥ 3` go to LF groups;
/// the rest go to pass groups.
#[must_use]
pub fn partition_channels(source: &ModularSource, group_dim: u32) -> ChannelPartition {
    let nb_meta = source.nb_meta_channels.min(source.channels.len());
    let mut lf_global = Vec::new();
    for i in 0..nb_meta {
        lf_global.push(i);
    }
    let mut i = nb_meta;
    while let Some(ch) = source.channels.get(i) {
        if ch.width > group_dim || ch.height > group_dim {
            break;
        }
        lf_global.push(i);
        i += 1;
    }
    let mut lf_group = Vec::new();
    let mut pass_group = Vec::new();
    while i < source.channels.len() {
        match source.channels.get(i) {
            Some(ch) if ch.hshift >= 3 && ch.vshift >= 3 => lf_group.push(i),
            Some(_) => pass_group.push(i),
            None => break,
        }
        i += 1;
    }
    ChannelPartition {
        lf_global,
        lf_group,
        pass_group,
        nb_meta,
    }
}

/// Transform list written in `LfGlobal` (H.2 / Table H.7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModularTransform {
    /// `kRCT` over three equal channels at `begin_c` (always 0 here).
    Rct {
        /// Combined permutation/type; we write YCoCg (`6`).
        rct_type: u32,
    },
    /// Exact-colour `kPalette` (wave 1: `nb_deltas = 0`).
    Palette(PaletteParams),
    /// `kSqueeze` with empty on-wire steps → decoder default list (H.6.2.1).
    SqueezeDefault,
}

/// The image the modular layer encodes after the transform list is applied.
#[derive(Debug, Clone)]
pub struct ModularSource {
    /// Frame width (group grid / index-channel geometry).
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Post-transform channels in decode order (meta first when paletted).
    pub channels: Vec<CodedChannel>,
    /// Leading meta-channel count (H.1 / G.1.3).
    pub nb_meta_channels: usize,
    /// Transforms declared in `LfGlobal` (empty, RCT, or Palette).
    pub transforms: Vec<ModularTransform>,
    /// MA tree over residual samples.
    pub tree: MaTree,
    /// When false, residual streams never enable LZ77 (policy scoring path).
    pub allow_lz77: bool,
}

impl ModularSource {
    /// Direct planes (optional RCT already applied to samples).
    ///
    /// Each plane is copied into an [`SharedPlane`] once. Prefer
    /// [`Self::direct_shared`] when the same planes are scored many times.
    #[must_use]
    pub fn direct(
        width: u32,
        height: u32,
        planes: &[Plane],
        rct: bool,
        tree: MaTree,
        allow_lz77: bool,
    ) -> Self {
        let shared: Vec<SharedPlane> = planes
            .iter()
            .map(|p| {
                let bytes = (p.len() as u64).saturating_mul(4);
                crate::lossless::note_plane_clone_bytes(bytes);
                std::sync::Arc::<[i32]>::from(p.as_slice())
            })
            .collect();
        Self::direct_shared(width, height, &shared, rct, tree, allow_lz77)
    }

    /// Direct planes already held as shared Arcs (Phase-1 plan trials).
    #[must_use]
    pub fn direct_shared(
        width: u32,
        height: u32,
        planes: &[SharedPlane],
        rct: bool,
        tree: MaTree,
        allow_lz77: bool,
    ) -> Self {
        let channels = planes
            .iter()
            .map(|p| CodedChannel::full_shared(width, height, std::sync::Arc::clone(p)))
            .collect();
        let transforms = if rct {
            vec![ModularTransform::Rct {
                rct_type: RCT_TYPE_YCOCG,
            }]
        } else {
            Vec::new()
        };
        Self {
            width,
            height,
            channels,
            nb_meta_channels: 0,
            transforms,
            tree,
            allow_lz77,
        }
    }

    /// Palette meta + index channels.
    #[must_use]
    pub fn from_palette(fwd: PaletteForward, tree: MaTree, allow_lz77: bool) -> Self {
        let meta_w = fwd.params.nb_colours;
        let meta_h = fwd.params.num_c;
        Self {
            width: fwd.width,
            height: fwd.height,
            channels: vec![
                CodedChannel::meta(meta_w, meta_h, fwd.meta),
                CodedChannel::full(fwd.width, fwd.height, fwd.index),
            ],
            nb_meta_channels: 1,
            transforms: vec![ModularTransform::Palette(fwd.params)],
            tree,
            allow_lz77,
        }
    }

    /// Direct or RCT samples, then default squeeze (empty on-wire step list).
    ///
    /// # Errors
    ///
    /// As [`squeeze::apply_default_to_planes`].
    pub fn with_default_squeeze(
        width: u32,
        height: u32,
        planes: &[Plane],
        rct: bool,
        tree: MaTree,
        allow_lz77: bool,
    ) -> Result<Self> {
        let channels = squeeze::apply_default_to_planes(width, height, planes)?;
        let mut transforms = Vec::new();
        if rct {
            transforms.push(ModularTransform::Rct {
                rct_type: RCT_TYPE_YCOCG,
            });
        }
        transforms.push(ModularTransform::SqueezeDefault);
        Ok(Self {
            width,
            height,
            channels,
            nb_meta_channels: 0,
            transforms,
            tree,
            allow_lz77,
        })
    }

    /// Checks channel lengths match their geometries.
    fn validate(&self) -> Result<()> {
        for ch in &self.channels {
            let expected = u64::from(ch.width) * u64::from(ch.height);
            let found = u64::try_from(ch.data.len()).unwrap_or(u64::MAX);
            if found != expected {
                return Err(EncodeError::SampleCountMismatch { expected, found });
            }
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
pub fn encode_lf_global(source: &ModularSource, geometry: &Geometry) -> Result<Vec<u8>> {
    source.validate()?;
    let mut w = BitWriter::new();

    // G.1.2: LfChannelDequantization is present whatever the encoding. Modular
    // mode never uses the weights, but the bit is still there.
    w.write_bool(true); // all_default

    // G.1.3: no global MA tree; every sub-bitstream carries its own.
    w.write_bool(false);

    write_modular_header(&mut w, &source.transforms)?;
    write_ma_tree(&mut w, &source.tree)?;

    // G.1.3: residual-code meta + channels that fit in group_dim.
    let part = partition_channels(source, geometry.group_dim());
    if part.lf_global.is_empty() {
        write_empty_residual_stream(&mut w, source.tree.num_contexts())?;
    } else {
        write_residual_payload_indices(&mut w, source, &part.lf_global, None)?;
    }

    w.zero_pad_to_byte();
    Ok(w.into_bytes())
}

/// Writes one LF-group modular section (18181-1 G.2.3).
///
/// # Errors
///
/// As [`encode_lf_global`].
pub fn encode_lf_group(source: &ModularSource, rect: Rect, geometry: &Geometry) -> Result<Vec<u8>> {
    source.validate()?;
    let mut w = BitWriter::new();
    write_modular_header(&mut w, &[])?;
    write_ma_tree(&mut w, &source.tree)?;
    let part = partition_channels(source, geometry.group_dim());
    if part.lf_group.is_empty() {
        write_empty_residual_stream(&mut w, source.tree.num_contexts())?;
    } else {
        write_residual_payload_indices(&mut w, source, &part.lf_group, Some(rect))?;
    }
    w.zero_pad_to_byte();
    Ok(w.into_bytes())
}

/// Writes one pass-group section (18181-1 G.4.2).
///
/// # Errors
///
/// As [`encode_lf_global`].
pub fn encode_group(source: &ModularSource, rect: Rect, geometry: &Geometry) -> Result<Vec<u8>> {
    source.validate()?;
    let mut w = BitWriter::new();

    // A group sub-bitstream declares no transforms of its own: the channel
    // list it works on is the one LfGlobal already transformed.
    write_modular_header(&mut w, &[])?;
    write_ma_tree(&mut w, &source.tree)?;
    let part = partition_channels(source, geometry.group_dim());
    if part.pass_group.is_empty() {
        write_empty_residual_stream(&mut w, source.tree.num_contexts())?;
    } else {
        write_residual_payload_indices(&mut w, source, &part.pass_group, Some(rect))?;
    }

    w.zero_pad_to_byte();
    Ok(w.into_bytes())
}

/// Writes Table H.1 and the transform list (H.2 / Table H.7).
fn write_modular_header(w: &mut BitWriter, transforms: &[ModularTransform]) -> Result<()> {
    w.write_bool(false); // use_global_tree
    w.write_bool(true); // WPHeader: default_wp
    let n = u32::try_from(transforms.len()).unwrap_or(0);
    w.write_u32(&NB_TRANSFORMS_SPEC, n)?;
    for tr in transforms {
        match *tr {
            ModularTransform::Rct { rct_type } => {
                w.write_bits(2, TRANSFORM_ID_RCT)?;
                w.write_u32(&BEGIN_C_SPEC, 0)?;
                w.write_u32(&RCT_TYPE_SPEC, rct_type)?;
            }
            ModularTransform::Palette(params) => {
                w.write_bits(2, TRANSFORM_ID_PALETTE)?;
                w.write_u32(&BEGIN_C_SPEC, params.begin_c)?;
                w.write_u32(&PALETTE_NUM_C_SPEC, params.num_c)?;
                w.write_u32(&NB_COLOURS_SPEC, params.nb_colours)?;
                w.write_u32(&NB_DELTAS_SPEC, params.nb_deltas)?;
                w.write_bits(4, params.d_pred)?;
            }
            ModularTransform::SqueezeDefault => {
                w.write_bits(2, TRANSFORM_ID_SQUEEZE)?;
                // num_sq = 0 → decoder substitutes H.6.2.1 defaults.
                w.write_u32(&NUM_SQ_SPEC, 0)?;
            }
        }
    }
    Ok(())
}

/// Writes the MA tree of H.4.2 (six-context stream), breadth-first.
///
/// Public so policy can measure exact tree bits when scoring candidates.
///
/// # Errors
///
/// Tree-code emission or a bit writer error.
pub fn write_ma_tree(w: &mut BitWriter, tree: &MaTree) -> Result<()> {
    TREE_CODE.write_bundle(w, TREE_NUM_CONTEXTS)?;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(&tree.root);
    while let Some(node) = queue.pop_front() {
        match node {
            MaNode::Decision {
                property,
                value,
                left,
                right,
            } => {
                TREE_CODE.write_uint(w, property.saturating_add(1))?; // ctx 1
                TREE_CODE.write_uint(w, pack_signed(*value))?; // ctx 0
                queue.push_back(left);
                queue.push_back(right);
            }
            MaNode::Leaf { predictor, .. } => write_leaf_node(w, *predictor)?,
        }
    }
    Ok(())
}

/// Bit length of a written MA tree (bundle + nodes).
///
/// # Errors
///
/// As [`write_ma_tree`].
pub fn ma_tree_bit_cost(tree: &MaTree) -> Result<u64> {
    let mut w = BitWriter::counting();
    write_ma_tree(&mut w, tree)?;
    Ok(w.bit_len())
}

/// Shannon-style residual bit estimate (no ANS table emission).
///
/// Collects residuals once, picks a hybrid-uint config by the same census cost
/// as the real encoder, and returns data bits plus a fixed table overhead.
/// Used by the Opt-M tiered planner for intermediate candidate ranking; exact
/// [`write_residual_payload_indices`] remains the finalist gate.
///
/// # Errors
///
/// Residual out of range or census/hybrid-config rejection.
pub fn estimate_residual_bits_indices(
    source: &ModularSource,
    indices: &[usize],
    rect: Option<Rect>,
) -> Result<u64> {
    let events = collect_residuals_indices(source, indices, rect)?;
    hybrid_estimate_from_events(&events, source.tree.num_contexts())
}

/// Full-frame residual bit estimate (see [`estimate_residual_bits_indices`]).
///
/// # Errors
///
/// As [`estimate_residual_bits_indices`].
pub fn estimate_residual_bits_full(source: &ModularSource) -> Result<u64> {
    let indices: Vec<usize> = (0..source.channels.len()).collect();
    estimate_residual_bits_indices(source, &indices, None)
}

/// Phase 4B: sampled variant of [`estimate_residual_bits_full`].
///
/// Visits every `row_stride`-th row of each channel (always including the
/// last row) instead of every row, so cost scales with sample count, not
/// pixel count. Predictions and MA-tree context properties still read exact
/// sample values -- lossless encode has ground-truth pixels, not a decode
/// reconstruction, so a sampled subset gives residuals that are exact for
/// the rows it visits; only the population shrinks, not the accuracy.
///
/// Ranking-only, like [`estimate_residual_bits_full`]: the Exact tier
/// re-prices the settled finalist exactly regardless of which tier's search
/// chose it. `row_stride < 1` is treated as 1 (every row, i.e. no sampling).
///
/// Falls back to [`estimate_residual_bits_full`] when `source.tree` contains
/// [`Predictor::Weighted`] anywhere: H.5.1's state cannot tolerate skipped or
/// reordered rows (see [`collect_plane_residuals_weighted`]), so sampling it
/// would silently desync the error state instead of merely losing ranking
/// precision the way sampling a stateless predictor does.
///
/// # Errors
///
/// Residual out of range or census/hybrid-config rejection.
pub fn estimate_residual_bits_sampled(source: &ModularSource, row_stride: u32) -> Result<u64> {
    if source.tree.contains_predictor(Predictor::Weighted) {
        return estimate_residual_bits_full(source);
    }
    let indices: Vec<usize> = (0..source.channels.len()).collect();
    let events = collect_residuals_indices_sampled(source, &indices, row_stride)?;
    hybrid_estimate_from_events(&events, source.tree.num_contexts())
}

/// Shared Shannon-hybrid cost tail for both the full and sampled scorers.
fn hybrid_estimate_from_events(events: &[(usize, u32)], num_contexts: usize) -> Result<u64> {
    let num_contexts = num_contexts.max(1);
    let mut census = TokenCensus::new(num_contexts)?;
    for &(ctx, value) in events {
        census.record(ctx, value)?;
    }
    seed_empty_contexts(&mut census, num_contexts, events.is_empty());
    let config = best_hybrid_config(&census, num_contexts, None)?;
    let data = hybrid_data_cost(&census, num_contexts, &config);
    // Clustered ANS tables are usually hundreds of bits; a fixed pad keeps the
    // estimate from systematically undercutting exact prices.
    const TABLE_OVERHEAD_BITS: f64 = 256.0;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "bit estimate for ranking only; floored to non-negative u64"
    )]
    let bits = if data.is_finite() {
        (data + TABLE_OVERHEAD_BITS).max(0.0).ceil() as u64
    } else {
        u64::MAX / 4
    };
    Ok(bits)
}

fn write_leaf_node(w: &mut BitWriter, predictor: Predictor) -> Result<()> {
    TREE_CODE.write_uint(w, 0)?; // ctx 1: property+1 == 0 → leaf
    TREE_CODE.write_uint(w, predictor.index())?; // ctx 2
    TREE_CODE.write_uint(w, pack_signed(0))?; // ctx 3: offset
    TREE_CODE.write_uint(w, 0)?; // ctx 4: mul_log
    TREE_CODE.write_uint(w, 0)?; // ctx 5: mul_bits → multiplier 1
    Ok(())
}

/// Residual-codes selected channels; `rect` clips when `Some` (group sections).
///
/// # Errors
///
/// Residual out of range, entropy-layer rejection, or a bit writer error.
pub fn write_residual_payload_indices(
    w: &mut BitWriter,
    source: &ModularSource,
    indices: &[usize],
    rect: Option<Rect>,
) -> Result<()> {
    let events = collect_residuals_indices(source, indices, rect)?;
    let dist_mul = indices
        .iter()
        .filter_map(|&i| source.channels.get(i).map(|c| c.width))
        .max()
        .unwrap_or(0);
    write_ans_stream(
        w,
        source.tree.num_contexts(),
        &events,
        dist_mul,
        source.allow_lz77,
    )
}

/// Residual-codes every channel at full geometry.
///
/// # Errors
///
/// As [`write_residual_payload_indices`].
pub fn write_residual_payload_full(w: &mut BitWriter, source: &ModularSource) -> Result<()> {
    let indices: Vec<usize> = (0..source.channels.len()).collect();
    write_residual_payload_indices(w, source, &indices, None)
}

/// Back-compat alias for policy scoring.
///
/// # Errors
///
/// As [`write_residual_payload_full`].
pub fn write_residual_payload(w: &mut BitWriter, source: &ModularSource, rect: Rect) -> Result<()> {
    if rect.x0 == 0 && rect.y0 == 0 && rect.width == source.width && rect.height == source.height {
        return write_residual_payload_full(w, source);
    }
    let indices: Vec<usize> = (0..source.channels.len()).collect();
    write_residual_payload_indices(w, source, &indices, Some(rect))
}

/// A legal residual entropy bundle with no sample symbols (multi-section
/// `LfGlobal`).
fn write_empty_residual_stream(w: &mut BitWriter, num_contexts: usize) -> Result<()> {
    write_ans_stream(w, num_contexts, &[], 0, false)
}

/// Builds ANS tables from `(context, value)` events and writes bundle + payload.
///
/// Tries a plain stream and an LZ77 stream (greedy matches); keeps the shorter
/// when `allow_lz77` is true. `dist_multiplier` is H.3's row stride for the
/// C.3.3 distance transform.
fn write_ans_stream(
    w: &mut BitWriter,
    num_contexts: usize,
    events: &[(usize, u32)],
    dist_multiplier: u32,
    allow_lz77: bool,
) -> Result<()> {
    let plain = encode_ans_stream_plain(num_contexts, events)?;
    let best = if allow_lz77 {
        match encode_ans_stream_lz77(num_contexts, events, dist_multiplier) {
            Ok(lz) if lz.bit_len() < plain.bit_len() => lz,
            _ => plain,
        }
    } else {
        plain
    };
    append_bits(w, &best);
    Ok(())
}

/// Literal-only ANS residual stream (LZ77 flag false).
fn encode_ans_stream_plain(num_contexts: usize, events: &[(usize, u32)]) -> Result<BitWriter> {
    let num_contexts = num_contexts.max(1);
    let mut census = TokenCensus::new(num_contexts)?;
    for &(ctx, value) in events {
        census.record(ctx, value)?;
    }
    seed_empty_contexts(&mut census, num_contexts, events.is_empty());
    let config = best_hybrid_config(&census, num_contexts, None)?;
    let plan = EncoderPlan::identity(num_contexts, CodingMode::Ans, config)?;
    let tables = EntropyTables::build(&plan, &census)?;
    let mut w = BitWriter::new();
    tables.write_bundle(&mut w)?;
    let mut encoder = SymbolEncoder::new(&tables);
    for &(ctx, value) in events {
        encoder.push_uint(ctx, value)?;
    }
    encoder.write_stream(&mut w)?;
    Ok(w)
}

/// ANS residual stream with LZ77, or an error if copies cannot be expressed.
///
/// `dist_multiplier` is H.3's value (largest channel width). Match distances
/// are 1-based positions in the residual sequence; raw wire distances use the
/// C.3.3 transform inverse so the modular decoder recovers them.
fn encode_ans_stream_lz77(
    num_contexts: usize,
    events: &[(usize, u32)],
    dist_multiplier: u32,
) -> Result<BitWriter> {
    if events.is_empty() {
        return Err(EncodeError::unsupported(
            "LZ77 on an empty residual stream",
            "C.3.3",
        ));
    }
    let num_contexts = num_contexts.max(1);

    // Probe hybrid-uint on literals alone so `min_symbol` can sit just above
    // the largest literal token (keeps the ANS alphabet small).
    let mut lit_census = TokenCensus::new(num_contexts)?;
    for &(ctx, value) in events {
        lit_census.record(ctx, value)?;
    }
    seed_empty_contexts(&mut lit_census, num_contexts, false);
    let config = best_hybrid_config(&lit_census, num_contexts, None)?;
    let max_lit_token = max_token_in_census(&lit_census, num_contexts, &config)?;
    let min_symbol = choose_min_symbol(max_lit_token).ok_or_else(|| {
        EncodeError::unsupported(
            "residual tokens too large for ANS LZ77 min_symbol",
            "Table C.1",
        )
    })?;
    // length_base ≤ 255 - min_symbol so the length trigger stays in 0..255.
    let max_length = RESIDUAL_LZ77_MIN_LENGTH + (255 - min_symbol);
    let lz77 = residual_lz77_params(min_symbol)?;
    let coded = greedy_lz77_events(events, lz77.min_length, max_length, dist_multiplier);
    if !coded.iter().any(|e| matches!(e, LzEvent::Copy { .. })) {
        return Err(EncodeError::unsupported(
            "LZ77 with no copies adopted",
            "C.3.3",
        ));
    }

    let dist_ctx = num_contexts;
    let num_dist = num_contexts + 1;
    let mut census = TokenCensus::new(num_dist)?;
    for ev in &coded {
        match *ev {
            LzEvent::Lit { ctx, value } => census.record(ctx, value)?,
            LzEvent::Copy {
                ctx,
                length,
                raw_distance,
            } => census.record_copy(ctx, length, raw_distance, dist_ctx, &lz77)?,
        }
    }
    seed_empty_contexts(&mut census, num_dist, false);

    if !literals_below_min_symbol(&census, num_contexts, &config, min_symbol) {
        return Err(EncodeError::unsupported(
            "literal tokens collide with LZ77 min_symbol",
            "C.3.3",
        ));
    }
    let plan = EncoderPlan::identity_with_lz77(num_contexts, CodingMode::Ans, config, lz77)?;
    let tables = EntropyTables::build(&plan, &census)?;
    let mut w = BitWriter::new();
    tables.write_bundle(&mut w)?;
    let mut encoder = SymbolEncoder::new(&tables);
    for ev in &coded {
        match *ev {
            LzEvent::Lit { ctx, value } => encoder.push_uint(ctx, value)?,
            LzEvent::Copy {
                ctx,
                length,
                raw_distance,
            } => encoder.push_copy(ctx, length, raw_distance)?,
        }
    }
    encoder.write_stream(&mut w)?;
    Ok(w)
}

/// Inverse of C.3.3 `resolve_distance` for a desired 1-based window distance.
///
/// With `dist_multiplier == 0` the wire stores `distance - 1`. Otherwise values
/// below 120 index the special table, and larger values encode as
/// `distance + 119`. We always use the `+ 119` form when `dist_multiplier > 0`
/// so every 1D residual distance is expressible without searching the table.
fn encode_raw_distance(distance: u64, dist_multiplier: u32) -> Result<u32> {
    if distance == 0 {
        return Err(EncodeError::unsupported("LZ77 distance of zero", "C.3.3"));
    }
    if dist_multiplier == 0 {
        return u32::try_from(distance - 1)
            .map_err(|_| EncodeError::unsupported("LZ77 distance exceeds u32", "C.3.3"));
    }
    // distance = raw - 119  ⇒  raw = distance + 119, for raw ≥ 120.
    let raw = distance
        .checked_add(119)
        .ok_or_else(|| EncodeError::unsupported("LZ77 raw distance overflows", "C.3.3"))?;
    u32::try_from(raw)
        .map_err(|_| EncodeError::unsupported("LZ77 raw distance exceeds u32", "C.3.3"))
}

fn residual_lz77_params(min_symbol: u32) -> Result<Lz77EncodeParams> {
    let length_config = HybridUintConfig::new(8, 0, 0)
        .map_err(|_| EncodeError::unsupported("lz_len_conf for residual LZ77", "C.2.3"))?;
    Lz77EncodeParams::new(min_symbol, RESIDUAL_LZ77_MIN_LENGTH, length_config)
        .map_err(|_| EncodeError::unsupported("residual LZ77 Table C.1 params", "Table C.1"))
}

/// Smallest Table C.1-representable `min_symbol` above every literal token,
/// still leaving ANS room for length symbols (`≤ 224`).
fn choose_min_symbol(max_literal_token: u32) -> Option<u32> {
    let need = max_literal_token.saturating_add(1).max(8);
    if need > RESIDUAL_LZ77_MAX_MIN_SYMBOL {
        return None;
    }
    // Every value in 8..=32775 is representable as `8 + u(15)`.
    Some(need)
}

fn max_token_in_census(
    census: &TokenCensus,
    num_contexts: usize,
    config: &HybridUintConfig,
) -> Result<u32> {
    let mut max_token = 0u32;
    for ctx in 0..num_contexts {
        let Some(hist) = census.context(ctx) else {
            continue;
        };
        for (value, _) in hist.iter() {
            let split = config.tokenize(value).map_err(EncodeError::from)?;
            max_token = max_token.max(split.token);
        }
    }
    Ok(max_token)
}

/// One residual-stream event after match finding.
#[derive(Debug, Clone, Copy)]
enum LzEvent {
    Lit {
        ctx: usize,
        value: u32,
    },
    Copy {
        ctx: usize,
        length: u32,
        raw_distance: u32,
    },
}

/// Greedy longest-match LZ77 over a residual value sequence.
///
/// Distance is 1-based in the residual window; `raw_distance` is the C.3.3 wire
/// value for `dist_multiplier`. Search is limited to
/// [`RESIDUAL_LZ77_LOOKBACK`] prior symbols.
fn greedy_lz77_events(
    events: &[(usize, u32)],
    min_length: u32,
    max_length: u32,
    dist_multiplier: u32,
) -> Vec<LzEvent> {
    let n = events.len();
    let mut out = Vec::with_capacity(n);
    let min_len = min_length as usize;
    let max_len = max_length as usize;
    let mut i = 0usize;
    while i < n {
        let mut best_len = 0usize;
        let mut best_dist = 0usize; // 1-based
        let search_start = i.saturating_sub(RESIDUAL_LZ77_LOOKBACK);
        for start in (search_start..i).rev() {
            let dist = i - start;
            if best_len == max_len {
                break;
            }
            if events.get(start).map(|e| e.1) != events.get(i).map(|e| e.1) {
                continue;
            }
            let mut len = 1usize;
            while len < max_len
                && i + len < n
                && events.get(start + len).map(|e| e.1) == events.get(i + len).map(|e| e.1)
            {
                len += 1;
            }
            if len >= min_len && len > best_len {
                best_len = len;
                best_dist = dist;
            }
        }
        if best_len >= min_len {
            let ctx = events.get(i).map_or(0, |e| e.0);
            let Ok(raw_distance) = encode_raw_distance(best_dist as u64, dist_multiplier) else {
                let (ctx, value) = events.get(i).copied().unwrap_or((0, 0));
                out.push(LzEvent::Lit { ctx, value });
                i += 1;
                continue;
            };
            out.push(LzEvent::Copy {
                ctx,
                length: u32::try_from(best_len).unwrap_or(max_length),
                raw_distance,
            });
            i += best_len;
        } else {
            let (ctx, value) = events.get(i).copied().unwrap_or((0, 0));
            out.push(LzEvent::Lit { ctx, value });
            i += 1;
        }
    }
    out
}

fn seed_empty_contexts(census: &mut TokenCensus, num_contexts: usize, force_all: bool) {
    if force_all {
        for ctx in 0..num_contexts {
            let _ = census.record(ctx, 0);
        }
        return;
    }
    for ctx in 0..num_contexts {
        if census
            .context(ctx)
            .is_none_or(|h| h.iter().next().is_none())
        {
            let _ = census.record(ctx, 0);
        }
    }
}

/// Appends every bit of `src` onto `dst` (order-preserving, not byte-aligned).
fn append_bits(dst: &mut BitWriter, src: &BitWriter) {
    // Prefer the bulk path: no clone, no per-bit replay.
    if let Err(err) = dst.append_writer(src) {
        // A well-formed source should never overflow accounting; fall back so
        // an unexpected edge cannot abort emission with a silent skip.
        debug_assert!(false, "append_writer failed: {err:?}");
        let bits = src.bit_len();
        let bytes = src.as_bytes();
        for i in 0..bits {
            let byte_index = usize::try_from(i / 8).unwrap_or(0);
            let bit_index = i % 8;
            let set = bytes
                .get(byte_index)
                .is_some_and(|b| (b >> bit_index) & 1 == 1);
            dst.write_bit(set);
        }
    }
}

/// Picks a hybrid-uint config by estimated token Shannon cost on the census.
///
/// When `max_literal_token` is `Some(m)`, every hybrid-uint token of a *value*
/// (not a direct LZ77 length token) must be `< m` so it cannot collide with
/// length-trigger symbols.
fn best_hybrid_config(
    census: &TokenCensus,
    num_contexts: usize,
    max_literal_token: Option<u32>,
) -> Result<HybridUintConfig> {
    let mut best: Option<(f64, HybridUintConfig)> = None;
    for &(split, msb, lsb) in HYBRID_CANDIDATES {
        let Ok(config) = HybridUintConfig::new(split, msb, lsb) else {
            continue;
        };
        if let Some(cap) = max_literal_token
            && !literals_below_min_symbol(census, num_contexts, &config, cap)
        {
            continue;
        }
        let cost = hybrid_data_cost(census, num_contexts, &config);
        if best.is_none_or(|(c, _)| cost < c) {
            best = Some((cost, config));
        }
    }
    best.map(|(_, c)| c).ok_or_else(|| {
        EncodeError::unsupported("any hybrid-uint configuration for residuals", "C.2.3")
    })
}

fn literals_below_min_symbol(
    census: &TokenCensus,
    num_contexts: usize,
    config: &HybridUintConfig,
    min_symbol: u32,
) -> bool {
    for ctx in 0..num_contexts {
        let Some(hist) = census.context(ctx) else {
            continue;
        };
        for (value, _) in hist.iter() {
            let Ok(split) = config.tokenize(value) else {
                return false;
            };
            if split.token >= min_symbol {
                return false;
            }
        }
    }
    true
}

fn hybrid_data_cost(census: &TokenCensus, num_contexts: usize, config: &HybridUintConfig) -> f64 {
    let mut token_counts: Vec<u64> = Vec::new();
    let mut extra_bits = 0.0f64;
    for ctx in 0..num_contexts {
        let Some(hist) = census.context(ctx) else {
            continue;
        };
        for (value, count) in hist.iter() {
            let Ok(split) = config.tokenize(value) else {
                return f64::INFINITY;
            };
            let token = split.token as usize;
            if token_counts.len() <= token {
                token_counts.resize(token + 1, 0);
            }
            if let Some(slot) = token_counts.get_mut(token) {
                *slot = slot.saturating_add(count);
            }
            extra_bits += count as f64 * f64::from(split.extra_bits);
        }
    }
    let total: u64 = token_counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let mut data = 0.0f64;
    for &c in &token_counts {
        if c == 0 {
            continue;
        }
        let p = c as f64 / total as f64;
        data += -(c as f64) * p.log2();
    }
    data + extra_bits
}

/// Residuals for selected channels at full geometry or a group rectangle.
fn collect_residuals_indices(
    source: &ModularSource,
    indices: &[usize],
    rect: Option<Rect>,
) -> Result<Vec<(usize, u32)>> {
    let mut out = Vec::new();
    for &i in indices {
        let Some(ch) = source.channels.get(i) else {
            continue;
        };
        let sample_rect = match rect {
            None => Rect {
                x0: 0,
                y0: 0,
                width: ch.width,
                height: ch.height,
            },
            Some(r) => {
                // G.2.3 / G.4.2: map frame rect into channel coordinates.
                let hs = ch.hshift.clamp(0, 31) as u32;
                let vs = ch.vshift.clamp(0, 31) as u32;
                let x0 = r.x0 >> hs;
                let y0 = r.y0 >> vs;
                let x1 = (r.x0 + r.width).div_ceil(1u32 << hs);
                let y1 = (r.y0 + r.height).div_ceil(1u32 << vs);
                let width = x1.saturating_sub(x0).min(ch.width.saturating_sub(x0));
                let height = y1.saturating_sub(y0).min(ch.height.saturating_sub(y0));
                Rect {
                    x0,
                    y0,
                    width,
                    height,
                }
            }
        };
        if sample_rect.width == 0 || sample_rect.height == 0 {
            continue;
        }
        collect_plane_residuals(&ch.data, ch.width, sample_rect, &source.tree, &mut out)?;
    }
    Ok(out)
}

/// Sampled variant of [`collect_residuals_indices`] for Phase 4B's scorer:
/// visits a row-strided subset of each channel's samples instead of a full
/// scan (no `rect` clipping -- always relative to the channel's own
/// dimensions, matching how `estimate_residual_bits_sampled` is used during
/// MA-tree topology search on the whole score plane).
fn collect_residuals_indices_sampled(
    source: &ModularSource,
    indices: &[usize],
    row_stride: u32,
) -> Result<Vec<(usize, u32)>> {
    let mut out = Vec::new();
    for &i in indices {
        let Some(ch) = source.channels.get(i) else {
            continue;
        };
        collect_plane_residuals_sampled(
            &ch.data,
            ch.width,
            ch.height,
            row_stride,
            &source.tree,
            &mut out,
        )?;
    }
    Ok(out)
}

/// Rows visited by the sampled scorer: every `row_stride`-th row, always
/// including the final row so the tree's bottom edge is represented.
fn sampled_rows(height: u32, row_stride: u32) -> Vec<u32> {
    if height == 0 {
        return Vec::new();
    }
    let row_stride = row_stride.max(1) as usize;
    let mut rows: Vec<u32> = (0..height).step_by(row_stride).collect();
    if rows.last().copied() != Some(height - 1) {
        rows.push(height - 1);
    }
    rows
}

fn collect_plane_residuals_sampled(
    plane: &[i32],
    width: u32,
    height: u32,
    row_stride: u32,
    tree: &MaTree,
    out: &mut Vec<(usize, u32)>,
) -> Result<()> {
    let stride = usize::try_from(width).unwrap_or(usize::MAX);
    // Absolute (not rect-relative) indexing: predictions and context
    // properties always read ground-truth samples, so a sampled row can look
    // at its true W/N/NW neighbours even when those rows aren't visited.
    let at = |x: u32, y: u32| -> i64 {
        let index = usize::try_from(y)
            .ok()
            .and_then(|row| row.checked_mul(stride))
            .and_then(|row| usize::try_from(x).ok().and_then(|x| row.checked_add(x)));
        index
            .and_then(|i| plane.get(i))
            .map_or(0, |&v| i64::from(v))
    };
    for y in sampled_rows(height, row_stride) {
        for x in 0..width {
            let (ctx, predictor) = tree.leaf_at(&at, x, y);
            let prediction = predict(&at, x, y, predictor);
            let residual = at(x, y) - prediction;
            let residual = i32::try_from(residual).map_err(|_| EncodeError::ValueOutOfRange {
                what: "modular residual",
                value: residual,
            })?;
            out.push((ctx, pack_signed(residual)));
        }
    }
    Ok(())
}

fn collect_plane_residuals(
    plane: &[i32],
    stride: u32,
    rect: Rect,
    tree: &MaTree,
    out: &mut Vec<(usize, u32)>,
) -> Result<()> {
    // Phase 4A: any tree with a Weighted leaf needs the full sequential,
    // state-carrying walk -- H.5.1 requires the self-correcting state to
    // advance on every sample regardless of which leaf it lands in, so this
    // cannot share the order-independent paths below.
    if tree.contains_predictor(Predictor::Weighted) {
        return collect_plane_residuals_weighted(plane, stride, rect, tree, out);
    }

    let stride = usize::try_from(stride).unwrap_or(usize::MAX);
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

    // Phase-1: specialized single-leaf paths skip MA tree walks.
    if tree.num_contexts() == 1 {
        let predictor = tree.primary_predictor();
        match predictor {
            Predictor::Zero => {
                for y in 0..rect.height {
                    for x in 0..rect.width {
                        let residual =
                            i32::try_from(at(x, y)).map_err(|_| EncodeError::ValueOutOfRange {
                                what: "modular residual",
                                value: at(x, y),
                            })?;
                        out.push((0, pack_signed(residual)));
                    }
                }
                return Ok(());
            }
            Predictor::West | Predictor::North | Predictor::Gradient => {
                for y in 0..rect.height {
                    for x in 0..rect.width {
                        let prediction = predict(&at, x, y, predictor);
                        let residual = at(x, y) - prediction;
                        let residual =
                            i32::try_from(residual).map_err(|_| EncodeError::ValueOutOfRange {
                                what: "modular residual",
                                value: residual,
                            })?;
                        out.push((0, pack_signed(residual)));
                    }
                }
                return Ok(());
            }
            Predictor::AverageWestNorth | Predictor::Select => {}
            Predictor::Weighted => {
                unreachable!("Weighted-containing trees are dispatched above")
            }
        }
    }

    for y in 0..rect.height {
        for x in 0..rect.width {
            let (ctx, predictor) = tree.leaf_at(&at, x, y);
            let prediction = predict(&at, x, y, predictor);
            let residual = at(x, y) - prediction;
            let residual = i32::try_from(residual).map_err(|_| EncodeError::ValueOutOfRange {
                what: "modular residual",
                value: residual,
            })?;
            out.push((ctx, pack_signed(residual)));
        }
    }
    Ok(())
}

/// Full sequential collection for a tree containing [`Predictor::Weighted`]
/// anywhere in it.
///
/// H.5.1 requires the self-correcting state to advance for every sample of
/// the scan, in raster order, unconditionally -- including samples whose
/// leaf selects a different predictor -- so unlike
/// [`collect_plane_residuals`] and [`collect_plane_residuals_sampled`], this
/// cannot skip rows, reorder them, or use a specialised fast path.
///
/// `rect` is a self-contained scan region, matching how this encoder already
/// treats group rects for every other predictor: H.3 edge substitution (and
/// here, the H.5 error state) resets at the rect's own boundaries, not the
/// full channel's -- see [`neighbours`]'s `x > 0` / `y > 0` tests, which are
/// rect-relative, not absolute-plane. This is what lets groups decode
/// independently on the decoder side (`jpxl_decode::modular::decode_channels`
/// is handed one group-local channel per call); [`WeightedState`] follows the
/// same scoping so the two sides agree on where state resets.
fn collect_plane_residuals_weighted(
    plane: &[i32],
    stride: u32,
    rect: Rect,
    tree: &MaTree,
    out: &mut Vec<(usize, u32)>,
) -> Result<()> {
    use jpxl_core::limits::{AllocGuard, Limits};
    use jpxl_core::modular_weighted::{WeightedState, WpHeader};

    let stride_usize = usize::try_from(stride).unwrap_or(usize::MAX);
    let at = |x: u32, y: u32| -> i64 {
        let index = usize::try_from(rect.y0 + y)
            .ok()
            .and_then(|row| row.checked_mul(stride_usize))
            .and_then(|row| {
                usize::try_from(rect.x0 + x)
                    .ok()
                    .and_then(|x| row.checked_add(x))
            });
        index
            .and_then(|i| plane.get(i))
            .map_or(0, |&v| i64::from(v))
    };

    // The encoder always emits WPHeader's default_wp = true (see
    // encode_lf_global); the search-time cost estimate must match what will
    // actually be emitted.
    let header = WpHeader::default_wp();
    let mut guard = AllocGuard::new(&Limits::relaxed());
    let mut wp_state = WeightedState::new(rect.width, &mut guard)?;

    for y in 0..rect.height {
        for x in 0..rect.width {
            let nb = weighted_neighbours(&at, x, y, rect.width);
            let wp = wp_state.predict(&header, &nb, x);
            let (ctx, predictor) = tree.leaf_at(&at, x, y);
            // Table H.3 row 6: (prediction + 3) >> 3 converts out of the <<3
            // domain H.5.2 works in.
            let prediction = if predictor == Predictor::Weighted {
                (wp.prediction + 3) >> 3
            } else {
                predict(&at, x, y, predictor)
            };
            let sample = at(x, y);
            let residual = sample - prediction;
            let residual = i32::try_from(residual).map_err(|_| EncodeError::ValueOutOfRange {
                what: "modular residual",
                value: residual,
            })?;
            let sample_i32 = i32::try_from(sample).map_err(|_| EncodeError::ValueOutOfRange {
                what: "modular sample",
                value: sample,
            })?;
            // H.5.1: the state advances on every sample, unconditionally --
            // not only ones that selected predictor 6.
            wp_state.update(x, &wp, sample_i32);
            out.push((ctx, pack_signed(residual)));
        }
        wp_state.advance_row();
    }
    Ok(())
}

/// Table H.3 prediction at rectangle-relative `(x, y)`.
///
/// Never called with [`Predictor::Weighted`]: that predictor has no
/// stateless form (H.5 is a running state machine, not a neighbour
/// function) and is handled entirely by
/// [`collect_plane_residuals_weighted`]'s own dispatch.
fn predict(at: &impl Fn(u32, u32) -> i64, x: u32, y: u32, predictor: Predictor) -> i64 {
    let (w, n, nw) = neighbours(at, x, y);
    match predictor {
        Predictor::Zero => 0,
        Predictor::West => w,
        Predictor::North => n,
        Predictor::AverageWestNorth => (w + n) / 2,
        Predictor::Select => {
            // Table H.3: abs(N - NW) < abs(W - NW) ? W : N
            if (n - nw).abs() < (w - nw).abs() {
                w
            } else {
                n
            }
        }
        Predictor::Gradient => (w + n - nw).clamp(w.min(n), w.max(n)),
        Predictor::Weighted => {
            unreachable!("Weighted is priced by collect_plane_residuals_weighted, not predict()")
        }
    }
}

/// H.3 edge substitutions for W, N, NW.
fn neighbours(at: &impl Fn(u32, u32) -> i64, x: u32, y: u32) -> (i64, i64, i64) {
    let w = if x > 0 {
        at(x - 1, y)
    } else if y > 0 {
        at(x, y - 1)
    } else {
        0
    };
    let n = if y > 0 { at(x, y - 1) } else { w };
    let nw = if x > 0 && y > 0 { at(x - 1, y - 1) } else { w };
    (w, n, nw)
}

/// Table H.2's full seven-neighbour set, with H.3's edge-substitution
/// cascade, for [`Predictor::Weighted`]'s [`jpxl_core::modular_weighted`]
/// state machine. `width` bounds the `NE`/`NEE` lookahead; `x`, `y` are
/// rect-relative, matching [`neighbours`].
fn weighted_neighbours(
    at: &impl Fn(u32, u32) -> i64,
    x: u32,
    y: u32,
    width: u32,
) -> jpxl_core::modular_weighted::SelfCorrectingNeighbours {
    let w = if x > 0 {
        at(x - 1, y)
    } else if y > 0 {
        at(x, y - 1)
    } else {
        0
    };
    let n = if y > 0 { at(x, y - 1) } else { w };
    let nw = if x > 0 && y > 0 { at(x - 1, y - 1) } else { w };
    let ne = if x + 1 < width && y > 0 {
        at(x + 1, y - 1)
    } else {
        n
    };
    let nn = if y > 1 { at(x, y - 2) } else { n };
    let nee = if x + 2 < width && y > 0 {
        at(x + 2, y - 1)
    } else {
        ne
    };
    let ww = if x > 1 { at(x - 2, y) } else { w };
    jpxl_core::modular_weighted::SelfCorrectingNeighbours {
        w,
        n,
        nw,
        ne,
        nn,
        nee,
        ww,
    }
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

        assert_eq!(predict(&at, 0, 0, Predictor::Gradient), 0);
        // Row 0, x > 0: N falls back to W, NW to W, so the gradient is W.
        assert_eq!(predict(&at, 1, 0, Predictor::Gradient), 10);
        assert_eq!(predict(&at, 2, 0, Predictor::Gradient), 20);
        // Column 0, y > 0: W falls back to N, so the gradient is N.
        assert_eq!(predict(&at, 0, 1, Predictor::Gradient), 10);
        // Interior: clamp(W + N - NW, min, max) = clamp(40 + 20 - 10, 20, 40).
        assert_eq!(predict(&at, 1, 1, Predictor::Gradient), 40);
    }

    #[test]
    fn select_matches_table_h3() {
        // Decoder unit test: N=20, NW=14, W=10 → abs(N-NW)=6, abs(W-NW)=4
        // 6 < 4 is false → Select yields N = 20.
        let at = |x: u32, y: u32| -> i64 {
            match (x, y) {
                (0, 0) => 14, // NW of (1,1)
                (1, 0) => 20, // N
                (0, 1) => 10, // W
                _ => 0,
            }
        };
        assert_eq!(predict(&at, 1, 1, Predictor::Select), 20);
    }

    #[test]
    fn binary_split_assigns_left_and_right_contexts() {
        let tree = MaTree::binary_split(5, 10, Predictor::Gradient); // abs(W)
        // W = 20 → abs(W)=20 > 10 → left ctx 0
        let high = |x: u32, y: u32| -> i64 { if x == 0 && y == 1 { 20 } else { 0 } };
        assert_eq!(tree.context(&high, 1, 1), 0);
        // W = 3 → abs(W)=3 ≤ 10 → right ctx 1
        let low = |x: u32, y: u32| -> i64 { if x == 0 && y == 1 { 3 } else { 0 } };
        assert_eq!(tree.context(&low, 1, 1), 1);
    }

    #[test]
    fn deeper_tree_assigns_ctx_ids_in_bfs_leaf_order() {
        // Root split, then split the left child → three leaves.
        // BFS decode order: D0, D1, L_right_of_root (ctx0), L_ll (ctx1), L_lr (ctx2).
        let tree = MaTree::binary_split(6, 0, Predictor::West) // N
            .split_leaf(0, 7, 0, Predictor::North, Predictor::Gradient)
            .expect("split left");
        assert_eq!(tree.num_contexts(), 3);
        assert_eq!(tree.depth(), 2);
        let preds = tree.leaf_predictors();
        // ctx0 is the unsplit right leaf of the root (West).
        assert_eq!(preds[0], Predictor::West);
        assert_eq!(preds[1], Predictor::North);
        assert_eq!(preds[2], Predictor::Gradient);
    }

    #[test]
    fn per_leaf_predictors_differ_on_binary_split() {
        let tree = MaTree::binary_split_preds(5, 0, Predictor::West, Predictor::North);
        assert_eq!(
            tree.leaf_predictors(),
            vec![Predictor::West, Predictor::North]
        );
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
        let source = ModularSource::direct(
            4,
            4,
            &planes,
            false,
            MaTree::single_leaf(Predictor::Gradient),
            true,
        );
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
    fn lf_global_omits_oversized_channels_when_multi_section() {
        let planes = vec![vec![7i32; 300 * 300]];
        let source = ModularSource::direct(
            300,
            300,
            &planes,
            false,
            MaTree::single_leaf(Predictor::Gradient),
            true,
        );
        // group_dim 512: single-section → full residual. group_dim 128: multi
        // → channel exceeds group_dim → empty residual (header-only GlobalModular).
        let with_samples = encode_lf_global(&source, &Geometry::new(300, 300, 2).expect("valid"))
            .expect("encodes");
        let multi = encode_lf_global(&source, &Geometry::new(300, 300, 0).expect("valid"))
            .expect("encodes");
        assert!(
            multi.len() < with_samples.len(),
            "multi-section LfGlobal without meta must omit frame-sized residuals: {} vs {}",
            multi.len(),
            with_samples.len()
        );
    }

    #[test]
    fn partition_puts_palette_meta_in_lf_global() {
        let n = 200 * 200;
        let plane = vec![0i32; n];
        let fwd = try_exact_palette(200, 200, &[plane], 0, 1)
            .expect("ok")
            .expect("palette");
        let source = ModularSource::from_palette(fwd, MaTree::single_leaf(Predictor::Zero), false);
        let part = partition_channels(&source, 128);
        assert_eq!(part.nb_meta, 1);
        assert_eq!(part.lf_global, vec![0]);
        assert_eq!(part.pass_group, vec![1]);
        assert!(part.lf_group.is_empty());
    }

    #[test]
    fn greedy_lz77_collapses_a_long_run() {
        let events: Vec<(usize, u32)> = (0..40).map(|_| (0usize, 0u32)).collect();
        // dist_multiplier 0: raw = distance - 1.
        let coded = greedy_lz77_events(&events, 3, 34, 0);
        assert!(
            coded.iter().any(|e| matches!(e, LzEvent::Copy { .. })),
            "expected at least one copy on a constant residual run"
        );
        let mut expanded = Vec::new();
        for ev in &coded {
            match *ev {
                LzEvent::Lit { value, .. } => expanded.push(value),
                LzEvent::Copy {
                    length,
                    raw_distance,
                    ..
                } => {
                    let dist = raw_distance as usize + 1;
                    for _ in 0..length {
                        let v = expanded[expanded.len() - dist];
                        expanded.push(v);
                    }
                }
            }
        }
        assert_eq!(expanded, vec![0u32; 40]);
    }

    #[test]
    fn encode_raw_distance_inverts_resolve_with_row_stride() {
        use jpxl_entropy::lz77::resolve_distance;
        for dist_multiplier in [0u32, 1, 64, 256] {
            for distance in [1u64, 2, 3, 64, 65, 100, 256] {
                let raw = encode_raw_distance(distance, dist_multiplier).expect("enc");
                let got = resolve_distance(raw, dist_multiplier).expect("dec");
                assert_eq!(
                    got, distance,
                    "M={dist_multiplier} distance={distance} raw={raw}"
                );
            }
        }
    }

    #[test]
    fn residual_lz77_is_not_forced_when_plain_wins() {
        // Single-symbol residual streams are already near-free under ANS; LZ77
        // must not be forced when it is larger (adopt-if-cheaper).
        let events: Vec<(usize, u32)> = vec![(0, 0); 256];
        let plain = encode_ans_stream_plain(1, &events).expect("plain");
        let mut w = BitWriter::new();
        write_ans_stream(&mut w, 1, &events, 16, true).expect("write");
        assert_eq!(
            w.bit_len(),
            plain.bit_len(),
            "constant residuals should keep the plain stream"
        );
    }

    #[test]
    fn residual_lz77_beats_plain_on_a_repeating_multi_value_pattern() {
        // 0..31 repeated 40 times: diverse enough that ANS pays ~5 bits/symbol,
        // repetitive enough that length/distance tokens win.
        let mut events = Vec::with_capacity(32 * 40);
        for _ in 0..40 {
            for v in 0..32u32 {
                events.push((0usize, v));
            }
        }
        let plain = encode_ans_stream_plain(1, &events).expect("plain");
        // dist_multiplier = period width, as H.3 would set for a 32-wide plane.
        let lz = encode_ans_stream_lz77(1, &events, 32).expect("lz77");
        assert!(
            lz.bit_len() < plain.bit_len(),
            "LZ77 must beat plain on a long repeating residual pattern: {} vs {}",
            lz.bit_len(),
            plain.bit_len()
        );
    }
}
