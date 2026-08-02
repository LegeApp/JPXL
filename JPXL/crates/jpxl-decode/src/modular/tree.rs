//! Meta-adaptive context modelling: properties and the MA tree (18181-1 H.4).
//!
//! # Properties (H.4.1, Table H.4)
//!
//! ```text
//!  0  i (channel index)        8  x > 0 ? W - property9(x-1, y) : W
//!  1  stream index             9  W + N - NW
//!  2  y                       10  W - NW
//!  3  x                       11  NW - N
//!  4  abs(N)                  12  N - NE
//!  5  abs(W)                  13  N - NN
//!  6  N                       14  W - WW
//!  7  W                       15  max_error (H.5.1)
//! ```
//!
//! followed by four "previous channel" properties for each earlier channel
//! whose width, height, `hshift` and `vshift` all match the current one.
//! Any property index the tree asks for that is not defined here reads as
//! zero, which H.4.1 states explicitly.
//!
//! **OCR note.** Rows 4 and 5 are damaged in both transcriptions: `latex`
//! reads `abs(N)` / `abs(i)` and the markdown reads `abs(1)` / `abs(i)`.
//! Rows 6 and 7 are `N` and `W`, so the only reading that makes rows 4–7 the
//! magnitude-then-value pair they obviously are is `abs(N)` / `abs(W)`. That
//! is what is implemented.
//!
//! # The tree (H.4.2)
//!
//! ```text
//! decode_tree() {
//!   ctx_id = 0; nodes_left = 1; tree.clear();
//!   while (nodes_left > 0) {
//!     nodes_left--;
//!     property = DecodeHybridVarLenUint(1) - 1;
//!     if (property >= 0) {
//!       decision_node.property = property;
//!       decision_node.value = UnpackSigned(DecodeHybridVarLenUint(0));
//!       decision_node.left_child  = tree.size() + nodes_left + 1;
//!       decision_node.right_child = tree.size() + nodes_left + 2;
//!       tree.push_back(decision_node); nodes_left += 2;
//!     } else {
//!       leaf_node.ctx = ctx_id++;
//!       leaf_node.predictor = DecodeHybridVarLenUint(2);
//!       leaf_node.offset = UnpackSigned(DecodeHybridVarLenUint(3));
//!       mul_log  = DecodeHybridVarLenUint(4);
//!       mul_bits = DecodeHybridVarLenUint(5);
//!       leaf_node.multiplier = (mul_bits + 1) << mul_log;
//!       tree.push_back(leaf_node);
//!     }
//!   }
//! }
//! ```
//!
//! The index arithmetic makes this a breadth-first queue: `nodes_left` counts
//! slots that have been promised but not yet written, and the node being
//! written always sits at `tree.size()`. Children therefore always have a
//! strictly larger index than their parent, which is what makes a single
//! forward pass enough to validate the shape and compute every depth.
//!
//! # Untrusted input
//!
//! The tree is fully attacker-controlled: node count, depth, property indices,
//! predictors and multipliers all come off the wire. Every one of them is
//! bounded here — node count by [`TreeLimits`] and by the `1 << 26` cap H.4.2
//! states, depth by [`TreeLimits`], predictors by Table H.3, and multipliers by
//! the `mul_log`/`mul_bits` constraints of H.4.2.

use jpxl_bitstream::BitReader;
use jpxl_core::limits::AllocGuard;
use jpxl_entropy::SymbolDecoder;

use super::channel::Channel;
use super::error::{Result, malformed};
use super::predictor::{Neighbours, Predictor};
use super::unpack_signed;
use super::weighted::narrow_to_i32;

/// Number of pre-clustered distributions the tree decoder itself uses (H.4.2).
pub const TREE_NUM_CONTEXTS: usize = 6;

/// H.4.2: `tree.size() <= (1 << 26)` for any conforming stream.
pub const SPEC_MAX_TREE_NODES: usize = 1 << 26;

/// H.4.2: `mul_log` is never strictly larger than 30.
pub const MAX_MUL_LOG: u32 = 30;

/// The number of properties defined by Table H.4 before the per-channel ones.
pub const NUM_STATIC_PROPERTIES: usize = 16;

/// Bounds on MA tree shape, from the Annex M level table.
///
/// Annex M caps "max_tree depth (H.4.2)" at 64 for level 5 and 2048 for level
/// 10, and the node count at `min(1 << 22, 1024 + fwidth * fheight *
/// nb_channels / 16)` globally or `min(1 << 20, 1024 + nb_local_samples)`
/// locally. Those formulas need frame geometry, which Annex H does not have,
/// so the caller supplies the resolved numbers. The defaults here are the
/// absolute caps H.4.2 itself states, so a caller that forgets is still bounded
/// — and every node is additionally charged to the [`AllocGuard`], which means
/// `Limits::max_alloc_bytes` bounds the tree even at the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeLimits {
    /// Maximum number of nodes (decision and leaf) in the decoded tree.
    pub max_nodes: usize,
    /// Maximum root-to-leaf depth, with the root at depth 0.
    pub max_depth: u32,
}

impl TreeLimits {
    /// The caps stated by H.4.2 itself: `1 << 26` nodes, and a depth that
    /// cannot exceed the node count.
    #[must_use]
    pub const fn spec_maximum() -> Self {
        Self {
            max_nodes: SPEC_MAX_TREE_NODES,
            max_depth: SPEC_MAX_TREE_NODES as u32,
        }
    }

    /// The Annex M level-5 bounds: 2^20 local nodes and depth 64.
    #[must_use]
    pub const fn level5() -> Self {
        Self {
            max_nodes: 1 << 20,
            max_depth: 64,
        }
    }
}

impl Default for TreeLimits {
    fn default() -> Self {
        Self::spec_maximum()
    }
}

/// A leaf of the MA tree (H.4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Leaf {
    /// `ln.ctx`: the entropy context this leaf selects, numbered in decode
    /// order starting at zero.
    pub ctx: usize,
    /// `ln.predictor`: the Table H.3 predictor to apply.
    pub predictor: Predictor,
    /// `ln.offset`: added to the residual after multiplication.
    pub offset: i32,
    /// `ln.multiplier`: `(mul_bits + 1) << mul_log`, at least 1.
    pub multiplier: i64,
}

/// A node of the MA tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeNode {
    /// An inner node testing `property[k] > value`; true takes `left`.
    Decision {
        /// Index into the property vector.
        property: u32,
        /// Threshold; the test is strictly greater.
        value: i32,
        /// Node taken when the test is true.
        left: usize,
        /// Node taken when the test is false.
        right: usize,
    },
    /// A terminal node.
    Leaf(Leaf),
}

/// A decoded MA tree (H.4.2), validated and ready to traverse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaTree {
    nodes: Vec<TreeNode>,
    num_leaves: usize,
    depth: u32,
}

impl MaTree {
    /// A single-leaf tree, the minimum a stream can signal.
    ///
    /// # Errors
    ///
    /// [`ModularError::Malformed`](super::ModularError::Malformed) if
    /// `predictor` has no row in Table H.3.
    pub fn single_leaf(predictor: u32, offset: i32, multiplier: i64) -> Result<Self> {
        Ok(Self {
            nodes: vec![TreeNode::Leaf(Leaf {
                ctx: 0,
                predictor: Predictor::from_value(predictor)?,
                offset,
                multiplier,
            })],
            num_leaves: 1,
            depth: 0,
        })
    }

    /// The nodes, in decode order. `nodes[0]` is the root.
    #[must_use]
    pub fn nodes(&self) -> &[TreeNode] {
        &self.nodes
    }

    /// Number of leaves, which is `(tree.size() + 1) / 2` and also the number
    /// of pre-clustered distributions the data stream needs (H.4.2).
    #[must_use]
    pub const fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    /// Longest root-to-leaf path, with the root at depth 0.
    #[must_use]
    pub const fn depth(&self) -> u32 {
        self.depth
    }

    /// `MA(properties)`: walks the tree and returns the selected leaf (H.4.1).
    ///
    /// A decision node takes the left branch when
    /// `property[d.property] > d.value`. Property indices past the end of
    /// `properties` read as zero, per H.4.1.
    ///
    /// # Errors
    ///
    /// [`ModularError::Malformed`](super::ModularError::Malformed) if the walk
    /// leaves the node array. Construction rules that out, so this is a
    /// belt-and-braces check rather than an expected path.
    pub fn traverse(&self, properties: &[i32]) -> Result<&Leaf> {
        let mut index = 0usize;
        // Child indices strictly increase, so the walk cannot cycle and cannot
        // take more steps than there are nodes.
        for _ in 0..=self.nodes.len() {
            match self.nodes.get(index) {
                Some(TreeNode::Leaf(leaf)) => return Ok(leaf),
                Some(TreeNode::Decision {
                    property,
                    value,
                    left,
                    right,
                }) => {
                    let p = properties.get(*property as usize).copied().unwrap_or(0);
                    index = if p > *value { *left } else { *right };
                }
                None => {
                    return Err(malformed!(
                        "H.4.1: MA tree traversal reached node {index}, past the {} decoded",
                        self.nodes.len()
                    ));
                }
            }
        }
        Err(malformed!("H.4.1: MA tree traversal did not terminate"))
    }

    /// Decodes an MA tree from an already-opened six-context stream (H.4.2).
    ///
    /// The caller owns the [`SymbolDecoder`] because H.4.2 splits the work:
    /// six pre-clustered distributions are read first (that is the decoder),
    /// then the tree symbols, and only then does the caller read the data
    /// stream's own distributions.
    ///
    /// # Errors
    ///
    /// [`ModularError::Malformed`](super::ModularError::Malformed) for a tree
    /// that breaks any H.4.2 constraint, or
    /// [`ModularError::Core`](super::ModularError::Core) when `limits` or the
    /// guard reject its size.
    pub fn decode(
        reader: &mut BitReader<'_>,
        decoder: &mut SymbolDecoder,
        limits: TreeLimits,
        guard: &mut AllocGuard,
    ) -> Result<Self> {
        let max_nodes = limits.max_nodes.min(SPEC_MAX_TREE_NODES);
        let mut nodes: Vec<TreeNode> = Vec::new();
        let mut ctx_id = 0usize;
        let mut nodes_left = 1usize;

        while nodes_left > 0 {
            nodes_left -= 1;
            if nodes.len() >= max_nodes {
                return Err(malformed!(
                    "H.4.2: MA tree exceeds the {max_nodes}-node limit in force"
                ));
            }
            // Charge before pushing, so `Limits::max_alloc_bytes` bounds the
            // tree independently of `TreeLimits`.
            guard.charge(size_of::<TreeNode>() as u64)?;

            let raw_property = decoder.read_uint(reader, 1)?;
            if raw_property > 0 {
                let property = raw_property - 1;
                let value = unpack_signed(decoder.read_uint(reader, 0)?);
                // `nodes_left` has already been decremented, exactly as H.4.2
                // writes it.
                let left = nodes
                    .len()
                    .checked_add(nodes_left)
                    .and_then(|v| v.checked_add(1))
                    .ok_or_else(|| malformed!("H.4.2: child index overflows"))?;
                let right = left
                    .checked_add(1)
                    .ok_or_else(|| malformed!("H.4.2: child index overflows"))?;
                nodes.push(TreeNode::Decision {
                    property,
                    value,
                    left,
                    right,
                });
                nodes_left = nodes_left
                    .checked_add(2)
                    .ok_or_else(|| malformed!("H.4.2: pending-node count overflows"))?;
            } else {
                let predictor = Predictor::from_value(decoder.read_uint(reader, 2)?)?;
                let offset = unpack_signed(decoder.read_uint(reader, 3)?);
                let mul_log = decoder.read_uint(reader, 4)?;
                let mul_bits = decoder.read_uint(reader, 5)?;
                if mul_log > MAX_MUL_LOG {
                    return Err(malformed!(
                        "H.4.2: mul_log = {mul_log} is larger than {MAX_MUL_LOG}"
                    ));
                }
                let max_bits = (1u32 << (31 - mul_log)) - 2;
                if mul_bits > max_bits {
                    return Err(malformed!(
                        "H.4.2: mul_bits = {mul_bits} exceeds (1 << (31 - {mul_log})) - 2 = \
                         {max_bits}"
                    ));
                }
                let multiplier = i64::from(mul_bits + 1) << mul_log;
                nodes.push(TreeNode::Leaf(Leaf {
                    ctx: ctx_id,
                    predictor,
                    offset,
                    multiplier,
                }));
                ctx_id += 1;
            }
        }

        Self::finish(nodes, ctx_id, limits)
    }

    /// Validates shape and computes depth in one forward pass.
    fn finish(nodes: Vec<TreeNode>, num_leaves: usize, limits: TreeLimits) -> Result<Self> {
        if nodes.is_empty() {
            return Err(malformed!("H.4.2: MA tree has no nodes"));
        }
        if nodes.len() % 2 == 0 {
            return Err(malformed!(
                "H.4.2: a full binary tree has an odd node count, got {}",
                nodes.len()
            ));
        }
        if (nodes.len() + 1) / 2 != num_leaves {
            return Err(malformed!(
                "H.4.2: {} nodes imply {} leaves but {num_leaves} were decoded",
                nodes.len(),
                (nodes.len() + 1) / 2
            ));
        }

        // Children always have a larger index than their parent, so assigning
        // depths in index order is enough: a node's depth is final before it is
        // read.
        let mut depths = vec![u32::MAX; nodes.len()];
        if let Some(root) = depths.first_mut() {
            *root = 0;
        }
        let mut max_depth = 0u32;
        for (index, node) in nodes.iter().enumerate() {
            let depth = depths.get(index).copied().unwrap_or(u32::MAX);
            if depth == u32::MAX {
                return Err(malformed!("H.4.2: node {index} is unreachable"));
            }
            match node {
                TreeNode::Leaf(_) => max_depth = max_depth.max(depth),
                TreeNode::Decision { left, right, .. } => {
                    let child_depth = depth
                        .checked_add(1)
                        .ok_or_else(|| malformed!("H.4.2: tree depth overflows"))?;
                    if child_depth > limits.max_depth {
                        return Err(malformed!(
                            "H.4.2: MA tree is deeper than the limit of {}",
                            limits.max_depth
                        ));
                    }
                    for child in [*left, *right] {
                        if child <= index {
                            return Err(malformed!(
                                "H.4.2: node {index} points backwards to child {child}"
                            ));
                        }
                        let Some(slot) = depths.get_mut(child) else {
                            return Err(malformed!(
                                "H.4.2: node {index} points to child {child}, past the tree"
                            ));
                        };
                        if *slot != u32::MAX {
                            return Err(malformed!("H.4.2: node {child} has two parents"));
                        }
                        *slot = child_depth;
                    }
                }
            }
        }

        Ok(Self {
            nodes,
            num_leaves,
            depth: max_depth,
        })
    }
}

/// Computes the property vector of Table H.4 for one channel.
///
/// The set of "previous channel" properties is fixed for a whole channel, so
/// it is resolved once when the builder is created and reused for every sample.
#[derive(Debug, Clone)]
pub struct PropertyBuilder {
    channel_index: usize,
    stream_index: i32,
    /// Earlier channels with identical dimensions and shifts, in the H.4.1
    /// iteration order (descending index).
    previous: Vec<usize>,
    values: Vec<i32>,
}

impl PropertyBuilder {
    /// Resolves the property layout for `channel_index` within `channels`.
    ///
    /// # Errors
    ///
    /// [`ModularError::Core`](super::ModularError::Core) if the property
    /// vector would exceed the guard's budget.
    pub fn new(
        channels: &[Channel],
        channel_index: usize,
        stream_index: u32,
        guard: &mut AllocGuard,
    ) -> Result<Self> {
        let Some(current) = channels.get(channel_index) else {
            return Err(malformed!(
                "H.4.1: property setup for channel {channel_index}, which does not exist"
            ));
        };
        let mut previous = Vec::new();
        for j in (0..channel_index).rev() {
            let Some(other) = channels.get(j) else { continue };
            if other.spec() == current.spec() {
                previous.push(j);
            }
        }
        let len = NUM_STATIC_PROPERTIES + 4 * previous.len();
        guard.charge(len as u64 * 4)?;
        Ok(Self {
            channel_index,
            // `stream index` is property 1; the channel-index property is 0.
            // Both are narrowed like every other property value.
            stream_index: narrow_to_i32(i64::from(stream_index)),
            previous,
            values: vec![0; len],
        })
    }

    /// Length of the property vector this builder produces.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the property vector is empty. It never is — it always has the
    /// sixteen static properties — but clippy asks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// `GetProperties(i, x, y)` of H.4.1.
    ///
    /// `nb` is the H.3 neighbourhood of the current sample and `max_error`
    /// is the H.5.1 output for it.
    pub fn compute(
        &mut self,
        channels: &[Channel],
        x: u32,
        y: u32,
        nb: &Neighbours,
        max_error: i32,
    ) -> &[i32] {
        // Property 9 evaluated at (x - 1, y). H.4.1 defines property 8 in terms
        // of it, and it is the only property that looks at another sample's
        // property vector.
        let west_gradient_error = if x > 0 {
            let left = channels
                .get(self.channel_index)
                .map(|c| Neighbours::gather(c, x - 1, y))
                .unwrap_or_default();
            nb.w - (left.w + left.n - left.nw)
        } else {
            nb.w
        };

        let statics = [
            self.channel_index as i64,
            i64::from(self.stream_index),
            i64::from(y),
            i64::from(x),
            nb.n.abs(),
            nb.w.abs(),
            nb.n,
            nb.w,
            west_gradient_error,
            nb.w + nb.n - nb.nw,
            nb.w - nb.nw,
            nb.nw - nb.n,
            nb.n - nb.ne,
            nb.n - nb.nn,
            nb.w - nb.ww,
            i64::from(max_error),
        ];
        for (slot, value) in self.values.iter_mut().zip(statics.iter()) {
            *slot = narrow_to_i32(*value);
        }

        let mut k = NUM_STATIC_PROPERTIES;
        for &j in &self.previous {
            let Some(other) = channels.get(j) else {
                continue;
            };
            // H.4.1's edge rules for the previous-channel neighbours are NOT
            // the H.3 ones: rW is zero at the left edge rather than wrapping to
            // the sample above, and rN/rNW fall back to rW.
            let rc = i64::from(other.get(x, y));
            let rw = if x > 0 {
                i64::from(other.get(x - 1, y))
            } else {
                0
            };
            let rn = if y > 0 {
                i64::from(other.get(x, y - 1))
            } else {
                rw
            };
            let rnw = if x > 0 && y > 0 {
                i64::from(other.get(x - 1, y - 1))
            } else {
                rw
            };
            let rg = Neighbours::gradient(rw, rn, rnw);
            for value in [rc.abs(), rc, (rc - rg).abs(), rc - rg] {
                if let Some(slot) = self.values.get_mut(k) {
                    *slot = narrow_to_i32(value);
                }
                k += 1;
            }
        }

        &self.values
    }
}

#[cfg(test)]
mod tests {
    use jpxl_core::limits::Limits;

    use super::super::channel::ChannelSpec;
    use super::*;

    fn guard() -> AllocGuard {
        AllocGuard::new(&Limits::relaxed())
    }

    fn leaf(ctx: usize) -> TreeNode {
        TreeNode::Leaf(Leaf {
            ctx,
            predictor: Predictor::Zero,
            offset: 0,
            multiplier: 1,
        })
    }

    #[test]
    fn single_leaf_tree_selects_itself_for_any_properties() {
        let tree = MaTree::single_leaf(5, -3, 2).expect("gradient leaf");
        assert_eq!(tree.num_leaves(), 1);
        assert_eq!(tree.depth(), 0);
        let l = tree.traverse(&[]).expect("one leaf");
        assert_eq!(l.predictor, Predictor::Gradient);
        assert_eq!(l.offset, -3);
        assert_eq!(l.multiplier, 2);
        assert_eq!(tree.traverse(&[i32::MIN, 7, 9]).expect("same leaf").ctx, 0);
    }

    #[test]
    fn traversal_takes_left_when_the_property_is_strictly_greater() {
        // Root: property[3] > 10 ? node 1 : node 2.
        let tree = MaTree::finish(
            vec![
                TreeNode::Decision {
                    property: 3,
                    value: 10,
                    left: 1,
                    right: 2,
                },
                leaf(0),
                leaf(1),
            ],
            2,
            TreeLimits::default(),
        )
        .expect("valid three-node tree");

        let mut props = vec![0i32; 4];
        props[3] = 11;
        assert_eq!(tree.traverse(&props).expect("left").ctx, 0);
        props[3] = 10;
        assert_eq!(tree.traverse(&props).expect("not strictly greater").ctx, 1);
        props[3] = 9;
        assert_eq!(tree.traverse(&props).expect("right").ctx, 1);
    }

    #[test]
    fn missing_properties_read_as_zero() {
        // property 99 is never populated; H.4.1 says it reads as zero, so a
        // test of `> -1` is true and `> 0` is false.
        let gt_minus_one = MaTree::finish(
            vec![
                TreeNode::Decision {
                    property: 99,
                    value: -1,
                    left: 1,
                    right: 2,
                },
                leaf(0),
                leaf(1),
            ],
            2,
            TreeLimits::default(),
        )
        .expect("valid");
        assert_eq!(gt_minus_one.traverse(&[1, 2, 3]).expect("left").ctx, 0);

        let gt_zero = MaTree::finish(
            vec![
                TreeNode::Decision {
                    property: 99,
                    value: 0,
                    left: 1,
                    right: 2,
                },
                leaf(0),
                leaf(1),
            ],
            2,
            TreeLimits::default(),
        )
        .expect("valid");
        assert_eq!(gt_zero.traverse(&[1, 2, 3]).expect("right").ctx, 1);
    }

    #[test]
    fn depth_is_computed_from_the_breadth_first_indices() {
        //        0
        //      /   \
        //     1     2        (2 is a leaf)
        //    / \
        //   3   4            (both leaves)
        let tree = MaTree::finish(
            vec![
                TreeNode::Decision {
                    property: 0,
                    value: 0,
                    left: 1,
                    right: 2,
                },
                TreeNode::Decision {
                    property: 1,
                    value: 0,
                    left: 3,
                    right: 4,
                },
                leaf(0),
                leaf(1),
                leaf(2),
            ],
            3,
            TreeLimits::default(),
        )
        .expect("valid five-node tree");
        assert_eq!(tree.depth(), 2);
        assert_eq!(tree.num_leaves(), 3);
    }

    #[test]
    fn shape_violations_are_rejected() {
        // Even node count cannot be a full binary tree.
        assert!(MaTree::finish(vec![leaf(0), leaf(1)], 2, TreeLimits::default()).is_err());
        // Empty.
        assert!(MaTree::finish(vec![], 0, TreeLimits::default()).is_err());
        // Child index past the end.
        assert!(
            MaTree::finish(
                vec![
                    TreeNode::Decision {
                        property: 0,
                        value: 0,
                        left: 1,
                        right: 9,
                    },
                    leaf(0),
                    leaf(1),
                ],
                2,
                TreeLimits::default(),
            )
            .is_err()
        );
        // Two parents for the same child.
        assert!(
            MaTree::finish(
                vec![
                    TreeNode::Decision {
                        property: 0,
                        value: 0,
                        left: 1,
                        right: 1,
                    },
                    leaf(0),
                    leaf(1),
                ],
                2,
                TreeLimits::default(),
            )
            .is_err()
        );
    }

    #[test]
    fn depth_limit_is_enforced() {
        let nodes = vec![
            TreeNode::Decision {
                property: 0,
                value: 0,
                left: 1,
                right: 2,
            },
            leaf(0),
            leaf(1),
        ];
        let limits = TreeLimits {
            max_nodes: 100,
            max_depth: 0,
        };
        let err = MaTree::finish(nodes, 2, limits).expect_err("depth 1 exceeds max_depth 0");
        assert!(err.to_string().contains("deeper than"));
    }

    #[test]
    fn property_layout_counts_matching_previous_channels() {
        let a = Channel::from_samples(ChannelSpec::new(2, 2), vec![0; 4]).expect("2x2");
        let b = Channel::from_samples(ChannelSpec::new(2, 2), vec![0; 4]).expect("2x2");
        // Different shifts, so this one does NOT contribute properties.
        let c = Channel::from_samples(
            ChannelSpec::with_shifts(2, 2, 1, 0),
            vec![0; 4],
        )
        .expect("2x2 shifted");
        let d = Channel::from_samples(ChannelSpec::new(2, 2), vec![0; 4]).expect("2x2");
        let channels = vec![a, b, c, d];

        let pb = PropertyBuilder::new(&channels, 3, 0, &mut guard()).expect("layout");
        // Channels 0 and 1 match; channel 2 has a different hshift.
        assert_eq!(pb.previous, vec![1, 0], "descending index order per H.4.1");
        assert_eq!(pb.len(), NUM_STATIC_PROPERTIES + 8);
    }

    #[test]
    fn static_properties_are_the_table_h4_values() {
        // A 4x3 ramp, same as the predictor tests.
        let ch = Channel::from_samples(ChannelSpec::new(4, 3), (0..12).collect()).expect("4x3");
        let channels = vec![ch];
        let mut pb = PropertyBuilder::new(&channels, 0, 7, &mut guard()).expect("layout");

        let x = 2;
        let y = 2;
        let nb = Neighbours::gather(&channels[0], x, y);
        // From the predictor tests: W = 9, N = 6, NW = 5, NE = 7, NN = 2,
        // NEE = 7, WW = 8.
        let props = pb.compute(&channels, x, y, &nb, -42);
        assert_eq!(props[0], 0, "channel index");
        assert_eq!(props[1], 7, "stream index");
        assert_eq!(props[2], 2, "y");
        assert_eq!(props[3], 2, "x");
        assert_eq!(props[4], 6, "abs(N)");
        assert_eq!(props[5], 9, "abs(W)");
        assert_eq!(props[6], 6, "N");
        assert_eq!(props[7], 9, "W");
        // Property 9 at (1, 2): W' = c(0,2) = 8, N' = c(1,1) = 5,
        // NW' = c(0,1) = 4, so property 9 there is 8 + 5 - 4 = 9.
        // Property 8 = W - 9 = 9 - 9 = 0.
        assert_eq!(props[8], 0, "W - property9(x-1, y)");
        assert_eq!(props[9], 9 + 6 - 5, "W + N - NW");
        assert_eq!(props[10], 9 - 5, "W - NW");
        assert_eq!(props[11], 5 - 6, "NW - N");
        assert_eq!(props[12], 6 - 7, "N - NE");
        assert_eq!(props[13], 6 - 2, "N - NN");
        assert_eq!(props[14], 9 - 8, "W - WW");
        assert_eq!(props[15], -42, "max_error");
    }

    #[test]
    fn property_eight_falls_back_to_w_in_the_first_column() {
        let ch = Channel::from_samples(ChannelSpec::new(4, 3), (0..12).collect()).expect("4x3");
        let channels = vec![ch];
        let mut pb = PropertyBuilder::new(&channels, 0, 0, &mut guard()).expect("layout");
        let nb = Neighbours::gather(&channels[0], 0, 1);
        let props = pb.compute(&channels, 0, 1, &nb, 0);
        assert_eq!(props[8], props[7], "x == 0 -> property 8 is just W");
    }

    #[test]
    fn previous_channel_properties_use_their_own_edge_rules() {
        // Channel 0 holds 1..=4; channel 1 is the one being decoded.
        let prev =
            Channel::from_samples(ChannelSpec::new(2, 2), vec![1, 2, 3, 4]).expect("2x2");
        let cur = Channel::from_samples(ChannelSpec::new(2, 2), vec![0; 4]).expect("2x2");
        let channels = vec![prev, cur];
        let mut pb = PropertyBuilder::new(&channels, 1, 0, &mut guard()).expect("layout");

        // At (0, 1) in channel 0: rC = 3, rW = 0 (x == 0 gives zero, NOT the
        // sample above as H.3 would), rN = c(0,0) = 1, rNW = rW = 0.
        // rG = clamp(0 + 1 - 0, min(0,1), max(0,1)) = clamp(1, 0, 1) = 1.
        // So the four properties are abs(3)=3, 3, abs(3-1)=2, 3-1=2.
        let nb = Neighbours::gather(&channels[1], 0, 1);
        let props = pb.compute(&channels, 0, 1, &nb, 0);
        assert_eq!(
            &props[NUM_STATIC_PROPERTIES..NUM_STATIC_PROPERTIES + 4],
            &[3, 3, 2, 2]
        );
    }
}
