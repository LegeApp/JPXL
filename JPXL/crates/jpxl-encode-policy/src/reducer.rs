//! The terminal coefficient reducer: spends a finalist's score reserve on
//! bytes by deleting the *last* nonzero HF coefficient of ranked varblock
//! channels, verified by the canonical score after every batch.
//!
//! # Why the last nonzero
//!
//! 18181-1 I.4 codes a channel's coefficients from the first HF order position
//! up to and including the last nonzero, then stops. Zeroing the last nonzero
//! therefore frees its own token *and* every interior zero token back to the
//! previous nonzero, while zeroing anything earlier frees nothing. The
//! quantizer's trailing-truncation pass (`quantize.rs`) already exploits this
//! with a crude bit-length rate proxy at quantization time; this pass runs
//! once more at the end, on the finalist, with the finalist's own trained
//! histograms pricing every token exactly, and with the full perceptual score
//! as the gate instead of a Lagrangian.
//!
//! # Contract
//!
//! * Every accepted batch is scored by the injected [`PerceptualEvaluator`]
//!   on the rendered candidate — the same canonical scorer the navigator
//!   uses — so the reducer can never emit a plan below `threshold`.
//! * A rejected batch is rolled back and halved; the work is bounded by
//!   [`ReducerLimits`] and reported in [`ReducerStats`].
//! * The pass is deterministic: edits are ranked by a fixed-point key and
//!   tie-broken by position, and every arithmetic path is sequential.
//!
//! The entropy model is *not* retrained between batches: the prices are the
//! finalist's, which is what the ranking needs. The caller retrains and
//! exact-prices the reduced plan once at the end (`attach_entropy`), and keeps
//! it only if the exact stream is smaller.

use std::sync::Arc;

use jpxl_core::varblock::TransformType;
use jpxl_encode::vardct::ids::PreContextId;
use jpxl_encode::vardct::plan::{
    EntropyPlan, NUM_CHANNELS, PixelPlan, QuantizedFrameIr, VarblockCoefficients,
};
use jpxl_encode::vardct::walk::CHANNEL_WALK_ORDER;
use jpxl_encode::vardct::{
    HfEventSink, OrderTables, PassGroupWalk, ValidatedPixelPlan, VardctGeometry, WalkVarblock,
    validate_pixels, walk_pass_group,
};

use crate::entropy_cost::EntropyCostView;
use crate::error::{PolicyError, Result};
use crate::quality::PerceptualEvaluator;
use crate::quantize::HfQuantizer;

/// Work bounds for one reducer run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReducerLimits {
    /// Maximum canonical score evaluations (each is one render + metric).
    pub max_evaluations: u32,
    /// Maximum accepted batches.
    pub max_rounds: u32,
    /// Edits in the first batch, or — when `batch_fraction` is positive —
    /// the floor of a first batch sized as that fraction of the candidates.
    pub initial_batch: usize,
    /// Fraction (0..=1) of the ranked candidates the first batch takes when
    /// positive; `0.0` uses `initial_batch` alone. A single-evaluation budget
    /// wants one batch sized to the frame, not a fixed count.
    pub batch_fraction: f64,
    /// A rejected batch is halved down to this size, then the pass stops.
    pub min_batch: usize,
    /// An accepted batch doubles the next one, up to this size.
    pub max_batch: usize,
    /// Edits whose estimated `bits saved / loss` key falls below this
    /// fraction of the best key in the same round are not attempted.
    pub key_floor: f64,
}

impl ReducerLimits {
    /// The bounded single-pass variant a production effort may use.
    pub const BALANCED: Self = Self {
        max_evaluations: 1,
        max_rounds: 1,
        initial_batch: 256,
        batch_fraction: 0.5,
        min_batch: 64,
        max_batch: 16_384,
        key_floor: 0.05,
    };

    /// The feature-gated Quality effort's variant.
    pub const QUALITY: Self = Self {
        max_evaluations: 6,
        max_rounds: 4,
        initial_batch: 512,
        batch_fraction: 0.0,
        min_batch: 32,
        max_batch: 8192,
        key_floor: 0.02,
    };
}

/// What one reducer run did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReducerStats {
    /// Canonical evaluations spent.
    pub evaluations: u32,
    /// Accepted batches.
    pub rounds: u32,
    /// Coefficients zeroed in accepted batches.
    pub edits_applied: u32,
    /// Coefficients zeroed in batches that were rolled back.
    pub edits_rolled_back: u32,
    /// Estimated Q8 bits the accepted edits freed under the finalist's model.
    pub estimated_bits_saved_q8: u64,
}

/// One legal terminal removal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerminalEdit {
    /// LF group index (raster order).
    pub lf_group: usize,
    /// Varblock index inside the LF group's coefficient list.
    pub varblock: usize,
    /// Channel in Table I.1 order (`0 = X`, `1 = Y`, `2 = B`).
    pub channel: usize,
    /// Coefficient cell to zero.
    pub cell: usize,
    /// The value being removed.
    pub value: i32,
    /// Q8 bits freed under the finalist's model: the coefficient's token, the
    /// interior zero tokens it exposes, and the `non_zeros` token change.
    pub bits_saved_q8: i64,
    /// Squared reconstructed magnitude of the removed coefficient (XYB units).
    pub est_loss: f32,
}

impl TerminalEdit {
    /// The ranking key: freed bits per unit of estimated loss. Fixed-point so
    /// the order is identical on every platform.
    fn key(&self) -> u64 {
        if self.bits_saved_q8 <= 0 {
            return 0;
        }
        let bits = u64::try_from(self.bits_saved_q8).unwrap_or(0);
        // Loss in a fixed 2^-24 grid, floored at one unit so a coefficient that
        // reconstructs to ~0 ranks first rather than dividing by zero.
        let loss = f64::from(self.est_loss.max(0.0)) * f64::from(1u32 << 24);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "loss is non-negative and capped well below u64::MAX by the plan's value range"
        )]
        let loss_units = (loss.min(1.8e19)) as u64;
        bits.saturating_mul(1u64 << 20) / loss_units.max(1)
    }
}

/// The reduced finalist.
#[derive(Debug)]
pub struct ReducedPlan {
    /// The edited plan, validated.
    pub pixels: ValidatedPixelPlan,
    /// Its canonical score (the last accepted evaluation).
    pub score: f64,
    /// The work done.
    pub stats: ReducerStats,
}

/// Runs the reducer on a finalist that already meets `threshold`.
///
/// `entropy` is the finalist's trained entropy stage, which prices every
/// token; `evaluator` is the canonical scorer. Returns `None` when no edit was
/// accepted (the caller keeps the finalist as it was).
///
/// # Errors
///
/// Anything the walk, the cost view, validation or the evaluator refuses.
pub fn reduce_terminal(
    finalist: &ValidatedPixelPlan,
    geometry: &VardctGeometry,
    entropy: &EntropyPlan,
    evaluator: &mut dyn PerceptualEvaluator,
    threshold: f64,
    limits: ReducerLimits,
) -> Result<Option<ReducedPlan>> {
    let pass = entropy.passes.first().ok_or(PolicyError::Unsupported {
        what: "a finalist with no entropy pass",
    })?;
    let orders = OrderTables::from_order_set(&pass.orders);
    let view = EntropyCostView::from_model(&pass.distributions)?;
    let mut quantizers = QuantizerCache::default();

    let mut stats = ReducerStats::default();
    let mut accepted: Option<(ValidatedPixelPlan, f64)> = None;
    let mut batch = limits.initial_batch.max(1);

    while stats.rounds < limits.max_rounds && stats.evaluations < limits.max_evaluations {
        let current = accepted.as_ref().map_or(finalist, |(p, _)| p);
        let mut edits =
            enumerate_terminal_edits(current, geometry, entropy, &orders, &view, &mut quantizers)?;
        if edits.is_empty() {
            break;
        }
        // Best key first, then by position so equal keys are ordered the same
        // way on every run.
        edits.sort_by(|a, b| {
            b.key()
                .cmp(&a.key())
                .then((a.lf_group, a.varblock, a.channel).cmp(&(b.lf_group, b.varblock, b.channel)))
        });
        let best_key = edits.first().map_or(0, TerminalEdit::key);
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a ranking key scaled by a fraction in 0..=1; precision beyond 2^53 is irrelevant"
        )]
        let floor = (best_key as f64 * limits.key_floor.clamp(0.0, 1.0)) as u64;
        edits.retain(|e| e.bits_saved_q8 > 0 && e.key() >= floor);
        if edits.is_empty() {
            break;
        }
        if stats.rounds == 0 && limits.batch_fraction > 0.0 {
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a candidate count scaled by a fraction in 0..=1"
            )]
            let sized = (edits.len() as f64 * limits.batch_fraction.clamp(0.0, 1.0)) as usize;
            batch = sized
                .max(limits.initial_batch)
                .min(limits.max_batch.max(1))
                .max(1);
        }

        // Inner loop: try the batch, halve on rejection.
        let mut applied_this_round = false;
        while stats.evaluations < limits.max_evaluations {
            let chosen: Vec<TerminalEdit> = edits.iter().copied().take(batch).collect();
            if chosen.is_empty() {
                break;
            }
            let candidate = apply_edits(current, &chosen)?;
            let score = evaluator.evaluate(&candidate)?.score;
            stats.evaluations = stats.evaluations.saturating_add(1);
            let count = u32::try_from(chosen.len()).unwrap_or(u32::MAX);
            if score >= threshold {
                stats.rounds = stats.rounds.saturating_add(1);
                stats.edits_applied = stats.edits_applied.saturating_add(count);
                stats.estimated_bits_saved_q8 =
                    chosen.iter().fold(stats.estimated_bits_saved_q8, |acc, e| {
                        acc.saturating_add(u64::try_from(e.bits_saved_q8).unwrap_or(0))
                    });
                accepted = Some((candidate, score));
                applied_this_round = true;
                // The score moved little: try a larger batch next round.
                batch = batch.saturating_mul(2).min(limits.max_batch.max(1));
                break;
            }
            stats.edits_rolled_back = stats.edits_rolled_back.saturating_add(count);
            if batch <= limits.min_batch.max(1) {
                break;
            }
            batch = (batch / 2).max(limits.min_batch.max(1));
        }
        if !applied_this_round {
            break;
        }
    }

    Ok(accepted.map(|(pixels, score)| ReducedPlan {
        pixels,
        score,
        stats,
    }))
}

/// Enumerates every legal terminal removal of `pixels` under `entropy`'s
/// prices: one candidate per varblock channel with at least one coded
/// nonzero.
///
/// # Errors
///
/// As [`walk_pass_group`] and the cost view.
pub(crate) fn enumerate_terminal_edits(
    pixels: &ValidatedPixelPlan,
    geometry: &VardctGeometry,
    entropy: &EntropyPlan,
    orders: &OrderTables,
    view: &EntropyCostView,
    quantizers: &mut QuantizerCache,
) -> Result<Vec<TerminalEdit>> {
    let plan = pixels.plan();
    let quantizer = &plan.spatial.quantizer;
    let global_scale = quantizer.global_scale.get();
    let (x_qm, b_qm) = (quantizer.x_qm_scale.get(), quantizer.b_qm_scale.get());
    let pass = entropy.passes.first().ok_or(PolicyError::Unsupported {
        what: "a finalist with no entropy pass",
    })?;
    let nb_block_ctx = entropy.block_context.nb_block_ctx();

    let mut edits = Vec::new();
    for group in 0..geometry.num_groups() {
        let Some((walk, varblocks, identities)) =
            pass_group_walk(plan, geometry, entropy, orders, pass, nb_block_ctx, group)?
        else {
            continue;
        };
        let mut sink = RecordingSink::default();
        walk_pass_group(&walk, &varblocks, &mut sink).map_err(|_| PolicyError::Unsupported {
            what: "a finalist whose coefficient walk the reducer could not replay",
        })?;
        if sink.varblocks.len() != varblocks.len() {
            return Err(PolicyError::Unsupported {
                what: "a coefficient walk that visited a different varblock count than planned",
            });
        }
        for ((record, vb), identity) in sink.varblocks.iter().zip(&varblocks).zip(&identities) {
            let transform = vb.transform;
            let num_blocks = transform.num_blocks();
            let order_id = transform.order_id();
            for (slot, &channel) in CHANNEL_WALK_ORDER.iter().enumerate() {
                let Some(ch) = record.channels.get(slot) else {
                    continue;
                };
                let Some(edit) = terminal_edit_for_channel(
                    ch,
                    vb,
                    channel,
                    num_blocks,
                    orders.order(order_id, channel),
                    view,
                    quantizers.get(transform, global_scale, vb.hf_mul, x_qm, b_qm)?,
                    *identity,
                )?
                else {
                    continue;
                };
                edits.push(edit);
            }
        }
    }
    Ok(edits)
}

/// Prices the removal of one channel's last nonzero.
#[allow(
    clippy::too_many_arguments,
    reason = "one walk record plus the tables it is priced against"
)]
fn terminal_edit_for_channel(
    ch: &ChannelRecord,
    vb: &WalkVarblock<'_>,
    channel: usize,
    num_blocks: usize,
    order: Option<&[u32]>,
    view: &EntropyCostView,
    quantizer: &HfQuantizer,
    identity: (usize, usize),
) -> Result<Option<TerminalEdit>> {
    let Some((nz_ctx, non_zeros)) = ch.non_zeros else {
        return Ok(None);
    };
    if non_zeros == 0 || ch.tokens.is_empty() {
        return Ok(None);
    }
    let Some(order) = order else {
        return Ok(None);
    };
    // The walk stops at the last nonzero, so the final token is it.
    let last_index = ch.tokens.len() - 1;
    let Some(&(last_ctx, packed)) = ch.tokens.get(last_index) else {
        return Ok(None);
    };
    let k = num_blocks + last_index;
    let Some(&cell_u32) = order.get(k) else {
        return Ok(None);
    };
    let cell = usize::try_from(cell_u32).unwrap_or(usize::MAX);
    let Some(coefficients) = vb.coefficients.channel(channel) else {
        return Ok(None);
    };
    let Some(&value) = coefficients.get(cell) else {
        return Ok(None);
    };
    if value == 0 || packed == 0 {
        return Err(PolicyError::Unsupported {
            what: "a coefficient walk whose final token is not a nonzero",
        });
    }

    let mut bits = i64::from(view.cost_q8(last_ctx, packed)?);
    // Interior zeros exposed by the removal: walk back to the previous nonzero.
    let mut i = last_index;
    while i > 0 {
        i -= 1;
        let Some(&(ctx, v)) = ch.tokens.get(i) else {
            break;
        };
        if v != 0 {
            break;
        }
        bits = bits.saturating_add(i64::from(view.cost_q8(ctx, 0)?));
    }
    // The non_zeros token changes from n to n-1.
    let before = i64::from(view.cost_q8(nz_ctx, non_zeros)?);
    let after = i64::from(view.cost_q8(nz_ctx, non_zeros - 1)?);
    bits = bits.saturating_add(before - after);

    let recon = quantizer.reconstruct(value, channel, cell);
    Ok(Some(TerminalEdit {
        lf_group: identity.0,
        varblock: identity.1,
        channel,
        cell,
        value,
        bits_saved_q8: bits,
        est_loss: recon * recon,
    }))
}

/// Applies `edits` to a copy of the plan's integers and validates the result.
///
/// Coefficient storage shared through arenas stays shared for every untouched
/// varblock; only edited varblocks are copied.
///
/// # Errors
///
/// [`PolicyError::Unsupported`] if an edit addresses a varblock the plan does
/// not have, plus anything validation refuses.
pub(crate) fn apply_edits(
    pixels: &ValidatedPixelPlan,
    edits: &[TerminalEdit],
) -> Result<ValidatedPixelPlan> {
    let plan = pixels.plan();
    let mut ir: QuantizedFrameIr = (*plan.quantized).clone();
    for edit in edits {
        let lf_group = ir
            .lf_groups
            .get_mut(edit.lf_group)
            .ok_or(PolicyError::Unsupported {
                what: "a terminal edit outside the plan's LF groups",
            })?;
        let transform = plan
            .spatial
            .lf_groups
            .get(edit.lf_group)
            .and_then(|g| g.blocks.get(edit.varblock))
            .map(|b| b.transform)
            .ok_or(PolicyError::Unsupported {
                what: "a terminal edit outside the plan's varblocks",
            })?;
        let slot =
            lf_group
                .coefficients
                .get_mut(edit.varblock)
                .ok_or(PolicyError::Unsupported {
                    what: "a terminal edit outside the plan's coefficients",
                })?;
        let mut channels: [Vec<i32>; NUM_CHANNELS] =
            core::array::from_fn(|c| slot.channel(c).map(<[i32]>::to_vec).unwrap_or_default());
        let target = channels
            .get_mut(edit.channel)
            .and_then(|ch| ch.get_mut(edit.cell))
            .ok_or(PolicyError::Unsupported {
                what: "a terminal edit outside its varblock's coefficients",
            })?;
        *target = 0;
        *slot = VarblockCoefficients::new(transform, channels).map_err(|_| {
            PolicyError::Unsupported {
                what: "an edited varblock whose coefficient count changed",
            }
        })?;
    }
    let edited = PixelPlan::from_shared(Arc::clone(&plan.spatial), Arc::new(ir));
    validate_pixels(edited).map_err(|_| PolicyError::Unsupported {
        what: "an edited plan that no longer validates",
    })
}

/// One pass group's walk: the tables, its varblocks in walk order, and each
/// varblock's `(lf_group, varblock)` identity in the plan.
type GroupWalk<'a> = (
    PassGroupWalk<'a>,
    Vec<WalkVarblock<'a>>,
    Vec<(usize, usize)>,
);

/// One LF group's pass-group walk, mirroring the writer's `pass_group_walk`
/// so the reducer replays exactly the events the stream will carry.
///
/// Returns `None` for a group with no varblocks.
#[allow(
    clippy::too_many_arguments,
    reason = "the walk needs every table the writer builds it from"
)]
fn pass_group_walk<'a>(
    plan: &'a PixelPlan,
    geometry: &VardctGeometry,
    entropy: &'a EntropyPlan,
    orders: &'a OrderTables,
    pass: &'a jpxl_encode::vardct::plan::HfPassEntropyPlan,
    nb_block_ctx: u64,
    group: u64,
) -> Result<Option<GroupWalk<'a>>> {
    let unsupported = |what: &'static str| PolicyError::Unsupported { what };
    let rect = geometry
        .group_rect(group)
        .ok_or_else(|| unsupported("a group index past the grid"))?;
    let id = geometry
        .lf_group_of(group)
        .ok_or_else(|| unsupported("a group outside every LF group"))?;
    let lf_rect = geometry
        .lf_group_rect(id)
        .ok_or_else(|| unsupported("an LF group past the grid"))?;
    let lf_grid = geometry
        .lf_group_blocks(id)
        .ok_or_else(|| unsupported("an LF group past the grid"))?;
    let index = usize::try_from(id.index()).unwrap_or(usize::MAX);
    let spatial = plan
        .spatial
        .lf_groups
        .get(index)
        .ok_or_else(|| unsupported("an LF group past the plan"))?;
    let quantized = plan
        .quantized
        .lf_groups
        .get(index)
        .ok_or_else(|| unsupported("an LF group past the plan's integers"))?;

    let origin_bx = (rect.x0 - lf_rect.x0) / 8;
    let origin_by = (rect.y0 - lf_rect.y0) / 8;
    let blocks_w = rect.width.div_ceil(8);
    let blocks_h = rect.height.div_ceil(8);

    let first = spatial
        .blocks
        .partition_point(|block| block.origin.by() < origin_by);
    let mut varblocks = Vec::new();
    let mut identities = Vec::new();
    for (i, block) in spatial.blocks.iter().enumerate().skip(first) {
        let (bx, by) = (block.origin.bx(), block.origin.by());
        if by >= origin_by + blocks_h {
            break;
        }
        if bx < origin_bx || by < origin_by {
            continue;
        }
        let (lx, ly) = (bx - origin_bx, by - origin_by);
        if lx >= blocks_w || ly >= blocks_h {
            continue;
        }
        let cell = usize::try_from(u64::from(by) * u64::from(lf_grid.width) + u64::from(bx))
            .unwrap_or(usize::MAX);
        let qdc = core::array::from_fn(|c| {
            quantized
                .lf
                .plane(c)
                .and_then(|p| p.get(cell))
                .copied()
                .unwrap_or(0)
        });
        let coefficients = quantized
            .coefficients
            .get(i)
            .ok_or_else(|| unsupported("a varblock with no coefficients"))?;
        varblocks.push(WalkVarblock {
            bx: lx,
            by: ly,
            transform: block.transform,
            hf_mul: block.hf_mul.get(),
            qdc,
            coefficients,
        });
        identities.push((index, i));
    }
    if varblocks.is_empty() {
        return Ok(None);
    }

    let group_index = usize::try_from(group).unwrap_or(usize::MAX);
    let hfp = u64::from(
        pass.group_presets
            .get(group_index)
            .ok_or_else(|| unsupported("a group past the preset table"))?
            .get(),
    );
    let offset = 495u64.saturating_mul(nb_block_ctx).saturating_mul(hfp);
    Ok(Some((
        PassGroupWalk {
            blocks_w,
            blocks_h,
            block_context: &entropy.block_context,
            orders,
            offset,
        },
        varblocks,
        identities,
    )))
}

/// The events of one varblock channel, in walk order.
#[derive(Debug, Default, Clone)]
struct ChannelRecord {
    non_zeros: Option<(PreContextId, u32)>,
    /// `(context, PackSigned value)` per coded coefficient.
    tokens: Vec<(PreContextId, u32)>,
}

#[derive(Debug, Default, Clone)]
struct VarblockRecord {
    /// In `CHANNEL_WALK_ORDER` slots (Y, X, B).
    channels: Vec<ChannelRecord>,
}

/// Records the walk so each token can be priced against its context.
#[derive(Debug, Default)]
struct RecordingSink {
    varblocks: Vec<VarblockRecord>,
}

impl HfEventSink for RecordingSink {
    fn varblock(&mut self, _transform: TransformType, _hf_mul: u32) {
        self.varblocks.push(VarblockRecord::default());
    }

    fn nonzeros(&mut self, context: PreContextId, value: u32) {
        if let Some(vb) = self.varblocks.last_mut() {
            vb.channels.push(ChannelRecord {
                non_zeros: Some((context, value)),
                tokens: Vec::new(),
            });
        }
    }

    fn coefficient(&mut self, context: PreContextId, value: u32) {
        if let Some(ch) = self
            .varblocks
            .last_mut()
            .and_then(|vb| vb.channels.last_mut())
        {
            ch.tokens.push((context, value));
        }
    }
}

/// `HfQuantizer`s by `(transform, HfMul)`, built on first use.
#[derive(Default)]
pub(crate) struct QuantizerCache {
    entries: Vec<(TransformType, u32, HfQuantizer)>,
}

impl QuantizerCache {
    fn get(
        &mut self,
        transform: TransformType,
        global_scale: u32,
        hf_mul: u32,
        x_qm: u32,
        b_qm: u32,
    ) -> Result<&HfQuantizer> {
        let position = self
            .entries
            .iter()
            .position(|(t, m, _)| *t == transform && *m == hf_mul);
        let index = match position {
            Some(i) => i,
            None => {
                let q = HfQuantizer::new(transform, global_scale, hf_mul, x_qm, b_qm)?;
                self.entries.push((transform, hf_mul, q));
                self.entries.len() - 1
            }
        };
        self.entries
            .get(index)
            .map(|(_, _, q)| q)
            .ok_or(PolicyError::Unsupported {
                what: "a quantizer cache slot that vanished",
            })
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn edit(bits: i64, loss: f32, pos: (usize, usize, usize)) -> TerminalEdit {
        TerminalEdit {
            lf_group: pos.0,
            varblock: pos.1,
            channel: pos.2,
            cell: 1,
            value: 1,
            bits_saved_q8: bits,
            est_loss: loss,
        }
    }

    #[test]
    fn the_key_prefers_more_bits_per_unit_loss_and_never_divides_by_zero() {
        let cheap = edit(256, 1.0, (0, 0, 1));
        let dear = edit(256, 4.0, (0, 0, 1));
        let free = edit(256, 0.0, (0, 0, 1));
        let useless = edit(0, 1.0, (0, 0, 1));
        assert!(cheap.key() > dear.key());
        assert!(free.key() > cheap.key());
        assert_eq!(useless.key(), 0);
        assert_eq!(edit(-5, 1.0, (0, 0, 1)).key(), 0);
    }

    #[test]
    fn the_recording_sink_groups_tokens_under_their_channel() {
        let mut sink = RecordingSink::default();
        sink.varblock(TransformType::Dct8x8, 1);
        sink.nonzeros(PreContextId::new(3), 2);
        sink.coefficient(PreContextId::new(4), 0);
        sink.coefficient(PreContextId::new(5), 3);
        sink.nonzeros(PreContextId::new(6), 0);
        sink.nonzeros(PreContextId::new(7), 1);
        sink.coefficient(PreContextId::new(8), 1);
        assert_eq!(sink.varblocks.len(), 1);
        let vb = &sink.varblocks[0];
        assert_eq!(vb.channels.len(), 3);
        assert_eq!(vb.channels[0].tokens.len(), 2);
        assert!(vb.channels[1].tokens.is_empty());
        assert_eq!(vb.channels[2].tokens, vec![(PreContextId::new(8), 1)]);
    }
}
