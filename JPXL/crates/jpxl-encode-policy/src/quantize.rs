//! Quantization against the exact decoder function (`Encoder-plan1.md` §7.3).
//!
//! An encoder that quantizes by dividing and rounding is guessing. The
//! decoder's reconstruction is a specific, non-linear, per-channel function of
//! the stored integer — I.5.2 for LF, I.5.3 for HF, with L.2.1's `quant_bias`
//! bending the small values — and the only way to know which integer is best is
//! to evaluate that function. So [`HfQuantizer::choose`] builds the candidate
//! integers arithmetically and then **scores them through the reconstruction**,
//! keeping the one whose reconstructed value is closest to the target.
//!
//! # The reconstruction, restated
//!
//! For a coefficient at cell `(x, y)` of channel `c` in a varblock with
//! `HfMul`:
//!
//! ```text
//! adj   = |q| <= 1 ? q * quant_bias[c] : q - quant_bias_numerator / q   (I.5.3)
//! Mul   = (1 << 16) / (global_scale * HfMul)                            (I.5.3)
//! qm    = pow(0.8, x_qm_scale - 2)   for X, ...b_qm_scale... for B      (I.5.3)
//! d     = adj * Mul * qm * dequant_matrix[c](x, y)                      (I.2.4)
//! ```
//!
//! and for an LF sample:
//!
//! ```text
//! mDC   = (1 << 16) * m_lf_unscaled[c] / (global_scale * quant_lf)      (I.2.1)
//! d     = mDC * q / (1 << extra_precision)                              (I.5.2)
//! ```
//!
//! The LF branch is linear and its inverse is exact; the HF branch is not, and
//! that is the whole reason this module exists rather than a division.
//!
//! # Chroma from luma is not optional
//!
//! I.6 runs on **every** dequantized coefficient, and its default parameters
//! are not neutral: `base_correlation_b` is `1.0`, so a decoder reconstructs
//! `B = dB + 1.0 * dY` whatever this encoder intends. The signalled factors
//! `XFromY`, `BFromY`, `x_factor_lf` and `b_factor_lf` are the *searchable*
//! part. Slice 15 estimates them, but the rule is unchanged: the resulting
//! whole `kB` still has to be subtracted on the way in, from the
//! **reconstructed** `dY`, not from source Y. Getting this wrong does not
//! produce a subtle error: it doubles the blue-yellow axis.
//!
//! `base_correlation_x` is `0.0`, so `kX` really is neutral and X passes
//! through.

use jpxl_core::dequant::{DequantMatrices, DequantMatrix};
use jpxl_core::varblock::TransformType;

use crate::error::{PolicyError, Result};

/// Number of coefficient channels.
pub const NUM_CHANNELS: usize = 3;

/// Cells in a DCT8x8 coefficient array.
pub const DCT8X8_CELLS: usize = 64;

/// I.5.3's `(1 << 16)` numerator, shared by the LF and HF multipliers.
const QUANT_NUMERATOR: f64 = 65536.0;

/// G.1.2's `/ 128`, applied to `m_x_lf`, `m_y_lf` and `m_b_lf`.
const LF_WEIGHT_SCALE: f64 = 128.0;

/// The G.1.2 default LF dequantization weights, before the `/ 128`.
const DEFAULT_LF_WEIGHTS: [f64; NUM_CHANNELS] = [1.0 / 32.0, 1.0 / 4.0, 1.0 / 2.0];

/// I.2.3's `base_correlation_x` default: chroma-from-luma leaves X alone.
const BASE_CORRELATION_X: f32 = 0.0;

/// I.2.3's `base_correlation_b` default: the decoder adds a whole `dY` to `dB`.
const BASE_CORRELATION_B: f32 = 1.0;

/// The largest magnitude a quantized coefficient may reach.
///
/// Not a clause limit — G.2.4 stores the coefficient as a modular sample, so
/// any `i32` is representable — but a working bound. A target that needs more
/// than this means the quantizer was handed a scale it cannot serve, and a
/// silent clamp there would show up as a saturated block nobody can explain.
const MAX_QUANT: i32 = 1 << 20;

/// I.5.2's LF quantizer: linear, and exactly invertible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LfQuantizer {
    m_dc: [f32; NUM_CHANNELS],
    extra_precision: u8,
}

impl LfQuantizer {
    /// Builds the quantizer for a frame's `global_scale` and `quant_lf`, with
    /// the G.1.2 default channel weights.
    #[must_use]
    pub fn new(global_scale: u32, quant_lf: u32, extra_precision: u8) -> Self {
        let denom = f64::from(global_scale) * f64::from(quant_lf);
        let m_dc = core::array::from_fn(|c| {
            let w = DEFAULT_LF_WEIGHTS.get(c).copied().unwrap_or(0.0) / LF_WEIGHT_SCALE;
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the single deliberate f64 -> f32 narrowing, mirroring \
                          the decoder's own lf_multipliers"
            )]
            if denom > 0.0 {
                (QUANT_NUMERATOR * w / denom) as f32
            } else {
                0.0
            }
        });
        Self {
            m_dc,
            extra_precision,
        }
    }

    /// I.5.2's `d = mDC * q / (1 << extra_precision)`.
    #[must_use]
    pub fn reconstruct(&self, q: i32, channel: usize) -> f32 {
        let m = self.m_dc.get(channel).copied().unwrap_or(0.0);
        #[allow(
            clippy::cast_precision_loss,
            reason = "|q| stays far inside f32's exact integer range; MAX_QUANT \
                      is 2^20"
        )]
        let q = q as f32;
        m * q / (1u32 << self.extra_precision) as f32
    }

    /// The integer whose reconstruction is nearest `target`.
    ///
    /// # Errors
    ///
    /// [`PolicyError::Unsupported`] if the quantizer is degenerate or the
    /// target needs an integer outside [`MAX_QUANT`].
    pub fn quantize(&self, target: f32, channel: usize) -> Result<i32> {
        let m = self.m_dc.get(channel).copied().unwrap_or(0.0);
        if !(m.is_finite() && m > 0.0) {
            return Err(PolicyError::Unsupported {
                what: "a degenerate LF dequantization multiplier",
            });
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "extra_precision is at most 3, so the shift is exact"
        )]
        let scaled = target * (1u32 << self.extra_precision) as f32 / m;
        clamp_round(scaled)
    }
}

/// I.5.3's HF quantizer for one transform type, over all three channels.
#[derive(Debug, Clone)]
pub struct HfQuantizer {
    quant_bias: [f32; NUM_CHANNELS],
    quant_bias_numerator: f32,
    /// Flattened `Mul * qm * matrix[c](x,y)` in coefficient order, per channel.
    /// Phase-1: avoid matrix lookups and multiplies on every [`Self::choose`].
    steps: [Box<[f32]>; NUM_CHANNELS],
    /// `0.5 * quant_bias[c] * steps[c][cell]` — zero wins under the nearest-
    /// reconstruction + lower-magnitude tie rule when `|target| <=` this.
    zero_threshold: [Box<[f32]>; NUM_CHANNELS],
    /// Phase 7.0: per-`(channel, cell)` Lagrange weight for the rate-aware
    /// choice — exactly the coefficient `block_cost_bounded` multiplies squared
    /// coefficient error by, i.e. `lambda[channel] * side^2 * size_penalty *
    /// frequency_weight[cell]`.
    ///
    /// `None` is the shipped nearest-reconstruction rule, and the branch on it
    /// is what keeps that path bit-identical.
    ///
    /// Phase 7.1 also needs this weight for its trailing-truncation decision
    /// *without* changing how `choose` picks, so the table's presence and its
    /// use by `choose` are separate: see [`Self::rd_choose`].
    rd: Option<[Box<[f32]>; NUM_CHANNELS]>,
    /// Whether [`Self::choose`] itself minimises the rate-distortion cost.
    /// Phase 7.0's mode; off by default and off under Phase 7.1, which wants
    /// the weight but keeps the nearest per-coefficient rule.
    rd_choose: bool,
}

impl HfQuantizer {
    /// Builds the quantizer for one transform, `global_scale` and `HfMul`.
    ///
    /// `qm_scale` is `[x_qm_scale, y_qm_scale (unused), b_qm_scale]` as the
    /// frame header signals them; the Y channel has no such factor.
    ///
    /// # Errors
    ///
    /// [`PolicyError::Unsupported`] if the I.2.5 default matrices cannot be
    /// built, which would mean the shared table is broken.
    pub fn new(
        transform: TransformType,
        global_scale: u32,
        hf_mul: u32,
        x_qm_scale: u32,
        b_qm_scale: u32,
    ) -> Result<Self> {
        let defaults = DequantMatrices::all_default().map_err(|_| PolicyError::Unsupported {
            what: "the I.2.5 default dequantization matrices",
        })?;
        let matrices: [DequantMatrix; NUM_CHANNELS] = [
            defaults
                .for_transform(transform, 0)
                .map_err(|_| PolicyError::Unsupported {
                    what: "a dequantization matrix for this transform",
                })?,
            defaults
                .for_transform(transform, 1)
                .map_err(|_| PolicyError::Unsupported {
                    what: "a dequantization matrix for this transform",
                })?,
            defaults
                .for_transform(transform, 2)
                .map_err(|_| PolicyError::Unsupported {
                    what: "a dequantization matrix for this transform",
                })?,
        ];

        let denom = f64::from(global_scale) * f64::from(hf_mul);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the deliberate f64 -> f32 narrowing of I.5.3's Mul"
        )]
        let mul = if denom > 0.0 {
            (QUANT_NUMERATOR / denom) as f32
        } else {
            0.0
        };
        let scale = [
            mul * qm_multiplier(x_qm_scale),
            mul,
            mul * qm_multiplier(b_qm_scale),
        ];
        let cols = transform.coeff_cols();
        let rows = transform.coeff_rows();
        let cells = cols.saturating_mul(rows);
        let quant_bias = jpxl_core::color::DEFAULT_QUANT_BIAS;
        let mut steps: [Box<[f32]>; NUM_CHANNELS] =
            core::array::from_fn(|_| vec![0.0f32; cells].into_boxed_slice());
        let mut zero_threshold: [Box<[f32]>; NUM_CHANNELS] =
            core::array::from_fn(|_| vec![0.0f32; cells].into_boxed_slice());
        for channel in 0..NUM_CHANNELS {
            let sc = scale.get(channel).copied().unwrap_or(0.0);
            let bias = quant_bias.get(channel).copied().unwrap_or(1.0);
            let matrix = &matrices[channel];
            for cell in 0..cells {
                let (x, y) = (cell % cols.max(1), cell / cols.max(1));
                let step = sc * matrix.at(x, y);
                if let Some(slot) = steps[channel].get_mut(cell) {
                    *slot = step;
                }
                if let Some(slot) = zero_threshold[channel].get_mut(cell) {
                    // |recon(±1)| = bias * step; ties prefer smaller |q|, so
                    // zero wins when |target| <= 0.5 * |recon(±1)|.
                    *slot = 0.5 * bias * step;
                }
            }
        }

        Ok(Self {
            quant_bias,
            quant_bias_numerator: jpxl_core::color::DEFAULT_QUANT_BIAS_NUMERATOR,
            steps,
            zero_threshold,
            rd: None,
            rd_choose: false,
        })
    }

    /// Installs Phase 7.0's per-cell Lagrange weights, switching this
    /// quantizer from nearest-reconstruction to rate-distortion choice.
    ///
    /// `weight_at(channel, cell)` must return exactly the coefficient
    /// [`crate::block_cost_bounded`] multiplies that cell's squared error by,
    /// or the quantizer and the cover search will be minimising different
    /// objectives — which is the whole defect this fixes.
    pub(crate) fn install_rd_weights(
        &mut self,
        rd_choose: bool,
        mut weight_at: impl FnMut(usize, usize) -> f64,
    ) {
        self.rd_choose = rd_choose;
        let cells = self.steps.first().map_or(0, |s| s.len());
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a Lagrange weight in bits per squared sample unit is well inside f32"
        )]
        let table: [Box<[f32]>; NUM_CHANNELS] = core::array::from_fn(|channel| {
            (0..cells)
                .map(|cell| weight_at(channel, cell) as f32)
                .collect::<Vec<f32>>()
                .into_boxed_slice()
        });
        self.rd = Some(table);
    }

    /// The rate-distortion weight for one cell, or `None` under the shipped
    /// nearest rule.
    #[inline]
    fn rd_weight(&self, channel: usize, cell: usize) -> Option<f32> {
        if !self.rd_choose {
            return None;
        }
        self.weight_at(channel, cell)
    }

    /// The Lagrange weight for one cell regardless of which rule `choose` uses.
    /// Phase 7.1's truncation pass reads this while `choose` stays nearest.
    pub(crate) fn weight_at(&self, channel: usize, cell: usize) -> Option<f32> {
        self.rd
            .as_ref()
            .and_then(|t| t.get(channel))
            .and_then(|c| c.get(cell).copied())
    }

    /// The rate-distortion cost of one candidate: `residual_bits(q) + rd * e^2`.
    ///
    /// The rate term mirrors `crate::residual_bits` exactly — zero costs
    /// nothing, and a nonzero costs its magnitude's bit length plus a sign bit.
    /// It is deliberately the *same* crude proxy the cover objective sums, so
    /// the two agree; `sources/outside-advice.md` is right that it knows
    /// nothing about zero runs or entropy context, which bounds what this mode
    /// can capture to the first-order "is this coefficient worth any bits at
    /// all" decision.
    #[inline]
    fn rd_cost(q: i32, error: f32, rd: f32) -> f32 {
        let bits = if q == 0 {
            0.0f32
        } else {
            #[allow(
                clippy::cast_precision_loss,
                reason = "a bit length is at most 33; exact in f32"
            )]
            let b = (33 - q.unsigned_abs().leading_zeros()) as f32;
            b
        };
        bits + rd * error * error
    }

    /// Phase 7.1: drop trailing nonzeros whose removal shortens the walk more
    /// than it costs in distortion.
    ///
    /// # Why this is a block pass and not a per-coefficient rule
    ///
    /// 18181-1 I.4 emits a coefficient token for every order position up to
    /// and including the **last** nonzero, then stops. So zeroing the last
    /// nonzero frees its own token *plus every interior zero back to the
    /// previous nonzero*, while zeroing mid-run frees nothing at all. Phase
    /// 7.1a measured that bonus at 0.91 / 1.56 / 1.43 tokens per
    /// varblock-channel at 0.5 / 1 / 2 bpp.
    ///
    /// `residual_bits` is indifferent between those two decisions, which is why
    /// Phase 7.0's per-coefficient rate term zeroed by magnitude alone and
    /// stripped texture uniformly. The last-nonzero position is a property of
    /// the whole block, so no per-coefficient rule can express this — hence a
    /// backward pass over the block.
    ///
    /// `order` is the coefficient order; `num_blocks` is where the HF positions
    /// start (LLF cells below it are written from the LF image and never coded
    /// here). `target_at` returns the pre-quantization coefficient for a cell,
    /// which is what the distortion of zeroing is measured against.
    ///
    /// Returns how many coefficients were dropped.
    pub(crate) fn truncate_trailing(
        &self,
        channel: usize,
        quant: &mut [i32],
        order: &[u32],
        num_blocks: usize,
        zero_token_bits: f32,
        mut target_at: impl FnMut(usize) -> f32,
    ) -> usize {
        let size = order.len().min(quant.len());
        if num_blocks >= size {
            return 0;
        }
        // Positions of the nonzeros, in order. The pass only ever removes from
        // the end, so this is built once.
        let mut nonzero_positions: Vec<usize> = (num_blocks..size)
            .filter(|&k| {
                order
                    .get(k)
                    .and_then(|&c| quant.get(c as usize))
                    .copied()
                    .unwrap_or(0)
                    != 0
            })
            .collect();

        let mut dropped = 0usize;
        while let Some(&last) = nonzero_positions.last() {
            let Some(&cell_u32) = order.get(last) else {
                break;
            };
            let cell = cell_u32 as usize;
            let Some(&q) = quant.get(cell) else { break };
            if q == 0 {
                nonzero_positions.pop();
                continue;
            }
            let Some(weight) = self.weight_at(channel, cell) else {
                break;
            };

            // Rate saved: this coefficient's own token, plus every interior
            // zero it currently keeps inside the walk. With no nonzero before
            // it, the walk collapses to nothing and every HF position from
            // `num_blocks` is freed.
            let previous = nonzero_positions
                .len()
                .checked_sub(2)
                .and_then(|i| nonzero_positions.get(i).copied());
            let exposed_zeros = match previous {
                Some(p) => last.saturating_sub(p).saturating_sub(1),
                None => last.saturating_sub(num_blocks),
            };
            #[allow(
                clippy::cast_precision_loss,
                reason = "a block has at most 1024 positions; exact in f32"
            )]
            let zeros = exposed_zeros as f32;
            let own_bits = {
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "a bit length is at most 33; exact in f32"
                )]
                let b = (33 - q.unsigned_abs().leading_zeros()) as f32;
                b
            };
            let bits_saved = own_bits + zeros * zero_token_bits;

            // Distortion added: the error goes from (recon - target)^2 to
            // target^2, because a dropped coefficient reconstructs as zero.
            let target = target_at(cell);
            let recon = self.reconstruct(q, channel, cell);
            let kept = recon - target;
            let added = target.mul_add(target, -(kept * kept));
            let cost = weight * added;

            if bits_saved <= cost {
                break;
            }
            if let Some(slot) = quant.get_mut(cell) {
                *slot = 0;
            }
            nonzero_positions.pop();
            dropped += 1;
        }
        dropped
    }

    /// I.5.3's bias adjustment.
    fn bias_adjust(&self, q: i32, channel: usize) -> f32 {
        #[allow(
            clippy::cast_precision_loss,
            reason = "|q| <= MAX_QUANT == 2^20, exact in f32"
        )]
        let f = q as f32;
        if q.abs() <= 1 {
            f * self.quant_bias.get(channel).copied().unwrap_or(1.0)
        } else {
            f - self.quant_bias_numerator / f
        }
    }

    /// Precomputed step for `(channel, cell)`, or 0 if out of range.
    #[inline]
    fn step_at(&self, channel: usize, cell: usize) -> f32 {
        self.steps
            .get(channel)
            .and_then(|s| s.get(cell).copied())
            .unwrap_or(0.0)
    }

    /// I.5.3's full reconstruction of one coefficient, before I.6.
    #[must_use]
    pub fn reconstruct(&self, q: i32, channel: usize, cell: usize) -> f32 {
        self.bias_adjust(q, channel) * self.step_at(channel, cell)
    }

    /// I.5.3's quantization step for one cell: `Mul * qm * dequant_matrix`.
    ///
    /// Zero for a cell outside the matrix; the block solver reads it to price
    /// distortion in the same units [`Self::choose`] quantizes in.
    #[must_use]
    pub fn step(&self, channel: usize, cell: usize) -> f32 {
        self.step_at(channel, cell)
    }

    /// Phase-2: quantize a contiguous coefficient lane (one channel of one
    /// varblock) into `out`, skipping LLF cells when `skip_llf` is set.
    /// Phase 8.5 batches each row's HF span through [`Self::choose_lane4`]
    /// with scalar tails, so batches never cross the top-left LLF boundary.
    ///
    /// Same integer rules as [`Self::choose`], applied cell-by-cell.
    ///
    /// # Errors
    ///
    /// As [`Self::choose`].
    pub fn quantize_lane(
        &self,
        channel: usize,
        coeffs: &[f32],
        out: &mut [i32],
        side: usize,
        n_blocks: usize,
        skip_llf: bool,
    ) -> Result<()> {
        let cells = coeffs.len().min(out.len());
        if side == 0 {
            return Err(PolicyError::Unsupported {
                what: "a zero-width HF coefficient lane",
            });
        }
        for row_start in (0..cells).step_by(side) {
            let row = row_start / side;
            let row_end = row_start.saturating_add(side).min(cells);
            let first_hf = if skip_llf && row < n_blocks {
                row_start.saturating_add(n_blocks).min(row_end)
            } else {
                row_start
            };
            if let Some(llf) = out.get_mut(row_start..first_hf) {
                llf.fill(0);
            }

            let mut cell = first_hf;
            while cell.saturating_add(4) <= row_end {
                let targets = [
                    coeffs.get(cell).copied().unwrap_or(0.0),
                    coeffs.get(cell + 1).copied().unwrap_or(0.0),
                    coeffs.get(cell + 2).copied().unwrap_or(0.0),
                    coeffs.get(cell + 3).copied().unwrap_or(0.0),
                ];
                let (quantized, _) = self.choose_lane4(channel, targets, cell)?;
                if let Some(slots) = out.get_mut(cell..cell + 4) {
                    slots.copy_from_slice(&quantized);
                }
                cell += 4;
            }
            while cell < row_end {
                let q = self.choose(coeffs.get(cell).copied().unwrap_or(0.0), channel, cell)?;
                if let Some(slot) = out.get_mut(cell) {
                    *slot = q;
                }
                cell += 1;
            }
        }
        Ok(())
    }

    /// [`Self::choose`], batched over four *contiguous, already non-LLF*
    /// cells (`cell_base..cell_base + 4`), plus each winner's reconstruction
    /// — for candidate *scoring* callers (`block_cost_bounded`) that need
    /// both without a second `reconstruct` pass.
    ///
    /// Phase-3 / outside-advice.md §3: vectorizes the *coefficient* axis
    /// (four adjacent cells at once) instead of `choose`'s existing
    /// `#[cfg(feature = "simd")]` path, which vectorizes the *four candidates
    /// of one cell* — outside-advice's own critique of that axis choice.
    /// Every result is bit-identical to four independent `choose` calls on
    /// the same `(channel, cell)` tuples: same zero-threshold shortcut, same
    /// `[0, estimate-1, estimate, estimate+1]` candidate order (so the same
    /// nearest-then-smaller-magnitude tie rule), same error semantics (a
    /// degenerate step or an out-of-range non-zero-threshold estimate is
    /// `Err`, matching `clamp_round`). This is deliberately *not* a
    /// distortion/rate estimate of any kind — see the module's `choose` doc
    /// and `sources/outside-advice.md` §8's contrast between this
    /// (safe, output-preserving) and a closed-form summary score
    /// (unbounded-tail risk without a calibration harness, not attempted).
    ///
    /// # Errors
    ///
    /// As [`Self::choose`], evaluated per lane.
    #[cfg(feature = "simd")]
    pub(crate) fn choose_lane4(
        &self,
        channel: usize,
        targets: [f32; 4],
        cell_base: usize,
    ) -> Result<([i32; 4], [f32; 4])> {
        use wide::{CmpLe, f32x4};

        // Phase 7.0: the vectorised body below implements the *nearest* rule.
        // Under rate-distortion choice, fall back to four scalar `choose`
        // calls — the same thing the non-simd build does — so the two paths
        // stay bit-identical by construction rather than by a second
        // hand-vectorised implementation of the RD comparison that would have
        // to be kept in sync.
        if self.rd_weight(channel, cell_base).is_some() {
            let mut q = [0i32; 4];
            let mut recon = [0.0f32; 4];
            for i in 0..4 {
                let cell = cell_base + i;
                let target = targets.get(i).copied().unwrap_or(0.0);
                let qi = self.choose(target, channel, cell)?;
                if let (Some(qs), Some(rs)) = (q.get_mut(i), recon.get_mut(i)) {
                    *qs = qi;
                    *rs = self.reconstruct(qi, channel, cell);
                }
            }
            return Ok((q, recon));
        }

        let mut step_arr = [0.0f32; 4];
        let mut thr_arr = [0.0f32; 4];
        for (i, slot) in step_arr.iter_mut().enumerate() {
            let cell = cell_base + i;
            let step = self.step_at(channel, cell);
            if !(step.is_finite() && step > 0.0) {
                return Err(PolicyError::Unsupported {
                    what: "a degenerate HF quantization step",
                });
            }
            *slot = step;
            if let Some(t) = thr_arr.get_mut(i) {
                *t = self
                    .zero_threshold
                    .get(channel)
                    .and_then(|t| t.get(cell).copied())
                    .unwrap_or(0.0);
            }
        }

        let target = f32x4::new(targets);
        let step = f32x4::new(step_arr);
        let thr = f32x4::new(thr_arr);
        // All-ones-bits per lane where zero wins outright (the scalar
        // shortcut), else all-zero-bits — `wide`'s comparison-mask idiom.
        let zero_mask = target.abs().cmp_le(thr);
        let zero_mask_arr = zero_mask.to_array();

        // The scalar zero-threshold shortcut is a ~2-instruction early
        // return for what is, on typical photo content, most HF
        // coefficients — a SIMD lane can't skip work per-element the way a
        // scalar early return can, so without this whole-lane fast path the
        // vectorized candidate search below (unconditionally run on every
        // lane) is *more* total arithmetic than four scalar `choose` calls
        // that mostly take the shortcut, not less. Measured: omitting this
        // cost ~35% more wall time on `cover_ms` despite fewer `choose_cover`
        // calls in the diagnostics counter (the counter tracks scalar
        // `choose` invocations, which this fast path also bypasses; see
        // its updated doc comment in `diagnostics.rs`).
        if zero_mask.all() {
            return Ok(([0i32; 4], [0.0f32; 4]));
        }

        // `wide::f32x4::round` lowers to scalar `roundf` calls on this target.
        // Convert the four quotients with the same checked round-away-from-zero
        // rule as scalar `choose`, then keep the exact integers in vector lanes.
        // This is both cheaper and makes the later candidate conversion a plain
        // cast: estimate +/- 1 is exactly representable below MAX_QUANT.
        let quotients = (target / step).to_array();
        let mut est_arr = [0.0f32; 4];
        for i in 0..4 {
            // A zero-threshold lane never reaches `clamp_round` in the
            // scalar path (it returns before computing `estimate` at all),
            // so an out-of-range speculative estimate there is not an error.
            if zero_mask_arr.get(i).copied().unwrap_or(0.0) == 0.0 {
                let quotient = quotients.get(i).copied().unwrap_or(0.0);
                let estimate = clamp_round(quotient)?;
                if let Some(slot) = est_arr.get_mut(i) {
                    #[allow(
                        clippy::cast_precision_loss,
                        reason = "MAX_QUANT == 2^20, exact in f32"
                    )]
                    {
                        *slot = estimate as f32;
                    }
                }
            }
        }
        let estimate = f32x4::new(est_arr);

        let bias = self.quant_bias.get(channel).copied().unwrap_or(1.0);
        let numerator = self.quant_bias_numerator;
        let bias_adjust_vec = |q: f32x4| -> f32x4 {
            let small = q.abs().cmp_le(f32x4::splat(1.0));
            let via_bias = q * f32x4::splat(bias);
            // Unused (discarded by `blend`) on lanes where `small` holds,
            // including any lane with q == 0; the resulting Inf/NaN there is
            // inert (never read) and not a panic in Rust's float semantics.
            let via_numerator = q - f32x4::splat(numerator) / q;
            small.blend(via_bias, via_numerator)
        };

        let mut best_q = [0i32; 4];
        let mut best_err = [f32::INFINITY; 4];
        let mut best_recon = [0.0f32; 4];
        // Same order as `choose`'s `[0, estimate-1, estimate, estimate+1]`,
        // so the same "first legal candidate with strictly smaller error, or
        // equal error and strictly smaller magnitude" tie rule applies.
        let candidate_slots: [f32x4; 4] = [
            f32x4::splat(0.0),
            estimate - f32x4::splat(1.0),
            estimate,
            estimate + f32x4::splat(1.0),
        ];
        for q_vec in candidate_slots {
            let q_arr = q_vec.to_array();
            let recon_vec = bias_adjust_vec(q_vec) * step;
            let err_arr = (recon_vec - target).abs().to_array();
            let recon_arr = recon_vec.to_array();
            for i in 0..4 {
                let q = q_arr.get(i).copied().unwrap_or(0.0);
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "estimate and estimate +/- 1 are exact integer f32 values below 2^24"
                )]
                let qi = q as i32;
                if qi.unsigned_abs() > MAX_QUANT.unsigned_abs() {
                    continue;
                }
                let (Some(be), Some(bq), Some(br), Some(&err)) = (
                    best_err.get_mut(i),
                    best_q.get_mut(i),
                    best_recon.get_mut(i),
                    err_arr.get(i),
                ) else {
                    continue;
                };
                if err < *be || (err == *be && qi.abs() < bq.abs()) {
                    *be = err;
                    *bq = qi;
                    *br = recon_arr.get(i).copied().unwrap_or(0.0);
                }
            }
        }

        for i in 0..4 {
            if zero_mask_arr.get(i).copied().unwrap_or(0.0) != 0.0 {
                if let (Some(q), Some(r)) = (best_q.get_mut(i), best_recon.get_mut(i)) {
                    *q = 0;
                    *r = 0.0;
                }
            }
        }

        Ok((best_q, best_recon))
    }

    /// [`Self::choose_lane4`] without the `simd` feature: four independent
    /// scalar [`Self::choose`] calls. Callers do not need to feature-gate.
    #[cfg(not(feature = "simd"))]
    pub(crate) fn choose_lane4(
        &self,
        channel: usize,
        targets: [f32; 4],
        cell_base: usize,
    ) -> Result<([i32; 4], [f32; 4])> {
        let mut q = [0i32; 4];
        let mut recon = [0.0f32; 4];
        for i in 0..4 {
            let cell = cell_base + i;
            let target = targets.get(i).copied().unwrap_or(0.0);
            let qi = self.choose(target, channel, cell)?;
            if let (Some(qs), Some(rs)) = (q.get_mut(i), recon.get_mut(i)) {
                *qs = qi;
                *rs = self.reconstruct(qi, channel, cell);
            }
        }
        Ok((q, recon))
    }

    /// The integer whose reconstruction is nearest `target`.
    ///
    /// Candidates come from the linear estimate — the bias adjustment is a
    /// small perturbation, never a sign change — plus zero, which the linear
    /// estimate can miss when `quant_bias` shrinks `±1` below half a step.
    /// Ties go to the smaller magnitude, so a coefficient that reconstructs
    /// equally well as `0` or `±1` costs the fewest bits.
    ///
    /// Phase-1: precomputed steps, exact zero early-out (output-identical),
    /// and optional `wide` comparison of the four candidates.
    ///
    /// # Errors
    ///
    /// [`PolicyError::Unsupported`] if the step is degenerate or the target
    /// needs an integer outside [`MAX_QUANT`].
    pub fn choose(&self, target: f32, channel: usize, cell: usize) -> Result<i32> {
        crate::diagnostics::note_choose();
        let step = self.step_at(channel, cell);
        if !(step.is_finite() && step > 0.0) {
            return Err(PolicyError::Unsupported {
                what: "a degenerate HF quantization step",
            });
        }
        // Exact zero shortcut: |target| <= 0.5 * quant_bias * step ⇒ zero
        // reconstructs at least as close as ±1 and wins ties by magnitude.
        //
        // Only sound under the nearest rule. Rate-distortion choice widens the
        // dead zone strictly (zero is the only candidate that costs no bits),
        // so the shortcut would still return the right answer *inside* the
        // threshold — but taking it would skip the wider zeroing the mode
        // exists for, so the RD path runs the full comparison.
        let rd = self.rd_weight(channel, cell);
        let thr = self
            .zero_threshold
            .get(channel)
            .and_then(|t| t.get(cell).copied())
            .unwrap_or(0.0);
        if rd.is_none() && target.abs() <= thr {
            return Ok(0);
        }

        let estimate = clamp_round(target / step)?;
        let candidates = [0, estimate - 1, estimate, estimate + 1];

        if let Some(rd) = rd {
            let mut best = 0i32;
            let mut best_cost = f32::INFINITY;
            for q in candidates {
                if q.abs() > MAX_QUANT {
                    continue;
                }
                let error = self.bias_adjust(q, channel) * step - target;
                let cost = Self::rd_cost(q, error, rd);
                // Same tie rule as the nearest path: equal cost prefers the
                // smaller magnitude, so the choice stays deterministic and
                // biased toward cheaper symbols.
                if cost < best_cost || (cost == best_cost && q.abs() < best.abs()) {
                    best = q;
                    best_cost = cost;
                }
            }
            return Ok(best);
        }

        #[cfg(feature = "simd")]
        {
            use wide::f32x4;
            let mut recon = [0.0f32; 4];
            let mut legal = [false; 4];
            for (i, &q) in candidates.iter().enumerate() {
                if q.abs() > MAX_QUANT {
                    recon[i] = f32::INFINITY;
                    legal[i] = false;
                } else {
                    recon[i] = self.bias_adjust(q, channel) * step;
                    legal[i] = true;
                }
            }
            let r = f32x4::new(recon);
            let t = f32x4::splat(target);
            let err = (r - t).abs().to_array();
            let mut best = 0i32;
            let mut best_error = f32::INFINITY;
            for (i, &q) in candidates.iter().enumerate() {
                if !legal[i] {
                    continue;
                }
                let error = err[i];
                if error < best_error || (error == best_error && q.abs() < best.abs()) {
                    best = q;
                    best_error = error;
                }
            }
            return Ok(best);
        }

        #[cfg(not(feature = "simd"))]
        {
            let mut best = 0i32;
            let mut best_error = f32::INFINITY;
            for q in candidates {
                if q.abs() > MAX_QUANT {
                    continue;
                }
                let error = (self.bias_adjust(q, channel) * step - target).abs();
                if error < best_error || (error == best_error && q.abs() < best.abs()) {
                    best = q;
                    best_error = error;
                }
            }
            Ok(best)
        }
    }

    /// S8 Phase C (`sources/outside-advice.md` §8's "lower bound to prune"
    /// primitive; `jpegxl-rs.work.arch-s8-full-redesign-scoped`): a cheap,
    /// *provable* lower bound on one coefficient's contribution to the
    /// `residual_bits(choose(target))` (rate) and
    /// `(reconstruct(choose(target)) - target)^2` (distortion) terms
    /// `block_cost_bounded` sums — computed **without** [`Self::choose`]'s
    /// 4-candidate search.
    ///
    /// Two cases, both provable from the same zero-threshold [`Self::choose`]
    /// already uses as its own exact fast path:
    /// - `|target| <= threshold`: the cell is *guaranteed* to quantize to
    ///   zero (this is the exact condition `choose` tests), so both returned
    ///   numbers are **exact**, not approximate — `0` bits and the true
    ///   distortion `target^2` (`reconstruct(0, ..) == 0.0`).
    /// - `|target| > threshold`: the cell is guaranteed *nonzero*. The
    ///   cheapest any nonzero symbol can cost is `2` bits (see
    ///   `residual_bits`, `q = ±1`); the true count could be higher, so `2`
    ///   is a genuine lower bound, not an estimate of the real value.
    ///   Distortion floor is `0.0` — true but uninformative; nothing cheaper
    ///   than the exact 4-candidate search bounds a nonzero cell's error
    ///   usefully (`sources/outside-advice.md` §8 finds the same limit: rate
    ///   is a discrete, non-smooth function of the quantized integer, so a
    ///   *tight* bound needs the per-cell magnitude data `choose` itself
    ///   uses — this returns a *safe*, not a *tight*, bound).
    ///
    /// Error semantics mirror [`Self::choose`] exactly (same degenerate-step
    /// and out-of-range checks, on the same inputs) so that whether or not a
    /// caller uses this bound to skip calling `choose` at all, the two paths
    /// agree on when a target is legal.
    ///
    /// # Errors
    ///
    /// As [`Self::choose`].
    pub fn cell_lower_bound(&self, target: f32, channel: usize, cell: usize) -> Result<(u64, f64)> {
        if !target.is_finite() {
            return Err(PolicyError::Unsupported {
                what: "a non-finite quantization target",
            });
        }
        let step = self.step_at(channel, cell);
        if !(step.is_finite() && step > 0.0) {
            return Err(PolicyError::Unsupported {
                what: "a degenerate HF quantization step",
            });
        }
        let thr = self
            .zero_threshold
            .get(channel)
            .and_then(|t| t.get(cell).copied())
            .unwrap_or(0.0);
        if target.abs() <= thr {
            let t = f64::from(target);
            return Ok((0, t * t));
        }
        // Guaranteed nonzero: mirror `clamp_round`'s MAX_QUANT check on the
        // same linear estimate `choose` would compute, so a target `choose`
        // would refuse is refused here too — never silently treated as a
        // cheap, legal bound.
        let estimate = (target / step).round();
        #[allow(
            clippy::cast_precision_loss,
            reason = "MAX_QUANT == 2^20, exact in f32"
        )]
        let max_quant_f = MAX_QUANT as f32;
        if estimate.abs() > max_quant_f {
            return Err(PolicyError::Unsupported {
                what: "a coefficient outside the quantizer's working range",
            });
        }
        Ok((2, 0.0))
    }
}

/// I.5.3's `pow(0.8, qm_scale - 2)`, exactly as the decoder computes it.
fn qm_multiplier(qm_scale: u32) -> f32 {
    let exponent = i32::try_from(qm_scale).unwrap_or(2) - 2;
    0.8f32.powi(exponent)
}

/// I.6's `kX` and `kB` with the I.2.3 defaults and neutral signalled factors.
///
/// Returned as a pair so a caller cannot apply one and forget the other.
#[must_use]
pub const fn neutral_cfl_factors() -> (f32, f32) {
    (BASE_CORRELATION_X, BASE_CORRELATION_B)
}

/// I.2.3's default `colour_factor`: the divisor I.6 turns a stored factor into
/// a correlation coefficient with. Slice 15 keeps it at the default so the
/// searched quantity is the biased factor alone, in a space every decoder
/// reads the same way.
pub const DEFAULT_COLOUR_FACTOR: u32 = 84;

/// I.6's `k = base_correlation + factor / colour_factor`, computed the way the
/// decoder computes it: the division is done in `f32`, from the same integer
/// the wire carries, so the encoder's `k` is bit-identical to the one the
/// decoder will apply.
#[must_use]
pub fn cfl_multiplier(base: f32, factor: i32, colour_factor: u32) -> f32 {
    #[allow(
        clippy::cast_precision_loss,
        reason = "colour_factor and the searched factor stay far inside f32's \
                  exact-integer range, mirroring the decoder's own widening"
    )]
    let ratio = factor as f32 / colour_factor as f32;
    base + ratio
}

/// The least-squares seed of one channel against luma, as sufficient
/// statistics of the unquantized residual `C - k*Y`
/// (`Encoder-plan1.md` §8).
///
/// The residual energy of a candidate correlation coefficient `k` is
/// `Σ(C - k·Y)² = ΣC² - 2k·ΣYC + k²·ΣY²`, so the three running sums are all a
/// seed calculation needs: every candidate is then an `O(1)` parabola
/// evaluation, not a re-scan of the coefficients. The policy layer follows
/// this regression with a separate integer refinement through reconstructed
/// `dY` and the exact quantizer arithmetic.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CflAccumulator {
    s_yy: f64,
    s_yc: f64,
    s_cc: f64,
}

impl CflAccumulator {
    /// Adds one `(unquantized Y, unquantized channel coefficient)` pair.
    pub fn add(&mut self, y: f32, coeff: f32) {
        let y = f64::from(y);
        let c = f64::from(coeff);
        self.s_yy += y * y;
        self.s_yc += y * c;
        self.s_cc += c * c;
    }

    /// Residual energy `Σ(C - k·Y)²` at a given `k`.
    fn residual_energy(&self, k: f64) -> f64 {
        self.s_cc - 2.0 * k * self.s_yc + k * k * self.s_yy
    }

    /// The integer factor in `[lo, hi]` whose correlation coefficient
    /// minimizes the residual energy, refining in the wire's representable
    /// space rather than rounding a float.
    ///
    /// The energy is convex in `k`, and `k` is affine in the factor, so the
    /// least-squares optimum's integer neighbours bracket the best factor;
    /// the neutral factor `0` is always among the candidates and wins every
    /// tie. That tie rule is what makes an already-decorrelated channel —
    /// grayscale's `X ≡ 0` and `B ≡ Y` — degenerate to exactly the neutral
    /// factor, byte for byte.
    #[must_use]
    pub fn best_factor(&self, base: f32, colour_factor: u32, lo: i32, hi: i32) -> i32 {
        if self.s_yy <= 0.0 {
            // No luma energy to correlate against: nothing to subtract.
            return 0;
        }
        let k_ls = self.s_yc / self.s_yy;
        let star = (k_ls - f64::from(base)) * f64::from(colour_factor);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "star is finite here (s_yy > 0), and the result is clamped \
                      into [lo, hi] before use"
        )]
        let centre = star.round() as i64;
        let mut best: Option<(i32, f64)> = None;
        // The floor/ceil of the continuous optimum bracket the best integer;
        // widen by one on each side against float slack, and always weigh the
        // neutral factor so the degenerate case cannot drift off zero.
        for delta in [-2i64, -1, 0, 1, 2] {
            let cand = centre.saturating_add(delta);
            let cand = i32::try_from(cand.clamp(i64::from(lo), i64::from(hi))).unwrap_or(0);
            consider(&mut best, cand, self, base, colour_factor);
        }
        consider(&mut best, 0, self, base, colour_factor);
        best.map_or(0, |(factor, _)| factor)
    }
}

/// Keeps the lower-energy candidate, breaking ties toward the smaller
/// magnitude and then toward zero (the neutral factor).
fn consider(
    best: &mut Option<(i32, f64)>,
    cand: i32,
    acc: &CflAccumulator,
    base: f32,
    colour_factor: u32,
) {
    let k = f64::from(cfl_multiplier(base, cand, colour_factor));
    let energy = acc.residual_energy(k);
    let better = match *best {
        None => true,
        Some((factor, best_energy)) => {
            energy < best_energy || (energy == best_energy && cand.abs() < factor.abs())
        }
    };
    if better {
        *best = Some((cand, energy));
    }
}

/// Rounds to the nearest integer and rejects anything past [`MAX_QUANT`].
fn clamp_round(v: f32) -> Result<i32> {
    if !v.is_finite() {
        return Err(PolicyError::Unsupported {
            what: "a non-finite quantization target",
        });
    }
    // Rust's float-to-int cast truncates toward zero. Moving a finite value by
    // half a unit first therefore implements `f32::round`'s ties-away rule
    // without a libm `roundf` call. Around MAX_QUANT (2^20), 0.5 is exactly
    // representable, so the adjustment and the resulting integer are exact.
    let adjusted = if v.is_sign_negative() {
        v - 0.5
    } else {
        v + 0.5
    };
    #[allow(
        clippy::cast_possible_truncation,
        reason = "finite float casts are defined and the result is range-checked immediately below"
    )]
    let value = adjusted as i32;
    if value.unsigned_abs() > MAX_QUANT.unsigned_abs() {
        return Err(PolicyError::Unsupported {
            what: "a coefficient outside the quantizer's working range",
        });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lf_quantizer_inverts_its_own_reconstruction() {
        let q = LfQuantizer::new(4096, 16, 0);
        for channel in 0..NUM_CHANNELS {
            for step in -40i32..=40 {
                let target = q.reconstruct(step, channel);
                assert_eq!(
                    q.quantize(target, channel).expect("in range"),
                    step,
                    "channel {channel} step {step}"
                );
            }
        }
    }

    #[test]
    fn the_lf_multipliers_order_the_channels_x_y_b() {
        // G.1.2's defaults are 1/32, 1/4, 1/2, so Y's step is eight times X's
        // and B's is sixteen times X's: chroma is quantized far more coarsely.
        let q = LfQuantizer::new(4096, 16, 0);
        let x = q.reconstruct(1, 0);
        let y = q.reconstruct(1, 1);
        let b = q.reconstruct(1, 2);
        assert!((y / x - 8.0).abs() < 1e-3, "{x} {y}");
        assert!((b / x - 16.0).abs() < 1e-3, "{x} {b}");
    }

    #[test]
    fn extra_precision_divides_the_lf_step() {
        let coarse = LfQuantizer::new(4096, 16, 0);
        let fine = LfQuantizer::new(4096, 16, 2);
        assert!((coarse.reconstruct(1, 1) / fine.reconstruct(1, 1) - 4.0).abs() < 1e-4);
    }

    #[test]
    fn checked_round_matches_f32_round_across_ties_and_working_range() {
        let offsets = [-0.75f32, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75];
        for integer in (-MAX_QUANT..=MAX_QUANT).step_by(257) {
            #[allow(
                clippy::cast_precision_loss,
                reason = "MAX_QUANT == 2^20, so every sampled integer is exact in f32"
            )]
            let base = integer as f32;
            for offset in offsets {
                let value = base + offset;
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "the expected value is range-checked before conversion"
                )]
                let expected = value.round() as i32;
                let actual = clamp_round(value);
                if expected.unsigned_abs() > MAX_QUANT.unsigned_abs() {
                    assert!(actual.is_err(), "{value}");
                } else {
                    assert_eq!(actual.expect("in range"), expected, "{value}");
                }
            }
        }
    }

    #[test]
    fn the_hf_quantizer_never_reconstructs_further_than_a_naive_rounding_would() {
        let q = HfQuantizer::new(TransformType::Dct8x8, 4096, 1, 2, 2).expect("defaults");
        for channel in 0..NUM_CHANNELS {
            for cell in 1..DCT8X8_CELLS {
                let step = q.reconstruct(1, channel, cell) / 0.945;
                for numerator in -12i32..=12 {
                    let target = step * (numerator as f32) / 4.0;
                    let chosen = q.choose(target, channel, cell).expect("in range");
                    let chosen_error = (q.reconstruct(chosen, channel, cell) - target).abs();
                    for other in [chosen - 1, chosen + 1, 0] {
                        let other_error = (q.reconstruct(other, channel, cell) - target).abs();
                        assert!(
                            chosen_error <= other_error + 1e-6,
                            "c{channel} cell{cell} target {target}: chose {chosen} \
                             (err {chosen_error}) over {other} (err {other_error})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_zero_target_quantizes_to_zero_in_every_channel_and_cell() {
        let q = HfQuantizer::new(TransformType::Dct8x8, 4096, 1, 2, 2).expect("defaults");
        for channel in 0..NUM_CHANNELS {
            for cell in 0..DCT8X8_CELLS {
                assert_eq!(q.choose(0.0, channel, cell).expect("in range"), 0);
                assert_eq!(q.reconstruct(0, channel, cell), 0.0);
            }
        }
    }

    #[test]
    fn quantize_lane_matches_scalar_choose_across_llf_edges_and_row_tails() {
        for transform in [
            TransformType::Dct8x8,
            TransformType::Dct16x16,
            TransformType::Dct32x32,
        ] {
            let q = HfQuantizer::new(transform, 4096, 1, 2, 2).expect("defaults");
            let side = transform.sample_cols();
            let n = transform.block_dims().0;
            let cells = side * side;
            for channel in 0..NUM_CHANNELS {
                let coeffs: Vec<f32> = (0..cells)
                    .map(|cell| {
                        let signed = i32::try_from(cell % 13).unwrap_or(0) - 6;
                        #[allow(
                            clippy::cast_precision_loss,
                            reason = "the test pattern is bounded to -6..=6"
                        )]
                        let signed = signed as f32;
                        q.step(channel, cell) * signed * 0.37
                    })
                    .collect();
                let expected: Vec<i32> = coeffs
                    .iter()
                    .enumerate()
                    .map(|(cell, &target)| {
                        if cell % side < n && cell / side < n {
                            Ok(0)
                        } else {
                            q.choose(target, channel, cell)
                        }
                    })
                    .collect::<Result<_>>()
                    .expect("scalar reference");
                let mut actual = vec![i32::MIN; cells];
                q.quantize_lane(channel, &coeffs, &mut actual, side, n, true)
                    .expect("lane quantization");
                assert_eq!(actual, expected, "{transform:?} channel {channel}");
            }
        }
    }

    #[test]
    fn the_dc_weight_of_dct8x8_is_the_coarsest_and_high_frequencies_are_finer() {
        // The I.2.5 DCT8x8 weights fall with distance from (0, 0), so the
        // dequantization matrix — their reciprocal — rises: one quantization
        // step reconstructs to a *larger* value at high frequency.
        let q = HfQuantizer::new(TransformType::Dct8x8, 4096, 1, 2, 2).expect("defaults");
        for channel in 0..NUM_CHANNELS {
            let low = q.reconstruct(1, channel, 1);
            let high = q.reconstruct(1, channel, 63);
            assert!(high > low, "channel {channel}: {low} then {high}");
        }
    }

    #[test]
    fn hf_mul_scales_every_step_together() {
        let one = HfQuantizer::new(TransformType::Dct8x8, 4096, 1, 2, 2).expect("defaults");
        let four = HfQuantizer::new(TransformType::Dct8x8, 4096, 4, 2, 2).expect("defaults");
        for cell in [1usize, 9, 63] {
            let ratio = one.reconstruct(2, 1, cell) / four.reconstruct(2, 1, cell);
            assert!((ratio - 4.0).abs() < 1e-3, "cell {cell}: {ratio}");
        }
    }

    #[test]
    fn the_qm_scale_is_neutral_at_two_and_shrinks_above_it() {
        assert!((qm_multiplier(2) - 1.0).abs() < 1e-7);
        assert!((qm_multiplier(3) - 0.8).abs() < 1e-6);
        assert!((qm_multiplier(1) - 1.25).abs() < 1e-6);
    }

    #[test]
    fn an_unreachable_target_is_refused_rather_than_clamped() {
        let q = HfQuantizer::new(TransformType::Dct8x8, 4096, 1, 2, 2).expect("defaults");
        assert!(q.choose(f32::INFINITY, 1, 1).is_err());
        assert!(q.choose(1e30, 1, 1).is_err());
        let lf = LfQuantizer::new(4096, 16, 0);
        assert!(lf.quantize(f32::NAN, 1).is_err());
    }

    /// Phase-3 safety net (outside-advice.md §8's contrast with a closed-form
    /// summary score): `choose_lane4` must be bit-identical to four
    /// independent `choose` calls, exhaustively, before `block_cost_bounded`
    /// is allowed to use it. Every target pattern below is deliberately
    /// chosen to stress a specific boundary: exactly at / just inside / just
    /// outside the zero threshold, exact half-integer ties, the `|q|<=1`
    /// bias-adjust branch edge, and both sides of `MAX_QUANT`.
    #[cfg(feature = "simd")]
    #[test]
    fn choose_lane4_is_bit_identical_to_four_scalar_choose_calls() {
        let quantizers = [
            HfQuantizer::new(TransformType::Dct8x8, 4096, 1, 2, 2).expect("defaults"),
            HfQuantizer::new(TransformType::Dct8x8, 4096, 4, 2, 2).expect("hf_mul 4"),
            HfQuantizer::new(TransformType::Dct8x8, 4096, 1, 1, 3).expect("asymmetric qm"),
            HfQuantizer::new(TransformType::Dct16x16, 4096, 1, 2, 2).expect("dct16"),
            HfQuantizer::new(TransformType::Dct32x32, 8192, 1, 2, 2).expect("dct32, coarse scale"),
            HfQuantizer::new(TransformType::Dct8x8, u32::from(u16::MAX), 1, 2, 2)
                .expect("very fine global_scale (small steps)"),
        ];

        for q in &quantizers {
            let cells = q.steps[0].len();
            for channel in 0..NUM_CHANNELS {
                for cell_base in 0..cells.saturating_sub(3) {
                    // One representative step, to build target patterns that
                    // land relative to *this* lane's own scale.
                    let step = q.step(channel, cell_base);
                    if !(step.is_finite() && step > 0.0) {
                        continue;
                    }
                    let thr = q
                        .zero_threshold
                        .get(channel)
                        .and_then(|t| t.get(cell_base).copied())
                        .unwrap_or(0.0);
                    let patterns: [[f32; 4]; 10] = [
                        [0.0, 0.0, 0.0, 0.0],
                        [thr * 0.99, -thr * 0.99, thr, -thr],
                        [thr * 1.01, -thr * 1.01, step * 0.5, -step * 0.5],
                        [step * 1.5, -step * 1.5, step * 2.5, -step * 2.5],
                        [step * 0.5, step * 1.5, step * 2.5, step * 3.5],
                        [step * 10.5, -step * 10.5, step * 100.25, -step * 100.75],
                        [
                            step * f32::from(i16::try_from(MAX_QUANT).unwrap_or(i16::MAX)),
                            -step * f32::from(i16::try_from(MAX_QUANT).unwrap_or(i16::MAX)),
                            step * 1_048_575.0,
                            -step * 1_048_575.0,
                        ],
                        [step, -step, step * 2.0, -step * 2.0],
                        [f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 0.0, thr],
                        [step * 3.333_333, -step * 7.777_777, step * 0.1, -step * 0.1],
                    ];
                    for targets in patterns {
                        let scalar: Vec<Result<(i32, f32)>> = (0..4)
                            .map(|i| {
                                let cell = cell_base + i;
                                let t = targets.get(i).copied().unwrap_or(0.0);
                                q.choose(t, channel, cell)
                                    .map(|qi| (qi, q.reconstruct(qi, channel, cell)))
                            })
                            .collect();
                        let lane = q.choose_lane4(channel, targets, cell_base);
                        match (scalar.iter().all(Result::is_ok), &lane) {
                            (true, Ok((qs, recons))) => {
                                for i in 0..4 {
                                    let (sq, sr) = scalar
                                        .get(i)
                                        .and_then(|r| r.as_ref().ok())
                                        .copied()
                                        .unwrap();
                                    let lq = qs.get(i).copied().unwrap_or(i32::MIN);
                                    let lr = recons.get(i).copied().unwrap_or(f32::NAN);
                                    assert_eq!(
                                        sq,
                                        lq,
                                        "channel {channel} cell {} target {}: scalar q={sq} \
                                         lane q={lq}",
                                        cell_base + i,
                                        targets[i]
                                    );
                                    assert_eq!(
                                        sr.to_bits(),
                                        lr.to_bits(),
                                        "channel {channel} cell {} target {}: scalar recon={sr} \
                                         lane recon={lr}",
                                        cell_base + i,
                                        targets[i]
                                    );
                                }
                            }
                            (false, Err(_)) => {
                                // Both sides agree the lane is unreachable —
                                // exact wording need not match.
                            }
                            (scalar_ok, lane_result) => panic!(
                                "channel {channel} cell_base {cell_base} targets {targets:?}: \
                                 scalar all-ok={scalar_ok} lane={lane_result:?}"
                            ),
                        }
                    }
                }
            }
        }
    }

    /// S8 Phase C safety net (`jpegxl-rs.work.arch-s8-full-redesign-scoped`):
    /// `cell_lower_bound` must never exceed what `choose`/`reconstruct`
    /// actually produce, exhaustively — a single counterexample means the
    /// bound is unsound and cannot safely prune anything. Reuses the
    /// `choose_lane4` test's quantizer configs and boundary-stressing target
    /// patterns (same rigor template, same targets that stress the zero
    /// threshold from both sides, `MAX_QUANT`, and extreme magnitudes) but
    /// asserts `<=`, not `==` — this bound is deliberately not tight, only
    /// safe (see the type's doc for why rate cannot be tightly bounded this
    /// cheaply). Also asserts the two ERROR cases agree exactly (degenerate
    /// step / non-finite / out-of-range), since a caller using this bound to
    /// skip `choose` entirely must not silently swallow an error `choose`
    /// would have raised.
    #[test]
    fn cell_lower_bound_never_exceeds_the_exact_cost() {
        let quantizers = [
            HfQuantizer::new(TransformType::Dct8x8, 4096, 1, 2, 2).expect("defaults"),
            HfQuantizer::new(TransformType::Dct8x8, 4096, 4, 2, 2).expect("hf_mul 4"),
            HfQuantizer::new(TransformType::Dct8x8, 4096, 1, 1, 3).expect("asymmetric qm"),
            HfQuantizer::new(TransformType::Dct16x16, 4096, 1, 2, 2).expect("dct16"),
            HfQuantizer::new(TransformType::Dct32x32, 8192, 1, 2, 2).expect("dct32, coarse scale"),
            HfQuantizer::new(TransformType::Dct8x8, u32::from(u16::MAX), 1, 2, 2)
                .expect("very fine global_scale (small steps)"),
        ];

        let mut cells_pruned = 0u64;
        let mut cells_total = 0u64;

        for q in &quantizers {
            let cells = q.steps[0].len();
            for channel in 0..NUM_CHANNELS {
                for cell in 0..cells {
                    let step = q.step(channel, cell);
                    if !(step.is_finite() && step > 0.0) {
                        continue;
                    }
                    let thr = q
                        .zero_threshold
                        .get(channel)
                        .and_then(|t| t.get(cell).copied())
                        .unwrap_or(0.0);
                    let targets: [f32; 12] = [
                        0.0,
                        thr * 0.99,
                        -thr * 0.99,
                        thr,
                        thr * 1.01,
                        -thr * 1.01,
                        step * 0.5,
                        step * 1.5,
                        -step * 2.5,
                        step * 10.5,
                        step * f32::from(i16::try_from(MAX_QUANT).unwrap_or(i16::MAX)),
                        f32::MIN_POSITIVE,
                    ];
                    for &target in &targets {
                        let exact = q.choose(target, channel, cell).map(|qi| {
                            (residual_bits_for_test(qi), q.reconstruct(qi, channel, cell))
                        });
                        let bound = q.cell_lower_bound(target, channel, cell);
                        match (exact, bound) {
                            (Ok((bits_exact, recon)), Ok((bits_lb, sse_lb))) => {
                                let sse_exact = f64::from(recon - target).powi(2);
                                assert!(
                                    bits_lb <= bits_exact,
                                    "channel {channel} cell {cell} target {target}: \
                                     bound bits {bits_lb} > exact bits {bits_exact}"
                                );
                                assert!(
                                    sse_lb <= sse_exact + 1e-9,
                                    "channel {channel} cell {cell} target {target}: \
                                     bound sse {sse_lb} > exact sse {sse_exact}"
                                );
                                cells_total += 1;
                                if bits_lb > 0 {
                                    cells_pruned += 1;
                                }
                            }
                            (Err(_), Err(_)) => {
                                // Both refuse the same unreachable target —
                                // exact wording need not match.
                            }
                            (exact, bound) => panic!(
                                "channel {channel} cell {cell} target {target}: \
                                 choose/reconstruct={exact:?} cell_lower_bound={bound:?} \
                                 disagree on whether this target is legal"
                            ),
                        }
                    }
                }
            }
        }

        assert!(cells_total > 0, "the sweep must exercise real cells");
        eprintln!(
            "S8_PHASE_C_BOUND_NONZERO_RATE cells_total={cells_total} \
             cells_with_nonzero_rate_floor={cells_pruned}"
        );
    }

    /// Mirrors `residual_bits` (defined in `lib.rs`, not importable here)
    /// exactly, so the test above can compute the same "exact bits" ground
    /// truth `block_cost_bounded` does, without creating a `quantize.rs` ->
    /// `lib.rs` dependency for one four-line function.
    fn residual_bits_for_test(q: i32) -> u64 {
        if q == 0 {
            0
        } else {
            u64::from(32 - q.unsigned_abs().leading_zeros()) + 1
        }
    }

    #[test]
    fn the_neutral_cfl_factors_are_the_i23_defaults() {
        // kX is genuinely zero; kB is *one*, which is why B has to be
        // decorrelated on the way in.
        assert_eq!(neutral_cfl_factors(), (0.0, 1.0));
    }

    #[test]
    fn cfl_regression_refines_in_the_integer_wire_space() {
        let mut acc = CflAccumulator::default();
        // C = 0.5 * Y, and colour_factor = 84, so factor 42 is exact.
        for y in [-2.0f32, -1.0, 0.5, 3.0] {
            acc.add(y, 0.5 * y);
        }
        assert_eq!(acc.best_factor(0.0, DEFAULT_COLOUR_FACTOR, -128, 127), 42);
        assert_eq!(cfl_multiplier(0.0, 42, DEFAULT_COLOUR_FACTOR), 0.5);
    }

    #[test]
    fn a_degenerate_cfl_regression_stays_neutral() {
        let mut acc = CflAccumulator::default();
        for value in [-3.0f32, 0.0, 8.0] {
            acc.add(0.0, value);
        }
        assert_eq!(acc.best_factor(1.0, DEFAULT_COLOUR_FACTOR, -128, 127), 0);
    }
}
