//! The self-correcting (weighted) predictor, 18181-1 H.5.
//!
//! ```text
//! Table H.5 — WPHeader bundle
//! condition      type    default  name
//!                Bool()  true     default_wp
//! !default_wp    u(5)    16       wp_p1
//! !default_wp    u(5)    10       wp_p2
//! !default_wp    u(5)     7       wp_p3a
//! !default_wp    u(5)     7       wp_p3b
//! !default_wp    u(5)     7       wp_p3c
//! !default_wp    u(5)     0       wp_p3d
//! !default_wp    u(5)     0       wp_p3e
//! !default_wp    u(4)    13       wp_w0
//! !default_wp    u(4)    12       wp_w1
//! !default_wp    u(4)    12       wp_w2
//! !default_wp    u(4)    12       wp_w3
//! ```
//!
//! # What makes this predictor different
//!
//! Predictor 6 is not a function of the neighbouring *samples* alone; it is a
//! function of the neighbouring *prediction errors*. H.5.1 requires it to be
//! invoked for **every** sample of a channel, including samples whose MA leaf
//! selected a different predictor, because
//!
//! * its `max_error` output is property 15 of the MA context model (H.4.1), and
//! * its error state has to stay in step with the raster scan; skipping a
//!   sample would leave `err[i]_W` holding a value from two samples ago.
//!
//! So the decoding loop calls [`WeightedState::predict`] before every sample
//! and [`WeightedState::update`] after every sample, unconditionally.
//!
//! # The error state
//!
//! Two quantities are carried per already-decoded sample:
//!
//! * `true_err`, a signed value: `NarrowToI32(prediction - (true_value << 3))`,
//! * `err[i]` for `i` in `[0, 4)`, a magnitude:
//!   `(abs(subpred[i] - (true_value << 3)) + 3) >> 3`.
//!
//! `predict` reads them at `W`, `N`, `NW`, `NE` and `WW`, which spans exactly
//! two raster rows, so [`WeightedState`] keeps two rows and rotates them.
//!
//! H.5.2 substitutes at the edges, and the two substitutions are *different*:
//! a missing `W`, `N` or `WW` contributes 0, while a missing `NW` or `NE`
//! contributes the value at `N` (which is itself 0 on the first row).
//!
//! # Arithmetic width
//!
//! `true_err` is narrowed to 32 bits by the spec itself. Everything derived
//! from it is evaluated in `i64`, which is what H.1 says the predictor
//! computations need — with one exception. The final
//! `s * ((1 << 24) Idiv sum_weights)` can reach ~2^62 for extreme sample
//! values, which is inside `i64` but with no headroom at all, so that single
//! product is taken in `i128`. The result is identical whenever `i64` would
//! not have overflowed, and this decoder cannot panic on a crafted stream.

use jpxl_bitstream::{BitReader, read_bool, trace_field};
use jpxl_core::limits::AllocGuard;

use super::error::Result;
use super::predictor::Neighbours;

/// Whether `true_err` is derived from the clamped prediction.
///
/// H.5.1 says `true_err = NarrowToI32(prediction - (true value << 3))`, and
/// H.5.2 computes `prediction`, then conditionally clamps it. Whether "the
/// prediction" means the clamped value is the question.
///
/// **RESOLVED (slice 7): clamped.** Setting this to `false` breaks fixtures
/// 08, 11, 12 and 13, which are otherwise bit-exact against `djxl`.
pub const EXPERIMENT_TRUE_ERR_USES_CLAMPED: bool = true;

/// Which comparison H.5.2's `max_error` walk uses.
/// 0 = `abs(x) > abs(max)` (the clause as transcribed), 1 = `x < max`
/// (minimum), 2 = `x > max` (maximum), 3 = `abs(x) >= abs(max)`.
///
/// **RESOLVED (slice 8): 0, the clause as written.** Slice 7 recorded an
/// apparent contradiction — fixture 10 seemed to need a plain minimum — but
/// that was a symptom, not a cause: the `true_err` values fed into the walk
/// were wrong because of the H.5.2 clamp (see [`EXPERIMENT_CLAMP_SYMMETRIC`]).
/// With the clamp corrected, rule 0 decodes all nine handmade fixtures
/// bit-exactly, and rules 1, 2 and 3 each break at least one.
pub const EXPERIMENT_MAX_ERROR_RULE: u32 = 0;

/// Whether H.5.2's clamp is the single symmetric clamp the clause prints.
///
/// H.5.2 prints
///
/// ```text
/// if (((true_err_N * true_err_W) | (true_err_N * true_err_NW)) <= 0)
///     prediction = clamp(prediction, min(W3, N3, NE3), max(W3, N3, NE3));
/// ```
///
/// **That is contradicted by the oracle.** Four `cjxl`-produced streams pin
/// the behaviour down, and no reading of one guard plus one symmetric clamp
/// satisfies all four (the full derivation, with the exact samples, is in
/// `docs/experiments/2026-08-03-h52-clamp-asymmetry.md`):
///
/// * fixture 08 `(2, 0)`, `true_err = (W -8456, N 0, NW 0)`: the prediction
///   **must** be pulled down to `max(W3, N3, NE3)`;
/// * fixture 20 `(3, 0)`, `true_err = (W +864, N 0, NW 0)`: the prediction
///   **must not** be pulled up to `min(W3, N3, NE3)`;
/// * fixture 10 `(2, 0)`, `true_err = (W +2040, N 0, NW 0)`: likewise, and
///   here the wrong `true_err` is what corrupts property 15 downstream;
/// * fixture 21 `(251, 5)`, `true_err = (0, 0, 0)`: the prediction **must**
///   be pulled up to `min(W3, N3, NE3)`.
///
/// The first two differ only in the sign of `true_err_W`, so no guard that is
/// a function of the two printed products can separate them: the upper and
/// lower halves of the clamp are gated differently. What satisfies every
/// sample of all nine handmade fixtures plus the debug fixtures is
///
/// * cap above by `max(W3, N3, NE3)` when the printed guard holds
///   (`p1 <= 0 || p2 <= 0`), and
/// * floor below by `min(W3, N3, NE3)` only on a *strict* sign disagreement
///   (`p1 < 0 || p2 < 0`) or when `true_err_N`, `true_err_W` and
///   `true_err_NW` are all zero.
///
/// Setting this constant to `true` restores the literal clause; it breaks
/// fixtures 05, 09, 10 and 20. `[provisional]`: this describes libjxl 0.13.0's
/// behaviour, which the printed clause does not.
pub const EXPERIMENT_CLAMP_SYMMETRIC: bool = false;

/// Which extra term H.5.2 adds to `err_sum` on the last column.
/// 0 = none, 1 = `err[i]_W` (the LaTeX reading), 2 = `err[i]_N`.
///
/// **NOT EXERCISED (slice 7).** All three values decode the same six fixtures
/// bit-exactly and leave the same three failing; only the bit position at
/// which fixture 05 gives up moves (3205 / 3205 / 3201). The LaTeX reading is
/// kept because `part1.md` is unreadable here and the LaTeX is canonical.
pub const EXPERIMENT_ERR_SUM_LAST_COLUMN: u32 = 1;

/// Number of sub-predictors (H.5.1: `subpred[i]` with `i` in `[0, 4)`).
pub const NUM_SUBPREDICTORS: usize = 4;

/// Weight parameters of the self-correcting predictor (18181-1 Table H.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WpHeader {
    /// `wp_p1`, the `subpred[1]` error weight. `u(5)`, default 16.
    pub p1: i64,
    /// `wp_p2`, the `subpred[2]` error weight. `u(5)`, default 10.
    pub p2: i64,
    /// `wp_p3a`..`wp_p3e`, the five `subpred[3]` weights. `u(5)`, defaults
    /// 7, 7, 7, 0, 0.
    pub p3: [i64; 5],
    /// `wp_w0`..`wp_w3`, the per-sub-predictor max weights. `u(4)`, defaults
    /// 13, 12, 12, 12.
    pub w: [i64; NUM_SUBPREDICTORS],
}

impl WpHeader {
    /// The Table H.5 defaults, used when `default_wp` is true.
    #[must_use]
    pub const fn default_wp() -> Self {
        Self {
            p1: 16,
            p2: 10,
            p3: [7, 7, 7, 0, 0],
            w: [13, 12, 12, 12],
        }
    }

    /// Reads a `WPHeader` bundle (Table H.5).
    ///
    /// # Errors
    ///
    /// [`ModularError::Bitstream`](super::ModularError::Bitstream) at end of
    /// input.
    pub fn read(reader: &mut BitReader<'_>) -> Result<Self> {
        let default_wp = trace_field!(reader, "wp.default_wp", read_bool(reader))?;
        if default_wp {
            return Ok(Self::default_wp());
        }
        let p5 = |reader: &mut BitReader<'_>, name| -> Result<i64> {
            Ok(i64::from(trace_field!(reader, name, reader.read_bits(5))?))
        };
        let p1 = p5(reader, "wp.wp_p1")?;
        let p2 = p5(reader, "wp.wp_p2")?;
        let p3 = [
            p5(reader, "wp.wp_p3a")?,
            p5(reader, "wp.wp_p3b")?,
            p5(reader, "wp.wp_p3c")?,
            p5(reader, "wp.wp_p3d")?,
            p5(reader, "wp.wp_p3e")?,
        ];
        let w4 = |reader: &mut BitReader<'_>, name| -> Result<i64> {
            Ok(i64::from(trace_field!(reader, name, reader.read_bits(4))?))
        };
        let w = [
            w4(reader, "wp.wp_w0")?,
            w4(reader, "wp.wp_w1")?,
            w4(reader, "wp.wp_w2")?,
            w4(reader, "wp.wp_w3")?,
        ];
        Ok(Self { p1, p2, p3, w })
    }
}

impl Default for WpHeader {
    fn default() -> Self {
        Self::default_wp()
    }
}

/// One sample's worth of self-correcting predictor output (H.5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WeightedPrediction {
    /// The prediction before H.5.2's sign-disagreement clamp, kept so the
    /// slice-7 experiment can ask which one `true_err` is derived from.
    pub unclamped: i64,
    /// The `prediction` value, in the `<< 3` domain. Table H.3 row 6 turns it
    /// into a sample estimate with `(prediction + 3) >> 3`.
    pub prediction: i64,
    /// Property 15 of Table H.4: the largest-magnitude neighbouring `true_err`.
    pub max_error: i32,
    /// The four sub-predictions, kept so `update` can derive `err[i]`.
    pub subpred: [i64; NUM_SUBPREDICTORS],
}

/// Per-sample error state carried across the raster scan (H.5.1).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ErrorEntry {
    /// `NarrowToI32(prediction - (true_value << 3))`.
    true_err: i32,
    /// `(abs(subpred[i] - (true_value << 3)) + 3) >> 3`, non-negative.
    err: [u64; NUM_SUBPREDICTORS],
}

/// The rolling error state of the self-correcting predictor for one channel.
///
/// Create one per channel — the state is defined in terms of that channel's
/// own neighbours and must not leak across channels.
#[derive(Debug, Clone)]
pub struct WeightedState {
    width: u32,
    /// Errors of the row above the current one; empty semantics on row 0 are
    /// carried by `has_prev`.
    prev: Vec<ErrorEntry>,
    /// Errors of the current row, filled left to right.
    curr: Vec<ErrorEntry>,
    has_prev: bool,
}

impl WeightedState {
    /// Allocates state for a channel `width` samples across.
    ///
    /// Two rows of error entries are charged to `guard` before allocation.
    ///
    /// # Errors
    ///
    /// [`ModularError::Core`](super::ModularError::Core) if the two rows would
    /// exceed the guard's budget.
    pub fn new(width: u32, guard: &mut AllocGuard) -> Result<Self> {
        let bytes = u64::from(width)
            .saturating_mul(size_of::<ErrorEntry>() as u64)
            .saturating_mul(2);
        guard.charge(bytes)?;
        let len = width as usize;
        Ok(Self {
            width,
            prev: vec![ErrorEntry::default(); len],
            curr: vec![ErrorEntry::default(); len],
            has_prev: false,
        })
    }

    /// The `true_err` of the sample to the west, or 0 if it does not exist.
    fn true_err_w(&self, x: u32) -> i64 {
        if x > 0 {
            i64::from(
                self.curr
                    .get(x as usize - 1)
                    .copied()
                    .unwrap_or_default()
                    .true_err,
            )
        } else {
            0
        }
    }

    /// The `true_err` of the sample to the north, or 0 on the first row.
    fn true_err_n(&self, x: u32) -> i64 {
        if self.has_prev {
            i64::from(
                self.prev
                    .get(x as usize)
                    .copied()
                    .unwrap_or_default()
                    .true_err,
            )
        } else {
            0
        }
    }

    /// The `true_err` to the north-west, falling back to `N` (H.5.2).
    fn true_err_nw(&self, x: u32) -> i64 {
        if x > 0 && self.has_prev {
            i64::from(
                self.prev
                    .get(x as usize - 1)
                    .copied()
                    .unwrap_or_default()
                    .true_err,
            )
        } else {
            self.true_err_n(x)
        }
    }

    /// The `true_err` to the north-east, falling back to `N` (H.5.2).
    fn true_err_ne(&self, x: u32) -> i64 {
        if x + 1 < self.width && self.has_prev {
            i64::from(
                self.prev
                    .get(x as usize + 1)
                    .copied()
                    .unwrap_or_default()
                    .true_err,
            )
        } else {
            self.true_err_n(x)
        }
    }

    fn err_w(&self, x: u32, i: usize) -> u64 {
        if x > 0 {
            self.curr
                .get(x as usize - 1)
                .and_then(|e| e.err.get(i))
                .copied()
                .unwrap_or(0)
        } else {
            0
        }
    }

    fn err_ww(&self, x: u32, i: usize) -> u64 {
        if x > 1 {
            self.curr
                .get(x as usize - 2)
                .and_then(|e| e.err.get(i))
                .copied()
                .unwrap_or(0)
        } else {
            0
        }
    }

    fn err_n(&self, x: u32, i: usize) -> u64 {
        if self.has_prev {
            self.prev
                .get(x as usize)
                .and_then(|e| e.err.get(i))
                .copied()
                .unwrap_or(0)
        } else {
            0
        }
    }

    fn err_nw(&self, x: u32, i: usize) -> u64 {
        if x > 0 && self.has_prev {
            self.prev
                .get(x as usize - 1)
                .and_then(|e| e.err.get(i))
                .copied()
                .unwrap_or(0)
        } else {
            self.err_n(x, i)
        }
    }

    fn err_ne(&self, x: u32, i: usize) -> u64 {
        if x + 1 < self.width && self.has_prev {
            self.prev
                .get(x as usize + 1)
                .and_then(|e| e.err.get(i))
                .copied()
                .unwrap_or(0)
        } else {
            self.err_n(x, i)
        }
    }

    /// Scratch accessor for slice-7 oracle experiments: the four neighbouring
    /// `true_err` values and the four `err_sum` values at `x`.
    #[must_use]
    pub fn debug_state(&self, x: u32) -> ([i64; 4], [u64; 4]) {
        let te = [
            self.true_err_w(x),
            self.true_err_n(x),
            self.true_err_nw(x),
            self.true_err_ne(x),
        ];
        let mut sums = [0u64; 4];
        for (i, slot) in sums.iter_mut().enumerate() {
            let mut sum = self.err_n(x, i)
                + self.err_w(x, i)
                + self.err_nw(x, i)
                + self.err_ww(x, i)
                + self.err_ne(x, i);
            if x + 1 == self.width {
                sum += self.err_w(x, i);
            }
            *slot = sum;
        }
        (te, sums)
    }

    /// Computes `prediction`, `max_error` and `subpred[0..4)` for `(x, y)`.
    ///
    /// `nb` must be the H.3 neighbourhood of the same sample in the same
    /// channel; H.5.2's `N3`, `NW3`, ... are exactly those neighbours shifted
    /// left by three.
    #[must_use]
    pub fn predict(&self, header: &WpHeader, nb: &Neighbours, x: u32) -> WeightedPrediction {
        let (n3, nw3, ne3, w3, nn3) = (nb.n << 3, nb.nw << 3, nb.ne << 3, nb.w << 3, nb.nn << 3);

        let te_w = self.true_err_w(x);
        let te_n = self.true_err_n(x);
        let te_nw = self.true_err_nw(x);
        let te_ne = self.true_err_ne(x);

        // H.5.2, verbatim.
        let subpred = [
            w3 + ne3 - n3,
            n3 - (((te_w + te_n + te_ne) * header.p1) >> 5),
            w3 - (((te_w + te_n + te_nw) * header.p2) >> 5),
            n3 - ((te_nw * header.p3[0]
                + te_n * header.p3[1]
                + te_ne * header.p3[2]
                + (nn3 - n3) * header.p3[3]
                + (nw3 - w3) * header.p3[4])
                >> 5),
        ];

        let mut weight = [0i64; NUM_SUBPREDICTORS];
        for (i, slot) in weight.iter_mut().enumerate() {
            // err_sum[i] = (err_N + err_W + err_NW + err_WW + err_NE) Umod 2^32
            let sum = self
                .err_n(x, i)
                .wrapping_add(self.err_w(x, i))
                .wrapping_add(self.err_nw(x, i))
                .wrapping_add(self.err_ww(x, i))
                .wrapping_add(self.err_ne(x, i));
            // H.5.2: on the last column an extra term is added. Which one is
            // EXPERIMENT_ERR_SUM_LAST_COLUMN.
            let sum = if x + 1 == self.width {
                match EXPERIMENT_ERR_SUM_LAST_COLUMN {
                    0 => sum,
                    1 => sum.wrapping_add(self.err_w(x, i)),
                    _ => sum.wrapping_add(self.err_n(x, i)),
                }
            } else {
                sum
            };
            #[expect(
                clippy::cast_possible_truncation,
                reason = "H.5.2 specifies `Umod (1 << 32)` on err_sum; truncation is the operation"
            )]
            let err_sum = sum as u32;
            *slot = error2weight(err_sum, header.w.get(i).copied().unwrap_or(0));
        }

        // Every error2weight result is at least 4, so sum_weights >= 16, hence
        // log_weight >= 5 and the shift below is never negative. The same bound
        // makes the final divisor non-zero.
        let sum_weights: i64 = weight.iter().sum();
        let log_weight = floor_log2(sum_weights.max(1) as u64) + 1;
        let shift = log_weight.saturating_sub(5);
        for slot in &mut weight {
            *slot >>= shift;
        }
        let sum_weights: i64 = weight.iter().sum();
        let divisor = sum_weights.max(1);

        let mut s = (sum_weights >> 1) - 1;
        for (sp, wt) in subpred.iter().zip(weight.iter()) {
            s += sp * wt;
        }
        // The one place `i64` has no headroom; see the module documentation.
        let wide = (i128::from(s) * i128::from((1i64 << 24) / divisor)) >> 24;
        // Saturation is unreachable for any sample set H.1 admits, but it keeps
        // the decoder total instead of relying on that.
        let prediction = i64::try_from(wide).unwrap_or(i64::MAX);

        // H.5.2's clamp. The clause prints one symmetric clamp behind one
        // guard; oracle evidence contradicts that (see
        // `EXPERIMENT_CLAMP_SYMMETRIC` and
        // `docs/experiments/2026-08-03-h52-clamp-asymmetry.md`), so the two
        // halves of the range are gated separately.
        //
        // The products are taken in `i64`, so two `i32` errors cannot overflow.
        let p1 = te_n * te_w;
        let p2 = te_n * te_nw;
        // The printed guard, `((p1) | (p2)) <= 0`, read as the pair of
        // comparisons its prose describes.
        let disagree = p1 <= 0 || p2 <= 0;
        // A real sign disagreement: some product is strictly negative.
        let disagree_strict = p1 < 0 || p2 < 0;
        // No error information at all in the three neighbours.
        let quiescent = te_n == 0 && te_w == 0 && te_nw == 0;

        let unclamped = prediction;
        let lo = w3.min(n3).min(ne3);
        let hi = w3.max(n3).max(ne3);
        let prediction = if EXPERIMENT_CLAMP_SYMMETRIC {
            if disagree {
                prediction.clamp(lo, hi)
            } else {
                prediction
            }
        } else {
            let capped = if disagree {
                prediction.min(hi)
            } else {
                prediction
            };
            if disagree_strict || quiescent {
                capped.max(lo)
            } else {
                capped
            }
        };

        // max_error = the neighbouring true_err of largest magnitude, tested in
        // the order W, N, NW, NE with strict `>` so ties keep the earlier one.
        let mut max_error = te_w;
        for candidate in [te_n, te_nw, te_ne] {
            let beats = match EXPERIMENT_MAX_ERROR_RULE {
                0 => candidate.abs() > max_error.abs(),
                1 => candidate < max_error,
                2 => candidate > max_error,
                _ => candidate.abs() >= max_error.abs(),
            };
            if beats {
                max_error = candidate;
            }
        }

        WeightedPrediction {
            unclamped,
            prediction,
            max_error: narrow_to_i32(max_error),
            subpred,
        }
    }

    /// Records the errors of the sample just decoded at `(x, _)`.
    ///
    /// Must be called once per sample, immediately after the sample value is
    /// known and before advancing `x`.
    pub fn update(&mut self, x: u32, wp: &WeightedPrediction, true_value: i32) {
        let shifted = i64::from(true_value) << 3;
        let source = if EXPERIMENT_TRUE_ERR_USES_CLAMPED {
            wp.prediction
        } else {
            wp.unclamped
        };
        let mut entry = ErrorEntry {
            true_err: narrow_to_i32(source - shifted),
            err: [0; NUM_SUBPREDICTORS],
        };
        for (slot, sp) in entry.err.iter_mut().zip(wp.subpred.iter()) {
            // err[i] = (abs(subpred[i] - (true_value << 3)) + 3) >> 3
            let magnitude = (sp - shifted).abs();
            *slot = u64::try_from((magnitude + 3) >> 3).unwrap_or(u64::MAX);
        }
        if let Some(cell) = self.curr.get_mut(x as usize) {
            *cell = entry;
        }
    }

    /// Rotates the current row into the "row above" slot. Call at end of row.
    pub fn advance_row(&mut self) {
        core::mem::swap(&mut self.prev, &mut self.curr);
        for cell in &mut self.curr {
            *cell = ErrorEntry::default();
        }
        self.has_prev = true;
    }
}

/// `NarrowToI32(x)` of 18181-1 4.2: the low 32 bits read as two's complement.
#[must_use]
pub const fn narrow_to_i32(x: i64) -> i32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "NarrowToI32 is defined as exactly this truncation (18181-1 4.2)"
    )]
    {
        x as i32
    }
}

/// `floor(log2(v))` for `v >= 1`.
const fn floor_log2(v: u64) -> u32 {
    63 - v.leading_zeros()
}

/// `error2weight(err_sum, maxweight)` of H.5.2.
///
/// ```text
/// shift = floor(log2(err_sum + 1)) - 5;
/// if (shift < 0) shift = 0;
/// return 4 + ((maxweight * ((1 << 24) Idiv ((err_sum >> shift) + 1))) >> shift);
/// ```
///
/// The `+ 1` inside the logarithm is what keeps `err_sum == 0` legal. After
/// the shift the denominator lies in `[1, 64]`, which is the range the H.5.2
/// note relies on for a table-driven implementation.
#[must_use]
pub fn error2weight(err_sum: u32, maxweight: i64) -> i64 {
    let shift = floor_log2(u64::from(err_sum) + 1).saturating_sub(5);
    let denominator = i64::from(err_sum >> shift) + 1;
    4 + ((maxweight * ((1i64 << 24) / denominator)) >> shift)
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

    #[test]
    fn default_header_matches_table_h5() {
        let h = WpHeader::default_wp();
        assert_eq!(h.p1, 16);
        assert_eq!(h.p2, 10);
        assert_eq!(h.p3, [7, 7, 7, 0, 0]);
        assert_eq!(h.w, [13, 12, 12, 12]);
    }

    #[test]
    fn default_wp_bit_costs_exactly_one_bit() {
        // Bool() true -> the eleven u(5)/u(4) fields are absent.
        let data = [0b0000_0001u8];
        let mut r = BitReader::new(&data);
        let h = WpHeader::read(&mut r).expect("default_wp");
        assert_eq!(h, WpHeader::default_wp());
        assert_eq!(r.total_bits_read(), 1);
    }

    #[test]
    fn explicit_header_reads_seven_u5_then_four_u4() {
        // default_wp = 0, then p1..p3e = 1,2,3,4,5,6,7 as u(5), then
        // w0..w3 = 8,9,10,11 as u(4). Total 1 + 35 + 16 = 52 bits.
        let mut bits: Vec<u8> = Vec::new();
        let mut push = |value: u32, n: u32| {
            for i in 0..n {
                bits.push(((value >> i) & 1) as u8);
            }
        };
        push(0, 1);
        for v in 1..=7u32 {
            push(v, 5);
        }
        for v in 8..=11u32 {
            push(v, 4);
        }
        let mut bytes = vec![0u8; bits.len().div_ceil(8)];
        for (i, bit) in bits.iter().enumerate() {
            bytes[i / 8] |= bit << (i % 8);
        }

        let mut r = BitReader::new(&bytes);
        let h = WpHeader::read(&mut r).expect("explicit wp header");
        assert_eq!(h.p1, 1);
        assert_eq!(h.p2, 2);
        assert_eq!(h.p3, [3, 4, 5, 6, 7]);
        assert_eq!(h.w, [8, 9, 10, 11]);
        assert_eq!(r.total_bits_read(), 52);
    }

    #[test]
    fn error2weight_hand_computed() {
        // err_sum = 0, maxweight = 13:
        //   shift = floor(log2(1)) - 5 = 0 - 5 -> clamped to 0
        //   denominator = (0 >> 0) + 1 = 1
        //   4 + ((13 * (1<<24)) >> 0) = 4 + 218103808 = 218103812
        assert_eq!(error2weight(0, 13), 4 + 13 * (1 << 24));

        // err_sum = 31, maxweight = 12:
        //   floor(log2(32)) = 5, shift = 0
        //   denominator = 32; (1<<24) Idiv 32 = 524288
        //   4 + 12 * 524288 = 4 + 6291456 = 6291460
        assert_eq!(error2weight(31, 12), 6_291_460);

        // err_sum = 64, maxweight = 12:
        //   floor(log2(65)) = 6, shift = 1
        //   denominator = (64 >> 1) + 1 = 33; (1<<24) Idiv 33 = 508400
        //   4 + ((12 * 508400) >> 1) = 4 + (6100800 >> 1) = 4 + 3050400
        assert_eq!(error2weight(64, 12), 3_050_404);

        // A maxweight of zero still yields the floor of 4.
        assert_eq!(error2weight(12345, 0), 4);
    }

    #[test]
    fn error2weight_denominator_stays_in_the_documented_range() {
        // The H.5.2 note asserts the divisor is an integer in [1, 64]. Prove it
        // across the whole exponent range rather than trusting the note.
        for exp in 0..32u32 {
            for delta in [0u64, 1, 2, 7] {
                let e = (1u64 << exp).saturating_add(delta).min(u64::from(u32::MAX));
                let err_sum = e as u32;
                let shift = floor_log2(u64::from(err_sum) + 1).saturating_sub(5);
                let denominator = u64::from(err_sum >> shift) + 1;
                assert!(
                    (1..=64).contains(&denominator),
                    "err_sum {err_sum} gave denominator {denominator}"
                );
            }
        }
    }

    #[test]
    fn narrow_to_i32_takes_the_low_thirty_two_bits() {
        assert_eq!(narrow_to_i32(0), 0);
        assert_eq!(narrow_to_i32(-1), -1);
        assert_eq!(narrow_to_i32(i64::from(i32::MAX) + 1), i32::MIN);
        assert_eq!(narrow_to_i32(1 << 32), 0);
        assert_eq!(narrow_to_i32((1 << 32) + 5), 5);
    }

    /// The first sample of a channel: no errors exist anywhere, so every
    /// `true_err_*` and `err[i]_*` is zero and the sub-predictors reduce to
    /// their sample terms.
    #[test]
    fn first_sample_reduces_to_the_sample_terms() {
        let state = WeightedState::new(4, &mut guard()).expect("4 wide");
        let h = WpHeader::default_wp();
        // At the origin all neighbours are 0 (H.3), so every subpred is 0.
        let wp = state.predict(&h, &Neighbours::default(), 0);
        assert_eq!(wp.subpred, [0, 0, 0, 0]);
        assert_eq!(wp.max_error, 0);
        // All errors zero -> every weight is error2weight(0, w_i).
        // w = [13, 12, 12, 12] -> [4 + 13*2^24, 4 + 12*2^24, ...]
        // sum = 16 + 49*2^24 = 822083600; floor(log2) = 29, log_weight = 30,
        // shift = 25. Each weight >> 25: 218103812>>25 = 6, 201326596>>25 = 5.
        // sum_weights = 6 + 5 + 5 + 5 = 21. s = (21 >> 1) - 1 = 9.
        // s += 0 for every subpred, so s = 9.
        // prediction = (9 * ((1<<24) Idiv 21)) >> 24 = (9 * 798915) >> 24
        //            = 7190235 >> 24 = 0.
        assert_eq!(wp.prediction, 0);
    }

    /// A three-row worked example on a 3-wide channel that asserts the
    /// *error state*, not just the outputs.
    ///
    /// The channel is driven with a constant true value of 8 and a
    /// neighbourhood that is also constant at 8, which makes the arithmetic
    /// hand-checkable while still exercising both rows of state.
    #[test]
    fn error_state_evolves_across_rows() {
        let h = WpHeader::default_wp();
        let mut state = WeightedState::new(3, &mut guard()).expect("3 wide");

        // Row 0, x = 0. No neighbours at all: everything is zero, prediction 0
        // (as proved by `first_sample_reduces_to_the_sample_terms`).
        let wp = state.predict(&h, &Neighbours::default(), 0);
        assert_eq!(wp.prediction, 0);
        // True value 8 -> shifted = 64.
        //   true_err = NarrowToI32(0 - 64) = -64
        //   err[i]   = (abs(0 - 64) + 3) >> 3 = 67 >> 3 = 8, for all i
        state.update(0, &wp, 8);
        assert_eq!(state.curr[0].true_err, -64);
        assert_eq!(state.curr[0].err, [8, 8, 8, 8]);

        // Row 0, x = 1. Now W exists. Per H.3 on the first row N = NW = NE =
        // NN = NEE = W, and W = c(0,0) = 8. All neighbours are 8, so
        // W3 = N3 = NE3 = NN3 = NW3 = 64.
        let nb = Neighbours {
            w: 8,
            n: 8,
            nw: 8,
            ne: 8,
            nn: 8,
            nee: 8,
            ww: 8,
        };
        // true_err_W = -64; true_err_N = 0 (no row above); true_err_NW and
        // true_err_NE both fall back to N, so both are 0.
        //   subpred[0] = 64 + 64 - 64 = 64
        //   subpred[1] = 64 - (((-64 + 0 + 0) * 16) >> 5) = 64 - (-1024 >> 5)
        //              = 64 - (-32) = 96
        //   subpred[2] = 64 - (((-64 + 0 + 0) * 10) >> 5) = 64 - (-640 >> 5)
        //              = 64 - (-20) = 84
        //     (-640 >> 5 is floor(-20.0) = -20 exactly.)
        //   subpred[3] = 64 - ((0*7 + 0*7 + 0*7 + 0*0 + 0*0) >> 5) = 64
        let wp = state.predict(&h, &nb, 1);
        assert_eq!(wp.subpred, [64, 96, 84, 64]);
        // max_error walks W, N, NW, NE with strict `>`: starts at
        // true_err_W = -64 and nothing beats |-64|, so max_error = -64.
        assert_eq!(wp.max_error, -64);
        state.update(1, &wp, 8);

        // Row 0, x = 2 (the last column) — this is where err_sum double-counts
        // the west error per H.5.2.
        let wp2 = state.predict(&h, &nb, 2);
        // Independently recompute err_sum[0] for x = 2:
        //   err_N = 0 (no row above), err_W = err[0] at x=1, err_NW = err_N = 0,
        //   err_WW = err[0] at x=0 = 8, err_NE = err_N = 0, plus the last
        //   column rule adds err_W again.
        let err_w0 = state.curr[1].err[0];
        let expected_sum = err_w0 + 8 + err_w0;
        // Re-derive weight[0] from that sum and confirm it is what predict used
        // by checking the whole prediction against a manual replay.
        let manual = replay_predict(&state, &h, &nb, 2);
        assert_eq!(wp2.prediction, manual, "err_sum[0] = {expected_sum}");

        // End of row 0: the current row becomes the row above.
        state.advance_row();
        assert!(state.has_prev);
        assert_eq!(state.prev[0].true_err, -64, "row 0 errors are now `N`");
        assert_eq!(
            state.curr[0],
            ErrorEntry::default(),
            "the new current row starts clean"
        );

        // Row 1, x = 0: true_err_N is now the row-0 value, true_err_W is 0
        // (no sample to the west), and true_err_NW falls back to N = -64.
        // true_err_NE is the row-0 value at x = 1.
        let nb_row1 = Neighbours {
            w: 8,
            n: 8,
            nw: 8,
            ne: 8,
            nn: 8,
            nee: 8,
            ww: 8,
        };
        let wp3 = state.predict(&h, &nb_row1, 0);
        let te_n = i64::from(state.prev[0].true_err);
        let te_ne = i64::from(state.prev[1].true_err);
        // subpred[1] = N3 - (((true_err_W + true_err_N + true_err_NE) * 16) >> 5)
        //            = 64 - (((0 + te_n + te_ne) * 16) >> 5)
        let te_w = 0i64; // no sample to the west of column 0
        assert_eq!(wp3.subpred[1], 64 - (((te_w + te_n + te_ne) * 16) >> 5));
        // subpred[2] = W3 - (((true_err_W + true_err_N + true_err_NW) * 10) >> 5)
        // with true_err_NW == true_err_N here.
        assert_eq!(wp3.subpred[2], 64 - (((te_w + te_n + te_n) * 10) >> 5));
        // max_error starts at true_err_W = 0 and is beaten by true_err_N.
        assert_eq!(i64::from(wp3.max_error), te_n);
    }

    /// Recomputes `predict`'s prediction from the public formula, independently
    /// of the implementation's loop structure.
    fn replay_predict(state: &WeightedState, h: &WpHeader, nb: &Neighbours, x: u32) -> i64 {
        let (n3, nw3, ne3, w3, nn3) = (nb.n << 3, nb.nw << 3, nb.ne << 3, nb.w << 3, nb.nn << 3);
        let te_w = state.true_err_w(x);
        let te_n = state.true_err_n(x);
        let te_nw = state.true_err_nw(x);
        let te_ne = state.true_err_ne(x);
        let subpred = [
            w3 + ne3 - n3,
            n3 - (((te_w + te_n + te_ne) * h.p1) >> 5),
            w3 - (((te_w + te_n + te_nw) * h.p2) >> 5),
            n3 - ((te_nw * h.p3[0]
                + te_n * h.p3[1]
                + te_ne * h.p3[2]
                + (nn3 - n3) * h.p3[3]
                + (nw3 - w3) * h.p3[4])
                >> 5),
        ];
        let mut weight = [0i64; 4];
        for (i, slot) in weight.iter_mut().enumerate() {
            let mut sum = state.err_n(x, i)
                + state.err_w(x, i)
                + state.err_nw(x, i)
                + state.err_ww(x, i)
                + state.err_ne(x, i);
            if x + 1 == state.width {
                sum += state.err_w(x, i);
            }
            *slot = error2weight(sum as u32, h.w[i]);
        }
        let sum: i64 = weight.iter().sum();
        let shift = (floor_log2(sum as u64) + 1) - 5;
        for w in &mut weight {
            *w >>= shift;
        }
        let sum: i64 = weight.iter().sum();
        let mut s = (sum >> 1) - 1;
        for (sp, wt) in subpred.iter().zip(weight.iter()) {
            s += sp * wt;
        }
        let prediction = ((i128::from(s) * i128::from((1i64 << 24) / sum)) >> 24) as i64;
        // The asymmetric clamp of §4.3 of
        // docs/experiments/2026-08-03-h52-clamp-asymmetry.md, written out
        // independently of `predict`'s control flow.
        let (p1, p2) = (te_n * te_w, te_n * te_nw);
        let (lo, hi) = (w3.min(n3).min(ne3), w3.max(n3).max(ne3));
        let capped = if p1 <= 0 || p2 <= 0 {
            prediction.min(hi)
        } else {
            prediction
        };
        if p1 < 0 || p2 < 0 || (te_n == 0 && te_w == 0 && te_nw == 0) {
            capped.max(lo)
        } else {
            capped
        }
    }

    #[test]
    fn clamping_triggers_only_on_mixed_error_signs() {
        // Craft a state where true_err_N and true_err_W share a sign and
        // true_err_NW does too, so the clamp is skipped; then flip one sign.
        let h = WpHeader::default_wp();
        let mut state = WeightedState::new(2, &mut guard()).expect("2 wide");

        // Fill row 0 with a known positive true_err, then move to row 1.
        state.curr[0] = ErrorEntry {
            true_err: 5,
            err: [1; 4],
        };
        state.curr[1] = ErrorEntry {
            true_err: 5,
            err: [1; 4],
        };
        state.advance_row();
        state.curr[0] = ErrorEntry {
            true_err: 5,
            err: [1; 4],
        };

        // At (1, 1): true_err_W = 5, true_err_N = 5, true_err_NW = 5.
        // (5*5) | (5*5) = 25 > 0, so no clamp.
        let nb = Neighbours {
            w: 0,
            n: 0,
            nw: 0,
            ne: 0,
            nn: 0,
            nee: 0,
            ww: 0,
        };
        let unclamped = state.predict(&h, &nb, 1);

        // Flip true_err_W negative: (5 * -5) is negative, so the clamp applies
        // and the prediction is forced into [min(W3,N3,NE3), max(...)] = [0, 0].
        state.curr[0].true_err = -5;
        let clamped = state.predict(&h, &nb, 1);
        assert_eq!(clamped.prediction, 0, "clamped into the degenerate range");
        assert_ne!(
            unclamped.prediction, clamped.prediction,
            "the sign test must actually change the result"
        );
    }

    /// The two halves of H.5.2's clamp are gated differently, and this is the
    /// pair of cases that proves it — fixtures 08 and 20 reduced to their
    /// arithmetic. Both have `true_err_N == true_err_NW == 0`, so the guard
    /// the clause prints sees identical inputs; they differ only in the sign
    /// of `true_err_W`, and they require opposite outcomes.
    ///
    /// See `docs/experiments/2026-08-03-h52-clamp-asymmetry.md`.
    #[test]
    fn the_clamp_caps_above_but_does_not_floor_below_on_a_zero_north_error() {
        let h = WpHeader::default_wp();

        // Row 0 of a 4-wide channel, sitting at x = 1, so N/NW/NE all fall
        // back to W (Table H.2) and every neighbouring error but W is zero.
        let mut over = WeightedState::new(4, &mut guard()).expect("4 wide");
        over.curr[0] = ErrorEntry {
            true_err: -8456,
            err: [1057; 4],
        };
        let nb = Neighbours {
            w: 1057,
            n: 1057,
            nw: 1057,
            ne: 1057,
            nn: 1057,
            nee: 1057,
            ww: 1057,
        };
        let wp = over.predict(&h, &nb, 1);
        // A negative west error makes the sub-predictors extrapolate upward,
        // above max(W3, N3, NE3) = 8456 — and the cap brings it back.
        assert!(wp.unclamped > 8456, "unclamped {}", wp.unclamped);
        assert_eq!(wp.prediction, 8456, "the upper bound must apply");

        // Same shape, opposite sign: the extrapolation goes below
        // min(W3, N3, NE3) = 576 and must be left there.
        let mut under = WeightedState::new(4, &mut guard()).expect("4 wide");
        under.curr[0] = ErrorEntry {
            true_err: 864,
            err: [108, 144, 130, 108],
        };
        let nb = Neighbours {
            w: 72,
            n: 72,
            nw: 72,
            ne: 72,
            nn: 72,
            nee: 72,
            ww: 72,
        };
        let wp = under.predict(&h, &nb, 1);
        assert!(wp.unclamped < 576, "unclamped {}", wp.unclamped);
        assert_eq!(
            wp.prediction, wp.unclamped,
            "the lower bound must NOT apply when only true_err_W is non-zero"
        );
    }

    /// The other half: with no error information at all, the lower bound is
    /// applied after all (fixture 21, coded channel 2 at `(251, 5)`).
    #[test]
    fn the_clamp_floors_below_when_every_neighbouring_error_is_zero() {
        let h = WpHeader::default_wp();
        let mut state = WeightedState::new(4, &mut guard()).expect("4 wide");
        // Two clean rows: every true_err is zero, but sub-predictor 0 has been
        // far more accurate than the others, so it takes nearly all the weight
        // — which is what lets the prediction leave the neighbour range at all.
        for cell in state.prev.iter_mut().chain(state.curr.iter_mut()) {
            *cell = ErrorEntry {
                true_err: 0,
                err: [0, 4000, 4000, 4000],
            };
        }
        state.has_prev = true;
        // W and N are large and NE is far below them, so subpred[0] =
        // W3 + NE3 - N3 drags the weighted prediction under min(W3, N3, NE3).
        let nb = Neighbours {
            w: 250,
            n: 251,
            nw: 250,
            ne: -4,
            nn: 250,
            nee: -4,
            ww: 249,
        };
        let wp = state.predict(&h, &nb, 1);
        assert!(wp.unclamped < -32, "unclamped {}", wp.unclamped);
        assert_eq!(wp.prediction, -32, "min(W3, N3, NE3) must apply");
    }

    #[test]
    fn max_error_picks_the_largest_magnitude_in_w_n_nw_ne_order() {
        let h = WpHeader::default_wp();
        let mut state = WeightedState::new(3, &mut guard()).expect("3 wide");
        state.prev[0] = ErrorEntry {
            true_err: -30,
            err: [0; 4],
        };
        state.prev[1] = ErrorEntry {
            true_err: 7,
            err: [0; 4],
        };
        state.prev[2] = ErrorEntry {
            true_err: 100,
            err: [0; 4],
        };
        state.has_prev = true;
        state.curr[0] = ErrorEntry {
            true_err: 20,
            err: [0; 4],
        };
        // At x = 1: W = 20, N = prev[1] = 7, NW = prev[0] = -30, NE = prev[2]
        // = 100. Largest magnitude is 100.
        let wp = state.predict(&h, &Neighbours::default(), 1);
        assert_eq!(wp.max_error, 100);
    }
}
