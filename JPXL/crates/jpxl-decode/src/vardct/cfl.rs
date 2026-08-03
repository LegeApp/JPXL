//! Chroma from luma (18181-1 I.6): per-tile X/B correlation factors and their
//! application to dequantized coefficients.
//!
//! I.6's own text (paraphrased per AGENTS.md — no ISO text is quoted):
//!
//! * Skipped entirely if any channel is subsampled.
//! * `kX = base_correlation_x + x_factor / colour_factor`, and likewise
//!   `kB = base_correlation_b + b_factor / colour_factor`, both computed from
//!   I.2.3's `LfChannelCorrelation` bundle.
//! * The reconstruction is `Y = dY; X = dX + kX*Y; B = dB + kB*Y` — Y itself
//!   is never modified, only X and B borrow from it.
//! * **Where it applies** (the part that matters for the caller): I.6 gives
//!   two different sources for `x_factor`/`b_factor` depending on which
//!   coefficients are being reconstructed. For **LF** coefficients,
//!   `x_factor`/`b_factor` are the single frame-wide constants
//!   `x_factor_lf - 128` / `b_factor_lf - 128` (I.2.3) — one `(kX, kB)` pair
//!   for the entire LF plane, applied once dequantization (I.5.2) has
//!   produced `dX, dY, dB`, and *before* the I.5.2 adaptive smoothing pass
//!   (I.5.2 explicitly runs smoothing "after ... applying chroma from
//!   luma"). For **HF** coefficients, `x_factor`/`b_factor` instead come
//!   from the `XFromY`/`BFromY` planes of G.2.4's `HfMetadata`, sampled once
//!   per 64x64-sample tile containing the coefficient's position — so HF
//!   reconstruction uses a *different* `(kX, kB)` pair per tile, applied by
//!   I.5.3's HF dequantization pipeline (a later slice's job to wire in).
//!
//! [`CflFactors`] models both sources behind one `at(tile_x, tile_y)`
//! lookup so a caller (8F, for the HF path; this module's own [`crate::vardct::lf`]
//! for the LF path) does not need to know which case it is in. [`apply`] is
//! I.6's linear reconstruction itself, shared by both.

use crate::modular::Channel;

use super::quantizer::LfChannelCorrelation;

/// Side length, in samples, of the rectangle each HF `XFromY`/`BFromY`
/// sample covers (I.6: "the 64x64 rectangle containing the current sample").
pub const CFL_TILE_DIM: u32 = 64;

/// I.6's per-tile chroma-from-luma correlation factors.
///
/// One value covers the whole frame for the LF case ([`CflFactors::for_lf`]);
/// one value per 64x64 tile for the HF case ([`CflFactors::for_hf`]). Both are
/// queried through the same [`CflFactors::at`].
#[derive(Debug, Clone)]
pub enum CflFactors {
    /// I.6's LF case: `x_factor`/`b_factor` are the frame-wide constants
    /// `x_factor_lf - 128` / `b_factor_lf - 128` (I.2.3), so `(kX, kB)` is a
    /// single pair for every position in the LF plane.
    Lf {
        /// `kX = base_correlation_x + x_factor_lf' / colour_factor`.
        k_x: f32,
        /// `kB = base_correlation_b + b_factor_lf' / colour_factor`.
        k_b: f32,
    },
    /// I.6's HF case: `x_factor`/`b_factor` are read from `XFromY`/`BFromY`
    /// (G.2.4), one sample per 64x64 tile.
    Hf {
        /// `base_correlation_x` (I.2.3).
        base_correlation_x: f32,
        /// `base_correlation_b` (I.2.3).
        base_correlation_b: f32,
        /// `colour_factor` (I.2.3), widened to `f32` once here rather than
        /// at every tile lookup.
        colour_factor: f32,
        /// `XFromY` (G.2.4): one `x_factor` sample per 64x64 tile.
        x_from_y: Channel,
        /// `BFromY` (G.2.4): one `b_factor` sample per 64x64 tile.
        b_from_y: Channel,
    },
}

impl CflFactors {
    /// Builds the LF case's single `(kX, kB)` pair from I.2.3's
    /// `LfChannelCorrelation` bundle.
    ///
    /// Per I.6, the LF `x_factor`/`b_factor` are `x_factor_lf - 128` and
    /// `b_factor_lf - 128` — exactly [`LfChannelCorrelation::x_factor`] and
    /// [`LfChannelCorrelation::b_factor`].
    #[must_use]
    pub fn for_lf(corr: &LfChannelCorrelation) -> Self {
        let colour_factor = colour_factor_f32(corr.colour_factor);
        Self::Lf {
            k_x: corr.base_correlation_x + correlation_ratio(corr.x_factor(), colour_factor),
            k_b: corr.base_correlation_b + correlation_ratio(corr.b_factor(), colour_factor),
        }
    }

    /// Builds the HF case from I.2.3's bundle plus G.2.4's `XFromY`/`BFromY`
    /// planes (one sample per 64x64 tile of the LF group).
    #[must_use]
    pub fn for_hf(corr: &LfChannelCorrelation, x_from_y: Channel, b_from_y: Channel) -> Self {
        Self::Hf {
            base_correlation_x: corr.base_correlation_x,
            base_correlation_b: corr.base_correlation_b,
            colour_factor: colour_factor_f32(corr.colour_factor),
            x_from_y,
            b_from_y,
        }
    }

    /// `(kX, kB)` at 64x64 tile `(tile_x, tile_y)`.
    ///
    /// The [`Self::Lf`] variant ignores the tile coordinate — I.6 states a
    /// single constant pair for the whole LF plane.
    #[must_use]
    pub fn at(&self, tile_x: u32, tile_y: u32) -> (f32, f32) {
        match self {
            Self::Lf { k_x, k_b } => (*k_x, *k_b),
            Self::Hf {
                base_correlation_x,
                base_correlation_b,
                colour_factor,
                x_from_y,
                b_from_y,
            } => {
                let x_factor = x_from_y.get(tile_x, tile_y);
                let b_factor = b_from_y.get(tile_x, tile_y);
                (
                    base_correlation_x + correlation_ratio(x_factor, *colour_factor),
                    base_correlation_b + correlation_ratio(b_factor, *colour_factor),
                )
            }
        }
    }

    /// [`Self::at`], from a pixel coordinate: tile `(x / 64, y / 64)`.
    #[must_use]
    pub fn at_pixel(&self, x: u32, y: u32) -> (f32, f32) {
        self.at(x / CFL_TILE_DIM, y / CFL_TILE_DIM)
    }
}

/// `colour_factor` widened to `f32`. Bounded well under `2^24` by I.2.3's
/// widest `U32` selector (`258 + u(16) <= 65793`), so the conversion is exact.
#[allow(
    clippy::cast_precision_loss,
    reason = "colour_factor is bounded far below f32's 24-bit exact-integer \
              range by I.2.3's widest U32 selector"
)]
fn colour_factor_f32(colour_factor: u32) -> f32 {
    colour_factor as f32
}

/// `factor / colour_factor`, 8B's invariant (`colour_factor >= 2` for every
/// I.2.3 selector) ruling out division by zero.
fn correlation_ratio(factor: i32, colour_factor: f32) -> f32 {
    factor as f32 / colour_factor
}

/// I.6's linear chroma-from-luma reconstruction: `Y = dY; X = dX + kX*Y;
/// B = dB + kB*Y`. Returns `(X, Y, B)`.
#[must_use]
pub fn apply(dx: f32, dy: f32, db: f32, k_x: f32, k_b: f32) -> (f32, f32, f32) {
    let y = dy;
    (dx + k_x * y, y, db + k_b * y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::ChannelSpec;

    #[test]
    fn for_lf_matches_the_hand_computed_formula() {
        // colour_factor = 4, x_factor_lf = 132 -> x_factor = 4,
        // b_factor_lf = 118 -> b_factor = -10; base_correlation_x = 0.5,
        // base_correlation_b = -0.25.
        // kX = 0.5 + 4/4 = 1.5; kB = -0.25 + (-10)/4 = -2.75.
        let corr = LfChannelCorrelation {
            colour_factor: 4,
            base_correlation_x: 0.5,
            base_correlation_b: -0.25,
            x_factor_lf: 132,
            b_factor_lf: 118,
        };
        let factors = CflFactors::for_lf(&corr);
        let (k_x, k_b) = factors.at(0, 0);
        assert!((k_x - 1.5).abs() < 1e-6, "kX = {k_x}");
        assert!((k_b - (-2.75)).abs() < 1e-6, "kB = {k_b}");
        // The LF variant ignores the tile coordinate entirely.
        assert_eq!(factors.at(0, 0), factors.at(7, 3));
    }

    #[test]
    fn for_lf_default_correlation_is_the_identity_pass_through() {
        // Table I.3 defaults: colour_factor 84, base_correlation_x 0.0,
        // base_correlation_b 1.0, x_factor_lf/b_factor_lf both 128 (factor 0).
        let corr = LfChannelCorrelation::default();
        let (k_x, k_b) = CflFactors::for_lf(&corr).at(0, 0);
        assert_eq!(k_x, 0.0);
        assert_eq!(k_b, 1.0);
    }

    #[test]
    fn for_hf_reads_a_distinct_pair_per_tile() {
        let corr = LfChannelCorrelation {
            colour_factor: 2,
            base_correlation_x: 0.0,
            base_correlation_b: 0.0,
            x_factor_lf: 128,
            b_factor_lf: 128,
        };
        // A 2x1 XFromY/BFromY grid: tile 0 has x_factor 4, tile 1 has -6.
        let x_from_y =
            Channel::from_samples(ChannelSpec::new(2, 1), vec![4, -6]).expect("2 samples");
        let b_from_y =
            Channel::from_samples(ChannelSpec::new(2, 1), vec![0, 8]).expect("2 samples");
        let factors = CflFactors::for_hf(&corr, x_from_y, b_from_y);

        let (k_x0, k_b0) = factors.at(0, 0);
        assert!((k_x0 - 2.0).abs() < 1e-6); // 4 / 2
        assert!((k_b0 - 0.0).abs() < 1e-6);

        let (k_x1, k_b1) = factors.at(1, 0);
        assert!((k_x1 - (-3.0)).abs() < 1e-6); // -6 / 2
        assert!((k_b1 - 4.0).abs() < 1e-6); // 8 / 2

        // `at_pixel` divides by the 64-sample tile.
        assert_eq!(factors.at_pixel(63, 0), factors.at(0, 0));
        assert_eq!(factors.at_pixel(64, 0), factors.at(1, 0));
    }

    #[test]
    fn apply_reconstructs_the_linear_model() {
        // dX=1, dY=2, dB=3, kX=0.5, kB=-1: X = 1 + 0.5*2 = 2, Y = 2,
        // B = 3 + -1*2 = 1.
        let (x, y, b) = apply(1.0, 2.0, 3.0, 0.5, -1.0);
        assert!((x - 2.0).abs() < 1e-6);
        assert!((y - 2.0).abs() < 1e-6);
        assert!((b - 1.0).abs() < 1e-6);
    }

    #[test]
    fn apply_leaves_y_unmodified() {
        let (_, y, _) = apply(100.0, -7.5, 100.0, 3.0, -3.0);
        assert_eq!(y, -7.5);
    }
}
