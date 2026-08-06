//! The desired-quantization field: `Encoder-plan1.md` §7.1's perceptual field
//! on the 8x8 atom grid, and §7.2's factorization of it into the integers the
//! wire can carry.
//!
//! # The model
//!
//! Per atom, the field holds a **log-step adjustment in octaves** relative to
//! the frame's base quantizer: `+1.0` means "quantize this atom one octave
//! coarser than the baseline", `-1.0` one octave finer. Log space because
//! quantization scales multiply (§7.1).
//!
//! The signal is the x265-family variance deviation: an atom's activity
//! `log2(1 + variance)` compared to the frame mean of the same quantity,
//! scaled by a strength and clamped. Two directions exist and they are honest
//! opposites:
//!
//! * [`AqMode::Masking`] — busy atoms coarser, flat atoms finer. Texture
//!   masks quantization noise and flat regions band; this optimizes perceived
//!   quality and is the production direction.
//! * [`AqMode::Uniform`] — busy atoms finer, flat atoms coarser. Busy atoms
//!   carry the largest reconstruction error at a uniform quantizer, so
//!   spending there equalizes error across the frame; this is the direction
//!   the slice-17 exit criterion ("spatial quality uniformity improves")
//!   measures directly.
//!
//! Chroma activity is folded into the luma term at a fixed down-weight —
//! chroma carries less perceptual rate than luma, so it modulates rather than
//! matches the luma vote.
//!
//! # The factorization (§7.2)
//!
//! The wire has one frame `global_scale` and one integer `HfMul >= 1` per
//! varblock. **`HfMul` runs the same counter-intuitive way as
//! `global_scale`**: I.2.1 divides by it (`Mul = (1 << 16) / (global_scale *
//! HfMul)`), so a *larger* `HfMul` is a *finer* quantizer — at baseline
//! `HfMul = 1` a varblock can only be refined, never coarsened. Bidirectional
//! adjustment is still representable exactly: halve `global_scale`, double
//! `quant_lf` (the LF step divides by their product, so every LF integer is
//! unchanged), and set the *baseline* `HfMul` to `2 x` the requested one, so
//! the baseline HF denominator `global_scale * HfMul` is exactly unchanged
//! too. `HfMul` one below the baseline is then one octave **coarser** and
//! one octave finer sits above. The caller falls back to refine-only
//! (adjustments clamped at `<= 0`) when the factorization is not
//! representable — see `AqSetup` in the crate root.

use jpxl_encode::vardct::ids::HfMul;

use crate::analysis::AnalysisAtlas;

/// How the adaptive-quantization field is pointed, if at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AqMode {
    /// No field: every varblock at the request's `HfMul`, byte-identical to
    /// a constant-mul encode. Opt out when measuring an un-AQ baseline.
    Off,
    /// Perceptual masking: busy coarser, flat finer.
    ///
    /// Production default since the M8 leftover wave: the entropy model now
    /// prices DctSelect/mul rows, Hierarchical cover is already the default
    /// cover, and flat content still collapses to a neutral field (byte-
    /// identical to [`AqMode::Off`]) so the switch costs nothing on flat.
    #[default]
    Masking,
    /// Error equalization: busy finer, flat coarser.
    Uniform,
}

/// The x265-family strength: how many octaves of adjustment one octave of
/// activity deviation buys. `0.25` keeps the default field inside `+-1`
/// octave on natural content, which is what the `HfMul` integer lattice
/// around a baseline of 2 can actually represent.
const AQ_STRENGTH: f32 = 0.25;

/// Hard clamp on the per-atom adjustment, in octaves. One octave either way:
/// the lattice `{1, 2, 3, 4, ...}` around baseline 2 cannot express more than
/// `-1` octave finer anyway, and unbounded coarsening blurs.
const AQ_CLAMP: f32 = 1.0;

/// The chroma share of an atom's activity (Prangnell-style luma+chroma
/// activity, down-weighted).
const CHROMA_AQ_WEIGHT: f32 = 0.35;

/// XYB is roughly unit-range float; variances are rescaled to 8-bit-squared
/// units before the `log2(1 + v)` activity so the constants above mean the
/// same thing they mean in the 8-bit literature.
const VARIANCE_TO_8BIT: f32 = 255.0 * 255.0;

/// §7.1's `DesiredQuantField`: one log-step adjustment per 8x8 atom, in
/// octaves relative to the frame baseline.
#[derive(Debug, Clone)]
pub struct DesiredQuantField {
    width: u32,
    adj: Box<[f32]>,
}

impl DesiredQuantField {
    /// Builds the field from the atlas, or `None` for [`AqMode::Off`].
    #[must_use]
    pub fn from_atlas(atlas: &AnalysisAtlas, mode: AqMode) -> Option<Self> {
        let sign = match mode {
            AqMode::Off => return None,
            AqMode::Masking => 1.0f32,
            AqMode::Uniform => -1.0f32,
        };
        let grid = atlas.grid();
        let count = usize::try_from(grid.area()).unwrap_or(0);

        let mut activity = Vec::with_capacity(count);
        let mut sum = 0.0f64;
        for atom_y in 0..grid.height {
            for atom_x in 0..grid.width {
                let a = atlas.atom(atom_x, atom_y).map_or(0.0, |f| {
                    let luma = (1.0 + f.variance_xyb[1] * VARIANCE_TO_8BIT).log2();
                    let chroma =
                        (1.0 + (f.variance_xyb[0] + f.variance_xyb[2]) * VARIANCE_TO_8BIT).log2();
                    luma + CHROMA_AQ_WEIGHT * chroma
                });
                sum += f64::from(a);
                activity.push(a);
            }
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a mean of f32 activities is far inside f32 range"
        )]
        let mean = if count > 0 {
            (sum / count as f64) as f32
        } else {
            0.0
        };

        let adj = activity
            .into_iter()
            .map(|a| (sign * AQ_STRENGTH * (a - mean)).clamp(-AQ_CLAMP, AQ_CLAMP))
            .collect();
        Some(Self {
            width: grid.width,
            adj,
        })
    }

    /// Whether every atom's snapped adjustment is zero: the field would
    /// assign the baseline everywhere, so carrying the §7.2 factorization's
    /// non-zero `mul` row would cost bytes and buy nothing. The caller treats
    /// a neutral field exactly like [`AqMode::Off`].
    #[must_use]
    pub fn is_neutral(&self) -> bool {
        self.adj.iter().all(|a| (a * 2.0).round() == 0.0)
    }

    /// The adjustment of atom `(x, y)`, zero outside the grid.
    #[must_use]
    pub fn adj_at(&self, x: u32, y: u32) -> f32 {
        let index = usize::try_from(u64::from(y) * u64::from(self.width) + u64::from(x))
            .unwrap_or(usize::MAX);
        self.adj.get(index).copied().unwrap_or(0.0)
    }

    /// §7.2: the integer `HfMul` for a varblock footprint of `rows x cols`
    /// atoms at `(bx, by)` (frame-global atom coordinates), around
    /// `baseline`.
    ///
    /// The footprint's adjustments are averaged in log space (they are log
    /// steps), optionally capped at zero for the refine-only fallback, and
    /// the multiplier `baseline * 2^-adj` — the sign flip because a larger
    /// `HfMul` is *finer* — is rounded to the nearest legal integer.
    #[must_use]
    pub fn mul_for_footprint(
        &self,
        bx: u32,
        by: u32,
        rows: u32,
        cols: u32,
        baseline: HfMul,
        refine_only: bool,
    ) -> HfMul {
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for dy in 0..rows {
            for dx in 0..cols {
                sum += self.adj_at(bx + dx, by + dy);
                count += 1;
            }
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "footprints are at most 16 atoms"
        )]
        let mut adj = if count > 0 { sum / count as f32 } else { 0.0 };
        // §7.2 wants a compact, predictable `HfMul` distribution: the field is
        // snapped to the half-octave lattice, so a frame carries at most the
        // five values of [`mul_lattice`] and the `mul` row's residuals stay
        // small and repetitive.
        adj = (adj * 2.0).round() / 2.0;
        if refine_only {
            adj = adj.min(0.0);
        }
        mul_of(baseline, adj)
    }
}

/// The multiplier at `baseline * 2^-adj` (`adj` in octaves-coarser; larger
/// `HfMul` is finer), rounded to the nearest legal integer and capped one
/// octave finer than the baseline — refinement beyond that buys quality the
/// request did not ask for.
fn mul_of(baseline: HfMul, adj: f32) -> HfMul {
    let base = baseline.get();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to [1, 2 * baseline] before the cast; baseline \
                  itself is a checked wire value"
    )]
    let mul = ((f64::from(base) * f64::from(-adj).exp2()).round() as u32)
        .clamp(1, base.saturating_mul(2));
    HfMul::new(mul).unwrap_or(baseline)
}

/// Every `HfMul` [`DesiredQuantField::mul_for_footprint`] can produce for a
/// baseline: the half-octave lattice over the clamp range, deduplicated. The
/// quantizer cache is prebuilt over exactly this set.
#[must_use]
pub fn mul_lattice(baseline: HfMul, refine_only: bool) -> Vec<HfMul> {
    let steps: &[f32] = if refine_only {
        &[-1.0, -0.5, 0.0]
    } else {
        &[-1.0, -0.5, 0.0, 0.5, 1.0]
    };
    let mut out: Vec<HfMul> = Vec::with_capacity(steps.len());
    for &adj in steps {
        let mul = mul_of(baseline, adj);
        if !out.contains(&mul) {
            out.push(mul);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::PreparedFrame;

    fn atlas_of(width: u32, height: u32, rgb: &[u8]) -> AnalysisAtlas {
        let frame = PreparedFrame::from_srgb8(width, height, rgb).expect("frame");
        AnalysisAtlas::analyze(&frame)
    }

    /// Left half flat, right half noisy.
    fn half_flat_half_noise(width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let luma = if x < width / 2 {
                    128u8
                } else {
                    let hash = (x.wrapping_mul(0x9E37).wrapping_add(y.wrapping_mul(0x79B9)))
                        .wrapping_mul(0x85EB_CA6B);
                    u8::try_from(96 + ((hash >> 24) & 0x3F)).unwrap_or(128)
                };
                out.extend_from_slice(&[luma, luma, luma]);
            }
        }
        out
    }

    #[test]
    fn masking_points_coarse_at_noise_and_fine_at_flat() {
        let atlas = atlas_of(128, 64, &half_flat_half_noise(128, 64));
        let field = DesiredQuantField::from_atlas(&atlas, AqMode::Masking).expect("field");
        assert!(field.adj_at(1, 1) < 0.0, "flat side must go finer");
        assert!(field.adj_at(12, 1) > 0.0, "noisy side must go coarser");
    }

    #[test]
    fn uniform_is_the_exact_opposite_direction() {
        let atlas = atlas_of(128, 64, &half_flat_half_noise(128, 64));
        let masking = DesiredQuantField::from_atlas(&atlas, AqMode::Masking).expect("field");
        let uniform = DesiredQuantField::from_atlas(&atlas, AqMode::Uniform).expect("field");
        for (x, y) in [(1u32, 1u32), (12, 1), (6, 4)] {
            let (m, u) = (masking.adj_at(x, y), uniform.adj_at(x, y));
            assert!(
                (m + u).abs() < 1e-6,
                "adjustments must be sign-mirrored, got {m} and {u}"
            );
        }
    }

    #[test]
    fn off_yields_no_field_and_a_flat_frame_yields_a_neutral_one() {
        let atlas = atlas_of(64, 64, &vec![100u8; 64 * 64 * 3]);
        assert!(DesiredQuantField::from_atlas(&atlas, AqMode::Off).is_none());
        let field = DesiredQuantField::from_atlas(&atlas, AqMode::Masking).expect("field");
        let baseline = HfMul::new(2).expect("legal");
        for atom in [(0u32, 0u32), (3, 3), (7, 7)] {
            assert_eq!(
                field.mul_for_footprint(atom.0, atom.1, 1, 1, baseline, false),
                baseline,
                "a flat frame has no deviation, so every mul is the baseline"
            );
        }
    }

    #[test]
    fn the_factorization_rounds_to_the_integer_lattice_and_respects_the_cap() {
        let atlas = atlas_of(128, 64, &half_flat_half_noise(128, 64));
        let field = DesiredQuantField::from_atlas(&atlas, AqMode::Masking).expect("field");
        let baseline = HfMul::new(2).expect("legal");
        let fine = field.mul_for_footprint(1, 1, 1, 1, baseline, false);
        let coarse = field.mul_for_footprint(12, 1, 1, 1, baseline, false);
        // Larger `HfMul` is *finer* (I.2.1 divides by it): flat atoms go
        // above the baseline, coarsened noisy atoms below.
        assert!(fine.get() > baseline.get(), "flat atoms refine");
        assert!(coarse.get() < baseline.get(), "noisy atoms coarsen");
        assert!(fine.get() <= baseline.get() * 2, "refinement cap");
        // Refine-only caps the coarse side at the baseline and leaves
        // refinement alone.
        let capped = field.mul_for_footprint(12, 1, 1, 1, baseline, true);
        assert_eq!(capped, baseline);
        let still_fine = field.mul_for_footprint(1, 1, 1, 1, baseline, true);
        assert_eq!(still_fine, fine);
    }
}
