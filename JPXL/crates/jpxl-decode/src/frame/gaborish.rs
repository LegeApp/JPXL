//! The Gabor-like transform (18181-1 J.3) and the 5.2 mirroring primitive.
//!
//! J.3 convolves every colour channel of the whole frame with one symmetric
//! 3x3 kernel. Before normalization the weights are
//!
//! ```text
//! w2 w1 w2       w1 = restoration_filter.gab_C_weight1   (the four edge
//! w1  1 w1                                                neighbours)
//! w2 w1 w2       w2 = restoration_filter.gab_C_weight2   (the four corners)
//! ```
//!
//! with `C` the channel being filtered, so each channel gets its own kernel.
//! The clause then requires the nine weights to be **rescaled uniformly so
//! that they sum to 1**: one scale factor `1 / (1 + 4*w1 + 4*w2)` multiplies
//! every tap, the centre included. That is the whole normalization — there is
//! no separate centre term. [`GaborKernel::new`] is the only place it is
//! computed, and [`GaborKernel::sum`] is what the exit test checks.
//!
//! Taps that fall outside the frame are redirected by `Mirror` (5.2), not
//! clamped: see [`mirror1d`].
//!
//! # Where this sits
//!
//! J.1: the Gabor-like transform runs on the whole frame, and the
//! edge-preserving filter (J.4, [`super::epf`]) runs immediately after it.
//! Wiring both into the decode pipeline is a later sub-slice; this module is a
//! pure function of (planes, parameters).

use crate::frame::error::{FrameError, Result};
use crate::frame::restoration::GaborWeights;

/// Dimensions of an `f32` sample plane stored in raster order.
///
/// The restoration filters work on plain `&[f32]` slices whose length is
/// exactly `width * height`; this pairs a slice with the shape it is read at
/// so the two cannot drift apart silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneDims {
    /// Samples per row.
    pub width: usize,
    /// Number of rows.
    pub height: usize,
}

impl PlaneDims {
    /// A plane of `width` by `height` samples.
    #[must_use]
    pub const fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Number of samples a plane of this shape holds.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.width.saturating_mul(self.height)
    }

    /// Whether the plane holds no samples.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Checks that `plane` has exactly [`PlaneDims::len`] samples.
    ///
    /// # Errors
    ///
    /// [`FrameError::FieldOutOfRange`] naming `what` when the length differs.
    pub fn check(&self, plane: &[f32], what: &'static str, clause: &'static str) -> Result<()> {
        if plane.len() == self.len() {
            Ok(())
        } else {
            Err(FrameError::out_of_range(
                what,
                clause,
                u64::try_from(plane.len()).unwrap_or(u64::MAX),
            ))
        }
    }
}

/// `usize` to `i64` without a lint-triggering `as` cast.
///
/// Plane dimensions are bounded far below `i64::MAX` by the `Limits` caps, so
/// the saturating fallback is unreachable in practice.
fn as_i64(v: usize) -> i64 {
    i64::try_from(v).unwrap_or(i64::MAX)
}

/// 18181-1 5.2 `Mirror1D`: folds an out-of-range coordinate back into
/// `[0, size)` by reflecting about the half-sample outside each edge.
///
/// The clause writes this recursively — `coord < 0` reflects to `-coord - 1`
/// and `coord >= size` to `2*size - 1 - coord`, each re-entering the function.
/// One reflection is not always enough: on a narrow plane a coordinate can
/// bounce between the two edges several times, which is exactly the case a
/// naive single-step mirror gets wrong. This is the same recursion written as
/// a loop.
///
/// `size == 0` has no valid sample to return; it yields 0 so the function is
/// total, and callers reject empty planes before they get here.
#[must_use]
pub fn mirror1d(coord: i64, size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    let size_i = as_i64(size);
    let mut c = coord;
    loop {
        if c < 0 {
            c = -c - 1;
        } else if c >= size_i {
            c = 2 * size_i - 1 - c;
        } else {
            return usize::try_from(c).unwrap_or(0);
        }
    }
}

/// Reads `plane` at `(x, y)`, mirroring per 5.2 when the coordinate is outside
/// the plane.
#[must_use]
pub fn sample_mirrored(plane: &[f32], dims: PlaneDims, x: i64, y: i64) -> f32 {
    let px = mirror1d(x, dims.width);
    let py = mirror1d(y, dims.height);
    // The index is in range whenever `plane.len() == dims.len()`, which every
    // public entry point checks first.
    plane
        .get(py.saturating_mul(dims.width).saturating_add(px))
        .copied()
        .unwrap_or(0.0)
}

/// The normalized J.3 kernel for one channel.
///
/// Constructed by [`GaborKernel::new`], which applies the clause's uniform
/// rescale; the three fields are already scaled, so [`GaborKernel::sum`] is 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaborKernel {
    /// Weight of the reference sample.
    pub centre: f32,
    /// Weight of each of the four edge neighbours.
    pub edge: f32,
    /// Weight of each of the four corner neighbours.
    pub corner: f32,
}

impl GaborKernel {
    /// Builds the normalized kernel from one channel's `weight1`/`weight2`.
    ///
    /// # Errors
    ///
    /// [`FrameError::FieldOutOfRange`] if the unnormalized weights do not sum
    /// to a finite non-zero value — a stream can signal arbitrary `F16()`
    /// weights, and `1 + 4*w1 + 4*w2 == 0` would otherwise divide by zero.
    pub fn new(weight1: f32, weight2: f32) -> Result<Self> {
        let unnormalized_sum = 4.0f32.mul_add(weight2, 4.0f32.mul_add(weight1, 1.0));
        if !unnormalized_sum.is_finite() || unnormalized_sum == 0.0 {
            // `value` carries no useful integer here; the field name and
            // clause are what identify the failure.
            return Err(FrameError::out_of_range(
                "gab_weight1/gab_weight2",
                "J.3",
                0,
            ));
        }
        let scale = 1.0 / unnormalized_sum;
        Ok(Self {
            centre: scale,
            edge: weight1 * scale,
            corner: weight2 * scale,
        })
    }

    /// The nine kernel weights summed. J.3 requires this to be 1.
    #[must_use]
    pub fn sum(&self) -> f32 {
        4.0f32.mul_add(self.corner, 4.0f32.mul_add(self.edge, self.centre))
    }

    /// Weight of the tap at offset `(dx, dy)`, each in `-1..=1`.
    #[must_use]
    pub const fn weight_at(&self, dx: i64, dy: i64) -> f32 {
        match (dx, dy) {
            (0, 0) => self.centre,
            (0, _) | (_, 0) => self.edge,
            _ => self.corner,
        }
    }
}

/// Applies J.3 to one channel, writing the result into `output`.
///
/// `input` and `output` must both hold `dims.len()` samples. The transform is
/// not in-place: every output sample reads the *unfiltered* neighbourhood.
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] if either slice's length disagrees with
/// `dims`.
pub fn gaborish_into(
    input: &[f32],
    output: &mut [f32],
    dims: PlaneDims,
    kernel: &GaborKernel,
) -> Result<()> {
    dims.check(input, "gaborish input plane length", "J.3")?;
    dims.check(output, "gaborish output plane length", "J.3")?;

    for y in 0..dims.height {
        let yi = as_i64(y);
        for x in 0..dims.width {
            let xi = as_i64(x);
            let mut acc = 0.0f32;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let w = kernel.weight_at(dx, dy);
                    acc = w.mul_add(sample_mirrored(input, dims, xi + dx, yi + dy), acc);
                }
            }
            if let Some(slot) = output.get_mut(y.saturating_mul(dims.width).saturating_add(x)) {
                *slot = acc;
            }
        }
    }
    Ok(())
}

/// Applies J.3 to one channel, returning a fresh plane.
///
/// # Errors
///
/// As [`gaborish_into`].
pub fn gaborish(input: &[f32], dims: PlaneDims, kernel: &GaborKernel) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; dims.len()];
    gaborish_into(input, &mut output, dims, kernel)?;
    Ok(output)
}

/// Applies J.3 to the three colour planes `[X, Y, B]`, each with its own
/// kernel built from the matching `gab_C_weight1`/`gab_C_weight2` pair.
///
/// Channel order is the `{x, y, b}` order of Table J.1, which is also the
/// order [`GaborWeights`] stores.
///
/// # Errors
///
/// As [`GaborKernel::new`] and [`gaborish_into`].
pub fn gaborish_planes(
    planes: [&[f32]; 3],
    dims: PlaneDims,
    weights: &GaborWeights,
) -> Result<[Vec<f32>; 3]> {
    let mut out = [
        vec![0.0f32; dims.len()],
        vec![0.0f32; dims.len()],
        vec![0.0f32; dims.len()],
    ];
    for c in 0..3 {
        let w1 = weights.weight1.get(c).copied().unwrap_or_default();
        let w2 = weights.weight2.get(c).copied().unwrap_or_default();
        let kernel = GaborKernel::new(w1, w2)?;
        let input = planes.get(c).copied().unwrap_or(&[]);
        if let Some(dst) = out.get_mut(c) {
            gaborish_into(input, dst, dims, &kernel)?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(plane: &[f32], dims: PlaneDims, x: usize, y: usize) -> f32 {
        plane.get(y * dims.width + x).copied().unwrap_or(f32::NAN)
    }

    #[test]
    fn mirror1d_matches_the_clause_on_a_wide_plane() {
        // size 4: ... 1 0 | 0 1 2 3 | 3 2 ...
        let got: Vec<usize> = (-4i64..8).map(|c| mirror1d(c, 4)).collect();
        assert_eq!(got, vec![3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0]);
    }

    #[test]
    fn mirror1d_terminates_on_a_one_sample_axis() {
        // The degenerate case: a single reflection off one edge lands outside
        // the other, so a naive one-step mirror produces an out-of-range index.
        // Every coordinate must fold to 0.
        for c in -8i64..8 {
            assert_eq!(mirror1d(c, 1), 0, "coord {c}");
        }
        // Two-wide is the other case a single reflection can get wrong: the
        // pattern has period 4 (0, 1, 1, 0), so a coordinate two steps outside
        // needs a second reflection off the opposite edge.
        let got: Vec<usize> = (-4i64..6).map(|c| mirror1d(c, 2)).collect();
        assert_eq!(got, vec![0, 1, 1, 0, 0, 1, 1, 0, 0, 1]);
    }

    #[test]
    fn rescaled_kernel_sums_to_one() {
        // Proves the J.3 rescale: whatever the signalled weights, the nine
        // taps sum to 1, so a constant plane survives the convolution.
        for (w1, w2) in [
            (0.115_169_525_f32, 0.061_248_592_f32), // Table J.1 defaults
            (0.0, 0.0),
            (2.0, -0.25),
            (-0.1, 0.5),
        ] {
            let k = GaborKernel::new(w1, w2).expect("finite non-zero sum");
            assert!(
                (k.sum() - 1.0).abs() < 1e-6,
                "w1={w1} w2={w2} sum={}",
                k.sum()
            );
        }
    }

    #[test]
    fn degenerate_weights_are_rejected_not_divided_by() {
        // 1 + 4*w1 + 4*w2 == 0 with w1 = -0.25, w2 = 0.
        assert!(GaborKernel::new(-0.25, 0.0).is_err());
        assert!(GaborKernel::new(f32::NAN, 0.0).is_err());
    }

    #[test]
    fn constant_plane_is_unchanged() {
        // The identity that the rescale exists to guarantee, checked on a
        // plane whose every pixel is an edge pixel in at least one axis, so
        // mirroring is exercised throughout.
        let dims = PlaneDims::new(5, 3);
        let input = vec![0.375f32; dims.len()];
        let k = GaborKernel::new(0.115_169_525, 0.061_248_592).expect("valid");
        let out = gaborish(&input, dims, &k).expect("valid");
        for (i, v) in out.iter().enumerate() {
            assert!((v - 0.375).abs() < 1e-6, "sample {i} = {v}");
        }
    }

    #[test]
    fn hand_computed_impulse_response() {
        // 3x3 plane, single 1.0 at the centre, w1 = 0.25, w2 = 0.125.
        // Unnormalized sum = 1 + 4*0.25 + 4*0.125 = 2.5, so scale = 0.4 and
        // the normalized taps are centre 0.4, edge 0.1, corner 0.05.
        //
        // Each output sample is (weight of the tap that lands on the impulse)
        // times 1.0. For the border samples the mirrored taps all land on
        // zeros, so only the real neighbour contributes.
        let dims = PlaneDims::new(3, 3);
        let mut input = vec![0.0f32; 9];
        if let Some(s) = input.get_mut(4) {
            *s = 1.0;
        }
        let k = GaborKernel::new(0.25, 0.125).expect("valid");
        assert_eq!((k.centre, k.edge, k.corner), (0.4, 0.1, 0.05));

        let out = gaborish(&input, dims, &k).expect("valid");
        let expected = [
            0.05, 0.1, 0.05, //
            0.1, 0.4, 0.1, //
            0.05, 0.1, 0.05,
        ];
        for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6, "sample {i}: {got} != {want}");
        }
    }

    #[test]
    fn hand_computed_mirroring_on_a_one_pixel_wide_plane() {
        // width 1, height 3. Every horizontal tap folds onto column 0, so the
        // three columns of the kernel collapse into one column of row weights:
        //   row weight = corner + edge + corner   for the rows above/below
        //   row weight = edge   + centre + edge   for the reference row
        // With w1 = 0.25, w2 = 0.125 (centre 0.4, edge 0.1, corner 0.05):
        //   own row  = 0.1 + 0.4 + 0.1 = 0.6
        //   next row = 0.05 + 0.1 + 0.05 = 0.2
        // At y = 0 the row above mirrors back onto row 0, so row 0 gets
        // 0.6 + 0.2 = 0.8 and row 1 gets 0.2.
        let dims = PlaneDims::new(1, 3);
        let input = vec![1.0f32, 0.0, 0.0];
        let k = GaborKernel::new(0.25, 0.125).expect("valid");
        let out = gaborish(&input, dims, &k).expect("valid");

        assert!((at(&out, dims, 0, 0) - 0.8).abs() < 1e-6, "{out:?}");
        assert!((at(&out, dims, 0, 1) - 0.2).abs() < 1e-6, "{out:?}");
        assert!((at(&out, dims, 0, 2) - 0.0).abs() < 1e-6, "{out:?}");
        // Total mass is preserved because the weights sum to 1.
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn single_pixel_plane_is_the_identity() {
        let dims = PlaneDims::new(1, 1);
        let k = GaborKernel::new(0.115_169_525, 0.061_248_592).expect("valid");
        let out = gaborish(&[0.25], dims, &k).expect("valid");
        assert!((out.first().copied().unwrap_or(0.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn each_channel_uses_its_own_weights() {
        // Table J.1 signals weight1/weight2 per channel; a shared kernel would
        // make all three outputs equal here.
        let dims = PlaneDims::new(3, 1);
        let plane = [0.0f32, 1.0, 0.0];
        let weights = GaborWeights {
            weight1: [0.25, 0.5, 0.0],
            weight2: [0.125, 0.0, 0.0],
        };
        let out = gaborish_planes([&plane, &plane, &plane], dims, &weights).expect("valid");
        // Channel b has both weights zero, so its kernel is the identity.
        assert_eq!(out.get(2).map(Vec::as_slice), Some(&plane[..]));
        let x_centre = out.first().and_then(|p| p.get(1)).copied().unwrap_or(0.0);
        let y_centre = out.get(1).and_then(|p| p.get(1)).copied().unwrap_or(0.0);
        assert!(
            (x_centre - y_centre).abs() > 1e-3,
            "{x_centre} vs {y_centre}"
        );
    }

    #[test]
    fn length_mismatch_is_an_error() {
        let dims = PlaneDims::new(4, 4);
        let k = GaborKernel::new(0.1, 0.1).expect("valid");
        assert!(gaborish(&[0.0; 15], dims, &k).is_err());
    }
}
