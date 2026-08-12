//! Approximate inverse of the J.3 Gabor-like transform (slice 20).
//!
//! The decoder applies a normalized 3×3 kernel with default weights
//! `w1 = 0.115169525`, `w2 = 0.061248592`. The encoder undoes that filter
//! before DCT so that decoder Gaborish restores the intended samples.
//!
//! The exact inverse of a space-varying mirror-boundary convolution is a
//! large sparse solve. This module uses a fixed number of Jacobi iterations
//! against the same kernel, which is enough for the lossy Part 3 peak-error
//! class when the planner enables gaborish. Bit-exact recovery is not claimed.

/// Table J.1 default edge weight.
pub const DEFAULT_GAB_WEIGHT1: f32 = 0.115_169_525;
/// Table J.1 default corner weight.
pub const DEFAULT_GAB_WEIGHT2: f32 = 0.061_248_592;

/// Normalized Gabor kernel (weights already sum to 1).
#[derive(Debug, Clone, Copy)]
pub struct GaborKernel {
    /// Centre weight.
    pub centre: f32,
    /// Edge weight (4-neighbour).
    pub edge: f32,
    /// Corner weight.
    pub corner: f32,
}

impl GaborKernel {
    /// Builds the normalized kernel from unnormalized `weight1` / `weight2`.
    #[must_use]
    pub fn new(weight1: f32, weight2: f32) -> Option<Self> {
        let unnorm = 1.0 + 4.0 * weight1 + 4.0 * weight2;
        if !unnorm.is_finite() || unnorm == 0.0 {
            return None;
        }
        let scale = 1.0 / unnorm;
        Some(Self {
            centre: scale,
            edge: weight1 * scale,
            corner: weight2 * scale,
        })
    }

    /// Default Table J.1 kernel.
    #[must_use]
    pub fn default_j1() -> Self {
        Self::new(DEFAULT_GAB_WEIGHT1, DEFAULT_GAB_WEIGHT2).expect("default weights are legal")
    }

    fn weight_at(self, dx: i32, dy: i32) -> f32 {
        let adx = dx.unsigned_abs();
        let ady = dy.unsigned_abs();
        match (adx, ady) {
            (0, 0) => self.centre,
            (1, 0) | (0, 1) => self.edge,
            (1, 1) => self.corner,
            _ => 0.0,
        }
    }
}

/// Forward J.3 convolution (mirror edges) for testing inverse quality.
///
/// With `simd`, the interior (no edge mirror) is evaluated in `f32x4` lanes
/// (Opt-P leaf). Boundary rows/columns stay scalar.
pub fn gaborish_forward(
    input: &[f32],
    width: usize,
    height: usize,
    kernel: GaborKernel,
) -> Vec<f32> {
    let mut out = vec![0.0f32; width.saturating_mul(height)];
    if width == 0 || height == 0 {
        return out;
    }

    // Boundary: full scalar path with mirror.
    for y in 0..height {
        for x in 0..width {
            let on_edge = x == 0 || y == 0 || x + 1 == width || y + 1 == height;
            if !on_edge {
                continue;
            }
            let mut acc = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let sx = mirror1d(i32::try_from(x).unwrap_or(i32::MAX) + dx, width);
                    let sy = mirror1d(i32::try_from(y).unwrap_or(i32::MAX) + dy, height);
                    let sample = input
                        .get(sy.saturating_mul(width).saturating_add(sx))
                        .copied()
                        .unwrap_or(0.0);
                    acc += kernel.weight_at(dx, dy) * sample;
                }
            }
            if let Some(slot) = out.get_mut(y.saturating_mul(width).saturating_add(x)) {
                *slot = acc;
            }
        }
    }

    // Interior: no mirror needed (neighbours are in-bounds).
    if height >= 3 && width >= 3 {
        #[cfg(feature = "simd")]
        {
            gaborish_forward_interior_simd(input, width, height, kernel, &mut out);
        }
        #[cfg(not(feature = "simd"))]
        {
            gaborish_forward_interior_scalar(input, width, height, kernel, &mut out);
        }
    }
    out
}

#[cfg(not(feature = "simd"))]
fn gaborish_forward_interior_scalar(
    input: &[f32],
    width: usize,
    height: usize,
    kernel: GaborKernel,
    out: &mut [f32],
) {
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let mut acc = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let sx = (x as i32 + dx) as usize;
                    let sy = (y as i32 + dy) as usize;
                    let sample = input
                        .get(sy.saturating_mul(width).saturating_add(sx))
                        .copied()
                        .unwrap_or(0.0);
                    acc += kernel.weight_at(dx, dy) * sample;
                }
            }
            if let Some(slot) = out.get_mut(y.saturating_mul(width).saturating_add(x)) {
                *slot = acc;
            }
        }
    }
}

#[cfg(feature = "simd")]
fn gaborish_forward_interior_simd(
    input: &[f32],
    width: usize,
    height: usize,
    kernel: GaborKernel,
    out: &mut [f32],
) {
    use wide::f32x4;
    let c = f32x4::splat(kernel.centre);
    let e = f32x4::splat(kernel.edge);
    let k = f32x4::splat(kernel.corner);
    for y in 1..height.saturating_sub(1) {
        let mut x = 1usize;
        let x_end = width.saturating_sub(1);
        while x + 4 <= x_end {
            // Load 3×6 window rows at y-1,y,y+1 starting at x-1 (6 samples for 4 outputs).
            let load = |row: usize, col: usize| -> f32x4 {
                let base = row.saturating_mul(width).saturating_add(col);
                f32x4::new([
                    input.get(base).copied().unwrap_or(0.0),
                    input.get(base + 1).copied().unwrap_or(0.0),
                    input.get(base + 2).copied().unwrap_or(0.0),
                    input.get(base + 3).copied().unwrap_or(0.0),
                ])
            };
            // For outputs at x..x+3, centre samples are input[y][x..x+3].
            let mid = load(y, x);
            let north = load(y - 1, x);
            let south = load(y + 1, x);
            let west = load(y, x - 1);
            let east = load(y, x + 1);
            let nw = load(y - 1, x - 1);
            let ne = load(y - 1, x + 1);
            let sw = load(y + 1, x - 1);
            let se = load(y + 1, x + 1);
            let acc = mid * c + (north + south + west + east) * e + (nw + ne + sw + se) * k;
            let a = acc.to_array();
            for (i, &val) in a.iter().enumerate() {
                if let Some(slot) = out.get_mut(y.saturating_mul(width).saturating_add(x + i)) {
                    *slot = val;
                }
            }
            x += 4;
        }
        // Tail interior columns.
        while x < x_end {
            let mut acc = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let sx = (x as i32 + dx) as usize;
                    let sy = (y as i32 + dy) as usize;
                    let sample = input
                        .get(sy.saturating_mul(width).saturating_add(sx))
                        .copied()
                        .unwrap_or(0.0);
                    acc += kernel.weight_at(dx, dy) * sample;
                }
            }
            if let Some(slot) = out.get_mut(y.saturating_mul(width).saturating_add(x)) {
                *slot = acc;
            }
            x += 1;
        }
    }
}

/// Precondition XYB planes for decoder Gaborish (default J.1 weights).
///
/// Call this **before** DCT when [`crate::vardct::plan::RestorationDecision::gaborish`]
/// is set. Uses a fixed Jacobi iteration count suitable for the lossy peak-
/// error class.
#[must_use]
pub fn precondition_xyb_planes(
    y: &[f32],
    x: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    const ITERS: usize = 12;
    let k = GaborKernel::default_j1();
    (
        gaborish_inverse(y, width, height, k, ITERS),
        gaborish_inverse(x, width, height, k, ITERS),
        gaborish_inverse(b, width, height, k, ITERS),
    )
}

/// Approximate inverse: `iters` Jacobi steps starting from `filtered`.
///
/// Solves `K * x ≈ filtered` where `K` is the Gaborish convolution.
#[must_use]
pub fn gaborish_inverse(
    filtered: &[f32],
    width: usize,
    height: usize,
    kernel: GaborKernel,
    iters: usize,
) -> Vec<f32> {
    let n = width.saturating_mul(height);
    let mut x = filtered.to_vec();
    if x.len() != n {
        x.resize(n, 0.0);
    }
    let centre = kernel.centre;
    if centre == 0.0 {
        return x;
    }
    for _ in 0..iters {
        let kx = gaborish_forward(&x, width, height, kernel);
        #[cfg(feature = "simd")]
        {
            use wide::f32x4;
            let inv_c = f32x4::splat(1.0 / centre);
            let mut i = 0usize;
            while i + 4 <= n {
                let f4 = f32x4::new([
                    filtered.get(i).copied().unwrap_or(0.0),
                    filtered.get(i + 1).copied().unwrap_or(0.0),
                    filtered.get(i + 2).copied().unwrap_or(0.0),
                    filtered.get(i + 3).copied().unwrap_or(0.0),
                ]);
                let kx4 = f32x4::new([
                    kx.get(i).copied().unwrap_or(0.0),
                    kx.get(i + 1).copied().unwrap_or(0.0),
                    kx.get(i + 2).copied().unwrap_or(0.0),
                    kx.get(i + 3).copied().unwrap_or(0.0),
                ]);
                let x4 = f32x4::new([
                    x.get(i).copied().unwrap_or(0.0),
                    x.get(i + 1).copied().unwrap_or(0.0),
                    x.get(i + 2).copied().unwrap_or(0.0),
                    x.get(i + 3).copied().unwrap_or(0.0),
                ]);
                // Jacobi: x' = x + (f - Kx) / centre
                let next = x4 + (f4 - kx4) * inv_c;
                let a = next.to_array();
                for (j, &val) in a.iter().enumerate() {
                    if let Some(slot) = x.get_mut(i + j) {
                        *slot = val;
                    }
                }
                i += 4;
            }
            while i < n {
                let f = filtered.get(i).copied().unwrap_or(0.0);
                let kxi = kx.get(i).copied().unwrap_or(0.0);
                let xi = x.get(i).copied().unwrap_or(0.0);
                if let Some(slot) = x.get_mut(i) {
                    *slot = xi + (f - kxi) / centre;
                }
                i += 1;
            }
        }
        #[cfg(not(feature = "simd"))]
        {
            for i in 0..n {
                let f = filtered.get(i).copied().unwrap_or(0.0);
                let kxi = kx.get(i).copied().unwrap_or(0.0);
                let xi = x.get(i).copied().unwrap_or(0.0);
                // Jacobi: x' = x + (f - Kx) / centre
                if let Some(slot) = x.get_mut(i) {
                    *slot = xi + (f - kxi) / centre;
                }
            }
        }
    }
    x
}

fn mirror1d(coord: i32, size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    let size_i = i32::try_from(size).unwrap_or(i32::MAX);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_recovers_a_smooth_field_within_tolerance() {
        let w = 16usize;
        let h = 12usize;
        let original: Vec<f32> = (0..w * h)
            .map(|i| {
                let x = (i % w) as f32;
                let y = (i / w) as f32;
                0.1 * x + 0.07 * y + 0.02 * (x * y).sin()
            })
            .collect();
        let k = GaborKernel::default_j1();
        let filtered = gaborish_forward(&original, w, h, k);
        let recovered = gaborish_inverse(&filtered, w, h, k, 12);
        let mut peak = 0.0f32;
        for (a, b) in original.iter().zip(recovered.iter()) {
            peak = peak.max((a - b).abs());
        }
        // Lossy class: well under Part 3 peak budgets for this smooth field.
        assert!(peak < 1e-3, "peak error {peak}");
    }

    #[test]
    fn default_kernel_weights_sum_to_one() {
        let k = GaborKernel::default_j1();
        let sum = k.centre + 4.0 * k.edge + 4.0 * k.corner;
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
