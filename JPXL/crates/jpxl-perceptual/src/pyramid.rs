//! The 2:1 box-filter pyramid the metric evaluates its maps on.
//!
//! SSIMULACRA2 downscales in **linear RGB**, before the opponent transform,
//! by averaging each 2×2 block; the last row and column are clamped (edge
//! replicated) when a dimension is odd. Halving continues while the scale
//! being halved is at least [`crate::MIN_DIMENSION`] in both dimensions, so
//! the last scale evaluated may be smaller than that floor.

/// Output dimension of one 2:1 step.
pub(crate) const fn half(dim: usize) -> usize {
    dim.div_ceil(2)
}

/// Averages each 2×2 block of `input` (`in_w × in_h`) into `output`
/// (`half(in_w) × half(in_h)`), replicating the last row/column for odd sizes.
///
/// The four samples are summed in raster order (top-left, top-right,
/// bottom-left, bottom-right) and scaled by exactly `0.25`.
pub(crate) fn downscale_by_2(input: &[f32], in_w: usize, in_h: usize, output: &mut [f32]) {
    let out_w = half(in_w);
    let out_h = half(in_h);
    debug_assert_eq!(input.len(), in_w * in_h);
    debug_assert_eq!(output.len(), out_w * out_h);
    if in_w == 0 || in_h == 0 {
        return;
    }
    let last_x = in_w - 1;
    let last_y = in_h - 1;
    for (oy, out_row) in output.chunks_exact_mut(out_w).enumerate().take(out_h) {
        let y0 = (oy * 2).min(last_y);
        let y1 = (oy * 2 + 1).min(last_y);
        let row0 = input.get(y0 * in_w..y0 * in_w + in_w).unwrap_or(&[]);
        let row1 = input.get(y1 * in_w..y1 * in_w + in_w).unwrap_or(&[]);
        for (ox, out) in out_row.iter_mut().enumerate() {
            let x0 = (ox * 2).min(last_x);
            let x1 = (ox * 2 + 1).min(last_x);
            let a = row0.get(x0).copied().unwrap_or(0.0);
            let b = row0.get(x1).copied().unwrap_or(0.0);
            let c = row1.get(x0).copied().unwrap_or(0.0);
            let d = row1.get(x1).copied().unwrap_or(0.0);
            *out = (a + b + c + d) * 0.25;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_sizes_replicate_the_last_row_and_column() {
        // 3×3 ramp; the bottom-right output averages the corner four times.
        let input: Vec<f32> = (0..9).map(|v| v as f32).collect();
        let mut out = vec![0.0; 4];
        downscale_by_2(&input, 3, 3, &mut out);
        assert_eq!(
            out,
            vec![
                (0.0 + 1.0 + 3.0 + 4.0) / 4.0,
                (2.0 + 2.0 + 5.0 + 5.0) / 4.0,
                (6.0 + 7.0 + 6.0 + 7.0) / 4.0,
                8.0
            ]
        );
    }
}
