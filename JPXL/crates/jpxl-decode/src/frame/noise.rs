//! Synthetic noise (18181-1 K.5): the LUT bundle (K.5.1) and the pixel
//! generator (K.5.2).
//!
//! ```text
//! Table G.1 — LfGlobal bundle
//! condition   type              name    subclause
//! kPatches    Patches           patches K.3.1
//! kSplines    Splines           splines K.4.1
//! kNoise      NoiseParameters   noise   K.5.1   <- this bundle
//! ```
//!
//! K.5.1 is the third row of `LfGlobal`, after patches and (an always-refused)
//! splines — reading it out of position shifts every field after it, same as
//! K.3.1 (see [`super::patches`]).
//!
//! # The `XorShift128Plus`/`SplitMix64` generator
//!
//! Both transcription sources render the state-update line with `*` where the
//! algorithm's own name says XOR: `s1_ *= (s1_ << 23)` and
//! `s1[i] = s1_ * s0_ * (s1_ >> 18) * (s0_ >> 5)`. This is the same
//! glyph-confusion OCR artifact already load-bearing at H.5.2 (see
//! `docs/HANDOFF.md`, "the sawtooth bug") — not a fresh claim, an instance of
//! a known, systemic transcription defect. Reading every `*` in the state
//! update as `^` reproduces Sebastiano Vigna's public-domain
//! `xorshift128plus` **exactly**, structural line for structural line
//! (`s[0] = s0; s1 ^= s1 << 23; s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5); return
//! s0 + s1;`), which is strong independent corroboration: the clause names
//! the algorithm, and the corrected reading matches the named algorithm
//! bit-for-bit. `SplitMix64` reads consistently the other way — its outer
//! `*` (against the two magic constants) is a real multiply, only the
//! XOR-into-`z` steps were misread — which also matches the canonical
//! `splitmix64` used to seed `xorshift128plus` families. No flip point: both
//! readings independently triangulate to the same, unambiguous conclusion.
//!
//! # Pipeline position and group grid
//!
//! K.1 puts noise last of the three image features (patches, splines,
//! noise), after K.2 upsampling — [`crate::vardct::render::apply_noise`]
//! runs after `apply_patches`, before Annex L. Noise groups are the frame's
//! ordinary G.1/G.2 group grid ([`FrameGeometry::group_rect`]); this decoder
//! refuses `frame_header.upsampling != 1` alongside noise rather than guess
//! whether "group" in K.5.2 means that un-upsampled grid mapped onto the
//! upsampled canvas, or a fresh group-sized tiling of it — nothing available
//! exercises the combination (see `docs/CONFORMANCE.md`).

use jpxl_bitstream::{BitReader, trace_field};

use crate::frame::error::Result;
use crate::frame::gaborish::{PlaneDims, mirror1d};
use crate::frame::geometry::Rect;

/// K.5.1's `NoiseParameters` bundle: an 8-entry noise-strength lookup table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseParams {
    /// `lut[i] = u(10) / (1 << 10)`, in reading order.
    pub lut: [f32; 8],
}

/// Reads a `NoiseParameters` bundle (18181-1 K.5.1).
///
/// # Errors
///
/// [`FrameError::Bitstream`](super::error::FrameError::Bitstream) on
/// truncation.
pub fn read_noise_params(reader: &mut BitReader<'_>) -> Result<NoiseParams> {
    let mut lut = [0.0f32; 8];
    for slot in &mut lut {
        let raw = trace_field!(reader, "noise.lut", reader.read_bits(10))?;
        *slot = raw as f32 / 1024.0;
    }
    Ok(NoiseParams { lut })
}

/// `SplitMix64`, used only to seed [`XorShiftState`] (K.5.2).
const fn split_mix64(seed: u64) -> u64 {
    let mut z = seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `SplitMix64`'s own initial-value convention: `SplitMix64(seed +
/// 0x9E3779B97F4A7C15)`.
const SPLITMIX_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

/// The eight-lane `XorShift128Plus` generator K.5.2 defines, seeded once per
/// group and consumed row by row, channel by channel (`rR` then `rG` then
/// `rB`), never reseeded mid-group.
struct XorShiftState {
    s0: [u64; 8],
    s1: [u64; 8],
}

impl XorShiftState {
    /// `seed0 = (vis_frame_idx << 32) + invis_frame_idx`, `seed1 = (x0 << 32)
    /// + y0` of the group's top-left pixel.
    fn seeded(seed0: u64, seed1: u64) -> Self {
        let mut s0 = [0u64; 8];
        let mut next = split_mix64(seed0.wrapping_add(SPLITMIX_INCREMENT));
        for slot in &mut s0 {
            *slot = next;
            next = split_mix64(next);
        }
        let mut s1 = [0u64; 8];
        let mut next = split_mix64(seed1.wrapping_add(SPLITMIX_INCREMENT));
        for slot in &mut s1 {
            *slot = next;
            next = split_mix64(next);
        }
        Self { s0, s1 }
    }

    /// One step of all eight lanes: returns `batch[i]` and advances state.
    fn next_batch(&mut self) -> [u64; 8] {
        let mut batch = [0u64; 8];
        let lanes = self
            .s0
            .iter_mut()
            .zip(self.s1.iter_mut())
            .zip(batch.iter_mut());
        for ((s0_slot, s1_slot), batch_slot) in lanes {
            let old_s0 = *s0_slot;
            let old_s1 = *s1_slot;
            *batch_slot = old_s1.wrapping_add(old_s0);
            *s0_slot = old_s1;
            let mut shifted = old_s0;
            shifted ^= shifted << 23;
            *s1_slot = shifted ^ old_s1 ^ (shifted >> 18) ^ (old_s1 >> 5);
        }
        batch
    }
}

/// Fills `channel` over `rect` (a `dims`-sized full-frame plane) with one
/// group's pseudorandom samples, in the row-major, 16-sample-segment order
/// K.5.2 specifies. `rng` is shared across the group's three channels — the
/// caller reseeds once per group, not per channel.
fn generate_channel_tile(
    rng: &mut XorShiftState,
    channel: &mut [f32],
    dims: PlaneDims,
    rect: Rect,
) {
    for row in 0..rect.height {
        let y = rect.y0 + row;
        let mut col = 0u32;
        while col < rect.width {
            let batch = rng.next_batch();
            let mut bits = [0u32; 16];
            for (i, &lane) in batch.iter().enumerate() {
                // Deliberate truncation: bits[2i]/bits[2i+1] are the lower and
                // upper 32-bit halves of the 64-bit lane (K.5.2), not a
                // narrowed count or index.
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "intentional 64->32-bit split, not a narrowing bug"
                )]
                let (lo, hi) = (lane as u32, (lane >> 32) as u32);
                if let Some(slot) = bits.get_mut(2 * i) {
                    *slot = lo;
                }
                if let Some(slot) = bits.get_mut(2 * i + 1) {
                    *slot = hi;
                }
            }
            let segment_len = (rect.width - col).min(16);
            for j in 0..segment_len {
                let Some(&bits_j) = bits.get(j as usize) else {
                    continue;
                };
                let sample = f32::from_bits((bits_j >> 9) | 0x3F80_0000);
                let x = rect.x0 + col + j;
                if let Some(idx) = plane_index(dims, x, y)
                    && let Some(slot) = channel.get_mut(idx)
                {
                    *slot = sample;
                }
            }
            col += segment_len;
        }
    }
}

fn plane_index(dims: PlaneDims, x: u32, y: u32) -> Option<usize> {
    if (x as usize) >= dims.width || (y as usize) >= dims.height {
        return None;
    }
    Some(y as usize * dims.width + x as usize)
}

/// K.5.2's Laplacian-like 5x5 kernel: `-4 * (Id - Bk)`, `Bk` the 5x5 box
/// kernel (`NOTE2`). Sums to zero, so the pseudorandom field's `[1, 2)` DC
/// bias (`InterpretAsF32` never produces a value below `1.0`) cancels here —
/// no separate centring step is needed before this stage.
const LAPLACIAN_5X5: [[f32; 5]; 5] = [
    [0.16, 0.16, 0.16, 0.16, 0.16],
    [0.16, 0.16, 0.16, 0.16, 0.16],
    [0.16, 0.16, -3.84, 0.16, 0.16],
    [0.16, 0.16, 0.16, 0.16, 0.16],
    [0.16, 0.16, 0.16, 0.16, 0.16],
];

/// Convolves `src` (a `dims`-sized plane) by [`LAPLACIAN_5X5`] into `dst`,
/// mirroring out-of-bounds taps per 5.2 (K.5.2: "every such access is
/// redirected to `Mirror(cx, cy)`"). Runs once over the whole channel, "not
/// in a per-group way".
pub fn convolve_laplacian(src: &[f32], dst: &mut [f32], dims: PlaneDims) {
    for y in 0..dims.height {
        for x in 0..dims.width {
            let mut acc = 0.0f32;
            for (ky, row) in LAPLACIAN_5X5.iter().enumerate() {
                let cy = y as i64 + ky as i64 - 2;
                let my = mirror1d(cy, dims.height);
                for (kx, &weight) in row.iter().enumerate() {
                    let cx = x as i64 + kx as i64 - 2;
                    let mx = mirror1d(cx, dims.width);
                    let tap = src.get(my * dims.width + mx).copied().unwrap_or(0.0);
                    acc += weight * tap;
                }
            }
            if let Some(slot) = dst.get_mut(y * dims.width + x) {
                *slot = acc;
            }
        }
    }
}

/// Generates the three (already-convolved) pseudorandom noise channels
/// `[rR, rG, rB]` for a `dims`-sized frame, tiled group by group per K.5.2.
///
/// `seed0` is `(vis_frame_idx << 32) + invis_frame_idx`; `group_rect(i)` must
/// return the frame's ordinary group grid (18181-1 G.1/G.2).
pub fn synthesize_noise_channels(
    dims: PlaneDims,
    seed0: u64,
    num_groups: u64,
    group_rect: impl Fn(u64) -> Option<Rect>,
) -> Option<[Vec<f32>; 3]> {
    let cells = dims.len();
    let mut raw = [
        vec![0.0f32; cells],
        vec![0.0f32; cells],
        vec![0.0f32; cells],
    ];

    for group_idx in 0..num_groups {
        let rect = group_rect(group_idx)?;
        let seed1 = (u64::from(rect.x0) << 32) + u64::from(rect.y0);
        let mut rng = XorShiftState::seeded(seed0, seed1);
        for channel in &mut raw {
            generate_channel_tile(&mut rng, channel, dims, rect);
        }
    }

    let mut convolved = [
        vec![0.0f32; cells],
        vec![0.0f32; cells],
        vec![0.0f32; cells],
    ];
    for (src, dst) in raw.iter().zip(convolved.iter_mut()) {
        convolve_laplacian(src, dst, dims);
    }
    Some(convolved)
}

/// K.5.2's strength lookup: given a pre-noise `x`/`y` sample pair, returns
/// `(sr, sg)`.
#[must_use]
pub fn noise_strength(lut: &[f32; 8], x_sample: f32, y_sample: f32) -> (f32, f32) {
    let in_r = (y_sample + x_sample) / 2.0;
    let in_g = (y_sample - x_sample) / 2.0;
    let scaled_r = (in_r * 6.0).max(0.0);
    let scaled_g = (in_g * 6.0).max(0.0);

    let (int_r, frac_r) = split_scaled(scaled_r);
    let (int_g, frac_g) = split_scaled(scaled_g);

    let at = |i: usize| lut.get(i).copied().unwrap_or(0.0);
    let sr = (at(int_r) * (1.0 - frac_r) + at(int_r + 1) * frac_r).clamp(0.0, 1.0);
    let sg = (at(int_g) * (1.0 - frac_g) + at(int_g + 1) * frac_g).clamp(0.0, 1.0);
    (sr, sg)
}

/// Splits a `[0, inf)` scaled value into its clamped integer/fractional parts
/// (K.5.2: values `>= 7` clamp back to integer `6`, fraction `1`).
fn split_scaled(scaled: f32) -> (usize, f32) {
    if scaled >= 7.0 {
        return (6, 1.0);
    }
    // `scaled` is finite and in `[0, 7)` here, so the floor is in `[0, 6]`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let int_part = scaled.floor() as usize;
    let int_part = int_part.min(6);
    (int_part, scaled - int_part as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_mix64_matches_hand_computed_reference_values() {
        // Independently computed (Python, same three-line formula) for
        // seed 12345, so this test can catch a transcription slip in this
        // file without depending on the rest of the module.
        assert_eq!(split_mix64(12345), 0xf36c_f116_4265_dd51);
        assert_eq!(split_mix64(0xf36c_f116_4265_dd51), 0x79a8_bd6c_f995_85ec);
    }

    #[test]
    fn xorshift_batch_advances_state_and_is_deterministic() {
        let mut rng = XorShiftState::seeded(0x1234_5678, 0x9abc_def0);
        let first = rng.next_batch();
        let mut rng2 = XorShiftState::seeded(0x1234_5678, 0x9abc_def0);
        let first2 = rng2.next_batch();
        assert_eq!(first, first2, "same seed must reproduce the same batch");
        let second = rng.next_batch();
        assert_ne!(first, second, "state must advance between batches");
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = XorShiftState::seeded(0, 0);
        let mut b = XorShiftState::seeded(0, 1);
        assert_ne!(a.next_batch(), b.next_batch());
    }

    #[test]
    fn laplacian_kernel_sums_to_zero() {
        let sum: f32 = LAPLACIAN_5X5.iter().flatten().sum();
        assert!(sum.abs() < 1e-6, "kernel must have zero DC gain, got {sum}");
    }

    #[test]
    fn convolve_laplacian_of_constant_plane_is_zero() {
        let dims = PlaneDims::new(6, 6);
        let src = vec![0.75f32; dims.len()];
        let mut dst = vec![f32::NAN; dims.len()];
        convolve_laplacian(&src, &mut dst, dims);
        for v in dst {
            assert!(
                v.abs() < 1e-5,
                "constant input must convolve to ~0, got {v}"
            );
        }
    }

    #[test]
    fn noise_strength_clamps_at_the_lut_edges() {
        let lut = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 2.0];
        // Deliberately out-of-range input clamps InScaledR/G to 6/1.
        let (sr, sg) = noise_strength(&lut, 100.0, 100.0);
        // in_r = (100+100)/2 = 100 -> scaled 600 -> clamp int=6, frac=1
        // lut[6]*(0) + lut[7]*1 = 2.0, then clamped to [0,1] => 1.0
        assert!((sr - 1.0).abs() < 1e-6);
        // in_g = (100-100)/2 = 0 -> scaled 0 -> int=0, frac=0 -> lut[0] = 0.0
        assert!((sg - 0.0).abs() < 1e-6);
    }

    #[test]
    fn split_scaled_matches_floor_below_seven() {
        assert_eq!(split_scaled(0.0), (0, 0.0));
        let (i, f) = split_scaled(3.25);
        assert_eq!(i, 3);
        assert!((f - 0.25).abs() < 1e-6);
        let (i, f) = split_scaled(6.999);
        assert_eq!(i, 6);
        assert!((f - 0.999).abs() < 1e-5);
        assert_eq!(split_scaled(7.0), (6, 1.0));
        assert_eq!(split_scaled(50.0), (6, 1.0));
    }
}
