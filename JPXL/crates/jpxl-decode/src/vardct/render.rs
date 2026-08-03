//! From quantized VarDCT coefficients to frame samples (18181-1 I.5.3, I.6,
//! I.8, I.9, Annex J, L.2.2).
//!
//! This is the last stage of the `kVarDCT` pipeline and the only one that sees
//! every earlier piece at once. Its shape follows the clause order exactly:
//!
//! ```text
//! I.5.3  bias-adjust the quantized HF coefficient, scale by Mul, by the
//!        channel's pow(0.8, qm_scale - 2), and by the I.2.4 matrix entry
//! I.6    chroma from luma, on the dequantized coefficients, with (kX, kB)
//!        from the 64x64 tile containing the varblock
//! I.8    overwrite the LLF sub-rectangle from the dequantized LF image
//! I.9    coefficients -> samples for the varblock's transform type
//! J.3/4  gaborish, then EPF, over the assembled XYB planes
//! L.2.2  inverse XYB -> linear sRGB
//! ```
//!
//! # Why the filters run before the colour transform
//!
//! Clause 4's pipeline summary is explicit and is the authority here: the
//! decoder reads frame data (Annex G), applies restoration filters (Annex J),
//! draws image features (Annex K), and "finally ... performs colour transforms
//! as specified in Annex L". J.4.2's distance metric weights the three
//! channels by `epf_channel_scale`, whose defaults `{40, 5, 3.5}` are only
//! meaningful for X, Y, B — another confirmation that EPF sees XYB, not RGB.
//!
//! # What lives here and what does not
//!
//! Everything in this module is a pure function of already-parsed structures.
//! Section walking, bit positions and the TOC belong to
//! [`crate::decode`]; the parameter bundles belong to the sibling modules.

use jpxl_core::color::OpsinInverse;
use jpxl_core::limits::AllocGuard;
use jpxl_core::varblock::{CoeffMatrix, SampleBlock, TransformType, llf_from_lf};

use crate::decode::ReferenceFrame;
use crate::error::{DecodeError, Result};
use crate::frame::gaborish::{GaborKernel, PlaneDims};
use crate::frame::patches::{PatchBlendMode, PatchDictionary, blend as patch_blend};
use crate::frame::restoration::RestorationFilter;
use crate::frame::{SigmaField, epf, gaborish_into, vardct_sigma};
use crate::vardct::cfl;
use crate::vardct::dequant_matrix::DequantMatrices;
use crate::vardct::hf_coeff::QuantCoeffBlock;
use crate::vardct::lf::DequantPlane;

/// Number of colour channels a `kVarDCT` frame always has (X, Y, B).
pub const NUM_CHANNELS: usize = 3;

/// I.5.3's `(1 << 16)` numerator, matching I.2.1's LF numerator.
const HF_NUMERATOR: f32 = 65536.0;

/// I.5.3's per-channel quantization-matrix scale base.
const QM_SCALE_BASE: f32 = 0.8;

// ---------------------------------------------------------------------------
// Frame-sized f32 planes
// ---------------------------------------------------------------------------

/// Three frame-sized `f32` planes in raster order.
///
/// Carries `[X, Y, B]` between I.9 and L.2.2 and `[R, G, B]` after it; which
/// one it is at a given moment is the caller's business, and the type name
/// deliberately does not claim either — the transition happens exactly once,
/// in [`to_linear_srgb`].
#[derive(Debug, Clone, PartialEq)]
pub struct ColourPlanes {
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
    /// The three planes, each `width * height` long.
    pub planes: [Vec<f32>; NUM_CHANNELS],
}

impl ColourPlanes {
    /// Allocates three zeroed planes, metering the allocation.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Core`] if the frame exceeds the allocation budget, or
    /// [`DecodeError::FieldOutOfRange`] if `width * height` overflows.
    pub fn zeros(width: u32, height: u32, guard: &mut AllocGuard) -> Result<Self> {
        let cells = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or_else(|| DecodeError::out_of_range("frame area", "I.9", u64::from(width)))?;
        guard.charge(cells * 4 * NUM_CHANNELS as u64)?;
        let len = usize::try_from(cells)
            .map_err(|_| DecodeError::out_of_range("frame area", "I.9", cells))?;
        Ok(Self {
            width,
            height,
            planes: [vec![0.0; len], vec![0.0; len], vec![0.0; len]],
        })
    }

    /// The plane dimensions in the form the J.3/J.4 filters want.
    #[must_use]
    pub fn dims(&self) -> PlaneDims {
        PlaneDims::new(self.width as usize, self.height as usize)
    }

    /// Writes one sample, ignoring coordinates outside the frame.
    ///
    /// Out-of-frame writes are normal: a varblock at the right or bottom edge
    /// of the last group covers samples the frame does not have, and I.9
    /// reconstructs the whole varblock regardless.
    pub fn set(&mut self, channel: usize, x: u32, y: u32, value: f32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = y as usize * self.width as usize + x as usize;
        if let Some(slot) = self.planes.get_mut(channel).and_then(|p| p.get_mut(idx)) {
            *slot = value;
        }
    }

    /// Reads one sample, or `0.0` outside the frame.
    #[must_use]
    pub fn get(&self, channel: usize, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        let idx = y as usize * self.width as usize + x as usize;
        self.planes
            .get(channel)
            .and_then(|p| p.get(idx))
            .copied()
            .unwrap_or(0.0)
    }

    /// Borrows the three planes as slices, in `[X, Y, B]` order.
    #[must_use]
    pub fn as_slices(&self) -> [&[f32]; NUM_CHANNELS] {
        self.planes.each_ref().map(Vec::as_slice)
    }
}

/// A frame's extra channels as `f32` planes on the nominal `[0, 1]` scale.
///
/// One plane per `metadata.ec_info` entry, in index order, each the size of
/// the frame. Unlike [`ColourPlanes`] these never pass through Annex L: an
/// extra channel is not colour, so G.4.2's interpretation "according to
/// `metadata.ec_info[i].bit_depth`" is the whole of it. Samples are left
/// unclipped for the same reason the colour planes are.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExtraPlanes {
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
    /// One plane per extra channel, each `width * height` long.
    pub planes: Vec<Vec<f32>>,
}

impl ExtraPlanes {
    /// An image with no extra channels.
    #[must_use]
    pub const fn empty(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            planes: Vec::new(),
        }
    }

    /// Allocates `count` zeroed frame-sized planes, metering the allocation.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Core`] if the planes exceed the allocation budget, or
    /// [`DecodeError::FieldOutOfRange`] if `width * height` overflows.
    pub fn zeros(width: u32, height: u32, count: usize, guard: &mut AllocGuard) -> Result<Self> {
        let cells = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or_else(|| DecodeError::out_of_range("frame area", "G.4.2", u64::from(width)))?;
        guard.charge(cells * 4 * count as u64)?;
        let len = usize::try_from(cells)
            .map_err(|_| DecodeError::out_of_range("frame area", "G.4.2", cells))?;
        Ok(Self {
            width,
            height,
            planes: vec![vec![0.0; len]; count],
        })
    }

    /// How many extra channels this holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.planes.len()
    }

    /// Whether the frame has no extra channels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.planes.is_empty()
    }

    /// Writes one sample, ignoring coordinates outside the frame.
    pub fn set(&mut self, channel: usize, x: u32, y: u32, value: f32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = y as usize * self.width as usize + x as usize;
        if let Some(slot) = self.planes.get_mut(channel).and_then(|p| p.get_mut(idx)) {
            *slot = value;
        }
    }

    /// Reads one sample, or `0.0` outside the frame or past the last channel.
    #[must_use]
    pub fn get(&self, channel: usize, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        let idx = y as usize * self.width as usize + x as usize;
        self.planes
            .get(channel)
            .and_then(|p| p.get(idx))
            .copied()
            .unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// I.5.3 — HF dequantization
// ---------------------------------------------------------------------------

/// Everything I.5.3 needs that does not vary per varblock.
#[derive(Debug, Clone, Copy)]
pub struct HfDequantParams<'a> {
    /// I.2.4 dequantization matrices, from `HfGlobal`.
    pub matrices: &'a DequantMatrices,
    /// `oim.quant_bias[0..3]` (Table L.1), indexed by the I.1 channel
    /// numbering `0 = X, 1 = Y, 2 = B`.
    pub quant_bias: [f32; NUM_CHANNELS],
    /// `oim.quant_bias_numerator` (Table L.1).
    pub quant_bias_numerator: f32,
    /// `quantizer.global_scale` (I.2.1).
    pub global_scale: u32,
    /// `frame_header.x_qm_scale` (F.2).
    pub x_qm_scale: u32,
    /// `frame_header.b_qm_scale` (F.2).
    pub b_qm_scale: u32,
}

impl HfDequantParams<'_> {
    /// I.5.3's per-channel `pow(0.8, qm_scale - 2)` factor.
    ///
    /// The Y channel has no such factor, i.e. the multiplier is exactly 1.
    /// `qm_scale` is a `u(3)`, so the exponent is in `-2..=5` and the `powi`
    /// is exact enough to be reproducible; it is written as `powi` on an
    /// `i32` rather than `powf` so no platform `pow` is involved.
    #[must_use]
    pub fn qm_multiplier(&self, channel: usize) -> f32 {
        let scale = match channel {
            0 => self.x_qm_scale,
            2 => self.b_qm_scale,
            _ => return 1.0,
        };
        let exponent = i32::try_from(scale).unwrap_or(2) - 2;
        QM_SCALE_BASE.powi(exponent)
    }

    /// I.5.3's `Mul = (1 << 16) / (global_scale * HfMul)`.
    ///
    /// Both factors are at least 1 (`global_scale` is `1 + u(11)` or a Table
    /// I.2 literal, `HfMul` is `1 + mul` with `mul >= 0` enforced by G.2.4),
    /// so this never divides by zero. The product reaches `2^11 * 2^31` in the
    /// worst case, so it is formed in `f64` and narrowed once.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the single deliberate f64 -> f32 narrowing at the end of the \
                  multiplier computation, mirroring Quantizer::lf_multipliers"
    )]
    #[must_use]
    pub fn hf_multiplier(&self, hf_mul: u32) -> f32 {
        let denom = f64::from(self.global_scale) * f64::from(hf_mul);
        if denom <= 0.0 {
            return 0.0;
        }
        (f64::from(HF_NUMERATOR) / denom) as f32
    }

    /// I.5.3's bias adjustment of one quantized coefficient.
    ///
    /// ```text
    /// if (abs(quant) <= 1) quant *= oim.quant_bias[channel];
    /// else                 quant -= oim.quant_bias_numerator / quant;
    /// ```
    ///
    /// The test is on the quantized **integer**, and the arithmetic is `f32`
    /// like the rest of the sample pipeline. Note that the `<= 1` branch
    /// covers `quant == 0` and maps it to `0` under any bias, so an all-zero
    /// varblock dequantizes to zero whatever the bundle says.
    #[must_use]
    pub fn bias_adjust(&self, quant: i32, channel: usize) -> f32 {
        let q = quant as f32;
        if quant.abs() <= 1 {
            q * self.quant_bias.get(channel).copied().unwrap_or(1.0)
        } else {
            q - self.quant_bias_numerator / q
        }
    }
}

/// I.5.3: dequantizes one varblock's coefficients in one channel.
///
/// The LLF cells (`k < num_blocks` in natural order, i.e. the top-left
/// `bheight/8 x bwidth/8` sub-rectangle) are dequantized along with the rest
/// and then overwritten by I.8; 8C never writes them, so what they hold here
/// is `bias_adjust(0) == 0` scaled by the matrix, which is exactly zero.
///
/// # Errors
///
/// [`DecodeError::Unsupported`] if the transform's dequantization matrix is a
/// RAW one that was never supplied; [`DecodeError::FieldOutOfRange`] if the
/// quantized block's shape does not match its transform.
pub fn dequantize_hf_block(
    transform: TransformType,
    channel: usize,
    quant: &QuantCoeffBlock,
    hf_mul: u32,
    params: &HfDequantParams<'_>,
) -> Result<CoeffMatrix> {
    let (rows, cols) = (transform.coeff_rows(), transform.coeff_cols());
    if quant.rows() != rows || quant.cols() != cols {
        return Err(DecodeError::out_of_range(
            "quantized block shape",
            "I.5.3",
            quant.rows() as u64,
        ));
    }
    let matrix = params.matrices.for_transform(transform, channel)?;
    if matrix.rows() != rows || matrix.cols() != cols {
        return Err(DecodeError::out_of_range(
            "dequantization matrix shape",
            "I.2.4",
            matrix.rows() as u64,
        ));
    }

    // Mul and the qm factor are constant over the varblock; only the matrix
    // entry varies per coefficient.
    let scale = params.hf_multiplier(hf_mul) * params.qm_multiplier(channel);

    let mut out = CoeffMatrix::zeros(rows, cols);
    for y in 0..rows {
        for x in 0..cols {
            let adjusted = params.bias_adjust(quant.at(x, y), channel);
            out.set(x, y, adjusted * scale * matrix.at(x, y));
        }
    }
    Ok(out)
}

/// I.6 for HF coefficients: `X = dX + kX*Y`, `B = dB + kB*Y`, in place over
/// three same-shaped coefficient matrices.
///
/// Applied to every cell including the LLF sub-rectangle. That is harmless and
/// deliberate: I.8 overwrites those cells immediately afterwards from the LF
/// image, which has already had its own (frame-wide) CfL applied by I.5.2.
pub fn apply_hf_cfl(coeffs: &mut [CoeffMatrix; NUM_CHANNELS], k_x: f32, k_b: f32) {
    let (rows, cols) = (coeffs[1].rows(), coeffs[1].cols());
    for y in 0..rows {
        for x in 0..cols {
            let d_y = coeffs[1].at(x, y);
            let (v_x, _, v_b) = cfl::apply(coeffs[0].at(x, y), d_y, coeffs[2].at(x, y), k_x, k_b);
            coeffs[0].set(x, y, v_x);
            coeffs[2].set(x, y, v_b);
        }
    }
}

/// I.8: overwrites the LLF sub-rectangle of `coeff` from the dequantized LF
/// plane of the varblock's LF group.
///
/// `(bx, by)` is the varblock's top-left 8x8 block, LF-group-relative — the
/// same coordinate `LfQuant` and `Sharpness` are indexed by.
pub fn write_llf(
    coeff: &mut CoeffMatrix,
    transform: TransformType,
    lf: &DequantPlane,
    bx: u32,
    by: u32,
) {
    let (block_rows, block_cols) = transform.block_dims();
    let mut rect = SampleBlock::zeros(block_rows, block_cols);
    for dy in 0..block_rows {
        for dx in 0..block_cols {
            // Table I.1's largest varblock is 32x32 blocks, so `narrow` never
            // saturates; it exists so no `as` cast appears on a coordinate.
            let value = lf.get(bx.saturating_add(narrow(dx)), by.saturating_add(narrow(dy)));
            rect.set(dx, dy, value);
        }
    }
    coeff.write_llf(&llf_from_lf(transform, &rect));
}

/// A small block-grid count as `u32`, saturating rather than wrapping.
///
/// Every caller passes a Table I.1 varblock extent (at most 32), so the
/// saturation is unreachable; it exists so no `as` cast appears on a
/// coordinate path.
fn narrow(v: usize) -> u32 {
    u32::try_from(v).unwrap_or(u32::MAX)
}

/// One varblock, from quantized coefficients to samples in all three channels.
///
/// Runs I.5.3, I.6 (HF), I.8 and I.9 in that order. `lf` are the three
/// dequantized LF planes of the varblock's LF group (I.5.2's output, already
/// CfL-corrected and smoothed), `(bx, by)` the varblock's LF-group-relative
/// block position, and `(k_x, k_b)` the HF chroma-from-luma factors of the
/// 64x64 tile containing it.
///
/// # Errors
///
/// As [`dequantize_hf_block`].
#[expect(
    clippy::too_many_arguments,
    reason = "I.5.3, I.6, I.8 and I.9 each contribute their own inputs and all \
              four run over one varblock; a bundle struct would only move the \
              list out of the clause order it deliberately follows"
)]
pub fn render_varblock(
    transform: TransformType,
    quant: [&QuantCoeffBlock; NUM_CHANNELS],
    hf_mul: u32,
    params: &HfDequantParams<'_>,
    cfl_factors: (f32, f32),
    lf: [&DequantPlane; NUM_CHANNELS],
    bx: u32,
    by: u32,
) -> Result<[SampleBlock; NUM_CHANNELS]> {
    let mut coeffs = [
        dequantize_hf_block(transform, 0, quant[0], hf_mul, params)?,
        dequantize_hf_block(transform, 1, quant[1], hf_mul, params)?,
        dequantize_hf_block(transform, 2, quant[2], hf_mul, params)?,
    ];

    let (k_x, k_b) = cfl_factors;
    apply_hf_cfl(&mut coeffs, k_x, k_b);

    for (coeff, plane) in coeffs.iter_mut().zip(lf) {
        write_llf(coeff, transform, plane, bx, by);
    }

    Ok([
        transform.samples_from_coefficients(&coeffs[0]),
        transform.samples_from_coefficients(&coeffs[1]),
        transform.samples_from_coefficients(&coeffs[2]),
    ])
}

// ---------------------------------------------------------------------------
// J — restoration filters
// ---------------------------------------------------------------------------

/// The per-8x8-block sigma planes J.4.3 needs, over the whole frame.
///
/// `sigma` is indexed by the frame's 8x8-block grid and holds the value J.4.3
/// computes from `mul` and `Sharpness` at that block; `varblock_sigma` holds,
/// for the same block, the sigma of the varblock covering it — that is what
/// the `< 0.3` skip test reads under
/// [`EPF_SKIP_IS_PER_VARBLOCK`](crate::frame::epf::EPF_SKIP_IS_PER_VARBLOCK).
#[derive(Debug, Clone, PartialEq)]
pub struct SigmaPlanes {
    /// Blocks per row of the frame's 8x8 grid.
    pub blocks_x: usize,
    /// Rows of blocks.
    pub blocks_y: usize,
    /// Per-block sigma.
    pub sigma: Vec<f32>,
    /// Per-block sigma of the covering varblock.
    pub varblock_sigma: Vec<f32>,
}

impl SigmaPlanes {
    /// Allocates a zeroed field for a frame's 8x8-block grid.
    ///
    /// The grid is the one [`crate::frame::epf::block_grid`] derives, so that
    /// [`SigmaField`] accepts it without a reshape.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Core`] if the field exceeds the allocation budget.
    pub fn zeros(width: u32, height: u32, guard: &mut AllocGuard) -> Result<Self> {
        let dims = PlaneDims::new(width as usize, height as usize);
        let (blocks_x, blocks_y) = crate::frame::epf::block_grid(dims);
        let cells = blocks_x
            .checked_mul(blocks_y)
            .ok_or_else(|| DecodeError::out_of_range("block grid", "J.4.3", u64::from(width)))?;
        guard.charge(cells as u64 * 8)?;
        Ok(Self {
            blocks_x,
            blocks_y,
            sigma: vec![0.0; cells],
            varblock_sigma: vec![0.0; cells],
        })
    }

    /// Sets both sigma values at one frame-relative 8x8 block.
    pub fn set(&mut self, bx: usize, by: usize, sigma: f32, varblock_sigma: f32) {
        if bx >= self.blocks_x || by >= self.blocks_y {
            return;
        }
        let idx = by * self.blocks_x + bx;
        if let Some(slot) = self.sigma.get_mut(idx) {
            *slot = sigma;
        }
        if let Some(slot) = self.varblock_sigma.get_mut(idx) {
            *slot = varblock_sigma;
        }
    }

    /// Borrows the two planes as the [`SigmaField`] the filter takes.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Frame`] if the two planes disagree with the grid, which
    /// only a bug in this module could cause.
    pub fn field(&self) -> Result<SigmaField<'_>> {
        Ok(SigmaField::new(&self.sigma, self.blocks_x, self.blocks_y)?
            .with_varblock_sigma(&self.varblock_sigma)?)
    }
}

/// J.4.3's sigma for one varblock: `mul * epf_quant_mul * epf_sharp_lut[s]`.
///
/// `mul` is I.5.3's `Mul` for the varblock, which is why this takes the
/// dequantization parameters rather than a bare number — the two clauses have
/// to agree on the same value, and computing it twice from different inputs is
/// how they would silently drift apart.
///
/// # Errors
///
/// [`DecodeError::Frame`] if `sharpness` is outside `epf_sharp_lut`.
pub fn varblock_epf_sigma(
    params: &HfDequantParams<'_>,
    hf_mul: u32,
    sharpness: u32,
    filter: &RestorationFilter,
) -> Result<f32> {
    Ok(vardct_sigma(
        params.hf_multiplier(hf_mul),
        sharpness,
        &filter.epf,
    )?)
}

/// Applies J.3 gaborish and J.4 EPF to the XYB planes, honouring the frame's
/// enable flags.
///
/// Both filters are no-ops when their flag is off (`!gab`, `epf_iters == 0`),
/// which is what makes the filters-off fixtures a clean test of I.5–I.9.
///
/// # Errors
///
/// [`DecodeError::Frame`] if a gaborish kernel is degenerate or a plane length
/// disagrees with the frame dimensions.
pub fn apply_restoration(
    planes: &mut ColourPlanes,
    filter: &RestorationFilter,
    sigma: &SigmaPlanes,
) -> Result<()> {
    let dims = planes.dims();

    if filter.gab {
        let mut out = [
            vec![0.0f32; dims.len()],
            vec![0.0f32; dims.len()],
            vec![0.0f32; dims.len()],
        ];
        for (c, (dst, src)) in out.iter_mut().zip(&planes.planes).enumerate() {
            let w1 = filter.gab_weights.weight1.get(c).copied().unwrap_or(0.0);
            let w2 = filter.gab_weights.weight2.get(c).copied().unwrap_or(0.0);
            let kernel = GaborKernel::new(w1, w2)?;
            gaborish_into(src, dst, dims, &kernel)?;
        }
        planes.planes = out;
    }

    if filter.epf.iters > 0 {
        let field = sigma.field()?;
        planes.planes = epf(planes.as_slices(), dims, &filter.epf, &field)?;
    }

    Ok(())
}

/// K.3.2: blends the patch dictionary onto the frame's planes.
///
/// Runs after Annex J and before Annex L, on the same XYB planes — see the
/// [`patches`](crate::frame::patches) module documentation for the two places
/// the standard pins that position.
///
/// `extra` carries the frame's extra channels on the `[0, 1]` scale of G.4.2's
/// last paragraph; they are blended too, and they are also where the alpha of
/// an alpha-using mode is read from.
///
/// # The per-channel-group loop
///
/// K.3.2 iterates `c` over `[0, num_extra]`, where `c == 0` means all three
/// colour channels together and `c > 0` means extra channel `c - 1`. Each `c`
/// has its own mode, alpha channel and clamp flag, so a patch can (and
/// `patches`/`patches_lossless` do) replace the colour and alpha-blend the
/// alpha, or vice versa.
///
/// # Which alpha
///
/// See [`PATCH_ALPHA_IS_THE_PATCHS_OWN`](crate::vardct::render::PATCH_ALPHA_IS_THE_PATCHS_OWN).
///
/// # Errors
///
/// [`DecodeError::Unsupported`] for an alpha-using blend mode on an image with
/// no extra channel to read alpha from; [`DecodeError::FieldOutOfRange`] for a
/// patch naming an unwritten reference slot.
pub fn apply_patches(
    planes: &mut ColourPlanes,
    extra: &mut ExtraPlanes,
    dictionary: &PatchDictionary,
    references: &[Option<ReferenceFrame>],
) -> Result<()> {
    if dictionary.patches.is_empty() {
        return Ok(());
    }

    for patch in &dictionary.patches {
        let slot = usize::try_from(patch.reference).unwrap_or(usize::MAX);
        let reference = references
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                DecodeError::out_of_range(
                    "patch reference slot",
                    "K.3.1",
                    u64::from(patch.reference),
                )
            })?;

        for position in &patch.positions {
            for (group, rule) in position.blending.iter().enumerate() {
                if rule.mode == PatchBlendMode::None {
                    continue;
                }
                let alpha_index = usize::try_from(rule.alpha_channel).unwrap_or(usize::MAX);
                if rule.mode.uses_alpha() && alpha_index >= extra.len() {
                    return Err(DecodeError::Unsupported {
                        feature: "an alpha-blending patch mode without an alpha channel",
                        clause: "18181-1 K.3.2",
                    });
                }

                for iy in 0..patch.height {
                    for ix in 0..patch.width {
                        // K.3.1 already proved (x, y) + (width, height) is
                        // inside the frame, so no clipping is needed here; the
                        // plane types still ignore an out-of-frame write,
                        // which keeps a future caller honest rather than
                        // silently corrupting.
                        let (dx, dy) = (position.x + ix, position.y + iy);
                        let (sx, sy) = (patch.x0 + ix, patch.y0 + iy);
                        let alpha = if rule.mode.uses_alpha() {
                            let a = if PATCH_ALPHA_IS_THE_PATCHS_OWN {
                                reference.get_extra(alpha_index, sx, sy)
                            } else {
                                extra.get(alpha_index, dx, dy)
                            };
                            // K.3.2: "If clamp is true, alpha values are
                            // clamped to the interval [0, 1] before blending."
                            if rule.clamp { a.clamp(0.0, 1.0) } else { a }
                        } else {
                            0.0
                        };

                        if group == 0 {
                            for c in 0..NUM_CHANNELS {
                                let new_sample = reference.get(c, sx, sy);
                                let old_sample = planes.get(c, dx, dy);
                                planes.set(
                                    c,
                                    dx,
                                    dy,
                                    patch_blend(rule.mode, old_sample, new_sample, alpha),
                                );
                            }
                        } else {
                            let c = group - 1;
                            let new_sample = reference.get_extra(c, sx, sy);
                            let old_sample = extra.get(c, dx, dy);
                            extra.set(
                                c,
                                dx,
                                dy,
                                patch_blend(rule.mode, old_sample, new_sample, alpha),
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// **Flip point — whose alpha K.3.2 blends with.**
///
/// Table K.1's alpha modes say "the alpha channel is the extra channel with
/// index `k = patch[i].blending[j].alpha_channel[c]`" and stop there: the
/// clause never says whether that channel is read from the *patch* (the
/// reference frame) or from the *canvas* (the frame being drawn on). Table
/// F.7's `kBlend`, which K.1 defers to, is a frame-blending rule where alpha
/// unambiguously belongs to the frame being composited — i.e. to the new
/// sample.
///
/// * `true` (shipped): alpha comes from the reference frame, at the patch's
///   own `(x0 + ix, y0 + iy)`. A patch then carries its own opacity, which is
///   the only reading under which a patch dictionary can express an
///   antialiased glyph over arbitrary background — the use K.3 exists for.
///   It also makes `kBlendAbove` on the colour group agree with what the
///   *same position's* alpha-channel rule writes into the alpha plane.
/// * `false`: alpha comes from the canvas at the blit position.
///
/// **Unexercised.** `patches`'s 654 positions blend the colour group with
/// `kAdd` and the alpha group with `kNone`, and no other available stream uses
/// a Table K.1 row above 3 at all, so nothing discriminates the two readings.
/// See `docs/experiments/2026-08-04-vardct-extra-channels-and-frame-blending.md`
/// §4.
pub const PATCH_ALPHA_IS_THE_PATCHS_OWN: bool = true;

/// L.2.2: converts the frame's XYB planes to linear sRGB in place.
pub fn to_linear_srgb(planes: &mut ColourPlanes, opsin: &OpsinInverse) {
    let [x, y, b] = &mut planes.planes;
    opsin.convert_planes(x, y, b);
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::unwrap_used,
    reason = "tests index fixed-size structures they just built; a panic here \
              is a failing test"
)]
mod tests {
    use super::*;
    use jpxl_core::limits::Limits;

    fn params(matrices: &DequantMatrices) -> HfDequantParams<'_> {
        HfDequantParams {
            matrices,
            quant_bias: [0.5, 0.25, 0.125],
            quant_bias_numerator: 0.145,
            global_scale: 4096,
            x_qm_scale: 3,
            b_qm_scale: 2,
        }
    }

    #[test]
    fn qm_multiplier_is_one_for_y_and_a_power_of_zero_point_eight_otherwise() {
        // Proves I.5.3's per-channel factor is applied to X and B only, with
        // the exponent offset by 2 (so the default b_qm_scale == 2 is a no-op
        // and the default x_qm_scale == 3 multiplies by exactly 0.8).
        let m = DequantMatrices::all_default().unwrap();
        let p = params(&m);
        assert_eq!(p.qm_multiplier(1), 1.0);
        assert!((p.qm_multiplier(0) - 0.8).abs() < 1e-7);
        assert_eq!(p.qm_multiplier(2), 1.0);

        let p = HfDequantParams {
            x_qm_scale: 0,
            b_qm_scale: 5,
            ..p
        };
        assert!((p.qm_multiplier(0) - 1.0 / 0.64).abs() < 1e-6);
        assert!((p.qm_multiplier(2) - 0.8f32.powi(3)).abs() < 1e-7);
    }

    #[test]
    fn hf_multiplier_matches_the_hand_computed_formula() {
        // Mul = 65536 / (global_scale * HfMul), hand-computed.
        let m = DequantMatrices::all_default().unwrap();
        let p = params(&m);
        assert!((p.hf_multiplier(1) - 65536.0 / 4096.0).abs() < 1e-6);
        assert!((p.hf_multiplier(16) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bias_adjust_follows_the_two_printed_branches() {
        // The branch boundary is on the quantized integer: |q| <= 1 scales,
        // |q| > 1 subtracts the reciprocal term. Both are checked at the
        // boundary itself, which is where a `<` / `<=` slip would show.
        let m = DequantMatrices::all_default().unwrap();
        let p = params(&m);

        assert_eq!(p.bias_adjust(0, 0), 0.0);
        assert!((p.bias_adjust(1, 0) - 0.5).abs() < 1e-7);
        assert!((p.bias_adjust(-1, 1) + 0.25).abs() < 1e-7);
        // |q| == 2 takes the other branch: 2 - 0.145/2.
        assert!((p.bias_adjust(2, 0) - (2.0 - 0.145 / 2.0)).abs() < 1e-6);
        assert!((p.bias_adjust(-2, 2) - (-2.0 + 0.145 / 2.0)).abs() < 1e-6);
    }

    #[test]
    fn an_all_zero_block_dequantizes_to_all_zeros() {
        // Proves the LLF cells 8C never writes contribute nothing before I.8
        // overwrites them, whatever the bias bundle says.
        let m = DequantMatrices::all_default().unwrap();
        let p = params(&m);
        for t in [
            TransformType::Dct8x8,
            TransformType::Dct16x8,
            TransformType::Afv0,
        ] {
            let q = QuantCoeffBlock::zeros(t);
            for c in 0..NUM_CHANNELS {
                let out = dequantize_hf_block(t, c, &q, 3, &p).unwrap();
                assert!(
                    out.as_slice().iter().all(|v| *v == 0.0),
                    "{t:?} channel {c}"
                );
            }
        }
    }

    #[test]
    fn dequantized_shape_is_the_transforms_landscape_coefficient_shape() {
        // The orientation trap: a DCT16x8's coefficients are 8 rows x 16
        // columns even though its samples are 16 rows x 8 columns.
        let m = DequantMatrices::all_default().unwrap();
        let p = params(&m);
        let t = TransformType::Dct16x8;
        let out = dequantize_hf_block(t, 1, &QuantCoeffBlock::zeros(t), 1, &p).unwrap();
        assert_eq!((out.rows(), out.cols()), (8, 16));
        assert_eq!((t.sample_rows(), t.sample_cols()), (16, 8));
    }

    #[test]
    fn hf_cfl_leaves_y_alone_and_shifts_x_and_b_by_multiples_of_y() {
        let t = TransformType::Dct8x8;
        let mut coeffs = [
            t.empty_coefficients(),
            t.empty_coefficients(),
            t.empty_coefficients(),
        ];
        coeffs[0].set(1, 0, 1.0);
        coeffs[1].set(1, 0, 4.0);
        coeffs[2].set(1, 0, -2.0);

        apply_hf_cfl(&mut coeffs, 0.5, -0.25);
        assert_eq!(coeffs[1].at(1, 0), 4.0, "Y is never modified");
        assert_eq!(coeffs[0].at(1, 0), 1.0 + 0.5 * 4.0);
        assert_eq!(coeffs[2].at(1, 0), -2.0 + -0.25 * 4.0);
    }

    #[test]
    fn a_flat_lf_and_no_hf_reconstructs_a_flat_varblock() {
        // The end-to-end identity for the whole render path: a varblock whose
        // HF coefficients are all zero and whose LF rectangle is a constant
        // must reconstruct as that same constant, for every transform type.
        // It proves the I.8 -> I.9 handoff (the LLF DC scaling, the DC-only
        // IDCT normalization, and the landscape/portrait flip) all at once.
        let m = DequantMatrices::all_default().unwrap();
        let p = params(&m);

        for t in TransformType::ALL {
            let (rows, cols) = t.block_dims();
            let lf = DequantPlane {
                width: cols as u32,
                height: rows as u32,
                samples: vec![0.375; rows * cols],
            };
            let q = QuantCoeffBlock::zeros(t);
            let blocks =
                render_varblock(t, [&q, &q, &q], 1, &p, (0.0, 0.0), [&lf, &lf, &lf], 0, 0).unwrap();
            for (c, block) in blocks.iter().enumerate() {
                assert_eq!(
                    (block.rows(), block.cols()),
                    (t.sample_rows(), t.sample_cols()),
                    "{t:?} channel {c}"
                );
                for (i, v) in block.as_slice().iter().enumerate() {
                    assert!(
                        (v - 0.375).abs() < 1e-4,
                        "{t:?} channel {c} sample {i} = {v}"
                    );
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // K.3.2 — patch rendering
    // ------------------------------------------------------------------

    fn one_slot(reference: ReferenceFrame) -> [Option<ReferenceFrame>; 4] {
        [Some(reference), None, None, None]
    }

    /// A 2x2 reference at the canvas origin whose three planes are constants.
    fn flat_reference(x: f32, y: f32, b: f32) -> ReferenceFrame {
        ReferenceFrame {
            origin_x: 0,
            origin_y: 0,
            width: 2,
            height: 2,
            planes: [vec![x; 4], vec![y; 4], vec![b; 4]],
            extra: Vec::new(),
        }
    }

    /// The same 2x2 reference with one constant extra channel.
    fn flat_reference_with_alpha(x: f32, y: f32, b: f32, alpha: f32) -> ReferenceFrame {
        ReferenceFrame {
            extra: vec![vec![alpha; 4]],
            ..flat_reference(x, y, b)
        }
    }

    /// A dictionary whose position carries one rule per channel group.
    fn dictionary_groups(rules: &[(PatchBlendMode, u32, bool)], at: (u32, u32)) -> PatchDictionary {
        PatchDictionary {
            patches: vec![crate::frame::patches::Patch {
                reference: 0,
                x0: 0,
                y0: 0,
                width: 2,
                height: 2,
                positions: vec![crate::frame::patches::PatchPosition {
                    x: at.0,
                    y: at.1,
                    blending: rules
                        .iter()
                        .map(
                            |&(mode, alpha_channel, clamp)| crate::frame::patches::PatchBlending {
                                mode,
                                alpha_channel,
                                clamp,
                            },
                        )
                        .collect(),
                }],
            }],
        }
    }

    /// No extra channels, for the tests that only exercise the colour group.
    fn no_extra() -> ExtraPlanes {
        ExtraPlanes::empty(0, 0)
    }

    fn dictionary(mode: PatchBlendMode, at: (u32, u32)) -> PatchDictionary {
        PatchDictionary {
            patches: vec![crate::frame::patches::Patch {
                reference: 0,
                x0: 0,
                y0: 0,
                width: 2,
                height: 2,
                positions: vec![crate::frame::patches::PatchPosition {
                    x: at.0,
                    y: at.1,
                    blending: vec![crate::frame::patches::PatchBlending {
                        mode,
                        alpha_channel: 0,
                        clamp: false,
                    }],
                }],
            }],
        }
    }

    #[test]
    fn a_patch_lands_only_on_its_own_rectangle() {
        // The blit's bounds are the whole geometry of K.3.2: a patch that
        // leaked one row or column would still look plausible on a flat
        // canvas, so the assertion is per-sample over the entire plane.
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut planes = ColourPlanes::zeros(5, 4, &mut guard).unwrap();
        let refs = one_slot(flat_reference(0.5, 0.25, -0.125));

        apply_patches(
            &mut planes,
            &mut no_extra(),
            &dictionary(PatchBlendMode::Replace, (1, 2)),
            &refs,
        )
        .unwrap();

        for y in 0..4 {
            for x in 0..5 {
                let inside = (1..3).contains(&x) && (2..4).contains(&y);
                let want = if inside {
                    [0.5, 0.25, -0.125]
                } else {
                    [0.0; 3]
                };
                for (c, w) in want.into_iter().enumerate() {
                    assert_eq!(planes.get(c, x, y), w, "channel {c} at ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn the_patch_reads_its_own_origin_in_the_reference() {
        // patch.x0/y0 index the reference; position.x/y index the canvas.
        // Swapping the two pairs is invisible when both are zero, so the test
        // makes them different.
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut planes = ColourPlanes::zeros(4, 4, &mut guard).unwrap();

        // A 4x4 reference whose Y plane is its raster index.
        let reference = ReferenceFrame {
            origin_x: 0,
            origin_y: 0,
            width: 4,
            height: 4,
            planes: [
                vec![0.0; 16],
                (0..16).map(|i| i as f32).collect(),
                vec![0.0; 16],
            ],
            extra: Vec::new(),
        };
        let dict = PatchDictionary {
            patches: vec![crate::frame::patches::Patch {
                reference: 0,
                x0: 2,
                y0: 1,
                width: 2,
                height: 2,
                positions: vec![crate::frame::patches::PatchPosition {
                    x: 0,
                    y: 0,
                    blending: vec![crate::frame::patches::PatchBlending {
                        mode: PatchBlendMode::Replace,
                        alpha_channel: 0,
                        clamp: false,
                    }],
                }],
            }],
        };
        apply_patches(&mut planes, &mut no_extra(), &dict, &one_slot(reference)).unwrap();

        // Reference rows 1 and 2, columns 2 and 3: indices 6, 7, 10, 11.
        assert_eq!(planes.get(1, 0, 0), 6.0);
        assert_eq!(planes.get(1, 1, 0), 7.0);
        assert_eq!(planes.get(1, 0, 1), 10.0);
        assert_eq!(planes.get(1, 1, 1), 11.0);
    }

    #[test]
    fn every_alpha_free_blend_mode_reaches_the_canvas() {
        // kAdd is the mode real encoders emit (all 94 of bike_5's patches use
        // it), but the other three cost nothing to pin.
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let refs = one_slot(flat_reference(0.5, 0.5, 0.5));

        for (mode, want) in [
            (PatchBlendMode::None, 0.25),
            (PatchBlendMode::Replace, 0.5),
            (PatchBlendMode::Add, 0.75),
            (PatchBlendMode::Mul, 0.125),
        ] {
            let mut planes = ColourPlanes::zeros(2, 2, &mut guard).unwrap();
            for plane in &mut planes.planes {
                plane.fill(0.25);
            }
            apply_patches(
                &mut planes,
                &mut no_extra(),
                &dictionary(mode, (0, 0)),
                &refs,
            )
            .unwrap();
            assert_eq!(planes.get(1, 0, 0), want, "{mode:?}");
        }
    }

    #[test]
    fn an_unwritten_reference_slot_is_rejected_not_ignored() {
        // A patch naming a slot no frame filled is a malformed stream; a
        // decoder that silently blitted zeros would produce plausible-looking
        // holes instead of an error.
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut planes = ColourPlanes::zeros(2, 2, &mut guard).unwrap();
        let empty: [Option<ReferenceFrame>; 4] = [None, None, None, None];
        assert!(
            apply_patches(
                &mut planes,
                &mut no_extra(),
                &dictionary(PatchBlendMode::Add, (0, 0)),
                &empty,
            )
            .is_err()
        );
    }

    #[test]
    fn alpha_modes_and_extra_channels_are_refused_not_guessed() {
        // These planes carry no alpha, so an alpha-blending mode has nothing
        // to read. Refusing is the difference between "not implemented" and
        // "wrong pixels".
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut planes = ColourPlanes::zeros(2, 2, &mut guard).unwrap();
        let refs = one_slot(flat_reference(0.5, 0.5, 0.5));
        for mode in [
            PatchBlendMode::BlendAbove,
            PatchBlendMode::BlendBelow,
            PatchBlendMode::MulAddAbove,
            PatchBlendMode::MulAddBelow,
        ] {
            assert!(
                apply_patches(
                    &mut planes,
                    &mut no_extra(),
                    &dictionary(mode, (0, 0)),
                    &refs
                )
                .is_err(),
                "{mode:?}"
            );
        }
    }

    #[test]
    fn each_channel_group_follows_its_own_rule() {
        // K.3.2 iterates c over [0, num_extra]: c == 0 is all three colour
        // channels together, c > 0 is extra channel c - 1. This is the shape
        // the `patches` corpus case has -- kAdd on colour, kNone on alpha --
        // and a decoder that applied the colour rule to the extra channel
        // would overwrite the alpha plane with the patch's.
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut planes = ColourPlanes::zeros(2, 2, &mut guard).unwrap();
        for plane in &mut planes.planes {
            plane.fill(0.25);
        }
        let mut extra = ExtraPlanes::zeros(2, 2, 1, &mut guard).unwrap();
        extra.planes[0].fill(0.5);
        let refs = one_slot(flat_reference_with_alpha(0.5, 0.5, 0.5, 0.125));

        apply_patches(
            &mut planes,
            &mut extra,
            &dictionary_groups(
                &[
                    (PatchBlendMode::Add, 0, false),
                    (PatchBlendMode::None, 0, false),
                ],
                (0, 0),
            ),
            &refs,
        )
        .unwrap();

        assert_eq!(planes.get(1, 0, 0), 0.75, "colour took kAdd");
        assert_eq!(extra.get(0, 0, 0), 0.5, "kNone left the alpha plane alone");
    }

    #[test]
    fn an_extra_channel_group_blends_into_the_extra_plane() {
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut planes = ColourPlanes::zeros(2, 2, &mut guard).unwrap();
        let mut extra = ExtraPlanes::zeros(2, 2, 1, &mut guard).unwrap();
        extra.planes[0].fill(0.25);
        let refs = one_slot(flat_reference_with_alpha(0.0, 0.0, 0.0, 0.5));

        apply_patches(
            &mut planes,
            &mut extra,
            &dictionary_groups(
                &[
                    (PatchBlendMode::None, 0, false),
                    (PatchBlendMode::Replace, 0, false),
                ],
                (0, 0),
            ),
            &refs,
        )
        .unwrap();

        assert_eq!(extra.get(0, 0, 0), 0.5, "the patch's own extra channel");
        assert_eq!(planes.get(1, 0, 0), 0.0, "kNone left the colour alone");
    }

    #[test]
    fn an_alpha_mode_reads_the_alpha_the_flip_point_names() {
        // With PATCH_ALPHA_IS_THE_PATCHS_OWN the alpha comes from the
        // reference frame, so kBlendAbove at alpha 0.5 lands halfway between
        // the canvas and the patch regardless of the canvas's own alpha.
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut planes = ColourPlanes::zeros(2, 2, &mut guard).unwrap();
        for plane in &mut planes.planes {
            plane.fill(0.25);
        }
        let mut extra = ExtraPlanes::zeros(2, 2, 1, &mut guard).unwrap();
        extra.planes[0].fill(1.0);
        let refs = one_slot(flat_reference_with_alpha(0.75, 0.75, 0.75, 0.5));

        apply_patches(
            &mut planes,
            &mut extra,
            &dictionary_groups(
                &[
                    (PatchBlendMode::BlendAbove, 0, false),
                    (PatchBlendMode::None, 0, false),
                ],
                (0, 0),
            ),
            &refs,
        )
        .unwrap();

        let want = if PATCH_ALPHA_IS_THE_PATCHS_OWN {
            // alpha 0.5 from the reference: 0.25 + 0.5 * (0.75 - 0.25).
            0.5
        } else {
            // alpha 1.0 from the canvas: the patch wins outright.
            0.75
        };
        assert_eq!(planes.get(1, 0, 0), want);
    }

    #[test]
    fn a_clamped_alpha_is_clamped_before_blending() {
        // K.3.2: "If clamp is true, alpha values are clamped to the interval
        // [0, 1] before blending." An alpha of 2.0 would otherwise overshoot
        // past the patch sample.
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut extra = ExtraPlanes::zeros(2, 2, 1, &mut guard).unwrap();
        let refs = one_slot(flat_reference_with_alpha(1.0, 1.0, 1.0, 2.0));

        for (clamp, want) in [(true, 1.0f32), (false, 1.75)] {
            let mut planes = ColourPlanes::zeros(2, 2, &mut guard).unwrap();
            for plane in &mut planes.planes {
                plane.fill(0.25);
            }
            apply_patches(
                &mut planes,
                &mut extra,
                &dictionary_groups(
                    &[
                        (PatchBlendMode::BlendAbove, 0, clamp),
                        (PatchBlendMode::None, 0, false),
                    ],
                    (0, 0),
                ),
                &refs,
            )
            .unwrap();
            assert_eq!(planes.get(1, 0, 0), want, "clamp = {clamp}");
        }
    }

    #[test]
    fn an_empty_dictionary_is_a_no_op() {
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut planes = ColourPlanes::zeros(3, 3, &mut guard).unwrap();
        for plane in &mut planes.planes {
            plane.fill(0.5);
        }
        let before = planes.clone();
        let empty: [Option<ReferenceFrame>; 4] = [None, None, None, None];
        apply_patches(
            &mut planes,
            &mut no_extra(),
            &PatchDictionary::default(),
            &empty,
        )
        .unwrap();
        assert_eq!(planes, before);
    }

    #[test]
    fn restoration_is_the_identity_when_both_filters_are_off() {
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut planes = ColourPlanes::zeros(17, 9, &mut guard).unwrap();
        for (i, v) in planes.planes[1].iter_mut().enumerate() {
            *v = i as f32 / 100.0;
        }
        let before = planes.clone();

        let filter = RestorationFilter {
            gab: false,
            epf: crate::frame::EpfParams {
                iters: 0,
                ..crate::frame::EpfParams::default()
            },
            ..RestorationFilter::default()
        };
        let sigma = SigmaPlanes::zeros(17, 9, &mut guard).unwrap();
        apply_restoration(&mut planes, &filter, &sigma).unwrap();
        assert_eq!(planes, before);
    }

    #[test]
    fn sigma_planes_cover_the_frames_block_grid() {
        // A frame whose dimensions are not multiples of 8 still has a block
        // for the partial edge; the filter would reject a mismatched grid.
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let planes = ColourPlanes::zeros(17, 9, &mut guard).unwrap();
        let sigma = SigmaPlanes::zeros(17, 9, &mut guard).unwrap();
        assert_eq!((sigma.blocks_x, sigma.blocks_y), (3, 2));
        assert_eq!(
            crate::frame::epf::block_grid(planes.dims()),
            (sigma.blocks_x, sigma.blocks_y)
        );
        sigma.field().unwrap();
    }
}
