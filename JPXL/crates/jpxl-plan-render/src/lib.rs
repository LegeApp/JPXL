//! Reconstructs the pixels a conforming decoder will produce from a validated
//! VarDCT pixel plan — on the encoder side, from the plan alone.
//!
//! A perceptual quality search needs reconstructed pixels for every probe.
//! Emitting a codestream and decoding it would pay for entropy training, ANS
//! tables, section layout and a full parse that the probe does not need, so
//! this crate runs the reconstruction directly on the plan's integers:
//!
//! ```text
//! ValidatedPixelPlan
//!   ├─ I.5.2  LF dequantization, I.6 LF chroma-from-luma
//!   ├─ I.5.3  HF dequantization        ─┐
//!   ├─ I.6    HF chroma-from-luma       │ per varblock
//!   ├─ I.8    LLF from the LF planes    │
//!   ├─ I.9    inverse transform        ─┘
//!   ├─ J.3    Gabor-like transform, J.4 edge-preserving filter
//!   ├─ L.2.2  XYB → linear sRGB
//!   └─ transfer function, then optional quantization to the frame's depth
//! ```
//!
//! # Boundary
//!
//! This crate is encoder-side. It never calls `jpxl-decode`; the kernels it
//! shares with the decoder live in `jpxl-core` (`dct`, `varblock`,
//! `dequant`, `color`, `reconstruct`, `restoration`), and the orchestration
//! here is its own. Agreement with `jpxl-decode` and the external decoders is
//! established by the parity tests, which is what keeps an encoder-side bug
//! from being accepted by its paired decoder.
//!
//! # What it does not model
//!
//! Adaptive LF smoothing (the encoder always signals
//! `kSkipAdaptiveLFSmoothing`), upsampling, patches, noise, splines, extra
//! channels, and non-sRGB output encodings. A plan asking for any of these
//! is refused with [`RenderError::Unsupported`] rather than rendered wrongly.

use jpxl_core::JpxlError;
use jpxl_core::color::{OpsinInverse, linear_to_srgb, srgb_to_linear};
use jpxl_core::dequant::{DequantMatrices, DequantMatrix};
use jpxl_core::reconstruct::{
    LF_WEIGHT_SCALE, bias_adjust, cfl_apply, cfl_factors, hf_multiplier, lf_dequantize,
    lf_multipliers, qm_multiplier,
};
use jpxl_core::restoration::{
    EPF_PAD, EpfParams, GaborKernel, PaddedPlane, PlaneDims, SigmaField, block_grid, epf_step_rows,
    epf_steps, gaborish_into, vardct_sigma,
};
use jpxl_core::varblock::{CoeffMatrix, SampleBlock, TransformType, llf_from_lf};
use jpxl_encode::EncodeExecutor;
use jpxl_encode::vardct::plan::NUM_CHANNELS;
use jpxl_encode::vardct::{PlanError, ValidatedPixelPlan, VardctGeometry};

/// Table L.1's default `OpsinInverseMatrix`, row-major, as the decoder
/// reads it for an `all_default` bundle.
const DEFAULT_INVERSE_MATRIX: [f32; 9] = [
    11.031_567,
    -9.866_944,
    -0.164_622_99,
    -3.254_147_4,
    4.418_770_5,
    -0.164_622_99,
    -3.658_851_3,
    2.712_923,
    1.945_928_2,
];

/// Table L.1's default `opsin_bias`, as signalled (negative).
const DEFAULT_OPSIN_BIAS: [f32; 3] = [-0.003_793_073_3; 3];

/// Table L.1's default `quant_bias`.
const DEFAULT_QUANT_BIAS: [f32; 3] = jpxl_core::color::DEFAULT_QUANT_BIAS;

/// Table L.1's default `quant_bias_numerator`.
const DEFAULT_QUANT_BIAS_NUMERATOR: f32 = jpxl_core::color::DEFAULT_QUANT_BIAS_NUMERATOR;

/// Side, in samples, of the tiles the HF chroma-from-luma factors cover.
const CFL_TILE_BLOCKS: u32 = 8;

/// Why a plan could not be rendered.
#[derive(Debug)]
pub enum RenderError {
    /// The plan's geometry or structure was rejected.
    Plan(PlanError),
    /// A core primitive failed (a dequantization matrix, a limit).
    Core(JpxlError),
    /// The plan uses a feature this renderer does not model.
    Unsupported(&'static str),
}

impl core::fmt::Display for RenderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Plan(e) => write!(f, "plan: {e}"),
            Self::Core(e) => write!(f, "core: {e}"),
            Self::Unsupported(what) => write!(f, "unsupported by the plan renderer: {what}"),
        }
    }
}

impl std::error::Error for RenderError {}

impl From<PlanError> for RenderError {
    fn from(e: PlanError) -> Self {
        Self::Plan(e)
    }
}

impl From<JpxlError> for RenderError {
    fn from(e: JpxlError) -> Self {
        Self::Core(e)
    }
}

/// The crate's result type.
pub type Result<T> = core::result::Result<T, RenderError>;

/// A rendered frame: three planes in the signalled (sRGB-encoded) colour
/// encoding, unclipped, exactly what a decoder holds before quantizing to
/// the frame's bit depth.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedFrame {
    width: u32,
    height: u32,
    bits_per_sample: u32,
    planes: [Vec<f32>; NUM_CHANNELS],
}

impl RenderedFrame {
    /// Width in samples.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in samples.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The frame's signalled bit depth.
    #[must_use]
    pub const fn bits_per_sample(&self) -> u32 {
        self.bits_per_sample
    }

    /// The sRGB-encoded planes, `[R, G, B]`, unclipped.
    #[must_use]
    pub fn encoded_planes(&self) -> [&[f32]; NUM_CHANNELS] {
        self.planes.each_ref().map(Vec::as_slice)
    }

    /// Full scale of a `bits` deep integer sample.
    fn full_scale(bits: u32) -> f32 {
        if bits >= 32 {
            f32::from(u16::MAX)
        } else {
            ((1u32 << bits.max(1)) - 1) as f32
        }
    }

    /// The planes quantized to `bits` per sample exactly as the decoder's
    /// integer output is: scaled, rounded and clamped.
    #[must_use]
    pub fn quantized(&self, bits: u32) -> [Vec<i32>; NUM_CHANNELS] {
        let max = Self::full_scale(bits);
        self.planes.each_ref().map(|plane| {
            plane
                .iter()
                .map(|&v| {
                    let scaled = (v * max).round();
                    if scaled.is_finite() {
                        // Clamped into [0, max] with max < 2^32 before the
                        // cast, so the narrowing is exact.
                        #[allow(
                            clippy::cast_possible_truncation,
                            reason = "the value is clamped to [0, max] first"
                        )]
                        let q = scaled.clamp(0.0, max) as i32;
                        q
                    } else {
                        0
                    }
                })
                .collect()
        })
    }

    /// Linear-sRGB planes after a round trip through `bits`-deep integer
    /// samples — the image a viewer of the decoded file actually sees, and
    /// therefore what a perceptual metric should score.
    #[must_use]
    pub fn linear_rgb_at_depth(&self, bits: u32) -> [Vec<f32>; NUM_CHANNELS] {
        let mut out: [Vec<f32>; NUM_CHANNELS] = [Vec::new(), Vec::new(), Vec::new()];
        self.linear_rgb_at_depth_into(bits, &mut out);
        out
    }

    /// [`Self::linear_rgb_at_depth`] writing into caller-owned buffers, so a
    /// search that scores many candidates reuses one set of planes instead of
    /// allocating three per probe. Each output is cleared and refilled to
    /// exactly `width * height` samples.
    ///
    /// The quantized integer is formed and looked up per sample in one pass,
    /// so no full-frame `i32` plane is materialised; the values are identical
    /// to `linear_rgb_at_depth` (and to [`Self::quantized`] followed by the
    /// same lookup) sample for sample.
    pub fn linear_rgb_at_depth_into(&self, bits: u32, out: &mut [Vec<f32>; NUM_CHANNELS]) {
        let max = Self::full_scale(bits);
        // One transfer-curve evaluation per representable integer, not per
        // sample: the round trip is a table lookup for every real bit depth.
        let entries = if bits >= 32 {
            65_536
        } else {
            1usize << bits.clamp(1, 16)
        };
        let lut: Vec<f32> = (0..entries)
            .map(|q| srgb_to_linear(q as f32 / max))
            .collect();
        for (dst, plane) in out.iter_mut().zip(self.planes.iter()) {
            dst.clear();
            dst.reserve(plane.len());
            dst.extend(plane.iter().map(|&v| {
                let scaled = (v * max).round();
                let q = if scaled.is_finite() {
                    // Clamped into [0, max] with max < 2^32 before the cast,
                    // so the narrowing is exact.
                    #[allow(
                        clippy::cast_possible_truncation,
                        reason = "the value is clamped to [0, max] first"
                    )]
                    let q = scaled.clamp(0.0, max) as i32;
                    q
                } else {
                    0
                };
                usize::try_from(q)
                    .ok()
                    .and_then(|q| lut.get(q))
                    .copied()
                    .unwrap_or_else(|| srgb_to_linear(q as f32 / max))
            }));
        }
    }
}

/// Wall time of one render's stages, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RenderTimings {
    /// LF dequantization, HF dequantization, CfL, LLF and inverse transforms.
    pub varblocks_ms: u64,
    /// J.3 Gabor-like transform.
    pub gaborish_ms: u64,
    /// J.4 edge-preserving filter, all steps.
    pub epf_ms: u64,
    /// XYB to linear sRGB and the transfer function.
    pub colour_ms: u64,
}

/// Reusable renderer: the dequantization matrices and the opsin inverse are
/// built once and shared across every plan rendered.
#[derive(Debug)]
pub struct PlanRenderer {
    matrices: DequantMatrices,
    cache: Vec<Option<[DequantMatrix; NUM_CHANNELS]>>,
    opsin: OpsinInverse,
    epf_params: EpfParams,
    gabor: GaborKernel,
}

impl PlanRenderer {
    /// A renderer over the Table I.6 default dequantization matrices and the
    /// Table L.1 default opsin inverse.
    ///
    /// # Errors
    ///
    /// [`RenderError::Core`] if the default matrices cannot be built.
    pub fn new() -> Result<Self> {
        Ok(Self {
            matrices: DequantMatrices::all_default()?,
            cache: vec![None; jpxl_core::varblock::NUM_DEQUANT_MATRICES],
            opsin: OpsinInverse::new(
                DEFAULT_INVERSE_MATRIX,
                DEFAULT_OPSIN_BIAS,
                jpxl_core::color::NOMINAL_INTENSITY_TARGET,
            ),
            epf_params: EpfParams::default(),
            gabor: GaborKernel::defaults(),
        })
    }

    /// The three dequantization matrices for a transform, built on first use.
    fn matrices_for(&mut self, transform: TransformType) -> Result<&[DequantMatrix; NUM_CHANNELS]> {
        let index = transform.dequant_matrix_index();
        let slot = self.cache.get_mut(index).ok_or(RenderError::Unsupported(
            "a dequantization matrix index past Table I.4",
        ))?;
        if slot.is_none() {
            *slot = Some([
                self.matrices.matrix(index, 0)?,
                self.matrices.matrix(index, 1)?,
                self.matrices.matrix(index, 2)?,
            ]);
        }
        slot.as_ref().ok_or(RenderError::Unsupported(
            "a dequantization matrix that failed to build",
        ))
    }

    /// Renders `pixels` to the signalled colour encoding.
    ///
    /// # Errors
    ///
    /// [`RenderError::Unsupported`] for a plan using a feature this renderer
    /// does not model (see the crate docs); [`RenderError::Plan`] or
    /// [`RenderError::Core`] for a structural failure.
    pub fn render(&mut self, pixels: &ValidatedPixelPlan) -> Result<RenderedFrame> {
        self.render_timed(pixels, None).map(|(frame, _)| frame)
    }

    /// [`Self::render`] with the restoration filters and the colour
    /// transform banded over `executor`'s workers. Output is identical to
    /// the serial render: every band is a fixed row range and no stage
    /// reduces across bands.
    ///
    /// # Errors
    ///
    /// As [`Self::render`].
    pub fn render_with(
        &mut self,
        pixels: &ValidatedPixelPlan,
        executor: &EncodeExecutor,
    ) -> Result<RenderedFrame> {
        self.render_timed(pixels, Some(executor))
            .map(|(frame, _)| frame)
    }

    /// [`Self::render_with`], also reporting where the time went.
    ///
    /// # Errors
    ///
    /// As [`Self::render`].
    pub fn render_timed(
        &mut self,
        pixels: &ValidatedPixelPlan,
        executor: Option<&EncodeExecutor>,
    ) -> Result<(RenderedFrame, RenderTimings)> {
        let mut timings = RenderTimings::default();
        let millis = |start: std::time::Instant| {
            u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
        };
        let stage_start = std::time::Instant::now();
        let plan = pixels.plan();
        let spatial = &*plan.spatial;
        let quantized = &*plan.quantized;
        let geometry: VardctGeometry = pixels.geometry()?;
        if spatial.lf.adaptive_smoothing {
            return Err(RenderError::Unsupported(
                "adaptive LF smoothing (the encoder always signals it skipped)",
            ));
        }
        if spatial.restoration.epf_iters > 3 {
            return Err(RenderError::Unsupported("epf_iters above 3"));
        }

        let (width, height) = (geometry.width(), geometry.height());
        let dims = PlaneDims::new(
            usize::try_from(width).unwrap_or(usize::MAX),
            usize::try_from(height).unwrap_or(usize::MAX),
        );
        let len = dims.len();
        let mut planes: [Vec<f32>; NUM_CHANNELS] = [vec![0.0; len], vec![0.0; len], vec![0.0; len]];
        let (blocks_x, blocks_y) = block_grid(dims);
        let mut sigma = vec![0.0f32; blocks_x * blocks_y];

        let global_scale = spatial.quantizer.global_scale.get();
        let qm = [
            qm_multiplier(spatial.quantizer.x_qm_scale.get()),
            1.0,
            qm_multiplier(spatial.quantizer.b_qm_scale.get()),
        ];
        let lf_mul = lf_multipliers(
            global_scale,
            spatial.quantizer.quant_lf.get(),
            spatial.lf.channel_dequant.map(|w| w / LF_WEIGHT_SCALE),
        );
        let corr = spatial.lf.correlation;
        let (k_x_lf, k_b_lf) = cfl_factors(
            corr.base_correlation_x,
            corr.base_correlation_b,
            corr.colour_factor,
            i32::from(corr.x_factor_lf) - 128,
            i32::from(corr.b_factor_lf) - 128,
        );

        for (group, ir) in spatial.lf_groups.iter().zip(quantized.lf_groups.iter()) {
            let rect = geometry
                .lf_group_rect(group.id)
                .ok_or(RenderError::Unsupported("an LF group outside the frame"))?;
            let blocks = ir.lf.blocks();
            let tiles = group.cfl.tiles();

            // I.5.2 + I.6 (LF): dequantize the three planes, then borrow
            // chroma from luma with the frame-wide LF factors.
            let lf = lf_planes(ir, &lf_mul, spatial.lf.extra_precision, (k_x_lf, k_b_lf))?;
            let lf_at = |c: usize, bx: u32, by: u32| -> f32 {
                if bx >= blocks.width || by >= blocks.height {
                    return 0.0;
                }
                let idx = usize::try_from(u64::from(by) * u64::from(blocks.width) + u64::from(bx))
                    .unwrap_or(usize::MAX);
                lf.get(c).and_then(|p| p.get(idx)).copied().unwrap_or(0.0)
            };

            for (vb, coeffs) in group.blocks.iter().zip(ir.coefficients.iter()) {
                let transform = vb.transform;
                let (bx, by) = (vb.origin.bx(), vb.origin.by());
                let hf_mul = vb.hf_mul.get();
                let mul = hf_multiplier(global_scale, hf_mul);

                // I.5.3: dequantize all three channels.
                let matrices = self.matrices_for(transform)?;
                let (rows, cols) = (transform.coeff_rows(), transform.coeff_cols());
                let mut coeff: [CoeffMatrix; NUM_CHANNELS] =
                    core::array::from_fn(|_| CoeffMatrix::zeros(rows, cols));
                for c in 0..NUM_CHANNELS {
                    let quant = coeffs.channel(c).ok_or(RenderError::Unsupported(
                        "a varblock with a missing channel",
                    ))?;
                    let matrix = matrices
                        .get(c)
                        .ok_or(RenderError::Unsupported("a missing dequantization channel"))?;
                    let scale = mul * qm.get(c).copied().unwrap_or(1.0);
                    let bias = DEFAULT_QUANT_BIAS.get(c).copied().unwrap_or(1.0);
                    let Some(out) = coeff.get_mut(c) else {
                        continue;
                    };
                    for (y, row) in quant.chunks_exact(cols).enumerate().take(rows) {
                        for (x, &q) in row.iter().enumerate() {
                            let adjusted = bias_adjust(q, bias, DEFAULT_QUANT_BIAS_NUMERATOR);
                            out.set(x, y, adjusted * scale * matrix.at(x, y));
                        }
                    }
                }

                // I.6 (HF): the tile's factors, applied to every cell; I.8
                // overwrites the LLF cells right after.
                let tile_index = usize::try_from(
                    u64::from(by / CFL_TILE_BLOCKS) * u64::from(tiles.width)
                        + u64::from(bx / CFL_TILE_BLOCKS),
                )
                .unwrap_or(usize::MAX);
                let x_factor = group.cfl.x_from_y().get(tile_index).map_or(0, |f| f.get());
                let b_factor = group.cfl.b_from_y().get(tile_index).map_or(0, |f| f.get());
                let (k_x, k_b) = cfl_factors(
                    corr.base_correlation_x,
                    corr.base_correlation_b,
                    corr.colour_factor,
                    x_factor,
                    b_factor,
                );
                apply_hf_cfl(&mut coeff, k_x, k_b);

                // I.8: the LLF rectangle from the LF planes.
                let (block_rows, block_cols) = transform.block_dims();
                for (c, matrix) in coeff.iter_mut().enumerate() {
                    let mut lf_rect = SampleBlock::zeros(block_rows, block_cols);
                    for dy in 0..block_rows {
                        for dx in 0..block_cols {
                            let value = lf_at(
                                c,
                                bx.saturating_add(narrow(dx)),
                                by.saturating_add(narrow(dy)),
                            );
                            lf_rect.set(dx, dy, value);
                        }
                    }
                    matrix.write_llf(&llf_from_lf(transform, &lf_rect));
                }

                // I.9: samples, placed at the varblock's frame position.
                let x0 = rect.x0 + bx * 8;
                let y0 = rect.y0 + by * 8;
                for (c, matrix) in coeff.iter().enumerate() {
                    let block = transform.samples_from_coefficients(matrix);
                    let Some(plane) = planes.get_mut(c) else {
                        continue;
                    };
                    for row in 0..block.rows() {
                        let fy = y0.saturating_add(narrow(row));
                        if fy >= height {
                            continue;
                        }
                        for col in 0..block.cols() {
                            let fx = x0.saturating_add(narrow(col));
                            if fx >= width {
                                continue;
                            }
                            let idx =
                                usize::try_from(u64::from(fy) * u64::from(width) + u64::from(fx))
                                    .unwrap_or(usize::MAX);
                            if let Some(slot) = plane.get_mut(idx) {
                                *slot = block.at(col, row);
                            }
                        }
                    }
                }

                // J.4.3: sigma per 8x8 block of the varblock, from `mul` and
                // the block's own `Sharpness`.
                let sharpness = group.sharpness.values();
                for dy in 0..block_rows {
                    for dx in 0..block_cols {
                        let (sbx, sby) =
                            (bx.saturating_add(narrow(dx)), by.saturating_add(narrow(dy)));
                        if sbx >= blocks.width || sby >= blocks.height {
                            continue;
                        }
                        let s_idx = usize::try_from(
                            u64::from(sby) * u64::from(blocks.width) + u64::from(sbx),
                        )
                        .unwrap_or(usize::MAX);
                        let s = sharpness.get(s_idx).copied().unwrap_or(0).min(7);
                        let fbx = usize::try_from(rect.x0 / 8 + sbx).unwrap_or(usize::MAX);
                        let fby = usize::try_from(rect.y0 / 8 + sby).unwrap_or(usize::MAX);
                        if fbx >= blocks_x || fby >= blocks_y {
                            continue;
                        }
                        if let Some(slot) = sigma.get_mut(fby * blocks_x + fbx) {
                            *slot = vardct_sigma(mul, s, &self.epf_params);
                        }
                    }
                }
            }
        }

        timings.varblocks_ms = millis(stage_start);

        // Annex J.
        let stage_start = std::time::Instant::now();
        if spatial.restoration.gaborish {
            let mut out: [Vec<f32>; NUM_CHANNELS] =
                [vec![0.0; len], vec![0.0; len], vec![0.0; len]];
            for (dst, src) in out.iter_mut().zip(planes.iter()) {
                gaborish_into(src, dst, dims, &self.gabor);
            }
            planes = out;
        }
        timings.gaborish_ms = millis(stage_start);
        let stage_start = std::time::Instant::now();
        if spatial.restoration.epf_iters > 0 {
            let field = SigmaField::new(&sigma, blocks_x, blocks_y).ok_or(
                RenderError::Unsupported("a sigma field that does not match the block grid"),
            )?;
            for step in epf_steps(spatial.restoration.epf_iters).iter().copied() {
                let padded = [
                    PaddedPlane::new(&planes[0], dims, EPF_PAD),
                    PaddedPlane::new(&planes[1], dims, EPF_PAD),
                    PaddedPlane::new(&planes[2], dims, EPF_PAD),
                ];
                let [Some(p0), Some(p1), Some(p2)] = padded else {
                    return Err(RenderError::Unsupported(
                        "EPF planes that do not match the frame",
                    ));
                };
                let padded = [p0, p1, p2];
                let mut out: [Vec<f32>; NUM_CHANNELS] =
                    [vec![0.0; len], vec![0.0; len], vec![0.0; len]];
                let bands = row_bands(&mut out, dims.width);
                let failed = std::sync::atomic::AtomicBool::new(false);
                run_items(executor, bands.len(), &|index| {
                    let Some((row0, mut slices)) = bands.take(index) else {
                        return;
                    };
                    let rows = row0..row0 + slices[0].len() / dims.width.max(1);
                    if epf_step_rows(step, &padded, &self.epf_params, &field, rows, &mut slices)
                        .is_none()
                    {
                        failed.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                });
                if failed.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err(RenderError::Unsupported(
                        "EPF planes that do not match the frame",
                    ));
                }
                planes = out;
            }
        }

        timings.epf_ms = millis(stage_start);

        // Annex L: XYB -> linear sRGB -> the signalled sRGB encoding.
        let stage_start = std::time::Instant::now();
        {
            let bands = row_bands(&mut planes, dims.width);
            let opsin = self.opsin;
            run_items(executor, bands.len(), &|index| {
                let Some((_, mut slices)) = bands.take(index) else {
                    return;
                };
                let [x, y, b] = &mut slices;
                opsin.convert_planes(x, y, b);
                for plane in slices.iter_mut() {
                    for v in plane.iter_mut() {
                        *v = linear_to_srgb(*v);
                    }
                }
            });
        }

        timings.colour_ms = millis(stage_start);

        Ok((
            RenderedFrame {
                width,
                height,
                bits_per_sample: spatial.frame.bits_per_sample,
                planes,
            },
            timings,
        ))
    }
}

/// I.5.2 and I.6 over one LF group: the three dequantized, CfL-corrected LF
/// planes on the group's block grid.
fn lf_planes(
    ir: &jpxl_encode::vardct::QuantizedLfGroup,
    multipliers: &[f32; NUM_CHANNELS],
    extra_precision: u8,
    (k_x, k_b): (f32, f32),
) -> Result<[Vec<f32>; NUM_CHANNELS]> {
    let mut out: [Vec<f32>; NUM_CHANNELS] = [Vec::new(), Vec::new(), Vec::new()];
    for (c, plane) in out.iter_mut().enumerate() {
        let quant = ir.lf.plane(c).ok_or(RenderError::Unsupported(
            "an LF group with a missing LF plane",
        ))?;
        let m = multipliers.get(c).copied().unwrap_or(0.0);
        *plane = quant
            .iter()
            .map(|&q| lf_dequantize(q, m, extra_precision))
            .collect();
    }
    let [x, y, b] = &mut out;
    for ((dx, dy), db) in x.iter_mut().zip(y.iter()).zip(b.iter_mut()) {
        let (vx, _, vb) = cfl_apply(*dx, *dy, *db, k_x, k_b);
        *dx = vx;
        *db = vb;
    }
    Ok(out)
}

/// I.6 for HF coefficients over three same-shaped matrices.
fn apply_hf_cfl(coeffs: &mut [CoeffMatrix; NUM_CHANNELS], k_x: f32, k_b: f32) {
    let (rows, cols) = (coeffs[1].rows(), coeffs[1].cols());
    for y in 0..rows {
        for x in 0..cols {
            let d_y = coeffs[1].at(x, y);
            let (v_x, _, v_b) = cfl_apply(coeffs[0].at(x, y), d_y, coeffs[2].at(x, y), k_x, k_b);
            coeffs[0].set(x, y, v_x);
            coeffs[2].set(x, y, v_b);
        }
    }
}

/// A small block extent as `u32`, saturating rather than wrapping.
fn narrow(v: usize) -> u32 {
    u32::try_from(v).unwrap_or(u32::MAX)
}

/// Rows per band when a stage is spread over workers. Fixed, so the band
/// partition — and with it nothing, since no stage reduces across bands — is
/// independent of the worker count.
const BAND_ROWS: usize = 64;

/// One parked band: its first row and the three planes' slices for it.
type Band<'a> = (usize, [&'a mut [f32]; NUM_CHANNELS]);

/// Row bands of three planes parked for one-shot pickup by executor items.
struct RowBands<'a> {
    items: Vec<std::sync::Mutex<Option<Band<'a>>>>,
}

impl<'a> RowBands<'a> {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn take(&self, index: usize) -> Option<Band<'a>> {
        self.items.get(index)?.lock().ok()?.take()
    }
}

/// Cuts three equally sized planes into [`BAND_ROWS`]-row bands.
fn row_bands(planes: &mut [Vec<f32>; NUM_CHANNELS], width: usize) -> RowBands<'_> {
    let band_len = width.saturating_mul(BAND_ROWS).max(1);
    let [p0, p1, p2] = planes;
    let items = p0
        .chunks_mut(band_len)
        .zip(p1.chunks_mut(band_len))
        .zip(p2.chunks_mut(band_len))
        .enumerate()
        .map(|(i, ((a, b), c))| std::sync::Mutex::new(Some((i * BAND_ROWS, [a, b, c]))))
        .collect();
    RowBands { items }
}

/// Runs `items` independent closures on `executor` (serially without one).
fn run_items(executor: Option<&EncodeExecutor>, items: usize, f: &(dyn Fn(usize) + Sync)) {
    match executor {
        Some(executor) => {
            let outcome: core::result::Result<Vec<()>, core::convert::Infallible> = executor
                .map_ordered(items, |i| {
                    f(i);
                    Ok(())
                });
            let _ = outcome;
        }
        None => {
            for i in 0..items {
                f(i);
            }
        }
    }
}
