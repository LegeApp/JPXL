//! Phase 6: what does a unit of the encoder's distortion currency actually cost
//! perceptually, per transform and per frequency?
//!
//! `jpxl-encode-policy`'s cover/CfL objective is
//! `J = bits + lambda_c * side^2 * sum_cells (recon - target)^2` — a **flat**
//! sum of squared dequantized-coefficient errors, with one scalar `lambda` per
//! channel. Every cell of every transform is priced identically. That objective
//! is a fidelity measure, and the measured gaps against `cjxl` line up with it:
//! 25-37% on PSNR, 12-20% on SSIMULACRA2, 39-53% on butteraugli. The encoder is
//! closest on the metric its own objective *is*.
//!
//! This harness measures the missing weight directly, without an encoder in the
//! loop. For each square transform, XYB channel and non-LLF coefficient cell it
//! injects a controlled error and reports what that cell costs on butteraugli
//! and SSIMULACRA2. Two sweeps, one test each:
//!
//! * `dct8x8_perceptual_frequency_response` injects **equal coefficient error**
//!   in every cell — exactly what the current objective prices identically — so
//!   the spread across cells *is* the weighting the objective is missing.
//! * `dct8x8_quantizer_normalised_frequency_response` injects error
//!   **proportional to each cell's I.2.5 dequant step**, which is what real
//!   quantization produces. A flat response there means the standard's own
//!   matrices already carry the weighting, and the objective's only mistake is
//!   measuring in dequantized rather than quantizer-normalised units.
//!
//! Both sweeps run over whichever squares `JPXL_CALIB_TRANSFORMS` asks for.
//! Phases 6.0 and 6.1 measured DCT8x8 alone (the default); Phase 6.2 adds
//! DCT16x16 and DCT32x32, because `block_cost_bounded`'s whole job is comparing
//! a large square against four sub-quadrants and a weight validated only for
//! 8x8 would bias that comparison rather than fix it. The `fu`/`fv` columns give
//! each cell's frequency as a fraction of Nyquist, which is what makes the
//! sizes comparable.
//!
//! Clean-room note (AGENTS.md §2, and the `butteraugli` note in the workspace
//! manifest): this is a **behavioural experiment against a black-box metric**,
//! the same category as running `cjxl`/`djxl` as an oracle. It reads no libjxl
//! source. What is done with the resulting numbers is a separate decision:
//! fitting an encoder weight table to butteraugli's measured response would make
//! this project's perceptual model a derivative of libjxl's, whereas
//! implementing a first-principles contrast-sensitivity weight and *checking* it
//! against this measurement would not. This file only measures.
//!
//! Ignored by default and skipped without a reference image. Run it as:
//!
//! ```text
//! JPXL_CALIB_REF=/path/to/ref.ppm \
//!   cargo test -p jpxl-conformance --features perceptual \
//!   --test perceptual_frequency -- --ignored --nocapture
//! ```
//!
//! Knobs, all optional: `JPXL_CALIB_AMPS` (comma-separated amplitudes, as a
//! fraction of the channel's own sample standard deviation, default
//! `0.25,0.5,1.0`), `JPXL_CALIB_CHANNELS` (`x`, `y`, `b`, or a comma list;
//! default all three), `JPXL_CALIB_TRANSFORMS` (`8`, `16`, `32`, or a comma
//! list; default `8`) and `JPXL_CALIB_SIDE` (centre-crop edge in pixels,
//! default 512, always rounded down to a whole DCT32x32 so every transform
//! tiles the identical crop).

#![cfg(all(feature = "butteraugli", feature = "ssimulacra2"))]

use jpxl_conformance::metrics::{Image, butteraugli_distance, ssimulacra2_score};
use jpxl_core::color::{
    linear_srgb_to_xyb_planes, linear_to_srgb, srgb_to_linear, xyb_to_linear_srgb_planes,
};
use jpxl_core::dct::idct_2d_raw;
use jpxl_core::dequant::DequantMatrices;
use jpxl_core::varblock::TransformType;

/// The largest square this harness measures, and the crop alignment: every
/// requested transform must tile the same pixels.
const MAX_SIDE: usize = 32;

/// Three planar XYB channels plus their shape.
struct Xyb {
    w: usize,
    h: usize,
    planes: [Vec<f32>; 3],
}

impl Xyb {
    /// Converts a centre crop of an 8-bit PPM into XYB planes.
    fn from_image(img: &Image, w: usize, h: usize, x0: usize, y0: usize) -> Self {
        let mut planes = [
            vec![0.0f32; w * h],
            vec![0.0f32; w * h],
            vec![0.0f32; w * h],
        ];
        let stride = img.w as usize * 3;
        for y in 0..h {
            for x in 0..w {
                let src = (y0 + y) * stride + (x0 + x) * 3;
                for (c, plane) in planes.iter_mut().enumerate() {
                    let raw = f32::from(img.samples[src + c]) / 255.0;
                    plane[y * w + x] = srgb_to_linear(raw);
                }
            }
        }
        let [r, g, b] = &mut planes;
        linear_srgb_to_xyb_planes(r, g, b);
        Self { w, h, planes }
    }

    /// Converts back to a gamma-encoded 8-bit sRGB image, as a decoder would
    /// deliver it to a metric.
    fn to_image(&self) -> Image {
        let [mut r, mut g, mut b] = self.planes.clone();
        xyb_to_linear_srgb_planes(&mut r, &mut g, &mut b);
        let mut samples = Vec::with_capacity(self.w * self.h * 3);
        for i in 0..self.w * self.h {
            for plane in [&r, &g, &b] {
                let v = linear_to_srgb(plane[i]) * 255.0;
                samples.push(v.clamp(0.0, 255.0).round() as u16);
            }
        }
        Image {
            w: self.w as u32,
            h: self.h as u32,
            channels: 3,
            max_value: 255,
            samples,
        }
    }

    /// Population standard deviation of one channel — the amplitude scale, so a
    /// perturbation means the same thing on X as it does on Y.
    fn std_dev(&self, channel: usize) -> f32 {
        let p = &self.planes[channel];
        if p.is_empty() {
            return 0.0;
        }
        let n = p.len() as f64;
        let mean = p.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = p
            .iter()
            .map(|&v| {
                let d = f64::from(v) - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        var.sqrt() as f32
    }
}

/// A sign in `{-1, +1}` from a cheap deterministic mix of block and cell.
///
/// Quantization error is not coherent across blocks; adding the *same* signed
/// basis function everywhere would build a global texture and measure that
/// instead. This keeps the injected field per-block-independent while staying
/// exactly reproducible.
fn sign_for(block: usize, cell: usize) -> f32 {
    let mut h = (block as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (cell as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 31;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 29;
    if h & 1 == 0 { 1.0 } else { -1.0 }
}

/// One square transform under test: its type, its coefficient/sample edge, and
/// the edge of its LLF sub-block.
#[derive(Clone, Copy)]
struct Square {
    ty: TransformType,
    /// Coefficient (and sample) edge: 8, 16 or 32.
    side: usize,
    /// LLF edge in cells — `block_dims().0`, i.e. 1, 2 or 4.
    llf: usize,
}

impl Square {
    fn new(ty: TransformType) -> Self {
        let side = ty.coeff_cols();
        assert_eq!(side, ty.coeff_rows(), "this harness measures square DCTs");
        Self {
            ty,
            side,
            llf: ty.block_dims().0,
        }
    }

    fn cells(&self) -> usize {
        self.side * self.side
    }

    /// Whether `cell` belongs to the LLF sub-block, which the HF scorer skips
    /// (`col_start = if row < n { n } else { 0 }`) because LF is a separate
    /// path. Excluded from every statistic here for the same reason.
    fn is_llf(&self, cell: usize) -> bool {
        cell / self.side < self.llf && cell % self.side < self.llf
    }

    fn name(&self) -> String {
        format!("DCT{s}x{s}", s = self.side)
    }
}

/// Adds `amplitude` of `square`'s basis function `cell` to every whole block of
/// one channel, with a per-block sign, and returns the realized sample-domain
/// sum of squared error in that channel.
fn inject(xyb: &mut Xyb, square: Square, channel: usize, cell: usize, amplitude: f32) -> f64 {
    let side = square.side;
    let blocks_x = xyb.w / side;
    let blocks_y = xyb.h / side;
    let w = xyb.w;
    let plane = &mut xyb.planes[channel];

    // The basis function is the same for every block; only its sign varies, so
    // the inverse transform runs once. Square transforms are their own
    // landscape layout, so this is plain row-major `side x side`.
    let mut coeffs = vec![0.0f32; square.cells()];
    coeffs[cell] = amplitude;
    let mut basis = idct_2d_raw(&coeffs, side, side);
    // I.7.2's inverse is the orthonormal DCT-III scaled by sqrt(s) per
    // dimension, so `idct_2d_raw` multiplies amplitude by `side` for a square.
    // Dividing it back out makes `amplitude` mean **sample-domain** injected
    // error: each block receives exactly `amplitude^2` of squared sample error
    // whatever the transform size.
    //
    // That is the only unit in which the sizes can be compared, and it is the
    // encoder's own choice: `block_cost_bounded` multiplies each candidate's
    // coefficient error by its `side^2` for exactly this reason, because the
    // forward transforms are not Parseval. Working in the common sample domain
    // here means the measured weight is the correction needed *on top of* that
    // normalisation, not a restatement of it.
    #[allow(
        clippy::cast_precision_loss,
        reason = "side is 8, 16 or 32; exact in f32"
    )]
    let orthonormal = 1.0 / side as f32;
    for slot in &mut basis {
        *slot *= orthonormal;
    }

    let mut sse = 0.0f64;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block = by * blocks_x + bx;
            let sign = sign_for(block, cell);
            for row in 0..side {
                let base = (by * side + row) * w + bx * side;
                for col in 0..side {
                    let d = sign * basis[row * side + col];
                    plane[base + col] += d;
                    sse += f64::from(d) * f64::from(d);
                }
            }
        }
    }
    sse
}

/// Sum of squared 8-bit sample error between two images, over all channels.
fn image_sse(a: &Image, b: &Image) -> f64 {
    a.samples
        .iter()
        .zip(&b.samples)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum()
}

fn env_list(key: &str) -> Option<Vec<String>> {
    std::env::var(key)
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_owned()).collect())
}

/// Everything the sweeps share: the crop, its XYB planes, the reference image
/// they score against, and the requested transform/channel/amplitude ladder.
struct Setup {
    reference: String,
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    clean: Xyb,
    base: Image,
    amplitudes: Vec<f32>,
    channels: Vec<usize>,
    squares: Vec<Square>,
}

impl Setup {
    /// Returns `None` when `JPXL_CALIB_REF` is unset, so the harness skips
    /// rather than fails on a machine without a corpus.
    fn from_env() -> Option<Self> {
        let reference = std::env::var("JPXL_CALIB_REF").ok()?;
        let bytes = std::fs::read(&reference).expect("read reference PPM");
        let img = Image::from_ppm(&bytes).expect("parse reference PPM");
        assert_eq!(img.channels, 3, "the harness scores RGB");
        assert_eq!(img.max_value, 255, "butteraugli scoring is 8-bit only");

        let squares: Vec<Square> = env_list("JPXL_CALIB_TRANSFORMS")
            .map(|v| {
                v.iter()
                    .filter_map(|s| match s.as_str() {
                        "8" => Some(Square::new(TransformType::Dct8x8)),
                        "16" => Some(Square::new(TransformType::Dct16x16)),
                        "32" => Some(Square::new(TransformType::Dct32x32)),
                        _ => None,
                    })
                    .collect::<Vec<Square>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![Square::new(TransformType::Dct8x8)]);

        let want_side: usize = std::env::var("JPXL_CALIB_SIDE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        // Aligned to the largest square this harness supports, not to the
        // largest one requested, so every transform tiles the *identical* crop
        // whichever subset a run asks for. A weight compared across sizes has
        // to come from the same pixels.
        let w = (img.w as usize).min(want_side) / MAX_SIDE * MAX_SIDE;
        let h = (img.h as usize).min(want_side) / MAX_SIDE * MAX_SIDE;
        assert!(
            w >= MAX_SIDE && h >= MAX_SIDE,
            "reference is smaller than one DCT32x32 block"
        );
        let x0 = ((img.w as usize) - w) / 2;
        let y0 = ((img.h as usize) - h) / 2;

        let amplitudes: Vec<f32> = env_list("JPXL_CALIB_AMPS")
            .map(|v| {
                v.iter()
                    .filter_map(|s| s.parse().ok())
                    .collect::<Vec<f32>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![0.25, 0.5, 1.0]);
        let channels: Vec<usize> = env_list("JPXL_CALIB_CHANNELS")
            .map(|v| {
                v.iter()
                    .filter_map(|s| match s.to_ascii_lowercase().as_str() {
                        "x" | "0" => Some(0),
                        "y" | "1" => Some(1),
                        "b" | "2" => Some(2),
                        _ => None,
                    })
                    .collect::<Vec<usize>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![0, 1, 2]);

        let clean = Xyb::from_image(&img, w, h, x0, y0);
        // The unperturbed round trip, not the original file: this cancels the
        // colour-conversion and 8-bit rounding loss so every reported number is
        // the cost of the injected coefficient error alone.
        let base = clean.to_image();
        let base_ba = butteraugli_distance(&base, &base).expect("self-distance");
        assert!(
            base_ba.distance < 1e-6,
            "self-distance must be zero, got {}",
            base_ba.distance
        );

        Some(Self {
            reference,
            w,
            h,
            x0,
            y0,
            clean,
            base,
            amplitudes,
            channels,
            squares,
        })
    }

    fn blocks(&self, square: Square) -> usize {
        (self.w / square.side) * (self.h / square.side)
    }

    fn header(&self, title: &str) {
        println!("# {title}");
        println!("# reference: {}", self.reference);
        println!("# crop: {}x{} at ({},{})", self.w, self.h, self.x0, self.y0);
        println!(
            "# channel_std: X={:.6} Y={:.6} B={:.6}",
            self.clean.std_dev(0),
            self.clean.std_dev(1),
            self.clean.std_dev(2)
        );
        for square in &self.squares {
            println!(
                "# transform {}: {} cells, {} LLF, {} blocks in the crop",
                square.name(),
                square.cells(),
                square.llf * square.llf,
                self.blocks(*square)
            );
        }
        // `fu`/`fv` are the cell's frequency as a fraction of Nyquist, which is
        // what makes a 16x16 cell comparable with the 8x8 cell at the same
        // spatial frequency. `cell`/`u`/`v` stay raw so a row can be traced
        // back to its coefficient.
        println!(
            "tf\tchan\tamp\tcell\tu\tv\tfu\tfv\tcoeff_sse\tsample_sse\tba_distance\tba_pnorm3\tssim2"
        );
    }

    /// Runs the cell sweep over every requested square. `cell_scale` returns the
    /// per-cell multiplier on the base amplitude: constant 1.0 is Phase 6.0's
    /// equal-coefficient-error probe, and the quantizer step shape is 6.1's.
    fn sweep(&self, cell_scale: &dyn Fn(Square, usize, usize) -> f32) {
        for &square in &self.squares {
            let side = square.side;
            for &channel in &self.channels {
                let scale = self.clean.std_dev(channel);
                assert!(scale > 0.0, "channel {channel} is constant");
                for &amp in &self.amplitudes {
                    for cell in 0..square.cells() {
                        if square.is_llf(cell) {
                            continue;
                        }
                        // A larger square tiles the same crop with fewer
                        // blocks, so equal per-block energy would mean
                        // 16x less *total* injected energy at DCT32x32 than
                        // at DCT8x8 — the three sizes would then be compared
                        // at three different distortion levels, and
                        // butteraugli's thresholds are not linear. Scaling by
                        // `side / 8` holds total injected sample-domain energy
                        // constant across sizes, so a shape difference between
                        // curves is a shape difference and not an operating
                        // point difference.
                        #[allow(
                            clippy::cast_precision_loss,
                            reason = "side is 8, 16 or 32; exact in f32"
                        )]
                        let size_norm = side as f32 / 8.0;
                        let amplitude = amp * scale * size_norm * cell_scale(square, channel, cell);
                        let mut probe = Xyb {
                            w: self.clean.w,
                            h: self.clean.h,
                            planes: self.clean.planes.clone(),
                        };
                        let coeff_sse = f64::from(amplitude)
                            * f64::from(amplitude)
                            * self.blocks(square) as f64;
                        let _realized = inject(&mut probe, square, channel, cell, amplitude);
                        let perturbed = probe.to_image();
                        let ba = butteraugli_distance(&self.base, &perturbed).expect("butteraugli");
                        let s2 = ssimulacra2_score(&self.base, &perturbed).expect("ssimulacra2");
                        let sample_sse = image_sse(&self.base, &perturbed);
                        let (u, v) = (cell / side, cell % side);
                        println!(
                            "{}\t{}\t{amp}\t{cell}\t{u}\t{v}\t{:.5}\t{:.5}\t{coeff_sse:.6}\t{sample_sse:.1}\t{:.6}\t{:.6}\t{:.4}",
                            side,
                            ["X", "Y", "B"][channel],
                            u as f32 / side as f32,
                            v as f32 / side as f32,
                            ba.distance,
                            ba.pnorm3,
                            s2
                        );
                    }
                }
            }
        }
    }
}

/// The I.2.5 default dequantization step shape for one square and channel,
/// normalised so its root-mean-square over the non-LLF cells is 1.
///
/// `HfQuantizer` builds its step as `scale[channel] * matrix.at(x, y)` with
/// `scale` constant across cells, so the matrix *is* the step shape and the
/// constant cancels under this normalisation. Normalising by RMS (not mean)
/// keeps the total injected energy equal to the constant-amplitude sweep, so
/// the two experiments sit at the same operating point and their butteraugli
/// numbers are directly comparable.
///
/// The LLF sub-block is excluded from the normalisation because it is not
/// quantized by this path, and its large steps would otherwise dominate.
fn step_shape(square: Square, channel: usize) -> Vec<f32> {
    let defaults = DequantMatrices::all_default().expect("I.2.5 default matrices");
    let matrix = defaults
        .for_transform(square.ty, channel)
        .expect("a dequantization matrix for this transform");
    let mut shape = vec![0.0f32; square.cells()];
    for (cell, slot) in shape.iter_mut().enumerate() {
        *slot = matrix.at(cell % square.side, cell / square.side);
    }
    let mut sum_sq = 0.0f64;
    let mut count = 0usize;
    for (cell, &v) in shape.iter().enumerate() {
        if square.is_llf(cell) {
            continue;
        }
        sum_sq += f64::from(v) * f64::from(v);
        count += 1;
    }
    #[allow(
        clippy::cast_possible_truncation,
        reason = "an RMS of finite matrix entries is well inside f32"
    )]
    let rms = (sum_sq / count.max(1) as f64).sqrt() as f32;
    assert!(
        rms > 0.0,
        "{} channel {channel} has a degenerate step shape",
        square.name()
    );
    for slot in &mut shape {
        *slot /= rms;
    }
    shape
}

/// Reports each requested square's step-shape spread, so a run whose extreme
/// cells sit far outside the linear regime is visible in its own log rather
/// than only in the scores. Phase 6.1's chroma numbers were unusable for
/// exactly this reason.
fn report_step_shapes(setup: &Setup) -> Vec<[Vec<f32>; 3]> {
    let mut all = Vec::with_capacity(setup.squares.len());
    for &square in &setup.squares {
        let shapes = [
            step_shape(square, 0),
            step_shape(square, 1),
            step_shape(square, 2),
        ];
        for (channel, shape) in shapes.iter().enumerate() {
            let mut lo = f32::MAX;
            let mut hi = f32::MIN;
            for (cell, &v) in shape.iter().enumerate() {
                if square.is_llf(cell) {
                    continue;
                }
                lo = lo.min(v);
                hi = hi.max(v);
            }
            println!(
                "# step_shape[{}][{}]: non-LLF range {lo:.4}..{hi:.4}, ratio {:.2}x",
                square.name(),
                ["X", "Y", "B"][channel],
                hi / lo
            );
        }
        all.push(shapes);
    }
    all
}

#[test]
#[ignore = "measurement harness: needs JPXL_CALIB_REF and minutes of butteraugli"]
fn dct8x8_perceptual_frequency_response() {
    let Some(setup) = Setup::from_env() else {
        eprintln!("skipped: set JPXL_CALIB_REF to an 8-bit binary PPM");
        return;
    };
    setup.header("Phase 6.0/6.2 perceptual frequency response, equal coefficient error");
    // Every cell gets the *same* injected coefficient energy, which is exactly
    // what the current objective prices identically.
    setup.sweep(&|_square, _channel, _cell| 1.0);
}

#[test]
#[ignore = "measurement harness: needs JPXL_CALIB_REF and minutes of butteraugli"]
fn dct8x8_quantizer_normalised_frequency_response() {
    let Some(setup) = Setup::from_env() else {
        eprintln!("skipped: set JPXL_CALIB_REF to an 8-bit binary PPM");
        return;
    };
    setup.header("Phase 6.1/6.2 quantizer-normalised frequency response");
    println!("# injection amplitude per cell is proportional to that cell's I.2.5 dequant step");
    let shapes = report_step_shapes(&setup);
    let index: std::collections::HashMap<usize, usize> = setup
        .squares
        .iter()
        .enumerate()
        .map(|(i, s)| (s.side, i))
        .collect();
    // Real quantization injects error proportional to each cell's step, so this
    // is what the encoder's decisions are actually made under. A flat response
    // here means the standard's matrices already carry the frequency weighting
    // and the objective's only mistake is measuring in dequantized rather than
    // quantizer-normalised units.
    setup.sweep(&|square, channel, cell| {
        index
            .get(&square.side)
            .and_then(|&i| shapes.get(i))
            .and_then(|s| s.get(channel))
            .and_then(|s| s.get(cell))
            .copied()
            .unwrap_or(1.0)
    });
}
