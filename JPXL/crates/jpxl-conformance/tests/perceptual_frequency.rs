//! Phase 6.0: what does a unit of the encoder's distortion currency actually
//! cost perceptually, per DCT8x8 frequency?
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
//! loop. For each XYB channel and each of the 64 DCT8x8 cells it injects a
//! controlled coefficient error of equal magnitude — so the current objective
//! assigns every cell exactly the same cost — and reports what each one costs on
//! butteraugli and SSIMULACRA2. The spread across cells *is* the weighting the
//! objective is missing; a flat objective is only correct if the response is
//! flat.
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
//! default all three) and `JPXL_CALIB_SIDE` (centre-crop edge in pixels,
//! default 512).

#![cfg(all(feature = "butteraugli", feature = "ssimulacra2"))]

use jpxl_conformance::metrics::{Image, butteraugli_distance, ssimulacra2_score};
use jpxl_core::color::{
    linear_srgb_to_xyb_planes, linear_to_srgb, srgb_to_linear, xyb_to_linear_srgb_planes,
};
use jpxl_core::dct::idct2d_8x8;
use jpxl_core::dequant::DequantMatrices;
use jpxl_core::varblock::TransformType;

/// One 8x8 block's worth of coefficients.
const CELLS: usize = 64;
/// The DCT8x8 edge, in samples.
const SIDE: usize = 8;

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

/// Adds `amplitude` of DCT8x8 basis function `cell` to every whole block of one
/// channel, with a per-block sign, and returns the realized sample-domain sum of
/// squared error in that channel.
fn inject(xyb: &mut Xyb, channel: usize, cell: usize, amplitude: f32) -> f64 {
    let blocks_x = xyb.w / SIDE;
    let blocks_y = xyb.h / SIDE;
    let w = xyb.w;
    let plane = &mut xyb.planes[channel];

    // The basis function is the same for every block; only its sign varies, so
    // the inverse transform runs once.
    let mut basis = [0.0f32; CELLS];
    basis[cell] = amplitude;
    idct2d_8x8(&mut basis);

    let mut sse = 0.0f64;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block = by * blocks_x + bx;
            let sign = sign_for(block, cell);
            for row in 0..SIDE {
                let base = (by * SIDE + row) * w + bx * SIDE;
                for col in 0..SIDE {
                    let d = sign * basis[row * SIDE + col];
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

/// Everything the two sweeps share: the crop, its XYB planes, the reference
/// image they score against, and the requested channel/amplitude ladder.
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

        let want_side: usize = std::env::var("JPXL_CALIB_SIDE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        // Whole blocks only, and centred so a crop of a photograph keeps its
        // subject.
        let w = (img.w as usize).min(want_side) / SIDE * SIDE;
        let h = (img.h as usize).min(want_side) / SIDE * SIDE;
        assert!(
            w >= SIDE && h >= SIDE,
            "reference is smaller than one block"
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
        })
    }

    fn blocks(&self) -> usize {
        (self.w / SIDE) * (self.h / SIDE)
    }

    fn header(&self, title: &str) {
        println!("# {title}");
        println!("# reference: {}", self.reference);
        println!(
            "# crop: {}x{} at ({},{}), {} blocks",
            self.w,
            self.h,
            self.x0,
            self.y0,
            self.blocks()
        );
        println!(
            "# channel_std: X={:.6} Y={:.6} B={:.6}",
            self.clean.std_dev(0),
            self.clean.std_dev(1),
            self.clean.std_dev(2)
        );
        println!("chan\tamp\tcell\tu\tv\tcoeff_sse\tsample_sse\tba_distance\tba_pnorm3\tssim2");
    }

    /// Runs the cell sweep. `cell_scale` returns the per-cell multiplier on the
    /// base amplitude: constant 1.0 is Phase 6.0's equal-coefficient-error
    /// probe, and the quantizer step shape is Phase 6.1's.
    fn sweep(&self, cell_scale: &dyn Fn(usize, usize) -> f32) {
        for &channel in &self.channels {
            let scale = self.clean.std_dev(channel);
            assert!(scale > 0.0, "channel {channel} is constant");
            for &amp in &self.amplitudes {
                for cell in 0..CELLS {
                    let amplitude = amp * scale * cell_scale(channel, cell);
                    let mut probe = Xyb {
                        w: self.clean.w,
                        h: self.clean.h,
                        planes: self.clean.planes.clone(),
                    };
                    let coeff_sse =
                        f64::from(amplitude) * f64::from(amplitude) * self.blocks() as f64;
                    let _realized = inject(&mut probe, channel, cell, amplitude);
                    let perturbed = probe.to_image();
                    let ba = butteraugli_distance(&self.base, &perturbed).expect("butteraugli");
                    let s2 = ssimulacra2_score(&self.base, &perturbed).expect("ssimulacra2");
                    let sample_sse = image_sse(&self.base, &perturbed);
                    println!(
                        "{}\t{amp}\t{cell}\t{}\t{}\t{coeff_sse:.6}\t{sample_sse:.1}\t{:.6}\t{:.6}\t{:.4}",
                        ["X", "Y", "B"][channel],
                        cell / SIDE,
                        cell % SIDE,
                        ba.distance,
                        ba.pnorm3,
                        s2
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "measurement harness: needs JPXL_CALIB_REF and minutes of butteraugli"]
fn dct8x8_perceptual_frequency_response() {
    let Some(setup) = Setup::from_env() else {
        eprintln!("skipped: set JPXL_CALIB_REF to an 8-bit binary PPM");
        return;
    };
    setup.header("Phase 6.0 DCT8x8 perceptual frequency response");
    // Every cell gets the *same* injected coefficient energy, which is exactly
    // what the current objective prices identically.
    setup.sweep(&|_channel, _cell| 1.0);
}

/// The I.2.5 default DCT8x8 dequantization step shape for one channel,
/// normalised so its root-mean-square over the 63 non-LLF cells is 1.
///
/// `HfQuantizer` builds its step as `scale[channel] * matrix.at(x, y)` with
/// `scale` constant across cells, so the matrix *is* the step shape and the
/// constant cancels under this normalisation. Normalising by RMS (not mean)
/// keeps the total injected energy equal to Phase 6.0's constant-amplitude
/// sweep, so the two experiments sit at the same operating point and their
/// butteraugli numbers are directly comparable.
fn dct8x8_step_shape(channel: usize) -> [f32; CELLS] {
    let defaults = DequantMatrices::all_default().expect("I.2.5 default matrices");
    let matrix = defaults
        .for_transform(TransformType::Dct8x8, channel)
        .expect("a DCT8x8 dequantization matrix");
    let mut shape = [0.0f32; CELLS];
    for (cell, slot) in shape.iter_mut().enumerate() {
        *slot = matrix.at(cell % SIDE, cell / SIDE);
    }
    // Cell 0 is the LLF and is quantized by the LF path, not this one; leaving
    // it out of the normalisation keeps a large DC weight from dominating.
    let sum_sq: f64 = shape
        .iter()
        .skip(1)
        .map(|&v| f64::from(v) * f64::from(v))
        .sum();
    #[allow(
        clippy::cast_possible_truncation,
        reason = "an RMS of 63 finite matrix entries is well inside f32"
    )]
    let rms = (sum_sq / (CELLS - 1) as f64).sqrt() as f32;
    assert!(rms > 0.0, "channel {channel} has a degenerate step shape");
    for slot in &mut shape {
        *slot /= rms;
    }
    shape
}

#[test]
#[ignore = "measurement harness: needs JPXL_CALIB_REF and minutes of butteraugli"]
fn dct8x8_quantizer_normalised_frequency_response() {
    let Some(setup) = Setup::from_env() else {
        eprintln!("skipped: set JPXL_CALIB_REF to an 8-bit binary PPM");
        return;
    };
    let shapes: [[f32; CELLS]; 3] = [
        dct8x8_step_shape(0),
        dct8x8_step_shape(1),
        dct8x8_step_shape(2),
    ];
    setup.header("Phase 6.1 DCT8x8 quantizer-normalised frequency response");
    println!("# injection amplitude per cell is proportional to that cell's I.2.5 dequant step");
    for (channel, shape) in shapes.iter().enumerate() {
        let name = ["X", "Y", "B"][channel];
        let lo = shape.iter().skip(1).copied().fold(f32::MAX, f32::min);
        let hi = shape.iter().skip(1).copied().fold(f32::MIN, f32::max);
        println!(
            "# step_shape[{name}]: non-LLF range {lo:.4}..{hi:.4}, ratio {:.2}x",
            hi / lo
        );
    }
    // Real quantization injects error proportional to each cell's step, so this
    // is what the encoder's decisions are actually made under. A flat response
    // here means the standard's matrices already carry the frequency weighting
    // and the objective's only mistake is measuring in dequantized rather than
    // quantizer-normalised units.
    setup.sweep(&|channel, cell| {
        shapes
            .get(channel)
            .and_then(|s| s.get(cell))
            .copied()
            .unwrap_or(1.0)
    });
}
