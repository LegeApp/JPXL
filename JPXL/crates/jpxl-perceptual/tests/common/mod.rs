//! Shared fixtures for the metric's integration tests: deterministic
//! synthetic images, simple distortions, and a minimal binary-PPM reader.

#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

/// A small deterministic PRNG (xorshift32) so fixtures never depend on a
/// random crate.
pub struct Lcg(u32);

impl Lcg {
    pub fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }

    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Three planar linear-RGB channels.
#[derive(Clone, Debug)]
pub struct Planes {
    pub width: u32,
    pub height: u32,
    pub r: Vec<f32>,
    pub g: Vec<f32>,
    pub b: Vec<f32>,
}

impl Planes {
    pub fn view(&self) -> jpxl_perceptual::LinearRgbView<'_> {
        jpxl_perceptual::LinearRgbView::new(self.width, self.height, &self.r, &self.g, &self.b)
            .expect("fixture planes are consistent")
    }

    pub fn interleaved(&self) -> Vec<[f32; 3]> {
        self.r
            .iter()
            .zip(&self.g)
            .zip(&self.b)
            .map(|((&r, &g), &b)| [r, g, b])
            .collect()
    }
}

/// IEC 61966-2-1 sRGB EOTF.
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 12.92 * 0.003_130_8 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// A photograph-like synthetic: a smooth colour gradient, a textured band,
/// a few sharp edges and mild noise.
pub fn synthetic(width: u32, height: u32, seed: u32) -> Planes {
    let (w, h) = (width as usize, height as usize);
    let mut rng = Lcg::new(seed);
    let mut r = vec![0.0; w * h];
    let mut g = vec![0.0; w * h];
    let mut b = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            let mut pr = 0.2 + 0.6 * fx;
            let mut pg = 0.25 + 0.5 * fy;
            let mut pb = 0.3 + 0.4 * (1.0 - fx) * fy;
            // Texture band.
            if (0.3..0.6).contains(&fy) {
                let t = ((x * 7 + y * 3) % 11) as f32 / 11.0 - 0.5;
                pr += 0.15 * t;
                pg += 0.1 * t;
                pb -= 0.1 * t;
            }
            // Sharp vertical edge and a bright square.
            if fx > 0.7 {
                pr *= 0.5;
                pg *= 0.5;
                pb *= 0.5;
            }
            if (0.1..0.2).contains(&fx) && (0.7..0.8).contains(&fy) {
                pr = 0.95;
                pg = 0.95;
                pb = 0.9;
            }
            let n = (rng.next_f32() - 0.5) * 0.02;
            let i = y * w + x;
            r[i] = (srgb_to_linear(pr + n)).clamp(0.0, 1.0);
            g[i] = (srgb_to_linear(pg + n)).clamp(0.0, 1.0);
            b[i] = (srgb_to_linear(pb + n)).clamp(0.0, 1.0);
        }
    }
    Planes {
        width,
        height,
        r,
        g,
        b,
    }
}

/// Adds uniform noise of amplitude `amp` (linear light) to every sample.
pub fn add_noise(src: &Planes, amp: f32, seed: u32) -> Planes {
    let mut rng = Lcg::new(seed);
    let mut out = src.clone();
    for plane in [&mut out.r, &mut out.g, &mut out.b] {
        for v in plane.iter_mut() {
            *v = (*v + (rng.next_f32() - 0.5) * 2.0 * amp).clamp(0.0, 1.0);
        }
    }
    out
}

/// A 3×3 box blur with edge clamping, applied `passes` times.
pub fn box_blur(src: &Planes, passes: usize) -> Planes {
    let (w, h) = (src.width as usize, src.height as usize);
    let mut out = src.clone();
    for _ in 0..passes {
        for plane in [&mut out.r, &mut out.g, &mut out.b] {
            let input = plane.clone();
            for y in 0..h {
                for x in 0..w {
                    let mut sum = 0.0;
                    for dy in [-1i64, 0, 1] {
                        for dx in [-1i64, 0, 1] {
                            let sx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                            let sy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                            sum += input[sy * w + sx];
                        }
                    }
                    plane[y * w + x] = sum / 9.0;
                }
            }
        }
    }
    out
}

/// Quantises every sample to `levels` steps (banding).
pub fn quantize(src: &Planes, levels: f32) -> Planes {
    let mut out = src.clone();
    for plane in [&mut out.r, &mut out.g, &mut out.b] {
        for v in plane.iter_mut() {
            *v = (*v * levels).round() / levels;
        }
    }
    out
}

/// Reads a binary PPM (`P6`, maxval 255 or 65535) into linear planes.
pub fn read_ppm(path: &std::path::Path) -> Option<Planes> {
    let bytes = std::fs::read(path).ok()?;
    let mut pos = 0usize;
    let mut fields = Vec::new();
    while fields.len() < 4 {
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos < bytes.len() && bytes[pos] == b'#' {
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        let start = pos;
        while pos < bytes.len() && !bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if start == pos {
            return None;
        }
        fields.push(String::from_utf8_lossy(&bytes[start..pos]).into_owned());
    }
    pos += 1;
    if fields[0] != "P6" {
        return None;
    }
    let width: u32 = fields[1].parse().ok()?;
    let height: u32 = fields[2].parse().ok()?;
    let maxval: u32 = fields[3].parse().ok()?;
    let pixels = width as usize * height as usize;
    let data = &bytes[pos..];
    let mut r = Vec::with_capacity(pixels);
    let mut g = Vec::with_capacity(pixels);
    let mut b = Vec::with_capacity(pixels);
    let scale = maxval as f32;
    let mut push = |lut: &[f32], idx: usize| -> Option<()> {
        r.push(*lut.get(idx)?);
        Some(())
    };
    let _ = &mut push;
    if maxval == 255 {
        if data.len() < pixels * 3 {
            return None;
        }
        let lut: Vec<f32> = (0..256).map(|v| srgb_to_linear(v as f32 / scale)).collect();
        for px in data[..pixels * 3].chunks_exact(3) {
            r.push(lut[px[0] as usize]);
            g.push(lut[px[1] as usize]);
            b.push(lut[px[2] as usize]);
        }
    } else {
        if data.len() < pixels * 6 {
            return None;
        }
        for px in data[..pixels * 6].chunks_exact(6) {
            let s =
                |i: usize| srgb_to_linear(u16::from_be_bytes([px[i], px[i + 1]]) as f32 / scale);
            r.push(s(0));
            g.push(s(2));
            b.push(s(4));
        }
    }
    Some(Planes {
        width,
        height,
        r,
        g,
        b,
    })
}

/// The repository root (two levels above the workspace crate).
pub fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
}
