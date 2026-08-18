//! `PreparedFrame`: the normalized encoder input (`Encoder-plan1.md` §2.1).
//!
//! This is the first of the two *search-input* IRs, and it lives on the policy
//! side because nothing in it reaches the wire. It is the source image after
//! colour conversion and nothing else.
//!
//! # Why the plane store is a type
//!
//! §2.1's point: a 50-megapixel three-channel `f32` XYB image is roughly
//! 600 MB before a single candidate transform. Full-frame residency has to be
//! a *choice*, not the only implementation. [`PlaneStore`] is therefore an
//! enum from day one, with [`PlaneStoreKind::TiledSpill`] deliberately absent
//! until milestone 10 — but with every accessor already shaped so adding it
//! does not change a caller.

use jpxl_core::color::linear_srgb_to_xyb_planes;

use crate::error::{PolicyError, Result};

/// One row band of a source image handed to a worker exactly once: the
/// interleaved sRGB bytes and the three disjoint plane slices they convert
/// into (see [`PreparedFrame::from_srgb8_with`]).
type SourceBand<'a> =
    std::sync::Mutex<Option<(&'a [u8], &'a mut [f32], &'a mut [f32], &'a mut [f32])>>;

/// [`SourceBand`] for a high-precision source (see
/// [`PreparedFrame::from_srgb16_with`]).
type WideSourceBand<'a> =
    std::sync::Mutex<Option<(&'a [u16], &'a mut [f32], &'a mut [f32], &'a mut [f32])>>;

/// How a plane store holds its samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneStoreKind {
    /// Every sample in RAM.
    Resident,
}

/// One channel's samples.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaneStore {
    samples: Box<[f32]>,
}

impl PlaneStore {
    /// A resident store over `samples`.
    #[must_use]
    pub fn resident(samples: Vec<f32>) -> Self {
        Self {
            samples: samples.into_boxed_slice(),
        }
    }

    /// How the samples are held.
    #[must_use]
    pub const fn kind(&self) -> PlaneStoreKind {
        PlaneStoreKind::Resident
    }

    /// The samples in raster order.
    #[must_use]
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// The sample at `(x, y)` of a `stride`-wide plane, or `None` past the
    /// end.
    #[must_use]
    pub fn at(&self, x: u32, y: u32, stride: u32) -> Option<f32> {
        let index = u64::from(y) * u64::from(stride) + u64::from(x);
        usize::try_from(index)
            .ok()
            .and_then(|i| self.samples.get(i))
            .copied()
    }
}

/// The three XYB planes.
#[derive(Debug, Clone, PartialEq)]
pub struct XybPlanes {
    /// X.
    pub x: PlaneStore,
    /// Y.
    pub y: PlaneStore,
    /// B.
    pub b: PlaneStore,
}

/// The normalized encoder input.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedFrame {
    width: u32,
    height: u32,
    xyb: XybPlanes,
    intensity_target: f32,
    grayscale: bool,
}

impl PreparedFrame {
    /// Converts linear sRGB planes to XYB and wraps them.
    ///
    /// # Errors
    ///
    /// [`PolicyError::Unsupported`] for a zero dimension and
    /// [`PolicyError::SampleCountMismatch`] if a plane is not
    /// `width * height` long.
    pub fn from_linear_srgb(
        width: u32,
        height: u32,
        mut r: Vec<f32>,
        mut g: Vec<f32>,
        mut b: Vec<f32>,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(PolicyError::Unsupported {
                what: "a zero frame dimension",
            });
        }
        let expected = u64::from(width) * u64::from(height);
        for plane in [&r, &g, &b] {
            let found = plane.len() as u64;
            if found != expected {
                return Err(PolicyError::SampleCountMismatch { expected, found });
            }
        }
        // Preserve this source-domain fact before the irreversible RGB -> XYB
        // conversion. Slice 15 uses it as a hard no-op gate: a true grayscale
        // input must retain the pre-CfL codestream byte for byte, rather than
        // relying on near-zero floating-point regression to rediscover that.
        let grayscale = r
            .iter()
            .zip(&g)
            .zip(&b)
            .all(|((&red, &green), &blue)| red == green && green == blue);
        linear_srgb_to_xyb_planes(&mut r, &mut g, &mut b);
        Ok(Self {
            width,
            height,
            xyb: XybPlanes {
                x: PlaneStore::resident(r),
                y: PlaneStore::resident(g),
                b: PlaneStore::resident(b),
            },
            intensity_target: jpxl_core::color::NOMINAL_INTENSITY_TARGET,
            grayscale,
        })
    }

    /// Wraps already-converted XYB planes (for example after inverse-Gaborish
    /// preconditioning).
    ///
    /// `grayscale` is a source-domain fact and must be preserved from the
    /// original RGB conversion — inverse Gaborish does not invent chroma.
    ///
    /// # Errors
    ///
    /// [`PolicyError::Unsupported`] for a zero dimension and
    /// [`PolicyError::SampleCountMismatch`] if a plane is not
    /// `width * height` long.
    pub fn from_xyb(
        width: u32,
        height: u32,
        x: Vec<f32>,
        y: Vec<f32>,
        b: Vec<f32>,
        grayscale: bool,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(PolicyError::Unsupported {
                what: "a zero frame dimension",
            });
        }
        let expected = u64::from(width) * u64::from(height);
        for plane in [&x, &y, &b] {
            let found = plane.len() as u64;
            if found != expected {
                return Err(PolicyError::SampleCountMismatch { expected, found });
            }
        }
        Ok(Self {
            width,
            height,
            xyb: XybPlanes {
                x: PlaneStore::resident(x),
                y: PlaneStore::resident(y),
                b: PlaneStore::resident(b),
            },
            intensity_target: jpxl_core::color::NOMINAL_INTENSITY_TARGET,
            grayscale,
        })
    }

    /// Converts an 8-bit sRGB image, interleaved RGB, to XYB.
    ///
    /// The two stages a decoder runs in reverse: the IEC 61966-2-1 EOTF turns
    /// each code point into linear light, and L.2's forward opsin transform
    /// turns linear light into XYB. `intensity_target` stays nominal, so
    /// L.2.2's `itscale` is exactly 1 and the decoder's inverse is the exact
    /// inverse of this.
    ///
    /// # Errors
    ///
    /// [`PolicyError::Unsupported`] for a zero dimension and
    /// [`PolicyError::SampleCountMismatch`] if `rgb` is not
    /// `width * height * 3` long.
    pub fn from_srgb8(width: u32, height: u32, rgb: &[u8]) -> Result<Self> {
        Self::from_srgb8_with(width, height, rgb, None)
    }

    /// [`Self::from_srgb8`], spreading the conversion over `executor`'s
    /// workers when one is given (Phase 38).
    ///
    /// The image is cut into bands of whole rows; each band deinterleaves,
    /// linearises and converts its own pixels into disjoint slices of the
    /// three planes, and the grayscale flag is the conjunction of the bands'
    /// flags. Every operation is per pixel, so the planes are bit-identical
    /// to the serial conversion whatever the band size or worker count.
    ///
    /// # Errors
    ///
    /// As [`Self::from_srgb8`].
    pub fn from_srgb8_with(
        width: u32,
        height: u32,
        rgb: &[u8],
        executor: Option<&jpxl_encode::EncodeExecutor>,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(PolicyError::Unsupported {
                what: "a zero frame dimension",
            });
        }
        let expected = u64::from(width) * u64::from(height) * 3;
        let found = rgb.len() as u64;
        if found != expected {
            return Err(PolicyError::SampleCountMismatch { expected, found });
        }
        let pixels = rgb.len() / 3;
        // Phase 29: an 8-bit sample has only 256 distinct values, so the
        // `powf`-based transfer curve is evaluated once per byte value here
        // instead of once per sample. `profile-guided`: a fresh perf/inferno
        // profile on this session's host found `<f32>::powf` (inside
        // `srgb_to_linear`) among the top self-time leaves, entirely from
        // this per-pixel call site. The lookup returns bit-identical values
        // to calling `srgb_to_linear` directly, since it is the same
        // function evaluated on the same exact float for each byte.
        let lut: [f32; 256] =
            core::array::from_fn(|byte| jpxl_core::color::srgb_to_linear(byte as f32 / 255.0));

        // One band's work: deinterleave through the LUT, note whether every
        // pixel is grey, then convert the band's rows to XYB in place.
        let convert_band = |src: &[u8], r: &mut [f32], g: &mut [f32], b: &mut [f32]| -> bool {
            let mut grayscale = true;
            for (((px, rs), gs), bs) in src
                .chunks_exact(3)
                .zip(r.iter_mut())
                .zip(g.iter_mut())
                .zip(b.iter_mut())
            {
                let (cr, cg, cb) = (
                    px.first().copied().unwrap_or(0),
                    px.get(1).copied().unwrap_or(0),
                    px.get(2).copied().unwrap_or(0),
                );
                grayscale &= cr == cg && cg == cb;
                *rs = lut.get(usize::from(cr)).copied().unwrap_or(0.0);
                *gs = lut.get(usize::from(cg)).copied().unwrap_or(0.0);
                *bs = lut.get(usize::from(cb)).copied().unwrap_or(0.0);
            }
            linear_srgb_to_xyb_planes(r, g, b);
            grayscale
        };

        let mut r = vec![0.0f32; pixels];
        let mut g = vec![0.0f32; pixels];
        let mut b = vec![0.0f32; pixels];
        let grayscale = match executor {
            Some(executor) if pixels > 0 => {
                // Whole-row bands of about 64 rows: enough items to fill the
                // workers on a small frame, large enough to amortise dispatch.
                let row = usize::try_from(width).unwrap_or(usize::MAX);
                let band_len = row.saturating_mul(64).max(1);
                let bands: Vec<SourceBand<'_>> = rgb
                    .chunks(band_len * 3)
                    .zip(r.chunks_mut(band_len))
                    .zip(g.chunks_mut(band_len))
                    .zip(b.chunks_mut(band_len))
                    .map(|(((src, rs), gs), bs)| std::sync::Mutex::new(Some((src, rs, gs, bs))))
                    .collect();
                let flags = executor.map_ordered(bands.len(), |index| {
                    let (src, rs, gs, bs) = bands
                        .get(index)
                        .ok_or(PolicyError::Unsupported {
                            what: "a source band index outside the frame",
                        })?
                        .lock()
                        .map_err(|_| PolicyError::Unsupported {
                            what: "a poisoned source band",
                        })?
                        .take()
                        .ok_or(PolicyError::Unsupported {
                            what: "a source band converted twice",
                        })?;
                    Ok::<bool, PolicyError>(convert_band(src, rs, gs, bs))
                })?;
                flags.iter().all(|&flag| flag)
            }
            _ => convert_band(rgb, &mut r, &mut g, &mut b),
        };
        Self::from_xyb_planes(width, height, r, g, b, grayscale)
    }

    /// Converts a high-precision sRGB image, interleaved RGB, to XYB.
    ///
    /// The same two stages as [`Self::from_srgb8`] — the IEC 61966-2-1 EOTF to
    /// linear light, then L.2's forward opsin transform — but reading samples
    /// that carry more than eight bits each. `bits_per_sample` is the source's
    /// own depth in `1..=16`; each sample is normalised by `2^bits - 1`, so a
    /// 10-, 12-, 14- or 16-bit original keeps every code point it arrived with
    /// instead of being rounded to 256 levels first.
    ///
    /// # Why this is not `from_srgb8` with wider inputs
    ///
    /// [`Self::from_srgb8`] evaluates the transfer curve through a 256-entry
    /// lookup table, which is exact there because an 8-bit sample has exactly
    /// 256 distinct values. That table cannot represent a 16-bit sample, and
    /// quantizing to it would throw away the precision this constructor exists
    /// to keep — so the curve is evaluated per sample here. The result is the
    /// same function, at the source's own resolution.
    ///
    /// # Errors
    ///
    /// [`PolicyError::Unsupported`] for a zero dimension or a
    /// `bits_per_sample` outside `1..=16`, and
    /// [`PolicyError::SampleCountMismatch`] if `rgb` is not
    /// `width * height * 3` long.
    pub fn from_srgb16(width: u32, height: u32, rgb: &[u16], bits_per_sample: u32) -> Result<Self> {
        Self::from_srgb16_with(width, height, rgb, bits_per_sample, None)
    }

    /// [`Self::from_srgb16`], spreading the conversion over `executor`'s
    /// workers when one is given.
    ///
    /// As with [`Self::from_srgb8_with`], every operation is per pixel, so the
    /// planes are bit-identical to the serial conversion whatever the band
    /// size or worker count.
    ///
    /// # Errors
    ///
    /// As [`Self::from_srgb16`].
    pub fn from_srgb16_with(
        width: u32,
        height: u32,
        rgb: &[u16],
        bits_per_sample: u32,
        executor: Option<&jpxl_encode::EncodeExecutor>,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(PolicyError::Unsupported {
                what: "a zero frame dimension",
            });
        }
        if bits_per_sample == 0 || bits_per_sample > 16 {
            return Err(PolicyError::Unsupported {
                what: "a source bit depth outside 1..=16",
            });
        }
        let expected = u64::from(width) * u64::from(height) * 3;
        let found = rgb.len() as u64;
        if found != expected {
            return Err(PolicyError::SampleCountMismatch { expected, found });
        }
        let pixels = rgb.len() / 3;
        // `2^bits - 1` is the source's white point: a 16-bit sample is full
        // scale at 65535, a 12-bit one at 4095. Normalising by anything else
        // would shift the whole tone curve.
        //
        // This *divides* rather than multiplying by a precomputed reciprocal.
        // `1.0 / 255.0` is not exactly representable, so `s * (1.0 / 255.0)`
        // and `s / 255.0` disagree in the last bit for some samples — and that
        // is enough to move a quantized coefficient and change the emitted
        // bytes. Dividing keeps this path bit-identical to the 8-bit LUT in
        // `from_srgb8_with`, which is what `the_wide_entry_point_agrees_with_
        // the_8_bit_one_on_8_bit_input` pins.
        let max = f32::from(u16::MAX).min(((1u32 << bits_per_sample) - 1) as f32);
        let divisor = if max > 0.0 { max } else { 1.0 };

        let convert_band = |src: &[u16], r: &mut [f32], g: &mut [f32], b: &mut [f32]| -> bool {
            let mut grayscale = true;
            for (((px, rs), gs), bs) in src
                .chunks_exact(3)
                .zip(r.iter_mut())
                .zip(g.iter_mut())
                .zip(b.iter_mut())
            {
                let (cr, cg, cb) = (
                    px.first().copied().unwrap_or(0),
                    px.get(1).copied().unwrap_or(0),
                    px.get(2).copied().unwrap_or(0),
                );
                grayscale &= cr == cg && cg == cb;
                *rs = jpxl_core::color::srgb_to_linear(f32::from(cr) / divisor);
                *gs = jpxl_core::color::srgb_to_linear(f32::from(cg) / divisor);
                *bs = jpxl_core::color::srgb_to_linear(f32::from(cb) / divisor);
            }
            linear_srgb_to_xyb_planes(r, g, b);
            grayscale
        };

        let mut r = vec![0.0f32; pixels];
        let mut g = vec![0.0f32; pixels];
        let mut b = vec![0.0f32; pixels];
        let grayscale = match executor {
            Some(executor) if pixels > 0 => {
                let row = usize::try_from(width).unwrap_or(usize::MAX);
                let band_len = row.saturating_mul(64).max(1);
                let bands: Vec<WideSourceBand<'_>> = rgb
                    .chunks(band_len * 3)
                    .zip(r.chunks_mut(band_len))
                    .zip(g.chunks_mut(band_len))
                    .zip(b.chunks_mut(band_len))
                    .map(|(((src, rs), gs), bs)| std::sync::Mutex::new(Some((src, rs, gs, bs))))
                    .collect();
                let flags = executor.map_ordered(bands.len(), |index| {
                    let (src, rs, gs, bs) = bands
                        .get(index)
                        .ok_or(PolicyError::Unsupported {
                            what: "a source band index outside the frame",
                        })?
                        .lock()
                        .map_err(|_| PolicyError::Unsupported {
                            what: "a poisoned source band",
                        })?
                        .take()
                        .ok_or(PolicyError::Unsupported {
                            what: "a source band converted twice",
                        })?;
                    Ok::<bool, PolicyError>(convert_band(src, rs, gs, bs))
                })?;
                flags.iter().all(|&flag| flag)
            }
            _ => convert_band(rgb, &mut r, &mut g, &mut b),
        };
        Self::from_xyb_planes(width, height, r, g, b, grayscale)
    }

    /// Wraps freshly converted XYB planes with a source-domain grayscale flag.
    fn from_xyb_planes(
        width: u32,
        height: u32,
        x: Vec<f32>,
        y: Vec<f32>,
        b: Vec<f32>,
        grayscale: bool,
    ) -> Result<Self> {
        let expected = u64::from(width) * u64::from(height);
        for plane in [&x, &y, &b] {
            let found = plane.len() as u64;
            if found != expected {
                return Err(PolicyError::SampleCountMismatch { expected, found });
            }
        }
        Ok(Self {
            width,
            height,
            xyb: XybPlanes {
                x: PlaneStore::resident(x),
                y: PlaneStore::resident(y),
                b: PlaneStore::resident(b),
            },
            intensity_target: jpxl_core::color::NOMINAL_INTENSITY_TARGET,
            grayscale,
        })
    }

    /// Frame width in samples.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in samples.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The XYB planes.
    #[must_use]
    pub const fn xyb(&self) -> &XybPlanes {
        &self.xyb
    }

    /// The intensity target the analysis stage weights against.
    #[must_use]
    pub const fn intensity_target(&self) -> f32 {
        self.intensity_target
    }

    /// Whether the source RGB planes were exactly identical before conversion
    /// to XYB.
    ///
    /// This is deliberately a source-domain property. Testing XYB after the
    /// opsin transform would make the grayscale gate depend on floating-point
    /// cancellation and could turn a byte-identical no-op into a searched CfL
    /// stream on another target.
    #[must_use]
    pub const fn is_grayscale(&self) -> bool {
        self.grayscale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prepared_frame_holds_three_converted_planes() {
        let frame = PreparedFrame::from_linear_srgb(2, 2, vec![0.5; 4], vec![0.5; 4], vec![0.5; 4])
            .expect("legal frame");
        assert_eq!((frame.width(), frame.height()), (2, 2));
        assert_eq!(frame.xyb().y.samples().len(), 4);
        assert_eq!(frame.xyb().x.kind(), PlaneStoreKind::Resident);
        // Neutral grey has (near) zero X.
        let x = frame.xyb().x.at(0, 0, 2).expect("in range");
        assert!(x.abs() < 1e-3, "grey should decorrelate to X ~= 0, got {x}");
        assert!(frame.is_grayscale());
    }

    #[test]
    fn a_short_plane_is_rejected() {
        assert!(matches!(
            PreparedFrame::from_linear_srgb(2, 2, vec![0.0; 3], vec![0.0; 4], vec![0.0; 4]),
            Err(PolicyError::SampleCountMismatch { .. })
        ));
        assert!(matches!(
            PreparedFrame::from_linear_srgb(0, 2, Vec::new(), Vec::new(), Vec::new()),
            Err(PolicyError::Unsupported { .. })
        ));
    }
}
