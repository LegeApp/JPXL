//! JPEG XL encoder — the lossless modular subset.
//!
//! `docs/PLAN.md` slices 7.5 and 10. This crate produces one shape of file:
//!
//! * a **naked codestream**, or a minimal `jxlc` container behind
//!   [`EncodeOptions::container`];
//! * **greyscale or RGB**, 1..=16-bit integer samples, non-XYB, no extra
//!   channels;
//! * one **`kRegularFrame`**, `is_last`, no crop, no blending, no animation;
//! * **modular** encoding, one pass, one group grid;
//! * **no transforms** for greyscale, one **`kRCT`** (YCoCg) for RGB;
//! * a **one-leaf** MA tree and the **gradient** predictor;
//! * **prefix-coded** entropy with one context, one cluster and no LZ77;
//! * **restoration filters explicitly disabled**, so the decoded samples are
//!   the encoded samples.
//!
//! Everything outside that is [`EncodeError::Unsupported`] rather than a guess.
//!
//! # What this is for
//!
//! Not compression. A gradient-predicted flat prefix code is not competitive
//! and is not meant to be; the point is to prove the syntax stack end to end
//! by producing files that **other** decoders accept. The acceptance criteria,
//! in the order they were established, are: `jpxl-decode` reproduces the input
//! samples exactly, then `djxl` does, then `jxl-oxide` does.
//!
//! # Peer, not a layer
//!
//! Per `AGENTS.md` this crate is a peer of `jpxl-decode` over the neutral
//! `jpxl-bitstream` and `jpxl-core`. It does not depend on the decoder: an
//! encoder bug that the paired decoder happens to accept would prove nothing,
//! so the write side of every clause is derived from the same spec tables
//! rather than from the read side's code. `jpxl-decode` appears only as a
//! `dev-dependency`, in the tests that check the two agree.
//!
//! # Example
//!
//! ```
//! use jpxl_encode::{GreyImage, encode_grey8};
//!
//! let image = GreyImage::new(4, 3, vec![0u8; 12])?;
//! let jxl = encode_grey8(&image)?;
//! assert_eq!(&jxl[..2], &[0xFF, 0x0A], "a naked codestream signature");
//! # Ok::<(), jpxl_encode::EncodeError>(())
//! ```

pub mod container;
pub mod entropy;
pub mod error;
pub mod frame;
pub mod headers;
pub mod modular;
pub mod section;

use jpxl_bitstream::BitWriter;

use entropy::{FlatCode, pack_signed};
use frame::Geometry;
use headers::ColourShape;
use modular::{ModularSource, Plane, Rect};
use section::SectionStore;

pub use error::{EncodeError, Result};

/// The largest bit depth this encoder writes (18181-1 D.7 carries more; the
/// modular residual range and the CLI's Netpbm I/O both stop at 16).
pub const MAX_BITS_PER_SAMPLE: u32 = 16;

/// An integer image: one plane for greyscale, three for RGB.
///
/// Validated on construction, so an existing `Image` always has nonzero
/// dimensions, a supported channel count and bit depth, and exactly
/// `width * height` samples per plane, each inside `[0, 2^bits)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    width: u32,
    height: u32,
    bits_per_sample: u32,
    planes: Vec<Plane>,
}

impl Image {
    /// Wraps planar `planes` as a `width` x `height` image.
    ///
    /// # Errors
    ///
    /// [`EncodeError::ValueOutOfRange`] for a zero dimension, an unsupported
    /// bit depth or a sample outside `[0, 2^bits)`,
    /// [`EncodeError::Unsupported`] for a channel count other than 1 or 3, and
    /// [`EncodeError::SampleCountMismatch`] for a wrong-length plane.
    pub fn new(width: u32, height: u32, bits_per_sample: u32, planes: Vec<Plane>) -> Result<Self> {
        for (what, value) in [("width", width), ("height", height)] {
            if value == 0 {
                return Err(EncodeError::ValueOutOfRange { what, value: 0 });
            }
        }
        if bits_per_sample == 0 || bits_per_sample > MAX_BITS_PER_SAMPLE {
            return Err(EncodeError::ValueOutOfRange {
                what: "bits_per_sample",
                value: i64::from(bits_per_sample),
            });
        }
        if planes.len() != 1 && planes.len() != 3 {
            return Err(EncodeError::unsupported(
                "a channel count other than 1 (grey) or 3 (RGB)",
                "G.1.3",
            ));
        }

        let expected = u64::from(width) * u64::from(height);
        let max = i32::try_from((1u64 << bits_per_sample) - 1).unwrap_or(i32::MAX);
        for plane in &planes {
            let found = u64::try_from(plane.len()).unwrap_or(u64::MAX);
            if found != expected {
                return Err(EncodeError::SampleCountMismatch { expected, found });
            }
            if let Some(&bad) = plane.iter().find(|&&s| s < 0 || s > max) {
                return Err(EncodeError::ValueOutOfRange {
                    what: "sample",
                    value: i64::from(bad),
                });
            }
        }

        Ok(Self {
            width,
            height,
            bits_per_sample,
            planes,
        })
    }

    /// Wraps interleaved samples (the Netpbm layout) as an image.
    ///
    /// # Errors
    ///
    /// As [`Image::new`].
    pub fn from_interleaved(
        width: u32,
        height: u32,
        channels: usize,
        bits_per_sample: u32,
        samples: &[u16],
    ) -> Result<Self> {
        if channels == 0 {
            return Err(EncodeError::unsupported(
                "a channel count other than 1 (grey) or 3 (RGB)",
                "G.1.3",
            ));
        }
        let expected = u64::from(width) * u64::from(height) * channels as u64;
        let found = u64::try_from(samples.len()).unwrap_or(u64::MAX);
        if found != expected {
            return Err(EncodeError::SampleCountMismatch { expected, found });
        }
        let per_plane = samples.len() / channels;
        let planes: Vec<Plane> = (0..channels)
            .map(|c| {
                (0..per_plane)
                    .map(|i| samples.get(i * channels + c).map_or(0, |&s| i32::from(s)))
                    .collect()
            })
            .collect();
        Self::new(width, height, bits_per_sample, planes)
    }

    /// Image width in samples.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Image height in samples.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Bits per sample.
    #[must_use]
    pub const fn bits_per_sample(&self) -> u32 {
        self.bits_per_sample
    }

    /// Number of colour channels: 1 or 3.
    #[must_use]
    pub fn num_channels(&self) -> usize {
        self.planes.len()
    }

    /// The planes, in G.1.3 channel order.
    #[must_use]
    pub fn planes(&self) -> &[Plane] {
        &self.planes
    }

    /// The colour shape written into `ImageMetadata`.
    fn shape(&self) -> ColourShape {
        if self.planes.len() == 1 {
            ColourShape::Grey
        } else {
            ColourShape::Rgb
        }
    }
}

/// Encoder knobs. Every one of them is an encoder-side choice the standard
/// leaves free; none of them changes what a conforming decoder reconstructs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EncodeOptions {
    /// Wrap the codestream in a Part 2 container instead of emitting it naked.
    pub container: bool,
    /// Force a `group_size_shift` (18181-1 F.2, `group_dim = 128 << shift`)
    /// instead of [`frame::DEFAULT_GROUP_SIZE_SHIFT`].
    pub group_size_shift: Option<u32>,
}

/// An 8-bit greyscale image in raster order.
///
/// A thin convenience wrapper over [`Image`]; the general path is
/// [`Image::new`] plus [`encode`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GreyImage {
    inner: Image,
}

impl GreyImage {
    /// Wraps `samples` as a `width` x `height` greyscale image.
    ///
    /// # Errors
    ///
    /// As [`Image::new`].
    pub fn new(width: u32, height: u32, samples: Vec<u8>) -> Result<Self> {
        let plane: Plane = samples.iter().map(|&s| i32::from(s)).collect();
        Ok(Self {
            inner: Image::new(width, height, 8, vec![plane])?,
        })
    }

    /// Image width in samples.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.inner.width()
    }

    /// Image height in samples.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.inner.height()
    }

    /// The samples, in raster order.
    #[must_use]
    pub fn samples(&self) -> Vec<u8> {
        self.inner
            .planes
            .first()
            .map(|p| p.iter().map(|&s| u8::try_from(s).unwrap_or(0)).collect())
            .unwrap_or_default()
    }

    /// The general image this wraps.
    #[must_use]
    pub const fn as_image(&self) -> &Image {
        &self.inner
    }
}

/// Encodes an 8-bit greyscale image losslessly as a naked JPEG XL codestream.
///
/// # Errors
///
/// Any error from the layers below.
pub fn encode_grey8(image: &GreyImage) -> Result<Vec<u8>> {
    encode(image.as_image(), &EncodeOptions::default())
}

/// Encodes `image` losslessly.
///
/// # Errors
///
/// [`EncodeError::ValueOutOfRange`] if a residual or a section is outside the
/// range its field can carry, and any error from the layers below.
pub fn encode(image: &Image, options: &EncodeOptions) -> Result<Vec<u8>> {
    let codestream = encode_codestream(image, options)?;
    if options.container {
        // 18181-2 9.3 with Annex M of Part 1: `modular_16bit_buffers` is false
        // for a >8-bit image, which level 5 does not permit.
        let level = if image.bits_per_sample() > 8 {
            container::EXTENDED_LEVEL
        } else {
            container::DEFAULT_LEVEL
        };
        return Ok(container::wrap(&codestream, level));
    }
    Ok(codestream)
}

/// Encodes the naked codestream, whatever [`EncodeOptions::container`] says.
fn encode_codestream(image: &Image, options: &EncodeOptions) -> Result<Vec<u8>> {
    let (width, height) = (image.width(), image.height());
    let shift = options
        .group_size_shift
        .unwrap_or(frame::DEFAULT_GROUP_SIZE_SHIFT);
    let geometry = Geometry::new(width, height, shift)?;

    // H.6.3 is declared in the transform list, so the samples written are the
    // transformed ones and the decoder inverts them after the last group.
    let mut planes = image.planes().to_vec();
    let rct = planes.len() == 3;
    if rct {
        modular::apply_rct(&mut planes)?;
    }

    let code = FlatCode::for_max_value(max_packed_residual(&planes))?;
    let source = ModularSource {
        width,
        height,
        planes: &planes,
        rct,
        code,
    };

    let store = build_sections(&source, &geometry)?;

    let mut w = BitWriter::new();
    headers::write_signature(&mut w)?;
    headers::write_size_header(&mut w, width, height)?;
    headers::write_metadata(&mut w, image.shape(), image.bits_per_sample())?;

    // F.1: every frame starts on a byte boundary.
    w.zero_pad_to_byte();
    frame::write_frame_header(&mut w, geometry.group_size_shift())?;
    store.write(&mut w)?;

    Ok(w.into_bytes())
}

/// Encodes every section of the frame into a [`SectionStore`], in TOC order.
fn build_sections(source: &ModularSource<'_>, geometry: &Geometry) -> Result<SectionStore> {
    let mut store = SectionStore::new();
    store.push(modular::encode_lf_global(source, geometry)?);
    if geometry.is_single_section() {
        // F.3.1: one group and one pass means one section carrying everything.
        return Ok(store);
    }

    // F.3.1 order: LfGlobal, one per LF group, HfGlobal, then the pass groups.
    // The LF-group sections are empty because nothing here shifts a channel by
    // 3 (G.2.3), and HfGlobal is VarDCT-only (G.3).
    for _ in 0..geometry.num_lf_groups() {
        store.push_empty();
    }
    store.push_empty();
    for index in 0..geometry.num_groups() {
        let (x0, y0, width, height) = geometry
            .group_rect(index)
            .ok_or_else(|| EncodeError::unsupported("a group index past the grid", "G.4"))?;
        store.push(modular::encode_group(
            source,
            Rect {
                x0,
                y0,
                width,
                height,
            },
        )?);
    }
    Ok(store)
}

/// An upper bound on `PackSigned(sample - prediction)` over every plane.
///
/// H.3's gradient prediction is a clamp between two neighbours, so it lies
/// inside the plane's own value range everywhere except the first sample,
/// where the substitutions make it zero. The residual is therefore bounded by
/// the wider of the plane's span and its distance from zero.
fn max_packed_residual(planes: &[Plane]) -> u32 {
    let mut worst = 0u32;
    for plane in planes {
        let (mut lo, mut hi) = (0i64, 0i64);
        for &s in plane {
            let s = i64::from(s);
            lo = lo.min(s);
            hi = hi.max(s);
        }
        let bound = (hi - lo).max(hi.abs()).max(lo.abs());
        let packed = pack_signed(i32::try_from(bound).unwrap_or(i32::MAX));
        // PackSigned is not monotone in the sign, so both directions count.
        let packed = packed.max(pack_signed(i32::try_from(-bound).unwrap_or(i32::MIN)));
        worst = worst.max(packed);
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_degenerate_images() {
        assert!(matches!(
            GreyImage::new(0, 4, Vec::new()),
            Err(EncodeError::ValueOutOfRange { .. })
        ));
        assert!(matches!(
            GreyImage::new(4, 4, vec![0; 15]),
            Err(EncodeError::SampleCountMismatch { .. })
        ));
        assert!(matches!(
            Image::new(4, 4, 8, vec![vec![0; 16], vec![0; 16]]),
            Err(EncodeError::Unsupported { .. })
        ));
        assert!(matches!(
            Image::new(4, 4, 8, vec![vec![256; 16]]),
            Err(EncodeError::ValueOutOfRange { .. })
        ));
        assert!(matches!(
            Image::new(4, 4, 17, vec![vec![0; 16]]),
            Err(EncodeError::ValueOutOfRange { .. })
        ));
    }

    #[test]
    fn output_starts_with_the_naked_codestream_signature() {
        let image = GreyImage::new(2, 2, vec![1, 2, 3, 4]).expect("valid image");
        let bytes = encode_grey8(&image).expect("encodes");
        assert_eq!(bytes.get(..2), Some(&[0xFFu8, 0x0A][..]));
    }

    #[test]
    fn a_container_output_starts_with_the_signature_box() {
        let image = GreyImage::new(2, 2, vec![1, 2, 3, 4]).expect("valid image");
        let options = EncodeOptions {
            container: true,
            ..EncodeOptions::default()
        };
        let bytes = encode(image.as_image(), &options).expect("encodes");
        assert_eq!(bytes.get(..4), Some(&[0x00u8, 0x00, 0x00, 0x0C][..]));
        assert_eq!(bytes.get(4..8), Some(&b"JXL "[..]));
    }

    #[test]
    fn interleaving_and_planar_construction_agree() {
        let interleaved: Vec<u16> = (0..24u16).collect();
        let image = Image::from_interleaved(4, 2, 3, 8, &interleaved).expect("valid");
        assert_eq!(image.num_channels(), 3);
        assert_eq!(image.planes().first().map(Vec::len), Some(8));
        assert_eq!(
            image.planes().first().map(|p| p.as_slice()),
            Some(&[0i32, 3, 6, 9, 12, 15, 18, 21][..])
        );
        assert_eq!(
            image.planes().get(2).map(|p| p.as_slice()),
            Some(&[2i32, 5, 8, 11, 14, 17, 20, 23][..])
        );
    }

    #[test]
    fn the_residual_bound_covers_the_widest_plane() {
        assert_eq!(max_packed_residual(&[vec![0, 0, 0]]), 0);
        // PackSigned doubles a positive residual, so a span of n bounds at 2n.
        assert_eq!(max_packed_residual(&[vec![0, 255]]), 510);
        assert_eq!(max_packed_residual(&[vec![0, 65535]]), 131_070);
        // Negative chroma from the RCT widens the span in both directions.
        assert_eq!(max_packed_residual(&[vec![-300, 300]]), 1200);
    }

    #[test]
    fn a_multi_group_image_produces_the_section_count_f31_requires() {
        let planes = vec![vec![0i32; 600 * 520]];
        let image = Image::new(600, 520, 8, planes).expect("valid");
        let geometry = Geometry::new(600, 520, 2).expect("valid");
        let source = ModularSource {
            width: 600,
            height: 520,
            planes: image.planes(),
            rct: false,
            code: FlatCode::new_const(4),
        };
        let store = build_sections(&source, &geometry).expect("sections");
        assert_eq!(store.len() as u64, geometry.num_sections());
        assert_eq!(store.len(), 2 + 1 + 4);
    }
}
