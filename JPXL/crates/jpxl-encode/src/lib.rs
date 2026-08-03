//! JPEG XL encoder — the deliberately tiny first interoperable subset.
//!
//! `docs/PLAN.md` slice 7.5. This crate produces exactly one shape of file:
//!
//! * a **naked codestream** (no Part 2 container);
//! * one **8-bit greyscale** image, non-XYB, no extra channels;
//! * one **`kRegularFrame`**, `is_last`, no crop, no blending, no animation;
//! * **modular** encoding, one group, one pass, one TOC section;
//! * **no transforms**, a **one-leaf** MA tree, the **gradient** predictor;
//! * **prefix-coded** entropy with one context, one cluster and no LZ77;
//! * **restoration filters explicitly disabled**, so the decoded samples are
//!   the encoded samples.
//!
//! Everything outside that is [`EncodeError::Unsupported`] rather than a guess.
//! Breadth — 16-bit, RGB with RCT, multi-group, containers — is slice 10.
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

pub mod entropy;
pub mod error;
pub mod frame;
pub mod headers;
pub mod modular;

use jpxl_bitstream::BitWriter;

pub use error::{EncodeError, Result};

/// An 8-bit greyscale image in raster order.
///
/// Validated on construction, so an existing `GreyImage` always has nonzero
/// dimensions and exactly `width * height` samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GreyImage {
    width: u32,
    height: u32,
    samples: Vec<u8>,
}

impl GreyImage {
    /// Wraps `samples` as a `width` x `height` greyscale image.
    ///
    /// # Errors
    ///
    /// [`EncodeError::ValueOutOfRange`] for a zero dimension, or
    /// [`EncodeError::SampleCountMismatch`] if the buffer is the wrong length.
    pub fn new(width: u32, height: u32, samples: Vec<u8>) -> Result<Self> {
        for (what, value) in [("width", width), ("height", height)] {
            if value == 0 {
                return Err(EncodeError::ValueOutOfRange { what, value: 0 });
            }
        }
        let expected = u64::from(width) * u64::from(height);
        let found = u64::try_from(samples.len()).unwrap_or(u64::MAX);
        if expected != found {
            return Err(EncodeError::SampleCountMismatch { expected, found });
        }
        Ok(Self {
            width,
            height,
            samples,
        })
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

    /// The samples, in raster order.
    #[must_use]
    pub fn samples(&self) -> &[u8] {
        &self.samples
    }
}

/// Encodes an 8-bit greyscale image losslessly as a naked JPEG XL codestream.
///
/// # Errors
///
/// [`EncodeError::Unsupported`] if either dimension exceeds
/// [`frame::MAX_SINGLE_GROUP_DIM`], or any error from the layers below.
pub fn encode_grey8(image: &GreyImage) -> Result<Vec<u8>> {
    let (width, height) = (image.width(), image.height());
    let group_size_shift = frame::single_group_size_shift(width, height)?;

    // F.3.3 puts the section size in the TOC, which precedes the section on
    // the wire, so the section is encoded first and spliced in afterwards.
    // With one section that is the whole of the `SectionStore` idea; slice 10
    // generalises it.
    let samples: Vec<i32> = image.samples().iter().map(|&s| i32::from(s)).collect();
    let section = modular::encode_section(width, height, &samples)?;

    let mut w = BitWriter::new();
    headers::write_signature(&mut w)?;
    headers::write_size_header(&mut w, width, height)?;
    headers::write_grey8_metadata(&mut w)?;

    // F.1: every frame starts on a byte boundary.
    w.zero_pad_to_byte();
    frame::write_frame_header(&mut w, group_size_shift)?;
    frame::write_single_entry_toc(&mut w, section.len())?;
    w.write_bytes(&section)?;

    Ok(w.into_bytes())
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
    }

    #[test]
    fn rejects_images_larger_than_one_group() {
        let image = GreyImage::new(1025, 1, vec![0; 1025]).expect("valid image");
        assert!(matches!(
            encode_grey8(&image),
            Err(EncodeError::Unsupported { .. })
        ));
    }

    #[test]
    fn output_starts_with_the_naked_codestream_signature() {
        let image = GreyImage::new(2, 2, vec![1, 2, 3, 4]).expect("valid image");
        let bytes = encode_grey8(&image).expect("encodes");
        assert_eq!(bytes.get(..2), Some(&[0xFFu8, 0x0A][..]));
    }
}
