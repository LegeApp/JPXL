//! End-to-end decoding of a JPEG XL codestream (18181-1 Annexes A, F, G, H).
//!
//! ```text
//! Table A.1 — codestream
//!                                              Headers          Annexes D, E
//!   headers.metadata.have_preview              Frame preview
//!                                              Frame frames[0]  Annex F
//!   for (i = 1; !frames[i-1].is_last; ++i)     Frame frames[i]
//! ```
//!
//! Each frame is `ZeroPadToByte()`-aligned and then reads as Table F.1:
//! `FrameHeader`, `TOC`, then the sections `LfGlobal`, `LfGroup[...]`,
//! `HfGlobal`, `PassGroup[...]`.
//!
//! # Scope
//!
//! Modular mode only. A `kVarDCT` frame returns
//! [`DecodeError::Unsupported`] naming Annex I, as do the frame features that
//! belong to later slices (patches, splines, noise, XYB, YCbCr, upsampling,
//! multi-frame blending). Nothing here guesses: an unimplemented construct is
//! always a typed error, never wrong pixels.
//!
//! # The modular group pipeline (Annex G)
//!
//! Annex G splits one logical modular image across many sections, and the key
//! to reading it is that there is only **one** channel list for the whole
//! frame:
//!
//! 1. `LfGlobal` (G.1.3) reads the optional global MA tree, then a modular
//!    sub-bitstream over the full channel list — but it decodes only the
//!    meta-channels and any channel small enough to fit in a group, and it
//!    applies **no** inverse transforms ([`ChannelStop::GlobalModular`]).
//! 2. Each `LfGroup` (G.2.3) decodes the channels whose `hshift` *and*
//!    `vshift` are at least 3, over the LF-group rectangle, and copies them in.
//! 3. Each `PassGroup` (G.4.2) decodes the channels whose
//!    `minshift <= min(hshift, vshift) < maxshift`, over the group rectangle,
//!    and copies them in.
//! 4. Only when every group is decoded do the inverse transforms of H.6 run,
//!    over the now-complete image.
//!
//! Steps 2 and 3 only have work to do when a transform has produced shifted
//! channels — in practice Squeeze. For an image with no squeeze and no channel
//! larger than `group_dim`, step 1 decodes everything.

use jpxl_bitstream::BitReader;
use jpxl_core::color::OpsinInverse;
use jpxl_core::limits::{AllocGuard, Limits};

use crate::container;
use crate::error::{DecodeError, Result};
use crate::frame::{
    Encoding, FrameGeometry, FrameHeader, FrameType, Rect, Toc, read_frame_header, read_toc,
    stream_index as stream_index_of,
};
use crate::headers::{ImageHeaders, ImageMetadata, decode_image_headers_metered};
use crate::modular::{
    Channel, ChannelSpec, ChannelStop, GlobalTree, ModularOptions, TreeSource,
    decode_sub_bitstream_partial, decode_sub_bitstream_with, read_global_tree,
};

/// One decoded channel of the output image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
    /// Bits per sample, from `metadata.bit_depth` or `ec_info[i].bit_depth`
    /// (18181-1 G.4.2, last paragraph).
    pub bits_per_sample: u32,
    /// Samples in raster order.
    pub samples: Vec<i32>,
}

impl Plane {
    /// The sample at `(x, y)`, or 0 out of bounds.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> i32 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.samples
            .get(y as usize * self.width as usize + x as usize)
            .copied()
            .unwrap_or(0)
    }

    /// The largest legal sample value, `(1 << bits_per_sample) - 1`.
    #[must_use]
    pub const fn max_value(&self) -> u32 {
        if self.bits_per_sample >= 32 {
            u32::MAX
        } else {
            (1u32 << self.bits_per_sample) - 1
        }
    }
}

/// One decoded colour channel as `f32` samples.
///
/// This is the representation 18181-3 §4.2 grades against: nominal `[0, 1]`,
/// **not clipped**, one value per sample per channel. A `kVarDCT` frame
/// produces these directly (XYB is a float space and quantizing at assembly
/// would throw away exactly the precision the conformance thresholds measure);
/// a modular frame does not have them at all.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatPlane {
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
    /// Samples in raster order, in the signalled colour encoding.
    pub samples: Vec<f32>,
}

impl FloatPlane {
    /// The sample at `(x, y)`, or `0.0` out of bounds.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        self.samples
            .get(y as usize * self.width as usize + x as usize)
            .copied()
            .unwrap_or(0.0)
    }
}

/// A fully decoded image.
///
/// # Integer and float representations
///
/// [`planes`](DecodedImage::planes) is always populated and is what the
/// PGM/PPM writer consumes. [`float_planes`](DecodedImage::float_planes) is
/// populated only by the `kVarDCT` path, and where it is present it is the
/// **authoritative** result: the integer planes are its quantization to
/// `bits_per_sample`, done once here so that every existing consumer keeps
/// working unchanged. Lossless modular decoding stays exactly integer and
/// gains nothing from a float copy, so it does not get one.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedImage {
    /// Image width in samples.
    pub width: u32,
    /// Image height in samples.
    pub height: u32,
    /// Colour channels first (1 for greyscale, 3 otherwise), then the extra
    /// channels in index order — the G.1.3 channel order.
    pub planes: Vec<Plane>,
    /// How many leading planes are colour rather than extra channels.
    pub num_colour_channels: usize,
    /// The embedded ICC profile, present exactly when
    /// `metadata.colour_encoding.want_icc` was set (Table A.1, E.4).
    ///
    /// These are the profile's own bytes, byte-identical to what the encoder
    /// was given. Interpreting them is out of scope for a codestream decoder.
    pub icc_profile: Option<Vec<u8>>,
    /// The colour channels as unclipped `f32`, for a frame that decoded to a
    /// float space (`kVarDCT`). `None` for modular frames. See the type
    /// documentation.
    pub float_planes: Option<Vec<FloatPlane>>,
}

impl DecodedImage {
    /// Interleaves the colour channels into one `u16` buffer, clamping each
    /// sample to its plane's range.
    ///
    /// This is what a PGM/PPM writer wants. Extra channels are not included:
    /// compositing alpha is Annex L's job, not this function's.
    #[must_use]
    pub fn interleaved_colour(&self) -> Vec<u16> {
        let n = self.num_colour_channels;
        let mut out = Vec::with_capacity(self.width as usize * self.height as usize * n);
        for y in 0..self.height {
            for x in 0..self.width {
                for plane in self.planes.iter().take(n) {
                    let max = i64::from(plane.max_value());
                    let v = i64::from(plane.get(x, y)).clamp(0, max);
                    out.push(u16::try_from(v).unwrap_or(u16::MAX));
                }
            }
        }
        out
    }

    /// Bits per sample of the colour channels.
    #[must_use]
    pub fn colour_bits_per_sample(&self) -> u32 {
        self.planes.first().map_or(8, |p| p.bits_per_sample)
    }
}

/// Number of reference slots F.2's `save_as_reference` can name.
///
/// The field is a `u(2)`, so four; slot 0 is a real slot, not "none" (F.2
/// distinguishes "not saved" by `can_reference` being false, not by the
/// index).
pub const NUM_REFERENCE_SLOTS: usize = 4;

/// **Flip point — what `Reference[ref]`'s coordinates mean (K.3.1).**
///
/// K.3.1 says a patch sample is "the sample from channel c at the position
/// `(x0 + x, y0 + y)` in `Reference[ref]`". A reference frame may carry a crop
/// (F.2), so its own rectangle need not start at the canvas origin, and the
/// clause never says which of the two `Reference[ref]` is.
///
/// `true` (shipped): `Reference[ref]` is canvas-sized, so `(x0 + x, y0 + y)`
/// is a canvas coordinate and a cropped reference contributes only its own
/// rectangle, zero elsewhere. This is the reading that makes "Reference" mean
/// the same thing here as it does for frame blending (F.2), where a reference
/// is unambiguously something composited onto the canvas.
///
/// `false`: coordinates are relative to the reference frame's own crop origin.
///
/// **Unexercised.** Every reference frame available has crop origin `(0, 0)`,
/// where the two readings coincide exactly. `bike`/`bike_5`/`progressive` all
/// carry a 22x20 or 29x28 reference at `(0, 0)`.
pub const PATCH_REFERENCE_IS_CANVAS_COORDINATES: bool = true;

/// A stored reference frame (F.2 `save_as_reference`), as K.3 reads it.
///
/// Held in the colour space *before* the inverse colour transform — XYB for an
/// `xyb_encoded` image — because that is where K.3.2 blends: "the sample
/// values `new_sample` are in the colour space before the inverse colour
/// transforms from L.2, L.3 and L.4 are applied". Storing display-space RGB
/// here and converting back would be lossy and, for the `kModular` reference
/// frames real encoders emit, circular.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceFrame {
    /// Canvas x of the stored rectangle's left edge (F.2 `x0`).
    pub origin_x: i32,
    /// Canvas y of the stored rectangle's top edge (F.2 `y0`).
    pub origin_y: i32,
    /// Rectangle width in samples.
    pub width: u32,
    /// Rectangle height in samples.
    pub height: u32,
    /// The three colour planes, raster order, `width * height` each.
    pub planes: [Vec<f32>; 3],
}

impl ReferenceFrame {
    /// The sample at a **canvas** coordinate, or `0.0` outside the stored
    /// rectangle.
    ///
    /// See [`PATCH_REFERENCE_IS_CANVAS_COORDINATES`] for why the argument is a
    /// canvas coordinate rather than a rectangle-relative one.
    #[must_use]
    pub fn get(&self, channel: usize, x: u32, y: u32) -> f32 {
        let (ox, oy) = if PATCH_REFERENCE_IS_CANVAS_COORDINATES {
            (self.origin_x, self.origin_y)
        } else {
            (0, 0)
        };
        let Some(lx) = i64::from(x).checked_sub(i64::from(ox)) else {
            return 0.0;
        };
        let Some(ly) = i64::from(y).checked_sub(i64::from(oy)) else {
            return 0.0;
        };
        if lx < 0 || ly < 0 || lx >= i64::from(self.width) || ly >= i64::from(self.height) {
            return 0.0;
        }
        // Both are in `0..width`/`0..height` after the range test above, so
        // the conversions cannot fail on any target.
        let (lx, ly) = (
            usize::try_from(lx).unwrap_or(0),
            usize::try_from(ly).unwrap_or(0),
        );
        let idx = ly * self.width as usize + lx;
        self.planes
            .get(channel)
            .and_then(|p| p.get(idx))
            .copied()
            .unwrap_or(0.0)
    }
}

/// The four `save_as_reference` slots of F.2.
type ReferenceSlots = [Option<ReferenceFrame>; NUM_REFERENCE_SLOTS];

/// Number of `LFFrame[]` slots F.2 can address.
///
/// `lf_level` is `1 + u(2)`, so a `kLFFrame` writes `LFFrame[lf_level - 1]`
/// with the index in `0..4`, and G.2.2 reads `LFFrame[lf_level]` with
/// `lf_level` the *consuming* frame's own field — 0 for a regular frame.
pub const NUM_LF_FRAME_SLOTS: usize = 4;

/// **Flip point — what an `LFFrame[]` slot holds (F.2 + G.2.2 + L.2.2).**
///
/// F.2 records "the samples of the frame before any colour transform is
/// applied" as `LFFrame[lf_level - 1]`, and G.2.2 says those samples are used
/// *instead of* what G.2.2 and I.5.2 would have produced — i.e. instead of
/// **dequantized** LF. The two sentences pull in different directions for a
/// `kModular` LF frame, whose decoded samples are integers:
///
/// * `true` (shipped): L.2.2's `kModular` pre-step (`X = x' *
///   m_x_lf_unscaled`, `Y = y' * m_y_lf_unscaled`, `B = (B' + Y') *
///   m_b_lf_unscaled`) is applied before storing, so the slot holds float XYB
///   in exactly the space I.5.2 outputs. This is the only reading that can
///   satisfy G.2.2 at all: a `kModular` frame's `LfGlobal` carries
///   `LfChannelDequantization` but **not** the `Quantizer` bundle, so
///   `mXDC = (1 << 16) * m_x_lf_unscaled / (global_scale * quant_lf)` is not
///   computable for it, and the raw integers are in no shared scale.
/// * `false`: the slot holds the raw modular integers and something later
///   scales them. Nothing in G.2.2 says what.
///
/// Same reading, and the same clause, as [`reference_from_modular`] applies
/// for `save_before_ct` reference frames. Probed end to end: see
/// `docs/experiments/2026-08-04-lf-frame-and-multipass.md`.
pub const LF_FRAME_IS_XYB_PRESTEP: bool = true;

/// **Flip point — G.4.2's "the channel dimensions exceed group_dim".**
///
/// G.1.3 stops the `GlobalModular` decode at the first channel that is not
/// "at most `group_dim`" in both dimensions; G.4.2 then admits a channel to
/// group decoding only if "the channel dimensions exceed `group_dim` x
/// `group_dim`". Taken together with unshifted dimensions those two tests
/// leave a hole: a Squeeze pyramid's channel list is **not** monotone in size,
/// so a small channel sitting *after* a large one is decoded by neither rule
/// and stays zero. `progressive`'s LF frame is exactly that shape — its
/// channels 37/38 are 127x169 with `(hshift, vshift) = (2, 1)`, sitting after
/// a 254x338 channel.
///
/// * `true` (shipped): the comparison is against the channel's *own* grid,
///   `group_dim >> hshift` by `group_dim >> vshift` — the same right shift
///   G.2.3 and G.4.2 apply to the group rectangle two sentences later. Every
///   channel after G.1.3's stop then exceeds its shifted group, so the hole
///   closes and "remaining channel" means what the NOTE at G.1.3 says it
///   means.
/// * `false`: the literal unshifted comparison.
///
/// The two readings agree on every channel whose shifts are zero, which is
/// every non-Squeeze stream — all of slices 5-9's modular fixtures. They
/// disagree only on a squeezed channel list, and there `false` desynchronises
/// the very first pass-group section. Settled by the section-exhaustion gate:
/// see `docs/experiments/2026-08-04-lf-frame-and-multipass.md`.
pub const G42_SIZE_TEST_IS_SHIFTED: bool = true;

/// A stored `kLFFrame` (F.2), in the space G.2.2 substitutes for I.5.2's
/// output.
///
/// One sample per 8x8 block of the consuming frame: a `kLFFrame` with
/// `lf_level = 1` has frame dimensions `ceil(width / 8) x ceil(height / 8)`
/// (F.1), which is exactly the LF grid.
#[derive(Debug, Clone, PartialEq)]
pub struct LfFrame {
    /// Width in LF samples.
    pub width: u32,
    /// Height in LF samples.
    pub height: u32,
    /// The three planes in Table I.1's `X, Y, B` numbering, raster order.
    pub planes: [Vec<f32>; 3],
}

/// The `LFFrame[]` slots of F.2.
type LfFrameSlots = [Option<LfFrame>; NUM_LF_FRAME_SLOTS];

/// Decodes a JPEG XL file or naked codestream.
///
/// Accepts either a naked codestream (starting `FF 0A`) or a Part 2 container,
/// from which the `jxlc`/`jxlp` codestream is extracted first.
///
/// # Errors
///
/// [`DecodeError::InvalidSignature`] if the bytes are not JPEG XL,
/// [`DecodeError::Unsupported`] for a construct outside modular mode, and any
/// header, frame, modular or entropy error otherwise.
pub fn decode(data: &[u8], limits: &Limits) -> Result<DecodedImage> {
    let mut guard = AllocGuard::new(limits);
    let extracted;
    let codestream = if container::is_container(data) {
        extracted = container::extract_codestream(data, &mut guard)?;
        extracted.as_slice()
    } else {
        data
    };

    let mut reader = BitReader::new(codestream);
    let headers = decode_image_headers_metered(&mut reader, limits, &mut guard)?;

    // A.1: the ICC profile sits between the headers and the first frame, in the
    // same bit stream and with no alignment in between. It must be consumed
    // even by a decoder that ignores colour management, or every frame offset
    // after it is wrong.
    let icc_profile = if headers.metadata.colour_encoding.want_icc {
        Some(crate::icc::read_icc_profile(&mut reader, &mut guard)?)
    } else {
        None
    };

    if headers.metadata.preview.is_some() {
        return Err(unsupported("preview frames", "18181-1 A.1"));
    }

    // F.1: each frame is byte-aligned by ZeroPadToByte() before it is read.
    reader.zero_pad_to_byte()?;
    let mut cursor = usize::try_from(reader.total_bits_read() / 8)
        .map_err(|_| unsupported("a codestream larger than the address space", "18181-1 A.1"))?;

    let mut decoded: Option<DecodedImage> = None;
    let mut references: ReferenceSlots = [None, None, None, None];
    let mut lf_frames: LfFrameSlots = [None, None, None, None];
    let mut frames = 0u32;
    loop {
        frames += 1;
        if u64::from(frames) > u64::from(limits.max_frames) {
            return Err(DecodeError::Core(jpxl_core::JpxlError::LimitExceeded(
                format!("codestream has more than {} frames", limits.max_frames),
            )));
        }
        let rest = codestream
            .get(cursor..)
            .ok_or_else(|| unsupported("a truncated frame", "18181-1 F.1"))?;
        let mut frame_reader = BitReader::new(rest);

        let header = read_frame_header(
            &mut frame_reader,
            &headers.metadata,
            headers.width(),
            headers.height(),
            limits,
            &mut guard,
        )?;
        let geometry = FrameGeometry::from_header(
            &header,
            headers.width(),
            headers.height(),
            limits,
            &mut guard,
        )?;
        let toc = read_toc(
            &mut frame_reader,
            geometry.num_sections(),
            limits,
            &mut guard,
        )?;
        let section_base = cursor
            + usize::try_from(frame_reader.total_bits_read() / 8)
                .map_err(|_| unsupported("an oversized TOC", "18181-1 F.3.3"))?;

        if header.frame_type == FrameType::RegularFrame {
            if decoded.is_some() {
                return Err(unsupported(
                    "a codestream with more than one regular frame (blending)",
                    "18181-1 F.2",
                ));
            }
            decoded = Some(decode_frame(
                codestream,
                section_base,
                &toc,
                &geometry,
                &header,
                &headers,
                &references,
                &lf_frames,
                limits,
                &mut guard,
            )?);
        } else if header.frame_type == FrameType::LfFrame {
            // F.2: "if lf_level != 0, the samples of the frame (before any
            // colour transform is applied) are recorded as
            // LFFrame[lf_level - 1]". The frame itself is never displayed.
            let frame = decode_lf_frame(
                codestream,
                section_base,
                &toc,
                &geometry,
                &header,
                &headers,
                limits,
                &mut guard,
            )?;
            let slot = usize::try_from(header.lf_level)
                .ok()
                .and_then(|level| level.checked_sub(1))
                .unwrap_or(usize::MAX);
            *lf_frames
                .get_mut(slot)
                .ok_or_else(|| unsupported("an LF frame level past four", "18181-1 F.2"))? =
                Some(frame);
        } else if header.frame_type == FrameType::ReferenceOnly {
            // F.2: a kReferenceOnly frame is not part of the displayed
            // sequence; it exists to be read back by K.3 patches or by a later
            // frame's blending. Decode it and park it in its slot.
            let frame = decode_reference_frame(
                codestream,
                section_base,
                &toc,
                &geometry,
                &header,
                &headers,
                limits,
                &mut guard,
            )?;
            let slot = usize::try_from(header.save_as_reference).unwrap_or(usize::MAX);
            *references
                .get_mut(slot)
                .ok_or_else(|| unsupported("a reference slot past three", "18181-1 F.2"))? =
                Some(frame);
        } else {
            return Err(unsupported("kSkipProgressive frames", "18181-1 F.2"));
        }

        let total = usize::try_from(toc.total_size())
            .map_err(|_| unsupported("an oversized frame", "18181-1 F.3.3"))?;
        cursor = section_base
            .checked_add(total)
            .ok_or_else(|| unsupported("a frame that overruns the codestream", "18181-1 F.3.3"))?;
        if header.is_last {
            break;
        }
    }

    let mut image =
        decoded.ok_or_else(|| unsupported("a codestream with no regular frame", "18181-1 A.1"))?;
    image.icc_profile = icc_profile;
    Ok(image)
}

/// The byte range of TOC section `index`.
fn section_slice<'a>(
    codestream: &'a [u8],
    base: usize,
    toc: &Toc,
    index: usize,
) -> Result<&'a [u8]> {
    let offset = toc
        .offset_of(index)
        .ok_or_else(|| unsupported("a TOC section index past the table", "18181-1 F.3.3"))?;
    // F.3.3 permutes the *offsets*, so the size of conceptual section `index`
    // is the entry the permutation maps it to.
    let entry_index = match &toc.permutation {
        Some(p) => p
            .get(index)
            .and_then(|&target| usize::try_from(target).ok())
            .unwrap_or(index),
        None => index,
    };
    let size = toc.entries.get(entry_index).copied().unwrap_or(0);
    let start = base
        .checked_add(usize::try_from(offset).unwrap_or(usize::MAX))
        .ok_or_else(|| unsupported("a section offset past the codestream", "18181-1 F.3.3"))?;
    let end = start
        .checked_add(usize::try_from(size).unwrap_or(usize::MAX))
        .unwrap_or(codestream.len())
        .min(codestream.len());
    codestream
        .get(start..end)
        .ok_or_else(|| unsupported("a section that overruns the codestream", "18181-1 F.3.3"))
}

/// The initial modular channel list of G.1.3.
fn initial_channels(
    header: &FrameHeader,
    metadata: &ImageMetadata,
    geometry: &FrameGeometry,
) -> Result<(Vec<ChannelSpec>, usize)> {
    let (width, height) = (geometry.width(), geometry.height());
    let mut specs = Vec::new();

    // G.1.3: num_channels = num_extra, plus 1 or 3 for kModular.
    let num_colour = if header.encoding == Encoding::Modular {
        if !header.do_ycbcr && !metadata.xyb_encoded && metadata.colour_encoding.is_grey() {
            1
        } else {
            3
        }
    } else {
        0
    };
    for _ in 0..num_colour {
        specs.push(ChannelSpec::new(width, height));
    }

    // "Then the extra channels (if any) ... in ascending order of index",
    // with dim_shift applied to both dimensions (D.3.6).
    for info in &metadata.ec_info {
        let shift = info.dim_shift;
        if shift > 30 {
            return Err(DecodeError::out_of_range(
                "dim_shift",
                "D.3.6",
                u64::from(shift),
            ));
        }
        specs.push(ChannelSpec::with_shifts(
            shifted_ceil(width, shift),
            shifted_ceil(height, shift),
            // H.1: extra channels start at their channel shift value.
            shift as i32,
            shift as i32,
        ));
    }
    Ok((specs, num_colour))
}

/// `ceil(value / (1 << shift))`.
const fn shifted_ceil(value: u32, shift: u32) -> u32 {
    if shift >= 32 {
        return 0;
    }
    let step = 1u32 << shift;
    value.div_ceil(step)
}

/// Rejects the frame constructs no encoding supports yet.
///
/// Split from the per-encoding checks so that a construct is rejected in
/// exactly one place: everything here is orthogonal to `encoding`.
fn check_supported_common(header: &FrameHeader, purpose: FramePurpose) -> Result<()> {
    if header.have_crop && purpose == FramePurpose::Displayed {
        // A cropped *displayed* frame has to be blended onto the canvas
        // (F.2); a cropped reference frame just occupies its rectangle.
        return Err(unsupported("cropped frames", "18181-1 F.2"));
    }
    if header.do_ycbcr {
        return Err(unsupported("do_YCbCr colour reconstruction", "18181-1 L.3"));
    }
    if header.upsampling != 1 {
        return Err(unsupported("frame upsampling", "18181-1 J.2"));
    }
    if header.flags.splines() {
        return Err(unsupported("splines", "18181-1 K.4"));
    }
    if header.flags.noise() {
        return Err(unsupported("noise synthesis", "18181-1 K.5"));
    }
    Ok(())
}

/// Rejects every frame construct outside the modular scope of this slice.
fn check_supported_modular(
    header: &FrameHeader,
    metadata: &ImageMetadata,
    purpose: FramePurpose,
) -> Result<()> {
    check_supported_common(header, purpose)?;
    if header.flags.use_lf_frame() {
        // G.2.2's substitution is defined against the `LF coefficients` row of
        // Table G.3, which only exists for `encoding == kVarDCT`. A kModular
        // frame has no LF coefficients to replace.
        return Err(unsupported(
            "kUseLfFrame in a kModular frame",
            "18181-1 G.2.2",
        ));
    }
    if header.flags.patches() {
        // K.3.1's dictionary is the first bundle of LfGlobal for every
        // encoding, and the modular path does not read it — so this is a
        // bit-position guard, not only a feature gate. K.3.2 for a modular
        // frame would also have to blend in the modular sample space.
        return Err(unsupported("patches in a kModular frame", "18181-1 K.3"));
    }
    if metadata.xyb_encoded && purpose == FramePurpose::Displayed {
        // A displayed modular frame in an XYB image would need L.2.2's
        // kModular pre-step and then the whole of L.2. A reference frame stops
        // before L.2 by definition, so it only needs the pre-step, which
        // `decode_reference_frame` applies.
        return Err(unsupported(
            "xyb_encoded colour reconstruction",
            "18181-1 L.2",
        ));
    }
    Ok(())
}

/// Rejects every frame construct outside the `kVarDCT` scope of slice 8.
fn check_supported_vardct(header: &FrameHeader, metadata: &ImageMetadata) -> Result<()> {
    check_supported_common(header, FramePurpose::Displayed)?;
    if !metadata.xyb_encoded {
        // A kVarDCT frame that is not XYB-encoded would need L.2 skipped and
        // the samples interpreted directly in the signalled colour encoding.
        // cjxl does not emit one; refusing is better than guessing.
        return Err(unsupported(
            "a kVarDCT frame that is not xyb_encoded",
            "18181-1 L.2",
        ));
    }
    if metadata.num_extra() != 0 {
        // Extra channels in a kVarDCT frame ride in the G.2.3/G.4.2 modular
        // sub-bitstreams, which interleave with the HF coefficients inside the
        // same sections. That wiring is a separate slice.
        return Err(unsupported(
            "extra channels in a kVarDCT frame",
            "18181-1 G.4.2",
        ));
    }
    if header.jpeg_upsampling != [0, 0, 0] {
        return Err(unsupported(
            "chroma subsampling in a kVarDCT frame",
            "18181-1 G.2.2",
        ));
    }
    Ok(())
}

/// Decodes one frame's sections into planes, dispatching on its encoding.
#[expect(
    clippy::too_many_arguments,
    reason = "the frame layer genuinely needs all of this; a bundle struct \
              would only move the argument list somewhere less visible"
)]
fn decode_frame(
    codestream: &[u8],
    base: usize,
    toc: &Toc,
    geometry: &FrameGeometry,
    header: &FrameHeader,
    headers: &ImageHeaders,
    references: &ReferenceSlots,
    lf_frames: &LfFrameSlots,
    limits: &Limits,
    guard: &mut AllocGuard,
) -> Result<DecodedImage> {
    if header.encoding == Encoding::VarDct {
        return decode_vardct_frame(
            codestream, base, toc, geometry, header, headers, references, lf_frames, guard,
        );
    }
    decode_modular_frame(
        codestream,
        base,
        toc,
        geometry,
        header,
        headers,
        FramePurpose::Displayed,
        limits,
        guard,
    )
    .map(|(image, _)| image)
}

/// Why a frame is being decoded, which is what decides how strict the
/// feature checks are.
///
/// A `kReferenceOnly` frame is never displayed: it is read back by K.3
/// patches or by a later frame's blending, in the colour space *before* the
/// inverse colour transform. That makes two constructs harmless for it which
/// are real work for a displayed frame — a crop, because the reference simply
/// occupies that rectangle of the canvas, and `xyb_encoded`, because L.2 is
/// exactly the step a reference is stored ahead of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FramePurpose {
    /// The frame the file decodes to.
    Displayed,
    /// A `kReferenceOnly` frame headed for a `save_as_reference` slot.
    Reference,
    /// A `kLFFrame` headed for an `LFFrame[]` slot. Like a reference frame it
    /// is stored before the inverse colour transform (F.2 fixes
    /// `save_before_ct` for it), so the same relaxations apply.
    LfFrame,
}

/// Decodes a `kLFFrame` into an [`LfFrame`] slot (18181-1 F.2, G.2.2).
///
/// A `kLFFrame` is a `kModular` frame whose dimensions F.1 has already divided
/// by `1 << (3 * lf_level)`, so at `lf_level = 1` it is exactly the consuming
/// frame's 8x8-block grid. Its samples stand in for what G.2.2 and I.5.2 would
/// have produced, which is why they go through L.2.2's `kModular` pre-step
/// here — see [`LF_FRAME_IS_XYB_PRESTEP`].
///
/// # Errors
///
/// As [`decode_modular_frame`], plus [`DecodeError::Unsupported`] for a
/// `kVarDCT` LF frame (no available encoder emits one) and for a non-XYB or
/// greyscale one, which L.2.2's pre-step is not defined for.
#[expect(
    clippy::too_many_arguments,
    reason = "the same frame-layer inputs decode_frame takes"
)]
fn decode_lf_frame(
    codestream: &[u8],
    base: usize,
    toc: &Toc,
    geometry: &FrameGeometry,
    header: &FrameHeader,
    headers: &ImageHeaders,
    limits: &Limits,
    guard: &mut AllocGuard,
) -> Result<LfFrame> {
    if header.encoding == Encoding::VarDct {
        return Err(unsupported("a kVarDCT LF frame", "18181-1 F.2"));
    }
    let (image, lf_dequant) = decode_modular_frame(
        codestream,
        base,
        toc,
        geometry,
        header,
        headers,
        FramePurpose::LfFrame,
        limits,
        guard,
    )?;
    let planes = if LF_FRAME_IS_XYB_PRESTEP {
        xyb_from_modular(&image, &lf_dequant, &headers.metadata)?
    } else {
        raw_planes_as_float(&image)?
    };
    Ok(LfFrame {
        width: image.width,
        height: image.height,
        planes,
    })
}

/// The decoded modular planes as `f32`, with no L.2.2 pre-step — the `false`
/// arm of [`LF_FRAME_IS_XYB_PRESTEP`].
fn raw_planes_as_float(image: &DecodedImage) -> Result<[Vec<f32>; 3]> {
    if image.num_colour_channels != 3 {
        return Err(unsupported(
            "a greyscale LF frame in an XYB image",
            "18181-1 L.2.2",
        ));
    }
    let plane = |i: usize| -> Vec<f32> {
        image
            .planes
            .get(i)
            .map(|p| p.samples.iter().map(|&v| v as f32).collect())
            .unwrap_or_default()
    };
    Ok([plane(1), plane(0), plane(2)])
}

/// L.2.2's `kModular` pre-step over a whole decoded modular frame.
///
/// The clause names the first three channels **y', x', B'** — channel 0 is
/// luma, not X — the same Y, X, B ordering `LfQuant` uses (G.2.2). Reading
/// them as X, Y, B swaps the two chroma-carrying planes. The result is in
/// Table I.1's `X, Y, B` numbering.
fn xyb_from_modular(
    image: &DecodedImage,
    lf_dequant: &crate::vardct::quantizer::LfChannelDequantization,
    metadata: &ImageMetadata,
) -> Result<[Vec<f32>; 3]> {
    if !metadata.xyb_encoded {
        // A non-XYB image needs no pre-step: the samples are already in the
        // space the consumer names. Nothing available exercises it, and
        // guessing the scale is worse than refusing.
        return Err(unsupported(
            "a stored frame in a non-XYB image",
            "18181-1 L.2.2",
        ));
    }
    if image.num_colour_channels != 3 {
        return Err(unsupported(
            "a greyscale stored frame in an XYB image",
            "18181-1 L.2.2",
        ));
    }
    let [m_x, m_y, m_b] = lf_dequant.unscaled();
    let cells = image.width as usize * image.height as usize;
    let mut planes = [
        vec![0.0f32; cells],
        vec![0.0f32; cells],
        vec![0.0f32; cells],
    ];
    for y in 0..image.height {
        for x in 0..image.width {
            let idx = y as usize * image.width as usize + x as usize;
            let y_prime = image.planes.first().map_or(0, |p| p.get(x, y)) as f32;
            let x_prime = image.planes.get(1).map_or(0, |p| p.get(x, y)) as f32;
            let b_prime = image.planes.get(2).map_or(0, |p| p.get(x, y)) as f32;
            if let Some(slot) = planes.first_mut().and_then(|p| p.get_mut(idx)) {
                *slot = x_prime * m_x;
            }
            if let Some(slot) = planes.get_mut(1).and_then(|p| p.get_mut(idx)) {
                *slot = y_prime * m_y;
            }
            if let Some(slot) = planes.get_mut(2).and_then(|p| p.get_mut(idx)) {
                *slot = (b_prime + y_prime) * m_b;
            }
        }
    }
    Ok(planes)
}

/// Decodes a `kReferenceOnly` frame into a [`ReferenceFrame`] (18181-1 F.2).
///
/// The stored samples are in the colour space *before* the inverse colour
/// transform, which is what K.3.2 blends in. For an `xyb_encoded` image and a
/// `kModular` reference frame — the shape every real encoder emits, and the
/// one `bike`/`bike_5`/`progressive` use — that means applying L.2.2's
/// `kModular` pre-step and stopping:
///
/// ```text
/// X = x' * m_x_lf_unscaled
/// Y = y' * m_y_lf_unscaled
/// B = (B' + Y') * m_b_lf_unscaled
/// ```
///
/// Note the channel order L.2.2 states for those "values in the first three
/// channels": **y', x', B'**. Channel 0 is luma, not X — the same Y, X, B
/// ordering `LfQuant` turned out to use (G.2.2). Reading them as X, Y, B
/// swaps the two chroma-carrying planes of every patch.
///
/// # Errors
///
/// As [`decode_modular_frame`], plus [`DecodeError::Unsupported`] for a
/// `kVarDCT` reference frame (no available stream emits one) and for extra
/// channels in a reference frame.
#[expect(
    clippy::too_many_arguments,
    reason = "the same frame-layer inputs decode_frame takes"
)]
fn decode_reference_frame(
    codestream: &[u8],
    base: usize,
    toc: &Toc,
    geometry: &FrameGeometry,
    header: &FrameHeader,
    headers: &ImageHeaders,
    limits: &Limits,
    guard: &mut AllocGuard,
) -> Result<ReferenceFrame> {
    if header.encoding != Encoding::VarDct {
        // The common case: a small kModular rectangle holding the patch atlas.
        let (image, lf_dequant) = decode_modular_frame(
            codestream,
            base,
            toc,
            geometry,
            header,
            headers,
            FramePurpose::Reference,
            limits,
            guard,
        )?;
        return reference_from_modular(&image, &lf_dequant, &headers.metadata, header);
    }
    Err(unsupported("a kVarDCT reference frame", "18181-1 F.2"))
}

/// L.2.2's `kModular` pre-step, turning a decoded modular reference frame into
/// the XYB planes K.3.2 blends.
fn reference_from_modular(
    image: &DecodedImage,
    lf_dequant: &crate::vardct::quantizer::LfChannelDequantization,
    metadata: &ImageMetadata,
    header: &FrameHeader,
) -> Result<ReferenceFrame> {
    let planes = xyb_from_modular(image, lf_dequant, metadata)?;
    let (width, height) = (image.width, image.height);
    Ok(ReferenceFrame {
        origin_x: if header.have_crop { header.x0 } else { 0 },
        origin_y: if header.have_crop { header.y0 } else { 0 },
        width,
        height,
        planes,
    })
}

/// Decodes one modular frame's sections into planes.
#[expect(
    clippy::too_many_arguments,
    reason = "the frame layer genuinely needs all of this; a bundle struct \
              would only move the argument list somewhere less visible"
)]
fn decode_modular_frame(
    codestream: &[u8],
    base: usize,
    toc: &Toc,
    geometry: &FrameGeometry,
    header: &FrameHeader,
    headers: &ImageHeaders,
    purpose: FramePurpose,
    limits: &Limits,
    guard: &mut AllocGuard,
) -> Result<(
    DecodedImage,
    crate::vardct::quantizer::LfChannelDequantization,
)> {
    let metadata = &headers.metadata;
    check_supported_modular(header, metadata, purpose)?;

    let (specs, num_colour) = initial_channels(header, metadata, geometry)?;
    let group_dim = geometry.group_dim();

    let mut options = ModularOptions {
        stream_index: 0,
        bits_per_sample: metadata.bit_depth.bits_per_sample(),
        ..ModularOptions::level10()
    };

    // A single-section frame carries every structure consecutively in one bit
    // stream (F.3.1); otherwise each section is addressed from `base`.
    let single = geometry.is_single_section();
    let whole = section_slice(codestream, base, toc, 0)?;
    let mut single_reader = BitReader::new(whole);

    // ---- LfGlobal (G.1) --------------------------------------------------
    let (lf_dequant, global_tree, mut partial) = {
        let mut owned;
        let reader: &mut BitReader<'_> = if single {
            &mut single_reader
        } else {
            owned = BitReader::new(section_slice(codestream, base, toc, 0)?);
            &mut owned
        };

        // G.1.2 LfChannelDequantization is present for every encoding. Modular
        // mode never uses the weights, but the bits are still there, and the
        // typed reader is the one place their layout is asserted.
        let lf_dequant = crate::vardct::quantizer::read_lf_channel_dequantization(reader)?;

        // G.1.3 GlobalModular.
        let have_global_tree = reader.read_bool()?;
        let global_tree = if have_global_tree {
            Some(read_global_tree(reader, &options, guard)?)
        } else {
            None
        };
        let source = tree_source(global_tree.as_ref(), true);
        let partial = decode_sub_bitstream_partial(
            reader,
            &specs,
            &options,
            source,
            ChannelStop::GlobalModular { group_dim },
            guard,
        )?;
        (lf_dequant, global_tree, partial)
    };

    let first_undecoded = partial.first_undecoded();
    let nb_meta = partial.header().layout().nb_meta_channels;

    // ---- LfGroup sections (G.2.3) ---------------------------------------
    let num_lf_groups = geometry.num_lf_groups();
    for lf_index in 0..num_lf_groups {
        let rect = geometry
            .lf_group_rect(lf_index)
            .ok_or_else(|| unsupported("an LF group index past the grid", "18181-1 G.2"))?;
        let selected: Vec<usize> = (first_undecoded..partial.channels().len())
            .filter(|&i| {
                partial
                    .channels()
                    .get(i)
                    .is_some_and(|c| c.hshift() >= 3 && c.vshift() >= 3)
            })
            .collect();
        // H.4.1: ModularLfGroup streams are numbered 1 + num_lf_groups + index.
        options.stream_index = stream_index_of::modular_lf_group(geometry, lf_index)?;
        decode_group_section(
            codestream,
            base,
            toc,
            single.then_some(&mut single_reader),
            usize::try_from(1 + lf_index).unwrap_or(0),
            &mut partial,
            &selected,
            rect,
            global_tree.as_ref(),
            &options,
            guard,
        )?;
    }

    // ---- PassGroup sections (G.4.2) -------------------------------------
    let num_groups = geometry.num_groups();
    let pairs = header.passes.pairs_with_implicit_final();
    let mut maxshift = 3i32;
    let mut decoded_in_pass = vec![false; partial.channels().len()];
    for pass in 0..header.passes.num_passes {
        // G.4.2: minshift is log2(downsample[n]) when this pass is last_pass[n],
        // and maxshift otherwise (in which case the pass carries no modular
        // data at all).
        let minshift = pairs
            .iter()
            .find(|&&(_, last_pass)| last_pass == pass)
            .map_or(maxshift, |&(downsample, _)| {
                downsample.trailing_zeros() as i32
            });

        let selected: Vec<usize> = (first_undecoded..partial.channels().len())
            .filter(|&i| {
                if i < nb_meta || decoded_in_pass.get(i).copied().unwrap_or(true) {
                    return false;
                }
                let Some(c) = partial.channels().get(i) else {
                    return false;
                };
                if !G42_SIZE_TEST_IS_SHIFTED && c.width() <= group_dim && c.height() <= group_dim {
                    return false;
                }
                if G42_SIZE_TEST_IS_SHIFTED
                    && c.width() <= (group_dim >> c.hshift().clamp(0, 31))
                    && c.height() <= (group_dim >> c.vshift().clamp(0, 31))
                {
                    return false;
                }
                if c.hshift() >= 3 && c.vshift() >= 3 {
                    return false;
                }
                let m = c.hshift().min(c.vshift());
                minshift <= m && m < maxshift
            })
            .collect();

        for group in 0..num_groups {
            let rect = geometry
                .group_rect(group)
                .ok_or_else(|| unsupported("a group index past the grid", "18181-1 G.4"))?;
            // H.4.1: ModularGroup streams are numbered
            // 1 + 3 * num_lf_groups + 17 + num_groups * pass + group.
            options.stream_index =
                stream_index_of::modular_group(geometry, u64::from(pass), group)?;
            let section = 2 + num_lf_groups + num_groups * u64::from(pass) + group;
            decode_group_section(
                codestream,
                base,
                toc,
                single.then_some(&mut single_reader),
                usize::try_from(section).unwrap_or(0),
                &mut partial,
                &selected,
                rect,
                global_tree.as_ref(),
                &options,
                guard,
            )?;
        }
        for &i in &selected {
            if let Some(slot) = decoded_in_pass.get_mut(i) {
                *slot = true;
            }
        }
        maxshift = minshift;
    }

    // ---- H.6 inverse transforms over the completed image -----------------
    options.stream_index = 0;
    let image = partial.into_image(&options, guard)?;
    let assembled = assemble(
        image.into_channels(),
        metadata,
        geometry,
        num_colour,
        limits,
    )?;
    Ok((assembled, lf_dequant))
}

/// Decodes one LF-group or pass-group modular sub-bitstream and copies the
/// result back into the frame-wide channel list.
#[expect(
    clippy::too_many_arguments,
    reason = "shared by G.2.3 and G.4.2, which differ only in these parameters"
)]
fn decode_group_section(
    codestream: &[u8],
    base: usize,
    toc: &Toc,
    single_reader: Option<&mut BitReader<'_>>,
    section: usize,
    partial: &mut crate::modular::PartialModular,
    selected: &[usize],
    rect: Rect,
    global_tree: Option<&GlobalTree>,
    options: &ModularOptions,
    guard: &mut AllocGuard,
) -> Result<()> {
    // H.1: a sub-bitstream with no channels is not read at all.
    if selected.is_empty() {
        return Ok(());
    }

    let mut specs = Vec::with_capacity(selected.len());
    let mut origins = Vec::with_capacity(selected.len());
    for &i in selected {
        let Some(c) = partial.channels().get(i) else {
            continue;
        };
        let (hshift, vshift) = (c.hshift().max(0) as u32, c.vshift().max(0) as u32);
        // G.2.3 / G.4.2: "the group dimensions and the x,y offsets are
        // right-shifted by hshift (for x and width) and vshift (for y and
        // height)". Taken as: the origin shifts exactly (group origins are
        // multiples of group_dim, hence of 2^shift), and the extent is the
        // difference of the two shifted edges rounded up, so the group
        // rectangles tile the shifted channel exactly even when the frame's
        // last group is partial. A literal `width >> hshift` would drop the
        // final column of an odd-sized channel.
        let x0 = rect.x0 >> hshift;
        let y0 = rect.y0 >> vshift;
        let x1 = (rect.x0 + rect.width).div_ceil(1u32 << hshift);
        let y1 = (rect.y0 + rect.height).div_ceil(1u32 << vshift);
        specs.push(ChannelSpec::with_shifts(
            x1.saturating_sub(x0).min(c.width().saturating_sub(x0)),
            y1.saturating_sub(y0).min(c.height().saturating_sub(y0)),
            c.hshift(),
            c.vshift(),
        ));
        origins.push((x0, y0));
    }

    let source = tree_source(global_tree, true);
    let decoded = match single_reader {
        Some(reader) => decode_sub_bitstream_with(reader, &specs, options, source, guard)?,
        None => {
            let slice = section_slice(codestream, base, toc, section)?;
            if slice.is_empty() {
                return Ok(());
            }
            let mut reader = BitReader::new(slice);
            decode_sub_bitstream_with(&mut reader, &specs, options, source, guard)?
        }
    };

    // "The decoded modular group data is then copied into the partially
    // decoded GlobalModular image in the corresponding positions."
    let channels = decoded.into_channels();
    for ((&target, source_channel), &(x0, y0)) in
        selected.iter().zip(channels.iter()).zip(origins.iter())
    {
        let Some(dest) = partial.channels_mut().get_mut(target) else {
            continue;
        };
        for y in 0..source_channel.height() {
            for x in 0..source_channel.width() {
                dest.set(x0 + x, y0 + y, source_channel.get(x, y));
            }
        }
    }
    Ok(())
}

/// Picks the tree source for a sub-bitstream (H.2).
const fn tree_source(global: Option<&GlobalTree>, restart: bool) -> TreeSource<'_> {
    match global {
        Some(global) => TreeSource::Global { global, restart },
        None => TreeSource::Local,
    }
}

/// Turns the final channel list into output planes (G.4.2, last paragraph).
fn assemble(
    channels: Vec<Channel>,
    metadata: &ImageMetadata,
    geometry: &FrameGeometry,
    num_colour: usize,
    limits: &Limits,
) -> Result<DecodedImage> {
    let expected = num_colour + metadata.num_extra();
    if channels.len() != expected {
        return Err(DecodeError::out_of_range(
            "decoded channel count",
            "G.1.3",
            channels.len() as u64,
        ));
    }
    let _ = limits;

    let mut planes = Vec::with_capacity(channels.len());
    for (index, channel) in channels.into_iter().enumerate() {
        let bits = if index < num_colour {
            metadata.bit_depth.bits_per_sample()
        } else {
            metadata
                .ec_info
                .get(index - num_colour)
                .map_or(8, |info| info.bit_depth.bits_per_sample())
        };
        planes.push(Plane {
            width: channel.width(),
            height: channel.height(),
            bits_per_sample: bits,
            samples: channel.samples().to_vec(),
        });
    }

    Ok(DecodedImage {
        width: geometry.width(),
        height: geometry.height(),
        planes,
        num_colour_channels: num_colour,
        // Filled in by `decode`: the profile belongs to the codestream, not to
        // any one frame (Table A.1).
        icc_profile: None,
        float_planes: None,
    })
}

// ---------------------------------------------------------------------------
// The kVarDCT frame (Annexes G and I, Annex J, L.2)
// ---------------------------------------------------------------------------

/// One LF group's decoded state, held until the pass groups that reference it
/// have been decoded.
struct LfGroupState {
    /// G.2.2's quantized LF planes. Kept alongside the dequantized ones
    /// because I.4's block context reads the *quantized* `qdc` while I.8
    /// reads the dequantized samples — two different numbers at the same
    /// coordinate, and conflating them is exactly the kind of mistake the
    /// typed split exists to prevent.
    ///
    /// `None` under `kUseLfFrame`, where G.2.2 is skipped outright: the clause
    /// then declares every `LfQuant` sample to be `-inf`, so `lf_idx` is
    /// always zero and `qdc` has no other reader.
    quant: Option<crate::vardct::lf::LfQuantPlanes>,
    /// I.5.2's output: dequantized, CfL-corrected, smoothed LF planes.
    lf: crate::vardct::lf::DequantizedLf,
    /// G.2.4's `Sharpness` plane, one sample per 8x8 block.
    sharpness: Channel,
    /// I.6's HF chroma-from-luma factors, one per 64x64 tile.
    cfl: crate::vardct::cfl::CflFactors,
    /// G.2.4's greedy varblock placement, LF-group-relative.
    placements: Vec<crate::vardct::hf_meta::VarblockPlacement>,
}

/// One pass group's decoded HF coefficients, held until every pass is in.
struct GroupState {
    /// Index into `geometry.group_rect`.
    group: u64,
    /// The LF group this group sits in.
    lf_index: u64,
    /// The group's varblocks, group-relative, in raster order.
    varblocks: Vec<crate::vardct::hf_coeff::HfVarblock>,
    /// Quantized coefficients, accumulated over every pass.
    coefficients: crate::vardct::hf_coeff::HfCoefficients,
}

/// Decodes one `kVarDCT` frame (Table F.1's four section kinds) into planes.
///
/// The section walk mirrors [`decode_modular_frame`]'s, with the two
/// differences that make VarDCT a different pipeline rather than a variant:
/// `LfGlobal` carries three extra bundles (I.2.1–I.2.3), and the `HfGlobal`
/// section — which a modular frame leaves empty — carries the dequantization
/// matrices and the per-pass coefficient orders and histograms.
#[expect(
    clippy::too_many_lines,
    reason = "this is Table F.1 read top to bottom; splitting it would hide \
              the one thing a reader needs to see, which is the order the \
              sections are consumed in"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the frame layer needs all of this; a bundle struct would only \
              move the argument list somewhere less visible"
)]
fn decode_vardct_frame(
    codestream: &[u8],
    base: usize,
    toc: &Toc,
    geometry: &FrameGeometry,
    header: &FrameHeader,
    headers: &ImageHeaders,
    references: &ReferenceSlots,
    lf_frames: &LfFrameSlots,
    guard: &mut AllocGuard,
) -> Result<DecodedImage> {
    use crate::vardct::cfl::CflFactors;
    use crate::vardct::dequant_matrix::read_hf_global_params;
    use crate::vardct::hf_coeff::{
        HfCoefficients, HfGroupParams, HfVarblock, decode_hf_group, read_hf_passes,
    };
    use crate::vardct::hf_meta::{place_varblocks, read_hf_metadata};
    use crate::vardct::lf::{dequantize_lf, read_lf_quant};
    use crate::vardct::quantizer::{read_lf_channel_dequantization, read_lf_global_vardct};
    use crate::vardct::render;

    let metadata = &headers.metadata;
    check_supported_vardct(header, metadata)?;

    let modular_options = ModularOptions {
        stream_index: 0,
        bits_per_sample: metadata.bit_depth.bits_per_sample(),
        ..ModularOptions::level10()
    };

    let single = geometry.is_single_section();
    let whole = section_slice(codestream, base, toc, 0)?;
    let mut single_reader = BitReader::new(whole);

    // ---- LfGlobal (G.1) --------------------------------------------------
    let (patches, lf_dequant, vardct, global_tree) = {
        let mut owned;
        let reader: &mut BitReader<'_> = if single {
            &mut single_reader
        } else {
            owned = BitReader::new(section_slice(codestream, base, toc, 0)?);
            &mut owned
        };
        // K.3.1 is the FIRST row of Table G.1, ahead of everything else in
        // LfGlobal. Reading it late shifts every subsequent field.
        let patches = if header.flags.patches() {
            crate::frame::patches::read_patches(
                reader,
                metadata.num_extra(),
                geometry.width(),
                geometry.height(),
                crate::frame::patches::max_num_patches(geometry.width(), geometry.height()),
                guard,
            )?
        } else {
            crate::frame::patches::PatchDictionary::default()
        };
        let lf_dequant = read_lf_channel_dequantization(reader)?;
        let vardct = read_lf_global_vardct(reader, guard)?;
        // G.1.3 GlobalModular: the leading Bool() is read whatever the channel
        // count. With no extra channels and no modular colour channels the
        // channel list is empty, and H.1 says an empty sub-bitstream is not
        // read at all — which `check_supported_vardct` has already ensured.
        let have_global_tree = reader.read_bool()?;
        let global_tree = if have_global_tree {
            Some(read_global_tree(reader, &modular_options, guard)?)
        } else {
            None
        };
        (patches, lf_dequant, vardct, global_tree)
    };

    let multipliers = vardct.quantizer.lf_multipliers(&lf_dequant);
    let smoothing = header.flags.adaptive_lf_smoothing();

    // G.2.2: with kUseLfFrame the whole `LF coefficients` row is skipped —
    // no `extra_precision`, no `LfQuant` sub-bitstream — and so is I.5.2,
    // which carries chroma-from-luma for LF (I.6 says so in its own words) and
    // the adaptive smoothing pass with it. The samples come from
    // `LFFrame[frame_header.lf_level]` instead.
    let use_lf_frame = header.flags.use_lf_frame();
    let lf_frame = if use_lf_frame {
        Some(lf_frame_slot(lf_frames, header.lf_level, geometry)?)
    } else {
        None
    };

    // ---- LfGroup sections (G.2) -----------------------------------------
    let num_lf_groups = geometry.num_lf_groups();
    let mut lf_groups: Vec<LfGroupState> = Vec::with_capacity(
        usize::try_from(num_lf_groups)
            .map_err(|_| unsupported("an oversized LF group grid", "18181-1 G.2"))?,
    );
    for lf_index in 0..num_lf_groups {
        let rect = geometry
            .lf_group_rect(lf_index)
            .ok_or_else(|| unsupported("an LF group index past the grid", "18181-1 G.2"))?;
        let mut owned;
        let reader: &mut BitReader<'_> = if single {
            &mut single_reader
        } else {
            owned = BitReader::new(section_slice(
                codestream,
                base,
                toc,
                usize::try_from(1 + lf_index).unwrap_or(0),
            )?);
            &mut owned
        };

        // G.2.2 LfQuant.
        let mut options = ModularOptions {
            stream_index: stream_index_of::lf_coefficients(geometry, lf_index)?,
            ..modular_options
        };
        let quant = if use_lf_frame {
            None
        } else {
            Some(read_lf_quant(
                reader,
                rect.width,
                rect.height,
                header.jpeg_upsampling,
                &options,
                tree_source(global_tree.as_ref(), true),
                guard,
            )?)
        };

        // G.2.3 ModularLfGroup: no channels, so H.1 reads nothing. (Extra
        // channels are rejected above; a squeezed colour channel cannot exist
        // in a kVarDCT frame.)

        // G.2.4 HfMetadata.
        options.stream_index = stream_index_of::hf_metadata(geometry, lf_index)?;
        let meta = read_hf_metadata(
            reader,
            rect.width,
            rect.height,
            &options,
            tree_source(global_tree.as_ref(), true),
            guard,
        )?;
        let placements = place_varblocks(
            &meta.block_info,
            rect.width.div_ceil(8),
            rect.height.div_ceil(8),
        )?;

        // I.5.2's dequantization and I.6's LF chroma-from-luma, or G.2.2's
        // substitution. I.5.2's *smoothing* pass is deliberately not run here:
        // it is scoped to the frame-wide LF image, not to one LF group, and
        // runs below once every group is in. See `smooth_lf_image`.
        let lf = match (&quant, lf_frame) {
            (Some(quant), _) => {
                dequantize_lf(quant, &multipliers, &vardct.lf_chan_corr, false, false)?
            }
            (None, Some(frame)) => lf_from_frame(frame, rect),
            (None, None) => {
                return Err(unsupported(
                    "an LF group with no LF source",
                    "18181-1 G.2.2",
                ));
            }
        };

        lf_groups.push(LfGroupState {
            quant,
            lf,
            sharpness: meta.sharpness,
            cfl: CflFactors::for_hf(&vardct.lf_chan_corr, meta.x_from_y, meta.b_from_y),
            placements,
        });
    }

    // ---- I.5.2's adaptive smoothing, over the frame-wide LF image --------
    //
    // Skipped under kUseLfFrame (I.5.2 is skipped in its entirety for such a
    // frame, and the smoothing paragraph is inside it) and under
    // kSkipAdaptiveLFSmoothing.
    if smoothing && !use_lf_frame {
        smooth_lf_image(&mut lf_groups, geometry, &multipliers, guard)?;
    }

    // ---- HfGlobal (G.3) --------------------------------------------------
    //
    // This section used to be skipped outright, which is why nothing before
    // slice 8 could decode a lossy frame: every dequantization matrix and
    // every coefficient histogram lives here.
    let num_groups = geometry.num_groups();
    let (hf_params, mut passes) = {
        let mut owned;
        let reader: &mut BitReader<'_> = if single {
            &mut single_reader
        } else {
            owned = BitReader::new(section_slice(
                codestream,
                base,
                toc,
                usize::try_from(1 + num_lf_groups).unwrap_or(0),
            )?);
            &mut owned
        };
        let params = read_hf_global_params(reader, num_groups, guard)?;
        // I.2.4 RAW mode reads its 3-channel matrix from a modular
        // sub-bitstream *inline*, at this bit position. `read_dequant_matrices`
        // does not consume it, so a RAW stream would desynchronise everything
        // after it rather than merely lack a matrix. Refuse loudly.
        if !params.matrices.raw_requests().is_empty() {
            return Err(unsupported("RAW dequantization matrices", "18181-1 I.2.4"));
        }
        let passes = read_hf_passes(
            reader,
            header.passes.num_passes,
            vardct.hf_block_ctx.nb_block_ctx(),
            params.num_hf_presets,
            guard,
        )?;
        (params, passes)
    };

    // ---- PassGroup sections (G.4) ---------------------------------------
    let lf_dim = geometry.group_dim() * 8;
    let lf_groups_x = geometry.width().div_ceil(lf_dim);
    let mut groups: Vec<GroupState> = Vec::new();

    for pass in 0..header.passes.num_passes {
        for group in 0..num_groups {
            let rect = geometry
                .group_rect(group)
                .ok_or_else(|| unsupported("a group index past the grid", "18181-1 G.4"))?;
            let lf_index =
                u64::from(rect.y0 / lf_dim) * u64::from(lf_groups_x) + u64::from(rect.x0 / lf_dim);
            let lf_rect = geometry
                .lf_group_rect(lf_index)
                .ok_or_else(|| unsupported("an LF group index past the grid", "18181-1 G.2"))?;
            let state = lf_groups
                .get(usize::try_from(lf_index).unwrap_or(usize::MAX))
                .ok_or_else(|| unsupported("an LF group index past the grid", "18181-1 G.2"))?;

            let origin_bx = (rect.x0 - lf_rect.x0) / 8;
            let origin_by = (rect.y0 - lf_rect.y0) / 8;
            let blocks_w = rect.width.div_ceil(8);
            let blocks_h = rect.height.div_ceil(8);

            // I.4 is defined per group; G.2.4's placement is per LF group.
            // Select the varblocks whose top-left block falls in this group
            // and translate them to group-relative coordinates.
            let varblocks: Vec<HfVarblock> = state
                .placements
                .iter()
                .filter(|p| {
                    let (x, y) = (p.position.bx(), p.position.by());
                    x >= origin_bx
                        && y >= origin_by
                        && x < origin_bx + blocks_w
                        && y < origin_by + blocks_h
                })
                .map(|p| {
                    let (x, y) = (p.position.bx(), p.position.by());
                    HfVarblock {
                        bx: x - origin_bx,
                        by: y - origin_by,
                        transform: p.transform,
                        hf_mul: p.hf_mul,
                        // I.4's qdc is LfQuant at the varblock's top-left
                        // block, in the clause's [X, Y, B] channel order.
                        // Under kUseLfFrame there is no LfQuant and G.2.2
                        // pins lf_idx to zero, so the value is never read.
                        qdc: state
                            .quant
                            .as_ref()
                            .map_or([0; 3], |q| [q.x.get(x, y), q.y.get(x, y), q.b.get(x, y)]),
                    }
                })
                .collect();

            let slot = usize::try_from(group).unwrap_or(usize::MAX);
            if pass == 0 {
                let coefficients = HfCoefficients::new(&varblocks, guard)?;
                groups.push(GroupState {
                    group,
                    lf_index,
                    varblocks,
                    coefficients,
                });
            }
            let state = groups
                .get_mut(slot)
                .ok_or_else(|| unsupported("a group index past the grid", "18181-1 G.4"))?;

            let section = 2 + num_lf_groups + num_groups * u64::from(pass) + group;
            let mut owned;
            let reader: &mut BitReader<'_> = if single {
                &mut single_reader
            } else {
                owned = BitReader::new(section_slice(
                    codestream,
                    base,
                    toc,
                    usize::try_from(section).unwrap_or(0),
                )?);
                &mut owned
            };

            let group_params = HfGroupParams {
                shift: header.passes.shift_for(pass),
                blocks_w,
                blocks_h,
                num_hf_presets: hf_params.num_hf_presets,
                // G.2.2: under kUseLfFrame every LfQuant sample counts as
                // -inf, so no lf_threshold is ever exceeded and lf_idx is
                // zero regardless of the thresholds.
                lf_idx_is_zero: use_lf_frame,
            };
            let (orders, histograms) = passes.split_pass(pass as usize)?;
            decode_hf_group(
                reader,
                &group_params,
                &state.varblocks,
                &orders,
                histograms,
                &vardct.hf_block_ctx,
                &mut state.coefficients,
                guard,
            )?;

            // G.4.2 modular group data: no channels, nothing read.
        }
    }

    // ---- I.5.3, I.6, I.8, I.9 -------------------------------------------
    let dequant = render::HfDequantParams {
        matrices: &hf_params.matrices,
        quant_bias: opsin(metadata).quant_bias,
        quant_bias_numerator: opsin(metadata).quant_bias_numerator,
        global_scale: vardct.quantizer.global_scale,
        x_qm_scale: header.x_qm_scale,
        b_qm_scale: header.b_qm_scale,
    };

    let (width, height) = (geometry.width(), geometry.height());
    let mut planes = render::ColourPlanes::zeros(width, height, guard)?;
    let mut sigma = render::SigmaPlanes::zeros(width, height, guard)?;
    let filter = &header.restoration_filter;

    for state in &groups {
        let rect = geometry
            .group_rect(state.group)
            .ok_or_else(|| unsupported("a group index past the grid", "18181-1 G.4"))?;
        let lf_rect = geometry
            .lf_group_rect(state.lf_index)
            .ok_or_else(|| unsupported("an LF group index past the grid", "18181-1 G.2"))?;
        let lf_state = lf_groups
            .get(usize::try_from(state.lf_index).unwrap_or(usize::MAX))
            .ok_or_else(|| unsupported("an LF group index past the grid", "18181-1 G.2"))?;
        let origin_bx = (rect.x0 - lf_rect.x0) / 8;
        let origin_by = (rect.y0 - lf_rect.y0) / 8;

        for (index, vb) in state.varblocks.iter().enumerate() {
            // Back to LF-group-relative coordinates: LfQuant, Sharpness and
            // the CfL tiles are all indexed that way.
            let (bx, by) = (vb.bx + origin_bx, vb.by + origin_by);
            let blocks = render::render_varblock(
                vb.transform,
                [
                    state.coefficients.block(index, 0)?,
                    state.coefficients.block(index, 1)?,
                    state.coefficients.block(index, 2)?,
                ],
                vb.hf_mul,
                &dequant,
                // I.6: the 64x64 rectangle containing the sample; a varblock
                // never spans two tiles, because every varblock is at most
                // 32x32 samples and is aligned to its own size.
                lf_state.cfl.at_pixel(bx * 8, by * 8),
                [&lf_state.lf.x, &lf_state.lf.y, &lf_state.lf.b],
                bx,
                by,
            )?;

            let x0 = lf_rect.x0 + bx * 8;
            let y0 = lf_rect.y0 + by * 8;
            for (c, block) in blocks.iter().enumerate() {
                for row in 0..block.rows() {
                    for col in 0..block.cols() {
                        // Sample extents come from Table I.1 (at most 256), so
                        // `small` never saturates here.
                        planes.set(c, x0 + small(col), y0 + small(row), block.at(col, row));
                    }
                }
            }

            // J.4.3's sigma field, over the frame's 8x8-block grid. `mul` is
            // per varblock; `Sharpness` is per 8x8 block. The skip test reads
            // the varblock's own sigma, taken at its top-left block.
            let (rows, cols) = vb.transform.block_dims();
            let (block_rows, block_cols) = (small(rows), small(cols));
            let vb_sigma = render::varblock_epf_sigma(
                &dequant,
                vb.hf_mul,
                sharpness_at(&lf_state.sharpness, bx, by),
                filter,
            )?;
            for dy in 0..block_rows {
                for dx in 0..block_cols {
                    let s = render::varblock_epf_sigma(
                        &dequant,
                        vb.hf_mul,
                        sharpness_at(&lf_state.sharpness, bx + dx, by + dy),
                        filter,
                    )?;
                    sigma.set(
                        wide((lf_rect.x0 / 8) + bx + dx),
                        wide((lf_rect.y0 / 8) + by + dy),
                        s,
                        vb_sigma,
                    );
                }
            }
        }
    }

    // ---- Annex J, then Annex K, then Annex L -----------------------------
    //
    // Clause 4 fixes this order: frame data (G), restoration filters (J),
    // image features (K), and "finally ... colour transforms as specified in
    // Annex L". K.3.2 repeats it from its own side.
    render::apply_restoration(&mut planes, filter, &sigma)?;
    render::apply_patches(&mut planes, &patches, references, metadata.num_extra())?;
    render::to_linear_srgb(&mut planes, &opsin_inverse(metadata));
    apply_transfer_function(&mut planes, metadata)?;

    assemble_float(planes, metadata, geometry)
}

/// The `LFFrame[frame_header.lf_level]` slot G.2.2 names, checked against the
/// consuming frame's LF grid.
///
/// The shape check is the whole point of doing this once rather than per LF
/// group: an LF frame whose dimensions do not match `ceil(width / 8) x
/// ceil(height / 8)` would otherwise silently contribute zeros at the edges,
/// which reads as a plausible image rather than as an error.
fn lf_frame_slot<'a>(
    lf_frames: &'a LfFrameSlots,
    lf_level: u32,
    geometry: &FrameGeometry,
) -> Result<&'a LfFrame> {
    let index = usize::try_from(lf_level).unwrap_or(usize::MAX);
    let frame = lf_frames
        .get(index)
        .and_then(Option::as_ref)
        .ok_or_else(|| unsupported("kUseLfFrame with no stored LF frame", "18181-1 G.2.2"))?;
    let expected = (geometry.width().div_ceil(8), geometry.height().div_ceil(8));
    if (frame.width, frame.height) != expected {
        return Err(DecodeError::out_of_range(
            "LF frame dimensions against the frame's LF grid",
            "F.1",
            u64::from(frame.width),
        ));
    }
    Ok(frame)
}

/// I.5.2's adaptive smoothing pass, over the **frame-wide** LF image.
///
/// # Why this is not a per-LF-group operation
///
/// I.5.2 scopes the pass to "each LF sample of the image that is not in the
/// first or last row or column", and closes by having I.8 read "the
/// dequantized LF image". Both name the frame, not the LF group. Running the
/// pass per LF group instead skips the first and last row and column of
/// *every* group, so each internal LF-group seam keeps its unsmoothed values
/// while both sides of it were smoothed — and, worse, the samples one step
/// inside the seam are smoothed against a 3x3 neighbourhood that stops at the
/// group edge instead of continuing into the next group.
///
/// The two readings coincide exactly for any frame with a single LF group,
/// which is every fixture below 2048x2048 — that is why this survived slice 8
/// and its whole flip-point probe. It is visible on `bike` (2048x2560, one
/// internal seam at y = 2048) as a 16-row band carrying the entire remaining
/// error, and on the corpus-free reproducer fixture 64. Evidence and
/// localisation are the concurrent agent's, in
/// `docs/experiments/2026-08-04-negative-transfer-function-branch.md` §6.
///
/// # Why assembling after I.6 is sound
///
/// The clause's order is dequantize, then I.6 chroma-from-luma, then smooth.
/// The first two are per-sample: dequantization uses the group's own
/// `extra_precision` and the frame's `mXDC`/`mYDC`/`mBDC`, and I.6's LF arm
/// uses `x_factor_lf`/`b_factor_lf`, which are frame-wide constants from
/// I.2.3 rather than the per-64x64-tile factors the HF arm uses. Neither
/// looks at a neighbouring sample, so doing them per group and only then
/// assembling gives the same image as assembling first. Smoothing is the one
/// stage with a spatial footprint, and it is the one done frame-wide.
///
/// # Errors
///
/// A limit error if the frame-wide LF image exceeds the allocation budget, or
/// [`DecodeError::FieldOutOfRange`] if the three planes disagree in shape.
fn smooth_lf_image(
    lf_groups: &mut [LfGroupState],
    geometry: &FrameGeometry,
    multipliers: &crate::vardct::quantizer::LfDequantMultipliers,
    guard: &mut AllocGuard,
) -> Result<()> {
    use crate::vardct::lf::{DequantPlane, adaptive_smoothing};

    let width = geometry.width().div_ceil(8);
    let height = geometry.height().div_ceil(8);
    let cells = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| unsupported("an oversized LF image", "18181-1 I.5.2"))?;
    guard.charge(cells as u64 * 4 * 3)?;

    // The LF group's block origin, for both the gather and the scatter. An LF
    // group rectangle starts at a multiple of `8 * group_dim`, so the division
    // is exact.
    let origin = |index: usize| -> Result<(u32, u32)> {
        let rect = geometry
            .lf_group_rect(index as u64)
            .ok_or_else(|| unsupported("an LF group index past the grid", "18181-1 G.2"))?;
        Ok((rect.x0 / 8, rect.y0 / 8))
    };

    let mut planes = [
        DequantPlane {
            width,
            height,
            samples: vec![0.0; cells],
        },
        DequantPlane {
            width,
            height,
            samples: vec![0.0; cells],
        },
        DequantPlane {
            width,
            height,
            samples: vec![0.0; cells],
        },
    ];
    for (index, state) in lf_groups.iter().enumerate() {
        let (x0, y0) = origin(index)?;
        for (c, source) in [&state.lf.x, &state.lf.y, &state.lf.b]
            .into_iter()
            .enumerate()
        {
            let Some(dest) = planes.get_mut(c) else {
                continue;
            };
            for y in 0..source.height {
                for x in 0..source.width {
                    let (fx, fy) = (x0 + x, y0 + y);
                    if fx >= width || fy >= height {
                        continue;
                    }
                    if let Some(slot) = dest
                        .samples
                        .get_mut(fy as usize * width as usize + fx as usize)
                    {
                        *slot = source.get(x, y);
                    }
                }
            }
        }
    }

    let [fx, fy, fb] = &planes;
    let (sx, sy, sb) = adaptive_smoothing(fx, fy, fb, multipliers)?;
    let smoothed = [sx, sy, sb];

    for (index, state) in lf_groups.iter_mut().enumerate() {
        let (x0, y0) = origin(index)?;
        for (c, dest) in [&mut state.lf.x, &mut state.lf.y, &mut state.lf.b]
            .into_iter()
            .enumerate()
        {
            let Some(source) = smoothed.get(c) else {
                continue;
            };
            for y in 0..dest.height {
                for x in 0..dest.width {
                    let value = source.get(x0 + x, y0 + y);
                    if let Some(slot) = dest
                        .samples
                        .get_mut(y as usize * dest.width as usize + x as usize)
                    {
                        *slot = value;
                    }
                }
            }
        }
    }
    Ok(())
}

/// G.2.2's substitution: one LF group's worth of `LFFrame[...]`.
///
/// `rect` is the LF group's *pixel* rectangle, so its origin is a multiple of
/// `8 * group_dim` and dividing by 8 is exact.
fn lf_from_frame(frame: &LfFrame, rect: Rect) -> crate::vardct::lf::DequantizedLf {
    use crate::vardct::lf::{DequantPlane, DequantizedLf};

    let (x0, y0) = (rect.x0 / 8, rect.y0 / 8);
    let width = (rect.x0 + rect.width).div_ceil(8).saturating_sub(x0);
    let height = (rect.y0 + rect.height).div_ceil(8).saturating_sub(y0);
    let cut = |plane: &Vec<f32>| -> DequantPlane {
        let mut samples = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let (sx, sy) = (x0 + x, y0 + y);
                let value = if sx < frame.width && sy < frame.height {
                    plane
                        .get(sy as usize * frame.width as usize + sx as usize)
                        .copied()
                        .unwrap_or(0.0)
                } else {
                    0.0
                };
                samples.push(value);
            }
        }
        DequantPlane {
            width,
            height,
            samples,
        }
    };
    let [x, y, b] = &frame.planes;
    DequantizedLf {
        x: cut(x),
        y: cut(y),
        b: cut(b),
    }
}

/// `Sharpness` at an LF-group-relative 8x8 block, clamped to the plane.
///
/// A varblock never crosses the LF-group edge (G.2.4, enforced by
/// `place_varblocks`), so the only way a lookup lands outside is the partial
/// last block row/column of a frame whose size is not a multiple of 8 — where
/// `Channel::get` returning 0 is the right answer anyway.
/// A Table I.1 sample or block extent as `u32`, saturating rather than
/// wrapping. Every caller passes a value at most 256.
fn small(v: usize) -> u32 {
    u32::try_from(v).unwrap_or(u32::MAX)
}

/// A frame-relative block coordinate as `usize`.
fn wide(v: u32) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}

fn sharpness_at(sharpness: &Channel, bx: u32, by: u32) -> u32 {
    sharpness.get(bx, by).clamp(0, 7).unsigned_abs()
}

/// The `OpsinInverseMatrix` bundle in force, signalled or default (Table L.1).
fn opsin(metadata: &ImageMetadata) -> crate::headers::opsin::OpsinInverseMatrix {
    metadata.opsin_inverse_matrix.unwrap_or_default()
}

/// L.2.2's transform, built from the signalled bundle and `intensity_target`.
fn opsin_inverse(metadata: &ImageMetadata) -> OpsinInverse {
    let oim = opsin(metadata);
    OpsinInverse::new(
        oim.inverse_matrix,
        oim.opsin_bias,
        metadata.tone_mapping.intensity_target,
    )
}

/// Applies the signalled transfer function to L.2.2's linear-light output.
///
/// # Which colour space the output is in
///
/// L.2.2 produces **linear** light. 18181-3 §4.2 compares samples "in the
/// colour space specified by the reference ICC profile", and for the
/// conformance corpus that profile is exactly the codestream's own signalled
/// encoding: `bike_5` signals `k709` and ships a BT.709 `reference.icc`,
/// `opsin_inverse` signals `kSRGB` and ships an sRGB one. So the decoder's
/// output space is the signalled encoding, and this function is the step that
/// gets there.
///
/// The exception is `want_icc`. Clause 4's note is explicit: when
/// `xyb_encoded` is set, `colour_encoding`, `bit_depth` and any embedded ICC
/// profile are **suggestions** about what to do after the conversion to linear
/// sRGB — so with `want_icc` there is no signalled enumerated encoding to
/// convert to, and the output stays linear. The corpus confirms it: the
/// `grayscale` cases set `want_icc` and their `reference.icc` is a linear grey
/// profile, while their `original.icc` is an unrelated Adobe printer profile.
///
/// Only sRGB, BT.709, a plain gamma and the identity are implemented — PQ,
/// HLG and DCI are refused rather than approximated.
fn apply_transfer_function(
    planes: &mut crate::vardct::render::ColourPlanes,
    metadata: &ImageMetadata,
) -> Result<()> {
    use crate::headers::colour::CustomTransferFunction;
    use crate::headers::enums::{Primaries, TransferFunction, WhitePoint};

    let ce = &metadata.colour_encoding;
    if ce.want_icc {
        return Ok(());
    }
    // L.2.2 hands back sRGB primaries at D65. Anything else needs a chromatic
    // adaptation the decoder does not implement.
    if !ce.is_grey() && ce.primaries != Primaries::KSrgb {
        return Err(unsupported("non-sRGB primaries", "18181-1 L.2.2"));
    }
    if ce.white_point != WhitePoint::KD65 {
        return Err(unsupported("a non-D65 white point", "18181-1 L.2.2"));
    }

    let map: fn(f32) -> f32 = match ce.tf {
        CustomTransferFunction::Enumerated(TransferFunction::KSrgb) => {
            jpxl_core::color::linear_to_srgb
        }
        CustomTransferFunction::Enumerated(TransferFunction::K709) => {
            jpxl_core::color::linear_to_rec709
        }
        CustomTransferFunction::Enumerated(TransferFunction::KLinear) => return Ok(()),
        CustomTransferFunction::Gamma(gamma) => {
            // E.7: gamma is the OETF exponent scaled by 10^7.
            let exponent = f64::from(gamma) / 1e7;
            #[allow(
                clippy::cast_possible_truncation,
                reason = "a deliberate one-time narrowing of a bounded exponent"
            )]
            let exponent = exponent as f32;
            for plane in &mut planes.planes {
                for v in plane.iter_mut() {
                    *v = jpxl_core::color::linear_to_gamma(*v, exponent);
                }
            }
            return Ok(());
        }
        _ => {
            return Err(unsupported(
                "a transfer function other than sRGB, BT.709, linear or gamma",
                "18181-1 E.6",
            ));
        }
    };

    for plane in &mut planes.planes {
        for v in plane.iter_mut() {
            *v = map(*v);
        }
    }
    Ok(())
}

/// Turns the frame's float colour planes into a [`DecodedImage`].
///
/// The float planes are kept verbatim (18181-3 §4.2 grades unclipped `f32`);
/// the integer planes are their quantization to `bits_per_sample`, which is
/// what the PGM/PPM writer consumes.
fn assemble_float(
    planes: crate::vardct::render::ColourPlanes,
    metadata: &ImageMetadata,
    geometry: &FrameGeometry,
) -> Result<DecodedImage> {
    let num_colour = if metadata.colour_encoding.is_grey() {
        1
    } else {
        3
    };
    let bits = metadata.bit_depth.bits_per_sample();
    let max = if bits >= 32 {
        f32::from(u16::MAX)
    } else {
        ((1u32 << bits) - 1) as f32
    };

    let (width, height) = (planes.width, planes.height);
    let mut float_planes = Vec::with_capacity(num_colour);
    let mut integer_planes = Vec::with_capacity(num_colour);
    for samples in planes.planes.into_iter().take(num_colour) {
        let quantized = samples
            .iter()
            .map(|v| {
                let scaled = (v * max).round();
                if scaled.is_finite() {
                    // Clamped into [0, max] with max < 2^32 before the cast,
                    // so the narrowing is exact for every reachable value.
                    #[allow(
                        clippy::cast_possible_truncation,
                        reason = "the value is clamped to [0, max] first"
                    )]
                    let quantized = scaled.clamp(0.0, max) as i32;
                    quantized
                } else {
                    0
                }
            })
            .collect();
        integer_planes.push(Plane {
            width,
            height,
            bits_per_sample: bits,
            samples: quantized,
        });
        float_planes.push(FloatPlane {
            width,
            height,
            samples,
        });
    }

    Ok(DecodedImage {
        width: geometry.width(),
        height: geometry.height(),
        planes: integer_planes,
        num_colour_channels: num_colour,
        icc_profile: None,
        float_planes: Some(float_planes),
    })
}

pub(crate) const fn unsupported(feature: &'static str, clause: &'static str) -> DecodeError {
    DecodeError::Unsupported { feature, clause }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    #[test]
    fn shifted_ceil_rounds_up() {
        assert_eq!(shifted_ceil(300, 1), 150);
        assert_eq!(shifted_ceil(301, 1), 151);
        assert_eq!(shifted_ceil(8, 3), 1);
        assert_eq!(shifted_ceil(9, 3), 2);
        assert_eq!(shifted_ceil(5, 0), 5);
    }

    #[test]
    fn plane_max_value_saturates_at_thirty_two_bits() {
        let p = Plane {
            width: 1,
            height: 1,
            bits_per_sample: 8,
            samples: vec![0],
        };
        assert_eq!(p.max_value(), 255);
        let p = Plane {
            bits_per_sample: 32,
            ..p
        };
        assert_eq!(p.max_value(), u32::MAX);
    }

    #[test]
    fn interleaving_clamps_out_of_range_samples() {
        let image = DecodedImage {
            width: 2,
            height: 1,
            num_colour_channels: 1,
            planes: vec![Plane {
                width: 2,
                height: 1,
                bits_per_sample: 8,
                samples: vec![-5, 999],
            }],
            icc_profile: None,
            float_planes: None,
        };
        assert_eq!(image.interleaved_colour(), vec![0u16, 255]);
    }

    #[test]
    fn garbage_input_is_an_error_not_a_panic() {
        let mut seed = 0x1234_5678u32;
        for _ in 0..500 {
            let mut bytes = vec![0xFFu8, 0x0A];
            for _ in 0..32 {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                bytes.push((seed >> 15) as u8);
            }
            let _ = decode(&bytes, &Limits::default());
        }
    }
}
