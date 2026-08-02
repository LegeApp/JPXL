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
use jpxl_core::limits::{AllocGuard, Limits};

use crate::container;
use crate::error::{DecodeError, Result};
use crate::frame::{
    Encoding, FrameGeometry, FrameHeader, FrameType, Rect, Toc, read_frame_header, read_toc,
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

/// A fully decoded image.
#[derive(Debug, Clone, PartialEq, Eq)]
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

    if headers.metadata.preview.is_some() {
        return Err(unsupported("preview frames", "18181-1 A.1"));
    }

    // F.1: each frame is byte-aligned by ZeroPadToByte() before it is read.
    reader.zero_pad_to_byte()?;
    let mut cursor = usize::try_from(reader.total_bits_read() / 8)
        .map_err(|_| unsupported("a codestream larger than the address space", "18181-1 A.1"))?;

    let mut decoded: Option<DecodedImage> = None;
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
                limits,
                &mut guard,
            )?);
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

    decoded.ok_or_else(|| unsupported("a codestream with no regular frame", "18181-1 A.1"))
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

/// Rejects every frame construct outside the modular scope of this slice.
fn check_supported(header: &FrameHeader, metadata: &ImageMetadata) -> Result<()> {
    if header.encoding != Encoding::Modular {
        return Err(unsupported("kVarDCT frame decoding", "18181-1 Annex I"));
    }
    if metadata.xyb_encoded {
        return Err(unsupported(
            "xyb_encoded colour reconstruction",
            "18181-1 L.2",
        ));
    }
    if header.do_ycbcr {
        return Err(unsupported("do_YCbCr colour reconstruction", "18181-1 L.3"));
    }
    if header.upsampling != 1 {
        return Err(unsupported("frame upsampling", "18181-1 J.2"));
    }
    if header.flags.patches() {
        return Err(unsupported("patches", "18181-1 K.3"));
    }
    if header.flags.splines() {
        return Err(unsupported("splines", "18181-1 K.4"));
    }
    if header.flags.noise() {
        return Err(unsupported("noise synthesis", "18181-1 K.5"));
    }
    if header.flags.use_lf_frame() {
        return Err(unsupported("kUseLfFrame", "18181-1 G.2.2"));
    }
    if header.have_crop {
        return Err(unsupported("cropped frames", "18181-1 F.2"));
    }
    Ok(())
}

/// Decodes one modular frame's sections into planes.
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
    limits: &Limits,
    guard: &mut AllocGuard,
) -> Result<DecodedImage> {
    let metadata = &headers.metadata;
    check_supported(header, metadata)?;

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
    let (global_tree, mut partial) = {
        let mut owned;
        let reader: &mut BitReader<'_> = if single {
            &mut single_reader
        } else {
            owned = BitReader::new(section_slice(codestream, base, toc, 0)?);
            &mut owned
        };

        // G.1.2 LfChannelDequantization is present for every encoding. Modular
        // mode never uses the weights, but the bits are still there.
        if !reader.read_bool()? {
            for _ in 0..3 {
                jpxl_bitstream::read_f16_as_f32(reader)?;
            }
        }

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
        (global_tree, partial)
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
        options.stream_index = stream_index(1 + num_lf_groups + lf_index)?;
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
                if c.width() <= group_dim && c.height() <= group_dim {
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
                stream_index(1 + 3 * num_lf_groups + 17 + num_groups * u64::from(pass) + group)?;
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
    assemble(
        image.into_channels(),
        metadata,
        geometry,
        num_colour,
        limits,
    )
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

/// H.4.1: property 1 is the stream index, narrowed to the property width.
fn stream_index(value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_| unsupported("a stream index past 2^32", "18181-1 H.4.1"))
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
    })
}

const fn unsupported(feature: &'static str, clause: &'static str) -> DecodeError {
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
