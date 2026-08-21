//! End-user raster-file adapters for the CLI.
//!
//! File codecs deliberately live here, not in `jpxl`, `jpxl-encode`, or
//! `jpxl-decode`: applications using raw pixel buffers should not compile a
//! PNG/JPEG/TIFF stack they never call.

use std::io::Cursor;
use std::path::Path;

use image::codecs::pnm::{PnmEncoder, PnmSubtype, SampleEncoding};
use image::{DynamicImage, ImageBuffer, ImageFormat, Luma, LumaA, Rgb, Rgba};
use jpxl_bitstream::BitReader;
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_decode::headers::enums::ExtraChannelType;

/// An RGB background in 8-bit sRGB, used only when transparency must be flattened.
pub type Background = [u8; 3];

/// Raster formats the CLI can write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RasterFormat {
    Png,
    Jpeg,
    WebP,
    Tiff,
    Bmp,
    Gif,
    Ico,
    Tga,
    Qoi,
    Pgm,
    Ppm,
    Pnm,
}

impl RasterFormat {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.trim_start_matches('.').to_ascii_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "jpg" | "jpeg" | "jpe" => Ok(Self::Jpeg),
            "webp" => Ok(Self::WebP),
            "tif" | "tiff" => Ok(Self::Tiff),
            "bmp" => Ok(Self::Bmp),
            "gif" => Ok(Self::Gif),
            "ico" => Ok(Self::Ico),
            "tga" => Ok(Self::Tga),
            "qoi" => Ok(Self::Qoi),
            "pgm" => Ok(Self::Pgm),
            "ppm" => Ok(Self::Ppm),
            "pnm" => Ok(Self::Pnm),
            _ => Err(format!(
                "unsupported output format `{name}`; choose png, jpg, webp, tiff, bmp, gif, ico, tga, qoi, pgm, or ppm"
            )),
        }
    }

    pub fn from_path(path: &str) -> Result<Self, String> {
        let extension = Path::new(path)
            .extension()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                format!("cannot infer the output format from `{path}`; pass `--format <name>`")
            })?;
        Self::parse(extension)
    }

    const fn image_format(self) -> ImageFormat {
        match self {
            Self::Png => ImageFormat::Png,
            Self::Jpeg => ImageFormat::Jpeg,
            Self::WebP => ImageFormat::WebP,
            Self::Tiff => ImageFormat::Tiff,
            Self::Bmp => ImageFormat::Bmp,
            Self::Gif => ImageFormat::Gif,
            Self::Ico => ImageFormat::Ico,
            Self::Tga => ImageFormat::Tga,
            Self::Qoi => ImageFormat::Qoi,
            Self::Pgm | Self::Ppm | Self::Pnm => ImageFormat::Pnm,
        }
    }

    const fn supports_16_bit(self) -> bool {
        matches!(
            self,
            Self::Png | Self::Tiff | Self::Pgm | Self::Ppm | Self::Pnm
        )
    }

    const fn supports_alpha(self) -> bool {
        !matches!(self, Self::Jpeg | Self::Pgm | Self::Ppm | Self::Pnm)
    }

    const fn pnm_subtype(self) -> Option<PnmSubtype> {
        match self {
            Self::Pgm => Some(PnmSubtype::Graymap(SampleEncoding::Binary)),
            Self::Ppm => Some(PnmSubtype::Pixmap(SampleEncoding::Binary)),
            _ => None,
        }
    }
}

/// Parse `#RGB`, `#RRGGBB`, `RGB`, or `RRGGBB`.
pub fn parse_background(value: &str) -> Result<Background, String> {
    let value = value.trim_start_matches('#');
    let expanded;
    let hex = match value.len() {
        3 => {
            expanded = value.chars().flat_map(|c| [c, c]).collect::<String>();
            expanded.as_str()
        }
        6 => value,
        _ => return Err("background must be #RGB or #RRGGBB".to_owned()),
    };
    let mut out = [0u8; 3];
    for (index, channel) in out.iter_mut().enumerate() {
        let start = index * 2;
        *channel = u8::from_str_radix(hex.get(start..start + 2).unwrap_or_default(), 16)
            .map_err(|_| "background must contain hexadecimal digits".to_owned())?;
    }
    Ok(out)
}

/// Decode any supported raster file into the codec's validated pixel type.
pub fn decode_input(
    bytes: &[u8],
    background: Option<Background>,
) -> Result<jpxl_encode::Image, String> {
    let dynamic =
        image::load_from_memory(bytes).map_err(|error| format!("cannot decode image: {error}"))?;
    let (width, height) = (dynamic.width(), dynamic.height());

    match dynamic {
        DynamicImage::ImageLuma8(buffer) => from_u8(width, height, 1, buffer.into_raw()),
        DynamicImage::ImageRgb8(buffer) => from_u8(width, height, 3, buffer.into_raw()),
        DynamicImage::ImageLuma16(buffer) => from_u16(width, height, 1, 16, buffer.into_raw()),
        DynamicImage::ImageRgb16(buffer) => from_u16(width, height, 3, 16, buffer.into_raw()),
        DynamicImage::ImageLumaA8(buffer) => {
            let raw = buffer.into_raw();
            if alpha_is_opaque(&raw, 2, u8::MAX) {
                let gray = raw
                    .chunks_exact(2)
                    .filter_map(|pixel| pixel.first().copied())
                    .collect();
                from_u8(width, height, 1, gray)
            } else {
                let bg = background.ok_or_else(transparency_message)?;
                let mut rgb = Vec::with_capacity(raw.len() / 2 * 3);
                for pixel in raw.chunks_exact(2) {
                    let [gray, alpha] = pixel else { continue };
                    rgb.extend_from_slice(&[
                        composite8(*gray, bg[0], *alpha),
                        composite8(*gray, bg[1], *alpha),
                        composite8(*gray, bg[2], *alpha),
                    ]);
                }
                from_u8(width, height, 3, rgb)
            }
        }
        DynamicImage::ImageRgba8(buffer) => {
            let raw = buffer.into_raw();
            let rgb = rgba8_to_rgb(&raw, background)?;
            from_u8(width, height, 3, rgb)
        }
        DynamicImage::ImageLumaA16(buffer) => {
            let raw = buffer.into_raw();
            if alpha_is_opaque(&raw, 2, u16::MAX) {
                let gray = raw
                    .chunks_exact(2)
                    .filter_map(|pixel| pixel.first().copied())
                    .collect();
                from_u16(width, height, 1, 16, gray)
            } else {
                let bg = background.ok_or_else(transparency_message)?;
                let bg = bg.map(|channel| u16::from(channel) * 257);
                let mut rgb = Vec::with_capacity(raw.len() / 2 * 3);
                for pixel in raw.chunks_exact(2) {
                    let [gray, alpha] = pixel else { continue };
                    rgb.extend_from_slice(&[
                        composite16(*gray, bg[0], *alpha),
                        composite16(*gray, bg[1], *alpha),
                        composite16(*gray, bg[2], *alpha),
                    ]);
                }
                from_u16(width, height, 3, 16, rgb)
            }
        }
        DynamicImage::ImageRgba16(buffer) => {
            let raw = buffer.into_raw();
            let rgb = rgba16_to_rgb(&raw, background)?;
            from_u16(width, height, 3, 16, rgb)
        }
        other => {
            // HDR/float formats are normalised by image's documented dynamic
            // conversion, then retained at 16-bit precision for JPXL.
            let rgb = other.into_rgb16().into_raw();
            from_u16(width, height, 3, 16, rgb)
        }
    }
}

fn transparency_message() -> String {
    "the JPEG XL encoder does not yet write alpha channels; pass `--background #RRGGBB` to flatten transparency explicitly".to_owned()
}

fn from_u8(
    width: u32,
    height: u32,
    channels: usize,
    raw: Vec<u8>,
) -> Result<jpxl_encode::Image, String> {
    let samples: Vec<u16> = raw.into_iter().map(u16::from).collect();
    from_u16(width, height, channels, 8, samples)
}

fn from_u16(
    width: u32,
    height: u32,
    channels: usize,
    bits_per_sample: u32,
    samples: Vec<u16>,
) -> Result<jpxl_encode::Image, String> {
    jpxl_encode::Image::from_interleaved(width, height, channels, bits_per_sample, &samples)
        .map_err(|error| error.to_string())
}

fn alpha_is_opaque<T: Copy + PartialEq>(raw: &[T], channels: usize, max: T) -> bool {
    raw.chunks_exact(channels)
        .all(|pixel| pixel.last().copied() == Some(max))
}

fn rgba8_to_rgb(raw: &[u8], background: Option<Background>) -> Result<Vec<u8>, String> {
    let opaque = alpha_is_opaque(raw, 4, u8::MAX);
    let bg = if opaque {
        [0; 3]
    } else {
        background.ok_or_else(transparency_message)?
    };
    let mut rgb = Vec::with_capacity(raw.len() / 4 * 3);
    for pixel in raw.chunks_exact(4) {
        let [red, green, blue, alpha] = pixel else {
            continue;
        };
        for (source, background) in [*red, *green, *blue].into_iter().zip(bg) {
            rgb.push(if opaque {
                source
            } else {
                composite8(source, background, *alpha)
            });
        }
    }
    Ok(rgb)
}

fn rgba16_to_rgb(raw: &[u16], background: Option<Background>) -> Result<Vec<u16>, String> {
    let opaque = alpha_is_opaque(raw, 4, u16::MAX);
    let bg8 = if opaque {
        [0; 3]
    } else {
        background.ok_or_else(transparency_message)?
    };
    let bg = bg8.map(|channel| u16::from(channel) * 257);
    let mut rgb = Vec::with_capacity(raw.len() / 4 * 3);
    for pixel in raw.chunks_exact(4) {
        let [red, green, blue, alpha] = pixel else {
            continue;
        };
        for (source, background) in [*red, *green, *blue].into_iter().zip(bg) {
            rgb.push(if opaque {
                source
            } else {
                composite16(source, background, *alpha)
            });
        }
    }
    Ok(rgb)
}

fn composite8(source: u8, background: u8, alpha: u8) -> u8 {
    let alpha = u32::from(alpha);
    let value = u32::from(source) * alpha
        + u32::from(background) * (u32::from(u8::MAX) - alpha)
        + u32::from(u8::MAX) / 2;
    u8::try_from(value / u32::from(u8::MAX)).unwrap_or(u8::MAX)
}

fn composite16(source: u16, background: u16, alpha: u16) -> u16 {
    let alpha = u64::from(alpha);
    let value = u64::from(source) * alpha
        + u64::from(background) * (u64::from(u16::MAX) - alpha)
        + u64::from(u16::MAX) / 2;
    u16::try_from(value / u64::from(u16::MAX)).unwrap_or(u16::MAX)
}

#[derive(Debug, Clone, Copy)]
struct AlphaInfo {
    plane: usize,
    associated: bool,
}

fn alpha_info(encoded: &[u8], image: &jpxl_decode::DecodedImage) -> Option<AlphaInfo> {
    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    let codestream = if jpxl_decode::container::is_container(encoded) {
        jpxl_decode::container::extract_codestream(encoded, &mut guard).ok()?
    } else {
        encoded.to_vec()
    };
    let mut reader = BitReader::new(&codestream);
    let headers = jpxl_decode::decode_image_headers(&mut reader, &limits).ok()?;
    let (extra_index, info) = headers
        .metadata
        .ec_info
        .iter()
        .enumerate()
        .find(|(_, info)| info.channel_type == ExtraChannelType::KAlpha)?;
    let plane = image.num_colour_channels.checked_add(extra_index)?;
    image.planes.get(plane)?;
    Some(AlphaInfo {
        plane,
        associated: info.alpha_associated,
    })
}

/// Convert decoded JPEG XL pixels to a requested raster format in memory.
pub fn encode_output(
    encoded: &[u8],
    image: &jpxl_decode::DecodedImage,
    format: RasterFormat,
    background: Option<Background>,
) -> Result<Vec<u8>, String> {
    let alpha = alpha_info(encoded, image);
    if alpha.is_some() && !format.supports_alpha() && background.is_none() {
        return Err("the output format cannot store alpha; pass `--background #RRGGBB` to flatten transparency".to_owned());
    }

    let want_16 = format.supports_16_bit()
        && image
            .planes
            .iter()
            .take(image.num_colour_channels + usize::from(alpha.is_some()))
            .any(|plane| plane.bits_per_sample > 8);
    let dynamic = if want_16 {
        dynamic16(image, alpha, background)?
    } else {
        dynamic8(image, alpha, background)?
    };

    let mut output = Cursor::new(Vec::new());
    if let Some(subtype) = format.pnm_subtype() {
        // `DynamicImage::write_to(ImageFormat::Pnm)` deliberately chooses a
        // generic PAM (`P7`) header.  An explicit `.pgm`/`.ppm` request is a
        // stronger format contract, and the compare command consumes binary
        // PPM (`P6`), so select the matching Netpbm subtype explicitly.
        dynamic
            .write_with_encoder(PnmEncoder::new(&mut output).with_subtype(subtype))
            .map_err(|error| format!("cannot encode output image: {error}"))?;
    } else {
        dynamic
            .write_to(&mut output, format.image_format())
            .map_err(|error| format!("cannot encode output image: {error}"))?;
    }
    Ok(output.into_inner())
}

fn dynamic8(
    image: &jpxl_decode::DecodedImage,
    alpha: Option<AlphaInfo>,
    background: Option<Background>,
) -> Result<DynamicImage, String> {
    let channels = image.num_colour_channels;
    let keep_alpha = alpha.is_some() && background.is_none();
    let flatten_gray_to_rgb = channels == 1 && alpha.is_some() && background.is_some();
    let output_channels = if flatten_gray_to_rgb { 3 } else { channels };
    let stride = output_channels + usize::from(keep_alpha);
    let pixels = pixel_count(image)?;
    let alpha_plane = alpha.and_then(|info| image.planes.get(info.plane));
    let mut raw = Vec::with_capacity(pixels.saturating_mul(stride));
    for index in 0..pixels {
        let alpha_value = alpha_plane.map(|plane| scaled_sample(plane, index, 255));
        if flatten_gray_to_rgb {
            let mut gray = image
                .planes
                .first()
                .map_or(0, |plane| scaled_sample(plane, index, 255));
            if alpha.is_some_and(|info| info.associated) {
                gray = unassociate(gray, alpha_value.unwrap_or(255), 255);
            }
            let bg = background.unwrap_or([0; 3]);
            let alpha_value = u8::try_from(alpha_value.unwrap_or(255)).unwrap_or(u8::MAX);
            for background in bg {
                raw.push(composite8(
                    u8::try_from(gray).unwrap_or(u8::MAX),
                    background,
                    alpha_value,
                ));
            }
            continue;
        }
        for (channel, plane) in image.planes.iter().take(channels).enumerate() {
            let mut value = scaled_sample(plane, index, 255);
            if let (Some(info), Some(a)) = (alpha, alpha_value) {
                if info.associated {
                    value = unassociate(value, a, 255);
                }
                if let Some(bg) = background {
                    value = u32::from(composite8(
                        u8::try_from(value).unwrap_or(u8::MAX),
                        bg.get(channel).copied().unwrap_or(0),
                        u8::try_from(a).unwrap_or(u8::MAX),
                    ));
                }
            }
            raw.push(u8::try_from(value).unwrap_or(u8::MAX));
        }
        if keep_alpha {
            raw.push(u8::try_from(alpha_value.unwrap_or(255)).unwrap_or(u8::MAX));
        }
    }
    dynamic_from_u8(image.width, image.height, output_channels, keep_alpha, raw)
}

fn dynamic16(
    image: &jpxl_decode::DecodedImage,
    alpha: Option<AlphaInfo>,
    background: Option<Background>,
) -> Result<DynamicImage, String> {
    let channels = image.num_colour_channels;
    let keep_alpha = alpha.is_some() && background.is_none();
    let flatten_gray_to_rgb = channels == 1 && alpha.is_some() && background.is_some();
    let output_channels = if flatten_gray_to_rgb { 3 } else { channels };
    let stride = output_channels + usize::from(keep_alpha);
    let pixels = pixel_count(image)?;
    let alpha_plane = alpha.and_then(|info| image.planes.get(info.plane));
    let mut raw = Vec::with_capacity(pixels.saturating_mul(stride));
    for index in 0..pixels {
        let alpha_value = alpha_plane.map(|plane| scaled_sample(plane, index, 65_535));
        if flatten_gray_to_rgb {
            let mut gray = image
                .planes
                .first()
                .map_or(0, |plane| scaled_sample(plane, index, 65_535));
            if alpha.is_some_and(|info| info.associated) {
                gray = unassociate(gray, alpha_value.unwrap_or(65_535), 65_535);
            }
            let bg = background.unwrap_or([0; 3]);
            let alpha_value = u16::try_from(alpha_value.unwrap_or(65_535)).unwrap_or(u16::MAX);
            for background in bg {
                raw.push(composite16(
                    u16::try_from(gray).unwrap_or(u16::MAX),
                    u16::from(background) * 257,
                    alpha_value,
                ));
            }
            continue;
        }
        for (channel, plane) in image.planes.iter().take(channels).enumerate() {
            let mut value = scaled_sample(plane, index, 65_535);
            if let (Some(info), Some(a)) = (alpha, alpha_value) {
                if info.associated {
                    value = unassociate(value, a, 65_535);
                }
                if let Some(bg) = background {
                    value = u32::from(composite16(
                        u16::try_from(value).unwrap_or(u16::MAX),
                        u16::from(bg.get(channel).copied().unwrap_or(0)) * 257,
                        u16::try_from(a).unwrap_or(u16::MAX),
                    ));
                }
            }
            raw.push(u16::try_from(value).unwrap_or(u16::MAX));
        }
        if keep_alpha {
            raw.push(u16::try_from(alpha_value.unwrap_or(65_535)).unwrap_or(u16::MAX));
        }
    }
    dynamic_from_u16(image.width, image.height, output_channels, keep_alpha, raw)
}

fn pixel_count(image: &jpxl_decode::DecodedImage) -> Result<usize, String> {
    usize::try_from(u64::from(image.width) * u64::from(image.height))
        .map_err(|_| "decoded image is too large for this host".to_owned())
}

fn scaled_sample(plane: &jpxl_decode::Plane, index: usize, target_max: u32) -> u32 {
    let source_max = plane.max_value().max(1);
    let sample = plane.samples.get(index).copied().unwrap_or(0);
    let sample = u64::try_from(i64::from(sample).clamp(0, i64::from(source_max))).unwrap_or(0);
    let scaled = sample * u64::from(target_max) + u64::from(source_max) / 2;
    u32::try_from(scaled / u64::from(source_max)).unwrap_or(target_max)
}

fn unassociate(value: u32, alpha: u32, max: u32) -> u32 {
    if alpha == 0 {
        return 0;
    }
    let expanded = (u64::from(value) * u64::from(max) + u64::from(alpha) / 2) / u64::from(alpha);
    u32::try_from(expanded.min(u64::from(max))).unwrap_or(max)
}

fn dynamic_from_u8(
    width: u32,
    height: u32,
    colour_channels: usize,
    alpha: bool,
    raw: Vec<u8>,
) -> Result<DynamicImage, String> {
    match (colour_channels, alpha) {
        (1, false) => {
            ImageBuffer::<Luma<u8>, _>::from_raw(width, height, raw).map(DynamicImage::ImageLuma8)
        }
        (1, true) => {
            ImageBuffer::<LumaA<u8>, _>::from_raw(width, height, raw).map(DynamicImage::ImageLumaA8)
        }
        (3, false) => {
            ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, raw).map(DynamicImage::ImageRgb8)
        }
        (3, true) => {
            ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, raw).map(DynamicImage::ImageRgba8)
        }
        _ => None,
    }
    .ok_or_else(|| "decoded image has an unsupported colour-channel layout".to_owned())
}

fn dynamic_from_u16(
    width: u32,
    height: u32,
    colour_channels: usize,
    alpha: bool,
    raw: Vec<u16>,
) -> Result<DynamicImage, String> {
    match (colour_channels, alpha) {
        (1, false) => {
            ImageBuffer::<Luma<u16>, _>::from_raw(width, height, raw).map(DynamicImage::ImageLuma16)
        }
        (1, true) => ImageBuffer::<LumaA<u16>, _>::from_raw(width, height, raw)
            .map(DynamicImage::ImageLumaA16),
        (3, false) => {
            ImageBuffer::<Rgb<u16>, _>::from_raw(width, height, raw).map(DynamicImage::ImageRgb16)
        }
        (3, true) => {
            ImageBuffer::<Rgba<u16>, _>::from_raw(width, height, raw).map(DynamicImage::ImageRgba16)
        }
        _ => None,
    }
    .ok_or_else(|| "decoded image has an unsupported colour-channel layout".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn background_spellings_parse() {
        assert_eq!(parse_background("#1aF"), Ok([0x11, 0xaa, 0xff]));
        assert_eq!(parse_background("102030"), Ok([0x10, 0x20, 0x30]));
        assert!(parse_background("white").is_err());
    }

    fn decoded_image(planes: Vec<Vec<i32>>, bits_per_sample: u32) -> jpxl_decode::DecodedImage {
        let num_colour_channels = planes.len();
        jpxl_decode::DecodedImage {
            width: 2,
            height: 1,
            planes: planes
                .into_iter()
                .map(|samples| jpxl_decode::Plane {
                    width: 2,
                    height: 1,
                    bits_per_sample,
                    samples,
                })
                .collect(),
            num_colour_channels,
            icc_profile: None,
            float_planes: None,
        }
    }

    #[test]
    fn explicit_ppm_and_pgm_requests_write_the_requested_binary_subtype() {
        assert_eq!(RasterFormat::parse("ppm"), Ok(RasterFormat::Ppm));
        assert_eq!(RasterFormat::parse("pgm"), Ok(RasterFormat::Pgm));
        assert_eq!(RasterFormat::parse("pnm"), Ok(RasterFormat::Pnm));

        let rgb = decoded_image(vec![vec![1, 2], vec![3, 4], vec![5, 6]], 8);
        let ppm = encode_output(&[], &rgb, RasterFormat::Ppm, None).expect("binary PPM");
        assert!(ppm.starts_with(b"P6\n"));

        let gray = decoded_image(vec![vec![7, 8]], 8);
        let pgm = encode_output(&[], &gray, RasterFormat::Pgm, None).expect("binary PGM");
        assert!(pgm.starts_with(b"P5\n"));
    }

    #[test]
    fn explicit_netpbm_subtype_rejects_the_wrong_channel_layout() {
        let rgb = decoded_image(vec![vec![1, 2], vec![3, 4], vec![5, 6]], 8);
        assert!(encode_output(&[], &rgb, RasterFormat::Pgm, None).is_err());

        let gray = decoded_image(vec![vec![7, 8]], 8);
        assert!(encode_output(&[], &gray, RasterFormat::Ppm, None).is_err());
    }

    #[test]
    fn opaque_png_is_a_normal_rgb_input() {
        let source = DynamicImage::ImageRgba8(
            ImageBuffer::from_raw(1, 1, vec![1, 2, 3, 255]).expect("shape"),
        );
        let mut bytes = Cursor::new(Vec::new());
        source.write_to(&mut bytes, ImageFormat::Png).expect("png");
        let image = decode_input(bytes.get_ref(), None).expect("decode PNG");
        assert_eq!(
            (image.width(), image.height(), image.num_channels()),
            (1, 1, 3)
        );
        assert_eq!(image.planes().first(), Some(&vec![1]));
        assert_eq!(image.planes().get(1), Some(&vec![2]));
        assert_eq!(image.planes().get(2), Some(&vec![3]));
    }

    #[test]
    fn transparent_input_requires_an_explicit_background() {
        let source = DynamicImage::ImageRgba8(
            ImageBuffer::from_raw(1, 1, vec![255, 0, 0, 128]).expect("shape"),
        );
        let mut bytes = Cursor::new(Vec::new());
        source.write_to(&mut bytes, ImageFormat::Png).expect("png");
        assert!(decode_input(bytes.get_ref(), None).is_err());
        let image = decode_input(bytes.get_ref(), Some([0, 0, 0])).expect("flatten");
        assert_eq!(image.planes().first(), Some(&vec![128]));
    }
}
