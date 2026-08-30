//! Frame-level content classification for text/UI-aware routing.
//!
//! Screenshots, terminal windows, diagrams and line art occupy a corner of
//! image space where the VarDCT quality path is weakest (measured 1.4-4.7x
//! the bytes of a matched-quality competitor on a 121-screenshot corpus,
//! 2026-08-30) and where palette-oriented Modular candidates can compete.
//! This module produces a per-frame [`ContentHint`] naming that corner so the
//! facade can *add* a competing candidate on classifier-positive frames. The
//! hint must never suppress or alter the existing VarDCT path: a false
//! positive costs bounded wall time, never a worse stream.
//!
//! The discriminating signal, calibrated on the operator's real screenshot
//! corpus against the locked photo corpus, is the unique-colour census: every
//! screenshot with a material byte gap had at most ~17k unique colours and at
//! most 0.9 colours per hundred pixels, while photographs saturate the census
//! within their first rows. The census therefore runs first and alone decides
//! [`ContentClass::PhotoLike`]; the Sobel cell statistics (ported from the
//! operator's bpg-rs `still265::preanalysis`, same thresholds) run only on
//! census-sparse frames, where their cost is a few percent of an encode that
//! the routing is about to spend extra candidates on anyway.
//!
//! Everything here is deterministic and read-only: integer arithmetic over
//! the source samples, no floating point in any decision, identical output
//! for identical pixels on every host and worker count.

use std::time::Instant;

/// The frame-level routing class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentClass {
    /// Colour-dense content: the census saturated (photographs, gradients
    /// with noise, video frames). The routed candidate competition is
    /// skipped; the existing paths run unchanged.
    PhotoLike,
    /// Colour-sparse content with the screenshot/line-art signature: worth
    /// pricing a palette-oriented Modular candidate alongside VarDCT.
    TextUiLineArt,
    /// Neither signature: colour-sparse but too colourful for the routed
    /// ladder's calibrated win region. No extra candidate.
    Unknown,
}

impl ContentClass {
    /// The class name as a stable lowercase token (for traces and logs).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PhotoLike => "photo",
            Self::TextUiLineArt => "text_ui",
            Self::Unknown => "unknown",
        }
    }
}

/// One frame's classification and the evidence it rests on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContentHint {
    /// The routing class.
    pub class: ContentClass,
    /// Unique RGB colours seen before the census cap (exact when
    /// `census_capped` is false).
    pub unique_colours: u32,
    /// The census stopped at [`CENSUS_CAP`]; `unique_colours` is a floor.
    pub census_capped: bool,
    /// Share of 32x32 cells classified `TextLike` (bpg-rs rule; diagnostic).
    pub textlike_share: f32,
    /// Share of cells classified `TextLike` or `DirectionalEdge`.
    pub structured_share: f32,
    /// Share of cells with dense crisp edges and no grain
    /// (`edge_density >= 25%` and `noise < 7`): the practical screenshot
    /// structure signal on anti-aliased text, where the orientation-entropy
    /// half of the bpg-rs `TextLike` rule does not fire.
    pub crisp_share: f32,
    /// Wall time the classifier itself spent, in microseconds.
    pub classifier_us: u64,
}

impl ContentHint {
    /// A one-line JSON object for the quality trace.
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            "{{\"class\":\"{}\",\"unique_colours\":{},\"census_capped\":{},\
             \"textlike_share\":{:.4},\"structured_share\":{:.4},\"crisp_share\":{:.4},\
             \"classifier_us\":{}}}",
            self.class.as_str(),
            self.unique_colours,
            self.census_capped,
            self.textlike_share,
            self.structured_share,
            self.crisp_share,
            self.classifier_us,
        )
    }
}

/// Census cap: counting stops here and the frame is `PhotoLike` outright.
pub const CENSUS_CAP: u32 = 1 << 17;

/// A routed frame has at most this many unique colours...
pub const ROUTE_MAX_COLOURS: u32 = 65_536;

/// ...and at most one unique colour per this many pixels (the corpus's
/// gap images peaked at one per ~119 pixels; photographs sit far denser).
pub const ROUTE_MIN_PIXELS_PER_COLOUR: u64 = 50;

// --- 32x32-cell classification, ported from bpg-rs still265::preanalysis
// (operator-owned code). Same thresholds, 8-bit units / q8 shares. ---

const CELL: usize = 32;
const FLAT_VAR: u32 = 12;
const FLAT_EDGE_Q8: u16 = 13;
const GRAD_EDGE_Q8: u16 = 26;
const EDGE_DENSE_Q8: u16 = 38;
const TEXT_EDGE_Q8: u16 = 64;
const DIR_DOMINANT_Q8: u16 = 128;
const AXIS_ALIGNED_Q8: u16 = 160;
const LOW_ENTROPY_Q8: u16 = 110;
const HIGH_ENTROPY_Q8: u16 = 150;
const TEXTURE_VAR: u32 = 64;
const NOISE_HI: u16 = 7;

const EDGE_GRAD: i32 = 48;
const WEAK_GRAD: i32 = 24;

/// Coarse structural class of one 32x32 luma cell (bpg-rs semantics, minus
/// the chroma-critical arm, which the routing aggregates never read).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellClass {
    Flat,
    Gradient,
    Noisy,
    Texture,
    DirectionalEdge,
    TextLike,
}

/// Classifies an 8-bit interleaved sRGB frame.
///
/// Stage 1 walks the samples once, counting unique colours into a fixed
/// 2 MiB bitset, and stops the moment the count reaches [`CENSUS_CAP`] —
/// on photographic content that is within the first rows, so the common
/// (unrouted) case pays almost nothing. Stage 2, the cell pass, runs only
/// when the census stayed under the routing bound.
#[must_use]
pub fn classify_srgb8(width: u32, height: u32, rgb: &[u8]) -> ContentHint {
    classify_impl(width, height, |i| match rgb.get(3 * i..3 * i + 3) {
        Some([r, g, b]) => [*r, *g, *b],
        _ => [0, 0, 0],
    })
}

/// [`classify_srgb8`] for interleaved samples wider than a byte: each sample
/// is reduced to its top eight bits before the census and the cell pass.
/// An 8-bit source widened to `u16` (the facade's common case) reduces to
/// exactly its original bytes; deeper sources lose only precision the
/// structure statistics never resolved anyway.
#[must_use]
pub fn classify_srgb16(width: u32, height: u32, rgb: &[u16], bits_per_sample: u32) -> ContentHint {
    let shift = bits_per_sample.saturating_sub(8);
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the shift and min bring every sample into 0..=255"
    )]
    let reduce = move |s: u16| (s >> shift).min(255) as u8;
    classify_impl(width, height, move |i| match rgb.get(3 * i..3 * i + 3) {
        Some([r, g, b]) => [reduce(*r), reduce(*g), reduce(*b)],
        _ => [0, 0, 0],
    })
}

fn classify_impl(width: u32, height: u32, pixel: impl Fn(usize) -> [u8; 3]) -> ContentHint {
    let start = Instant::now();
    let pixels = u64::from(width) * u64::from(height);
    let (unique_colours, census_capped) =
        colour_census(usize::try_from(pixels).unwrap_or(usize::MAX), &pixel);

    let sparse = !census_capped
        && unique_colours <= ROUTE_MAX_COLOURS
        && u64::from(unique_colours).saturating_mul(ROUTE_MIN_PIXELS_PER_COLOUR) <= pixels;

    let mut textlike_share = 0.0f32;
    let mut structured_share = 0.0f32;
    let mut crisp_share = 0.0f32;
    let class = if census_capped {
        ContentClass::PhotoLike
    } else if sparse {
        let stats = cell_stats(width as usize, height as usize, &pixel);
        textlike_share = stats.textlike_share;
        structured_share = stats.structured_share;
        crisp_share = stats.crisp_share;
        ContentClass::TextUiLineArt
    } else {
        ContentClass::Unknown
    };

    ContentHint {
        class,
        unique_colours,
        census_capped,
        textlike_share,
        structured_share,
        crisp_share,
        classifier_us: u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX),
    }
}

/// Counts unique RGB triples into a 2^24-bit set, stopping at [`CENSUS_CAP`].
fn colour_census(pixels: usize, pixel: &impl Fn(usize) -> [u8; 3]) -> (u32, bool) {
    let mut seen = vec![0u64; (1usize << 24) / 64];
    let mut count = 0u32;
    for i in 0..pixels {
        let px = pixel(i);
        let key = (usize::from(px[0]) << 16) | (usize::from(px[1]) << 8) | usize::from(px[2]);
        let word = key >> 6;
        let bit = 1u64 << (key & 63);
        // The index is 24 bits by construction; `get_mut` keeps the checker
        // happy without a panic path.
        if let Some(slot) = seen.get_mut(word)
            && *slot & bit == 0
        {
            *slot |= bit;
            count += 1;
            if count >= CENSUS_CAP {
                return (count, true);
            }
        }
    }
    (count, false)
}

struct CellStats {
    textlike_share: f32,
    structured_share: f32,
    crisp_share: f32,
}

/// The bpg-rs cell rule over the extracted features.
fn classify_cell(
    variance: u32,
    edge_density_q8: u16,
    orient_entropy_q8: u16,
    dir_dominance_q8: u16,
    axis_aligned_q8: u16,
    noise: u16,
) -> CellClass {
    if edge_density_q8 >= TEXT_EDGE_Q8
        && orient_entropy_q8 < LOW_ENTROPY_Q8
        && axis_aligned_q8 >= AXIS_ALIGNED_Q8
        && noise < NOISE_HI
    {
        return CellClass::TextLike;
    }
    if edge_density_q8 >= EDGE_DENSE_Q8 && dir_dominance_q8 >= DIR_DOMINANT_Q8 {
        return CellClass::DirectionalEdge;
    }
    if noise >= NOISE_HI && variance < TEXTURE_VAR {
        return CellClass::Noisy;
    }
    if variance >= TEXTURE_VAR && orient_entropy_q8 >= HIGH_ENTROPY_Q8 {
        return CellClass::Texture;
    }
    if variance < FLAT_VAR && edge_density_q8 < FLAT_EDGE_Q8 {
        return CellClass::Flat;
    }
    if edge_density_q8 < GRAD_EDGE_Q8 {
        return CellClass::Gradient;
    }
    CellClass::Texture
}

/// Shannon entropy of the 4-bin direction histogram, in q8 of 2 bits.
///
/// The single floating-point corner of the pass; its result is rounded once
/// and every operation is an IEEE basic operation on exactly representable
/// integer ratios, so it is host-independent.
fn entropy4_q8(dir: &[i64; 4], total: i64) -> u16 {
    if total <= 0 {
        return 0;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "cell pixel counts are far inside f64's exact integer range"
    )]
    let t = total as f64;
    let mut h = 0.0f64;
    for &c in dir {
        if c > 0 {
            #[allow(
                clippy::cast_precision_loss,
                reason = "cell pixel counts are far inside f64's exact integer range"
            )]
            let p = c as f64 / t;
            h -= p * p.log2();
        }
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is clamped into 0..=256 before the cast"
    )]
    {
        ((h / 2.0) * 256.0).round().clamp(0.0, 256.0) as u16
    }
}

/// One pass of 32x32-cell luma structure statistics.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "cell sums fit i64 and the q8 shares are bounded by construction"
)]
fn cell_stats(width: usize, height: usize, pixel: &impl Fn(usize) -> [u8; 3]) -> CellStats {
    // Integer BT.601 luma, 8-bit; enough for structure statistics.
    let mut luma = vec![0u8; width * height];
    for (i, l) in luma.iter_mut().enumerate() {
        let [r, g, b] = pixel(i);
        *l = ((77 * i32::from(r) + 150 * i32::from(g) + 29 * i32::from(b) + 128) >> 8) as u8;
    }
    let lum = |x: i64, y: i64| -> i32 {
        let sx = x.clamp(0, width as i64 - 1) as usize;
        let sy = y.clamp(0, height as i64 - 1) as usize;
        luma.get(sy * width + sx).copied().map_or(0, i32::from)
    };

    let cells_x = width.div_ceil(CELL).max(1);
    let cells_y = height.div_ceil(CELL).max(1);
    let mut textlike = 0u32;
    let mut structured = 0u32;
    let mut crisp = 0u32;
    for cy in 0..cells_y {
        for cx in 0..cells_x {
            let x0 = (cx * CELL) as i64;
            let y0 = (cy * CELL) as i64;
            let x1 = ((cx + 1) * CELL).min(width) as i64;
            let y1 = ((cy + 1) * CELL).min(height) as i64;
            let n = ((x1 - x0) * (y1 - y0)).max(1);

            let mut sum = 0i64;
            let mut sum_sq = 0i64;
            let mut edge_count = 0i64;
            let mut noise_sum = 0i64;
            let mut weak_count = 0i64;
            let mut dir = [0i64; 4];
            for y in y0..y1 {
                for x in x0..x1 {
                    let c = lum(x, y);
                    sum += i64::from(c);
                    sum_sq += i64::from(c * c);
                    let tl = lum(x - 1, y - 1);
                    let tc = lum(x, y - 1);
                    let tr = lum(x + 1, y - 1);
                    let ml = lum(x - 1, y);
                    let mr = lum(x + 1, y);
                    let bl = lum(x - 1, y + 1);
                    let bc = lum(x, y + 1);
                    let br = lum(x + 1, y + 1);
                    let gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
                    let gy = (bl + 2 * bc + br) - (tl + 2 * tc + tr);
                    let grad = gx.abs() + gy.abs();
                    if grad > EDGE_GRAD {
                        edge_count += 1;
                        let ax = gx.abs();
                        let ay = gy.abs();
                        let bin = if ax >= 2 * ay {
                            0
                        } else if ay >= 2 * ax {
                            1
                        } else if (gx > 0) == (gy > 0) {
                            2
                        } else {
                            3
                        };
                        if let Some(d) = dir.get_mut(bin) {
                            *d += 1;
                        }
                    }
                    if grad < WEAK_GRAD {
                        let box_mean = (tl + tc + tr + ml + c + mr + bl + bc + br) / 9;
                        noise_sum += i64::from((c - box_mean).abs());
                        weak_count += 1;
                    }
                }
            }
            let mean = sum / n;
            let variance = ((sum_sq / n) - mean * mean).max(0) as u32;
            let edge_density_q8 = ((edge_count * 256) / n) as u16;
            let noise = if weak_count > 0 {
                (noise_sum / weak_count) as u16
            } else {
                0
            };
            let edge_total = dir.iter().sum::<i64>().max(1);
            let dir_max = dir.iter().copied().max().unwrap_or(0);
            let dir_dominance_q8 = ((dir_max * 256) / edge_total) as u16;
            let axis_aligned_q8 = (((dir[0] + dir[1]) * 256) / edge_total) as u16;
            let orient_entropy_q8 = entropy4_q8(&dir, edge_total);

            let class = classify_cell(
                variance,
                edge_density_q8,
                orient_entropy_q8,
                dir_dominance_q8,
                axis_aligned_q8,
                noise,
            );
            if class == CellClass::TextLike {
                textlike += 1;
            }
            if matches!(class, CellClass::TextLike | CellClass::DirectionalEdge) {
                structured += 1;
            }
            if edge_density_q8 >= TEXT_EDGE_Q8 && noise < NOISE_HI {
                crisp += 1;
            }
        }
    }
    let total = (cells_x * cells_y).max(1) as f32;
    CellStats {
        textlike_share: textlike as f32 / total,
        structured_share: structured as f32 / total,
        crisp_share: crisp as f32 / total,
    }
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    reason = "test fixtures index buffers they just sized"
)]
mod tests {
    use super::*;

    fn checkerboard_text(width: usize, height: usize) -> Vec<u8> {
        // Two-colour axis-aligned strokes: dense crisp edges, tiny census.
        let mut rgb = vec![255u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                if (x / 3 + y / 7) % 2 == 0 {
                    let i = (y * width + x) * 3;
                    rgb[i] = 20;
                    rgb[i + 1] = 20;
                    rgb[i + 2] = 20;
                }
            }
        }
        rgb
    }

    fn colour_noise(width: usize, height: usize) -> Vec<u8> {
        // A cheap deterministic PRNG walk: colour-dense like a photograph.
        let mut state = 0x9e37_79b9u32;
        (0..width * height * 3)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn a_two_colour_text_frame_routes_and_a_noise_frame_does_not() {
        let text = classify_srgb8(256, 256, &checkerboard_text(256, 256));
        assert_eq!(text.class, ContentClass::TextUiLineArt);
        assert_eq!(text.unique_colours, 2);
        assert!(!text.census_capped);
        assert!(text.crisp_share > 0.5, "crisp {}", text.crisp_share);

        let noise = classify_srgb8(512, 512, &colour_noise(512, 512));
        assert_eq!(noise.class, ContentClass::PhotoLike);
        assert!(noise.census_capped);
    }

    #[test]
    fn a_sparse_but_colourful_frame_is_unknown() {
        // ~21k unique colours on 64k pixels: under the cap, but denser than
        // one colour per ROUTE_MIN_PIXELS_PER_COLOUR pixels.
        let mut rgb = Vec::with_capacity(256 * 256 * 3);
        for i in 0..256 * 256u32 {
            let v = i / 3; // ~21.8k distinct triples
            rgb.extend_from_slice(&[(v >> 8) as u8, (v & 0xff) as u8, 7]);
        }
        let hint = classify_srgb8(256, 256, &rgb);
        assert!(!hint.census_capped);
        assert_eq!(hint.class, ContentClass::Unknown);
    }

    #[test]
    fn the_hint_serialises_the_schema_fields() {
        let hint = classify_srgb8(64, 64, &checkerboard_text(64, 64));
        let json = hint.to_json();
        for key in [
            "\"class\":\"text_ui\"",
            "\"unique_colours\":2",
            "\"census_capped\":false",
            "\"textlike_share\":",
            "\"structured_share\":",
            "\"crisp_share\":",
            "\"classifier_us\":",
        ] {
            assert!(json.contains(key), "{json} missing {key}");
        }
    }
}
