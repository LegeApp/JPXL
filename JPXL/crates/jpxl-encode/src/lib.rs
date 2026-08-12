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
//! * greyscale: no transform or optional **palette**; RGB: **`kRCT`** or
//!   exact-colour **palette** (mutually exclusive in wave 1);
//! * learned MA tree + ANS residuals + LZ77 when cheaper;
//! * **restoration filters explicitly disabled**, so the decoded samples are
//!   the encoded samples.
//!
//! Everything outside that is [`EncodeError::Unsupported`] rather than a guess.
//!
//! # What this is for
//!
//! Slice 19 densifies this path (ANS residuals + predictor selection). The
//! acceptance criteria remain: `jpxl-decode` reproduces the input samples
//! exactly, then `djxl` does, then `jxl-oxide` does.
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

//! # The policy boundary (slice 11)
//!
//! From slice 11 this crate is the **normative lowering and emission** half of
//! a one-way split (`docs/PLAN.md`, `docs/Encoder-plan1.md`): it defines what a
//! plan is, checks that a plan is structurally legal, and turns a legal plan
//! into bits. It does not search. The search half — block tiling, adaptive
//! quantization, chroma-from-luma, entropy clustering, rate control — is
//! `jpxl-encode-policy`, which depends on this crate and is never depended on
//! by it.
//!
//! * [`vardct`] holds the VarDCT plan IR, its validator and its writer gate.
//! * [`lossless`] holds the modular track's plan, whose (three-line) policy
//!   half moves to `jpxl-encode-policy` in slice 19.

pub mod container;
pub mod entropy;
pub mod error;
pub mod frame;
pub mod headers;
pub mod lossless;
pub mod modular;
pub mod resources;
pub mod section;
pub mod vardct;

use jpxl_bitstream::BitWriter;

use frame::Geometry;
use headers::ColourShape;
use lossless::ValidatedLosslessPlan;
use modular::{ModularSource, Plane, Rect};
use section::SectionStore;

pub use error::{EncodeError, Result};
pub use lossless::Effort;
pub use resources::{EncodeResources, ParallelAxis};

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
    /// Split the codestream across `jxlp` boxes (18181-2 9.10) of at most this
    /// many payload bytes each, instead of one `jxlc` box.
    ///
    /// Implies [`EncodeOptions::container`]: a fragmented codestream has
    /// nowhere to live outside a container.
    pub jxlp_fragment_size: Option<usize>,
    /// Resource policy for coarse section parallelism (Opt-P).
    ///
    /// Default is [`EncodeResources::auto`] (group axis + host parallelism)
    /// when the `parallel` feature is on. Under Contract A the codestream must
    /// not depend on [`EncodeResources::threads`].
    pub resources: EncodeResources,
    /// Lossless-modular search effort (speed/size dial), 1..=9.
    ///
    /// [`Effort::DEFAULT`] is the lean level 1: fastest, and byte-identical to
    /// the full search on photographic/smooth content. Higher levels spend more
    /// time and only occasionally (a few percent, on specific content) produce
    /// a smaller file; level 7 is the full search retained as a density anchor.
    /// Every level is exact-lossless, so this changes only the byte count and
    /// the encode time, never the decoded pixels.
    pub effort: Effort,
    /// Measurement escape hatch: override individual levers of the search
    /// budget [`EncodeOptions::effort`] would have selected.
    ///
    /// This exists so the effort ladder's *constants* can be swept and chosen
    /// from evidence (`jpxl bench modular --modular-max-depth/--modular-sample-budget`)
    /// instead of guessed. It is deliberately not a quality dial: `None` — the
    /// default — is the shipped behaviour, and nothing in the encoder sets it.
    /// Like [`EncodeOptions::effort`], every setting stays exact-lossless, so
    /// this can change the byte count and the encode time but never the
    /// decoded pixels.
    pub modular_search_overrides: ModularSearchOverrides,
}

/// Per-lever overrides of the [`Effort`]-selected modular search budget.
///
/// Each `Some` replaces one lever; each `None` keeps what the effort chose.
/// See [`EncodeOptions::modular_search_overrides`] for why this exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModularSearchOverrides {
    /// Cap on MA-tree depth (root = 0).
    pub max_tree_depth: Option<u32>,
    /// Cap on MA-tree leaves / residual contexts.
    pub max_tree_leaves: Option<usize>,
    /// Target samples the cheap ranker may score per candidate, summed across
    /// planes. `u64::MAX` means unbounded.
    pub cheap_sample_budget: Option<u64>,
    /// Frame sample count above which the tree search collapses to one split.
    /// `u64::MAX` retires the collapse.
    pub deep_search_sample_cap: Option<u64>,
}

impl ModularSearchOverrides {
    /// Whether any lever is overridden.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.max_tree_depth.is_none()
            && self.max_tree_leaves.is_none()
            && self.cheap_sample_budget.is_none()
            && self.deep_search_sample_cap.is_none()
    }
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
    if !options.container && options.jxlp_fragment_size.is_none() {
        return Ok(codestream);
    }
    // 18181-2 9.3 with Annex M of Part 1: `modular_16bit_buffers` is false
    // for a >8-bit image, which level 5 does not permit.
    let level = if image.bits_per_sample() > 8 {
        container::EXTENDED_LEVEL
    } else {
        container::DEFAULT_LEVEL
    };
    Ok(match options.jxlp_fragment_size {
        Some(size) => container::wrap_fragmented(&codestream, level, size),
        None => container::wrap(&codestream, level),
    })
}

/// Encodes the naked codestream, whatever [`EncodeOptions::container`] says.
///
/// Two steps, in this order and no other: **choose a plan**, then **emit it**.
/// The choosing is [`lossless::plan_for`] (policy, slice 19); the emitting is
/// [`encode_codestream_with_plan`], which decides nothing.
fn encode_codestream(image: &Image, options: &EncodeOptions) -> Result<Vec<u8>> {
    // Plan over *source* samples. RCT is applied only when the plan keeps it
    // (palette and RCT are mutually exclusive in wave 1).
    let source_planes = image.planes().to_vec();
    let want_rct = source_planes.len() == 3;
    let plan = lossless::plan_for(
        image.width(),
        image.height(),
        &source_planes,
        want_rct,
        options,
    )?;

    let planes = if plan.plan().rct {
        let mut p = source_planes;
        modular::apply_rct(&mut p)?;
        p
    } else {
        source_planes
    };
    encode_codestream_with_plan_resources(image, &planes, &plan, options.resources)
}

/// Emits the codestream a validated plan describes.
///
/// `planes` are the already-transformed planes the plan was chosen for.
///
/// # Errors
///
/// [`EncodeError::ValueOutOfRange`] if a residual or a section is outside the
/// range its field can carry, and any error from the layers below.
pub fn encode_codestream_with_plan(
    image: &Image,
    planes: &[Plane],
    plan: &ValidatedLosslessPlan,
) -> Result<Vec<u8>> {
    encode_codestream_with_plan_resources(image, planes, plan, EncodeResources::serial())
}

/// As [`encode_codestream_with_plan`], with an explicit resource policy.
///
/// # Errors
///
/// As [`encode_codestream_with_plan`].
pub fn encode_codestream_with_plan_resources(
    image: &Image,
    planes: &[Plane],
    plan: &ValidatedLosslessPlan,
    resources: EncodeResources,
) -> Result<Vec<u8>> {
    let (width, height) = (image.width(), image.height());
    let plan = plan.plan();
    let geometry = Geometry::new(width, height, plan.group_size_shift)?;

    let source = if let Some(ref palette) = plan.palette {
        ModularSource::from_palette(palette.clone(), plan.tree.clone(), true)
    } else if plan.squeeze {
        ModularSource::with_default_squeeze(
            width,
            height,
            planes,
            plan.rct,
            plan.tree.clone(),
            true,
        )?
    } else {
        ModularSource::direct(width, height, planes, plan.rct, plan.tree.clone(), true)
    };

    let store = build_sections(&source, &geometry, resources)?;

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
///
/// Independent LF-group and pass-group bodies may run on multiple workers;
/// results are reduced in F.3.1 index order (Contract A).
///
/// Multi-section frames try Phase 4C's global MA tree (tree + residual D bundle
/// once at G.1.3) and keep it only when the total section payload is no larger
/// than the historical per-section local-tree emission.
fn build_sections(
    source: &ModularSource,
    geometry: &Geometry,
    resources: EncodeResources,
) -> Result<SectionStore> {
    if geometry.is_single_section() {
        // F.3.1: one group and one pass means one section carrying everything.
        let mut store = SectionStore::new();
        store.push(modular::encode_lf_global(source, geometry)?);
        return Ok(store);
    }

    let local = build_sections_local(source, geometry, resources)?;
    let global = build_sections_global(source, geometry, resources)?;
    if global.total_len() <= local.total_len() {
        Ok(global)
    } else {
        Ok(local)
    }
}

/// Pre-4C multi-section emission: every modular section carries its own tree.
fn build_sections_local(
    source: &ModularSource,
    geometry: &Geometry,
    resources: EncodeResources,
) -> Result<SectionStore> {
    let mut store = SectionStore::new();
    store.push(modular::encode_lf_global(source, geometry)?);
    let n_lf = usize::try_from(geometry.num_lf_groups()).unwrap_or(0);
    let lf_workers = resources.workers_for(n_lf);
    let lf_bodies = resources::ordered_map(n_lf, lf_workers, |index| {
        let (x0, y0, width, height) = geometry
            .lf_group_rect(u64::try_from(index).unwrap_or(u64::MAX))
            .ok_or_else(|| EncodeError::unsupported("an LF group index past the grid", "G.2"))?;
        modular::encode_lf_group(
            source,
            Rect {
                x0,
                y0,
                width,
                height,
            },
            geometry,
        )
    })?;
    for body in lf_bodies {
        store.push(body);
    }
    store.push_empty(); // HfGlobal — VarDCT-only (G.3)
    let n_pg = usize::try_from(geometry.num_groups()).unwrap_or(0);
    let pg_workers = resources.workers_for(n_pg);
    let pg_bodies = resources::ordered_map(n_pg, pg_workers, |index| {
        let (x0, y0, width, height) = geometry
            .group_rect(u64::try_from(index).unwrap_or(u64::MAX))
            .ok_or_else(|| EncodeError::unsupported("a group index past the grid", "G.4"))?;
        modular::encode_group(
            source,
            Rect {
                x0,
                y0,
                width,
                height,
            },
            geometry,
        )
    })?;
    for body in pg_bodies {
        store.push(body);
    }
    Ok(store)
}

/// Phase 4C multi-section emission with G.1.3 global tree + shared residual D.
fn build_sections_global(
    source: &ModularSource,
    geometry: &Geometry,
    resources: EncodeResources,
) -> Result<SectionStore> {
    let model = modular::build_global_residual_model(source, geometry)?;
    let mut store = SectionStore::new();
    store.push(modular::encode_lf_global_with_global_tree(
        source, geometry, &model,
    )?);
    let n_lf = usize::try_from(geometry.num_lf_groups()).unwrap_or(0);
    let lf_workers = resources.workers_for(n_lf);
    let lf_bodies = resources::ordered_map(n_lf, lf_workers, |index| {
        let (x0, y0, width, height) = geometry
            .lf_group_rect(u64::try_from(index).unwrap_or(u64::MAX))
            .ok_or_else(|| EncodeError::unsupported("an LF group index past the grid", "G.2"))?;
        modular::encode_lf_group_with_global_tree(
            source,
            Rect {
                x0,
                y0,
                width,
                height,
            },
            geometry,
            &model,
        )
    })?;
    for body in lf_bodies {
        store.push(body);
    }
    store.push_empty(); // HfGlobal — VarDCT-only (G.3)
    let n_pg = usize::try_from(geometry.num_groups()).unwrap_or(0);
    let pg_workers = resources.workers_for(n_pg);
    let pg_bodies = resources::ordered_map(n_pg, pg_workers, |index| {
        let (x0, y0, width, height) = geometry
            .group_rect(u64::try_from(index).unwrap_or(u64::MAX))
            .ok_or_else(|| EncodeError::unsupported("a group index past the grid", "G.4"))?;
        modular::encode_group_with_global_tree(
            source,
            Rect {
                x0,
                y0,
                width,
                height,
            },
            geometry,
            &model,
        )
    })?;
    for body in pg_bodies {
        store.push(body);
    }
    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use modular::{MaTree, Predictor};

    /// Opt-P Contract A: multi-section modular encode is byte-identical at
    /// serial, fixed-N, and host-auto worker budgets.
    #[test]
    fn multi_section_modular_is_byte_identical_across_thread_counts() {
        let width = 300u32;
        let height = 200u32;
        let plane: Plane = (0..height)
            .flat_map(|y| (0..width).map(move |x| ((x + 3 * y) % 200) as i32))
            .collect();
        let image = Image::new(width, height, 8, vec![plane]).expect("image");
        let base = EncodeOptions {
            group_size_shift: Some(0), // group_dim 128 → multi-section
            ..EncodeOptions::default()
        };
        let mut serial = base;
        serial.resources = EncodeResources::serial();
        let mut parallel = base;
        parallel.resources = EncodeResources::groups(4);
        let auto = base; // default resources = auto

        let a = encode(&image, &serial).expect("serial");
        let b = encode(&image, &parallel).expect("parallel");
        let c = encode(&image, &auto).expect("auto");
        assert_eq!(
            a, b,
            "Contract A: 1-thread and 4-thread modular multi-section must match"
        );
        assert_eq!(
            a, c,
            "Contract A: EncodeResources::auto must match serial emission"
        );
    }

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
    fn a_multi_group_image_produces_the_section_count_f31_requires() {
        let planes = vec![vec![0i32; 600 * 520]];
        let image = Image::new(600, 520, 8, planes).expect("valid");
        let geometry = Geometry::new(600, 520, 2).expect("valid");
        let source = ModularSource::direct(
            600,
            520,
            image.planes(),
            false,
            MaTree::single_leaf(Predictor::Gradient),
            true,
        );
        let store =
            build_sections(&source, &geometry, EncodeResources::serial()).expect("sections");
        assert_eq!(store.len() as u64, geometry.num_sections());
        assert_eq!(store.len(), 2 + 1 + 4);
    }
}
