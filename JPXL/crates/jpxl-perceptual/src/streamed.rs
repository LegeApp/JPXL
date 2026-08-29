//! The banded candidate-side moment pipeline (Phase S2, exact variant).
//!
//! [`channel_terms`] produces exactly what the former full-plane walk
//! produced — every channel's pooled terms at one scale — without ever
//! materialising a full-frame moment plane. The moment maps (the candidate's
//! blurred mean, second moment and cross moment, plus the source's two when
//! the reference retains only planes) exist as rings of 32-row batches:
//!
//! ```text
//! superstep s:   horizontal blur of batch s      → temp ring (4 slots)
//!                vertical recursion emits batch s−2 → out ring (2 slots)
//!                pooling folds batch s−3          → per-channel sums
//! ```
//!
//! The three stages of one superstep touch three different batches, so they
//! are independent items of a single executor fan-out; the pipeline keeps
//! every worker busy while each stage's arithmetic stays in its original
//! order. Bit-identity with the full-plane walk is by construction:
//!
//! * the horizontal pass already ran per fixed 32-row band
//!   ([`crate::bands::BAND_ROWS`]), and these are the same bands;
//! * the vertical recursion is per column in `f64`; the per-strip state now
//!   persists across batches instead of walking a full plane, visiting the
//!   same rows in the same order ([`crate::blur`]'s own documented
//!   partition-independence);
//! * pooling folds the same 32-row band partials in the same band order.
//!
//! The candidate's positive-XYB planes and the retained reference are the
//! only full frames left; everything between them is `O(width)` state.

use crate::bands::{BAND_ROWS, Handoff, band_count, band_of};
use crate::blur::{
    BlurInput, DisjointColumns, RADIUS, STRIP, StripState, horizontal_band, vertical_row,
};
use crate::executor::BandExecutor;
use crate::pool::{ChannelTerms, MapSums, MomentBands, accumulate_band};
use crate::reference::{ReferenceRetention, ReferenceScale};

/// Maps streamed per channel when the reference retains its moments.
const CANDIDATE_MAPS: usize = 3;

/// Maps streamed per channel when the source's moments are recomputed.
const ALL_MAPS: usize = 5;

/// Temp-ring depth: the vertical stage reads batches `s−3 ..= s−1` while the
/// horizontal stage writes batch `s`, and those four occupy distinct slots
/// modulo four.
const TEMP_SLOTS: usize = 4;

/// Out-ring depth: the vertical stage writes batch `s−2` while pooling reads
/// batch `s−3`, distinct slots modulo two.
const OUT_SLOTS: usize = 2;

/// One moment map's pipeline: its temp ring, its out ring, and one recursion
/// state per column strip.
#[derive(Debug, Default)]
struct MapPipeline {
    temp: [Vec<f32>; TEMP_SLOTS],
    out: [Vec<f32>; OUT_SLOTS],
    states: Vec<StripState>,
}

/// Reusable scratch for the streamed walk: one pipeline per (channel, map).
///
/// Everything here is `O(width)`; dropping it (the scorer's
/// `release_scratch`) costs nothing semantic.
#[derive(Debug, Default)]
pub(crate) struct Scratch {
    maps: Vec<MapPipeline>,
}

/// Read access to the live window of a temp ring, by absolute row index.
///
/// Rows outside the plane are the metric's zero padding; rows inside it are
/// served from whichever ring slot holds their batch. The caller only ever
/// asks for rows of the three live batches (the slot being written this
/// superstep is not in `slots`).
#[derive(Clone, Copy)]
struct RowSource<'a> {
    slots: [&'a [f32]; TEMP_SLOTS],
    zeros: &'a [f32],
    width: usize,
    height: isize,
}

impl RowSource<'_> {
    fn row(&self, i: isize) -> &[f32] {
        if i < 0 || i >= self.height {
            return self.zeros;
        }
        let Ok(i) = usize::try_from(i) else {
            return self.zeros;
        };
        let slot = self
            .slots
            .get((i / BAND_ROWS) % TEMP_SLOTS)
            .copied()
            .unwrap_or(&[]);
        let start = (i % BAND_ROWS) * self.width;
        slot.get(start..start + self.width).unwrap_or(self.zeros)
    }
}

/// One strip's vertical-recursion advance over one output batch.
struct VertJob<'a> {
    rows: RowSource<'a>,
    state: &'a mut StripState,
    out: &'a DisjointColumns,
    x0: usize,
    cols: usize,
    batch: usize,
    batch_rows: usize,
    width: usize,
}

/// One channel's pooling of one completed batch into its running sums.
struct PoolJob<'a> {
    bands: MomentBands<'a>,
    total: &'a mut MapSums,
}

/// One executor item of one superstep.
enum Item<'a> {
    Horiz(BlurInput<'a>, &'a mut [f32], usize),
    Vert(VertJob<'a>),
    Pool(PoolJob<'a>),
}

/// Per-map borrows carried from the splitting pass to item construction.
struct MapCtx<'a> {
    horiz: Option<(BlurInput<'a>, &'a mut [f32])>,
    rows: Option<RowSource<'a>>,
    states: &'a mut Vec<StripState>,
}

/// One channel's completed-batch map slices for pooling.
#[derive(Default, Clone, Copy)]
struct PoolInputs<'a> {
    mu2: &'a [f32],
    s22: &'a [f32],
    s12: &'a [f32],
    ref_mu: Option<&'a [f32]>,
    ref_s11: Option<&'a [f32]>,
}

/// Pools every channel of one scale through the streamed pipeline.
///
/// `xyb` is the candidate's positive-XYB planes at this scale; the result is
/// bit-identical to blurring full mu/s22/s12 (and, for
/// [`ReferenceRetention::PlanesOnly`], the source's mu/s11) planes and
/// pooling them in fixed row bands.
pub(crate) fn channel_terms(
    scratch: &mut Scratch,
    reference: &ReferenceScale,
    retention: ReferenceRetention,
    xyb: &[Vec<f32>; 3],
    width: usize,
    height: usize,
    executor: &dyn BandExecutor,
) -> [ChannelTerms; 3] {
    let per_channel = match retention {
        ReferenceRetention::Moments => CANDIDATE_MAPS,
        ReferenceRetention::PlanesOnly => ALL_MAPS,
    };
    let strips = width.div_ceil(STRIP);
    scratch
        .maps
        .resize_with(3 * per_channel, MapPipeline::default);
    for pipeline in &mut scratch.maps {
        pipeline.states.resize_with(strips, || StripState::new(0));
        pipeline.states.truncate(strips);
        for (i, state) in pipeline.states.iter_mut().enumerate() {
            let x0 = i * STRIP;
            state.reset(width.saturating_sub(x0).min(STRIP));
        }
    }

    let batches = band_count(height);
    let band_len = width.saturating_mul(BAND_ROWS);
    let rows_in = |b: usize| height.saturating_sub(b * BAND_ROWS).min(BAND_ROWS);
    let mut totals = [MapSums::default(); 3];
    let zeros = vec![0.0f32; width];
    let height_i = isize::try_from(height).unwrap_or(isize::MAX);

    for s in 0..batches.saturating_add(3) {
        let hb = (s < batches).then_some(s);
        let vb = s.checked_sub(2).filter(|&b| b < batches);
        let pb = s.checked_sub(3).filter(|&b| b < batches);

        // Split every pipeline's rings for this superstep: the slot the
        // horizontal stage writes is exclusive, the slots the vertical stage
        // reads are shared, and the two out slots go one to the vertical
        // stage (exclusive, behind `DisjointColumns`) and one to pooling
        // (shared). The batch indices guarantee the parities never collide.
        let mut ctxs: Vec<MapCtx<'_>> = Vec::with_capacity(scratch.maps.len());
        let mut disjoints: Vec<Option<DisjointColumns>> = Vec::with_capacity(scratch.maps.len());
        let mut pool_inputs = [PoolInputs::default(); 3];
        for (mi, pipeline) in scratch.maps.iter_mut().enumerate() {
            let channel = mi / per_channel;
            let kind = mi % per_channel;
            let MapPipeline { temp, out, states } = pipeline;

            let mut slot_refs: [&[f32]; TEMP_SLOTS] = [&[]; TEMP_SLOTS];
            let mut horiz_out: Option<&mut [f32]> = None;
            let [t0, t1, t2, t3] = temp;
            for (si, slot) in [t0, t1, t2, t3].into_iter().enumerate() {
                match hb {
                    Some(b) if b % TEMP_SLOTS == si => {
                        slot.resize(rows_in(b) * width, 0.0);
                        horiz_out = Some(slot.as_mut_slice());
                    }
                    _ => {
                        if let Some(shared) = slot_refs.get_mut(si) {
                            *shared = slot.as_slice();
                        }
                    }
                }
            }

            let [o0, o1] = out;
            let mut pool_slot: Option<&[f32]> = None;
            let mut vert_out: Option<DisjointColumns> = None;
            for (oi, slot) in [o0, o1].into_iter().enumerate() {
                if let Some(b) = vb.filter(|&b| b % OUT_SLOTS == oi) {
                    slot.resize(rows_in(b) * width, 0.0);
                    vert_out = Some(DisjointColumns::new(slot.as_mut_slice()));
                } else if pb.is_some_and(|b| b % OUT_SLOTS == oi) {
                    pool_slot = Some(slot.as_slice());
                }
            }
            disjoints.push(vert_out);

            if let (Some(slice), Some(inputs)) = (pool_slot, pool_inputs.get_mut(channel)) {
                match kind {
                    0 => inputs.mu2 = slice,
                    1 => inputs.s22 = slice,
                    2 => inputs.s12 = slice,
                    3 => inputs.ref_mu = Some(slice),
                    _ => inputs.ref_s11 = Some(slice),
                }
            }

            let horiz = match (hb, horiz_out) {
                (Some(b), Some(out_band)) => {
                    let cand = xyb.get(channel).map_or(&[][..], Vec::as_slice);
                    let refp = reference.xyb.get(channel).map_or(&[][..], Vec::as_slice);
                    let cband = band_of(cand, b, band_len);
                    let rband = band_of(refp, b, band_len);
                    let input = match kind {
                        0 => BlurInput::Plane(cband),
                        1 => BlurInput::Product(cband, cband),
                        2 => BlurInput::Product(rband, cband),
                        3 => BlurInput::Plane(rband),
                        _ => BlurInput::Product(rband, rband),
                    };
                    Some((input, out_band))
                }
                _ => None,
            };
            let rows = vb.map(|_| RowSource {
                slots: slot_refs,
                zeros: zeros.as_slice(),
                width,
                height: height_i,
            });
            ctxs.push(MapCtx {
                horiz,
                rows,
                states,
            });
        }

        let mut items: Vec<Item<'_>> = Vec::new();
        for (mi, ctx) in ctxs.into_iter().enumerate() {
            if let Some((input, out_band)) = ctx.horiz {
                items.push(Item::Horiz(input, out_band, width));
            }
            if let (Some(rows), Some(b), Some(Some(out))) = (ctx.rows, vb, disjoints.get(mi)) {
                let batch_rows = rows_in(b);
                for (t, state) in ctx.states.iter_mut().enumerate() {
                    let x0 = t * STRIP;
                    items.push(Item::Vert(VertJob {
                        rows,
                        state,
                        out,
                        x0,
                        cols: width.saturating_sub(x0).min(STRIP),
                        batch: b,
                        batch_rows,
                        width,
                    }));
                }
            }
        }
        if let Some(b) = pb {
            for ((channel, inputs), total) in pool_inputs.iter().enumerate().zip(totals.iter_mut())
            {
                let img1 = reference.xyb.get(channel).map_or(&[][..], Vec::as_slice);
                let img2 = xyb.get(channel).map_or(&[][..], Vec::as_slice);
                let (mu1, s11): (&[f32], &[f32]) = match retention {
                    ReferenceRetention::Moments => (
                        reference
                            .mu
                            .as_ref()
                            .and_then(|m| m.get(channel))
                            .map_or(&[][..], |p| band_of(p, b, band_len)),
                        reference
                            .s11
                            .as_ref()
                            .and_then(|m| m.get(channel))
                            .map_or(&[][..], |p| band_of(p, b, band_len)),
                    ),
                    ReferenceRetention::PlanesOnly => {
                        (inputs.ref_mu.unwrap_or(&[]), inputs.ref_s11.unwrap_or(&[]))
                    }
                };
                items.push(Item::Pool(PoolJob {
                    bands: MomentBands {
                        img1: band_of(img1, b, band_len),
                        mu1,
                        s11,
                        img2: band_of(img2, b, band_len),
                        mu2: inputs.mu2,
                        s22: inputs.s22,
                        s12: inputs.s12,
                    },
                    total,
                }));
            }
        }

        let handoff = Handoff::new(items);
        executor.run(handoff.len(), &|index| match handoff.take(index) {
            Some(Item::Horiz(input, out_band, w)) => horizontal_band(input, out_band, w),
            Some(Item::Vert(job)) => vert_batch(job),
            Some(Item::Pool(job)) => {
                let mut partial = MapSums::default();
                accumulate_band(&job.bands, &mut partial);
                job.total.add(&partial);
            }
            None => {}
        });
    }

    let pixels = width * height;
    let mut channels = [ChannelTerms::default(); 3];
    for (out, total) in channels.iter_mut().zip(&totals) {
        *out = ChannelTerms::from_sums(total, pixels);
    }
    channels
}

/// One strip's vertical advance over one output batch. Dispatched to an AVX2
/// build where the host supports it.
fn vert_batch(job: VertJob<'_>) {
    #[cfg(target_arch = "x86_64")]
    if jpxl_core::cpu::has_avx2() {
        // SAFETY: `vert_batch_avx2` only requires that the host support AVX2,
        // which `has_avx2` has just confirmed.
        #[allow(unsafe_code)]
        unsafe {
            vert_batch_avx2(job);
        }
        return;
    }
    vert_batch_impl(job);
}

/// [`vert_batch`] compiled for AVX2.
///
/// Calling it is `unsafe` unless the host supports AVX2 (see
/// [`jpxl_core::cpu::has_avx2`]); that is the whole contract.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn vert_batch_avx2(job: VertJob<'_>) {
    vert_batch_impl(job);
}

#[inline(always)]
fn vert_batch_impl(job: VertJob<'_>) {
    let VertJob {
        rows,
        state,
        out,
        x0,
        cols,
        batch,
        batch_rows,
        width,
    } = job;
    let base = isize::try_from(batch * BAND_ROWS).unwrap_or(isize::MAX);
    // The recursion starts `RADIUS - 1` rows above the plane; the first batch
    // carries that warm-up, every later batch resumes its persisted state.
    let start = if batch == 0 { 1 - RADIUS } else { base };
    let end = base.saturating_add(isize::try_from(batch_rows).unwrap_or(0));
    let mut n = start;
    while n < end {
        let top = rows.row(n - RADIUS - 1);
        let bottom = rows.row(n + RADIUS - 1);
        let top = top.get(x0..x0 + cols).unwrap_or(&[]);
        let bottom = bottom.get(x0..x0 + cols).unwrap_or(&[]);
        // Warm-up rows (`n < base`, first batch only) advance the state
        // without emitting; `try_from` filters them.
        #[allow(unsafe_code)]
        let out_row = usize::try_from(n - base).ok().and_then(|local| {
            // SAFETY: this strip's `x0 .. x0 + cols` columns are its own —
            // the strip partition is disjoint — and `local` is within the
            // batch the out slot was sized for.
            unsafe { out.window(local * width + x0, cols) }
        });
        vertical_row(top, bottom, state, out_row);
        n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blur::Blur;
    use crate::executor::{ScopedThreadExecutor, SerialExecutor};

    /// The former full-plane walk, kept as the test oracle: blur every moment
    /// map whole, then pool fixed row bands in band order.
    fn naive_channel_terms(
        reference: &ReferenceScale,
        retention: ReferenceRetention,
        xyb: &[Vec<f32>; 3],
        width: usize,
        height: usize,
        executor: &dyn BandExecutor,
    ) -> [ChannelTerms; 3] {
        let pixels = width * height;
        let mut mu2 = vec![0.0f32; pixels];
        let mut s22 = vec![0.0f32; pixels];
        let mut s12 = vec![0.0f32; pixels];
        let mut ref_mu = vec![0.0f32; pixels];
        let mut ref_s11 = vec![0.0f32; pixels];
        let mut channels = [ChannelTerms::default(); 3];
        for (c, terms) in channels.iter_mut().enumerate() {
            let img1 = reference.xyb.get(c).map_or(&[][..], Vec::as_slice);
            let img2 = xyb.get(c).map_or(&[][..], Vec::as_slice);
            let mut blur = Blur::new();
            blur.blur_plane(img2, &mut mu2, width, height, executor);
            blur.blur_product_plane(img2, img2, &mut s22, width, height, executor);
            blur.blur_product_plane(img1, img2, &mut s12, width, height, executor);
            let (mu1, s11): (&[f32], &[f32]) = match retention {
                ReferenceRetention::Moments => (
                    reference
                        .mu
                        .as_ref()
                        .and_then(|m| m.get(c))
                        .map_or(&[][..], Vec::as_slice),
                    reference
                        .s11
                        .as_ref()
                        .and_then(|m| m.get(c))
                        .map_or(&[][..], Vec::as_slice),
                ),
                ReferenceRetention::PlanesOnly => {
                    blur.blur_plane(img1, &mut ref_mu, width, height, executor);
                    blur.blur_product_plane(img1, img1, &mut ref_s11, width, height, executor);
                    (&ref_mu, &ref_s11)
                }
            };
            let mut total = MapSums::default();
            for band in 0..crate::bands::band_count(height) {
                let mut partial = MapSums::default();
                accumulate_band(
                    &MomentBands {
                        img1: band_of(img1, band, width * BAND_ROWS),
                        mu1: band_of(mu1, band, width * BAND_ROWS),
                        s11: band_of(s11, band, width * BAND_ROWS),
                        img2: band_of(img2, band, width * BAND_ROWS),
                        mu2: band_of(&mu2, band, width * BAND_ROWS),
                        s22: band_of(&s22, band, width * BAND_ROWS),
                        s12: band_of(&s12, band, width * BAND_ROWS),
                    },
                    &mut partial,
                );
                total.add(&partial);
            }
            *terms = ChannelTerms::from_sums(&total, pixels);
        }
        channels
    }

    fn plane(seed: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = u32::try_from(i)
                    .unwrap_or(0)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed);
                #[allow(clippy::cast_possible_truncation)]
                let v = (f64::from(x % 2003) / 2003.0) as f32;
                v
            })
            .collect()
    }

    fn scale_for(width: usize, height: usize, retained: bool) -> (ReferenceScale, [Vec<f32>; 3]) {
        let pixels = width * height;
        let xyb_ref: [Vec<f32>; 3] =
            core::array::from_fn(|c| plane(u32::try_from(c).unwrap_or(0) * 101 + 7, pixels));
        let candidate: [Vec<f32>; 3] =
            core::array::from_fn(|c| plane(u32::try_from(c).unwrap_or(0) * 211 + 31, pixels));
        let (mu, s11) = if retained {
            let mut blur = Blur::new();
            let mut mu: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; pixels]);
            let mut s11: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; pixels]);
            for ((p, mu), s11) in xyb_ref.iter().zip(mu.iter_mut()).zip(s11.iter_mut()) {
                blur.blur_plane(p, mu, width, height, &SerialExecutor);
                blur.blur_product_plane(p, p, s11, width, height, &SerialExecutor);
            }
            (Some(mu), Some(s11))
        } else {
            (None, None)
        };
        (
            ReferenceScale {
                width,
                height,
                xyb: xyb_ref,
                mu,
                s11,
            },
            candidate,
        )
    }

    #[test]
    fn the_streamed_walk_matches_the_full_plane_walk_bit_for_bit() {
        for &(width, height) in &[
            (8usize, 4usize),
            (8, 6),
            (8, 8),
            (20, 100),
            (37, 130),
            (64, 64),
            (100, 20),
            (33, 40),
            (129, 65),
            (96, 96),
            (200, 32),
        ] {
            for retention in [ReferenceRetention::Moments, ReferenceRetention::PlanesOnly] {
                let retained = retention == ReferenceRetention::Moments;
                let (reference, candidate) = scale_for(width, height, retained);
                let naive = naive_channel_terms(
                    &reference,
                    retention,
                    &candidate,
                    width,
                    height,
                    &SerialExecutor,
                );
                let mut scratch = Scratch::default();
                let serial = channel_terms(
                    &mut scratch,
                    &reference,
                    retention,
                    &candidate,
                    width,
                    height,
                    &SerialExecutor,
                );
                assert_eq!(serial, naive, "{width}x{height} {retention:?} serial");
                let threaded = channel_terms(
                    &mut scratch,
                    &reference,
                    retention,
                    &candidate,
                    width,
                    height,
                    &ScopedThreadExecutor { workers: 3 },
                );
                assert_eq!(threaded, naive, "{width}x{height} {retention:?} threaded");
            }
        }
    }

    #[test]
    fn scratch_reuse_across_differing_scales_does_not_change_results() {
        // A larger scale followed by a smaller one reuses (and truncates)
        // rings and strip states; results must match a fresh scratch.
        let mut scratch = Scratch::default();
        for &(width, height) in &[(129, 65), (37, 130), (64, 64), (8, 6)] {
            let (reference, candidate) = scale_for(width, height, true);
            let reused = channel_terms(
                &mut scratch,
                &reference,
                ReferenceRetention::Moments,
                &candidate,
                width,
                height,
                &SerialExecutor,
            );
            let fresh = channel_terms(
                &mut Scratch::default(),
                &reference,
                ReferenceRetention::Moments,
                &candidate,
                width,
                height,
                &SerialExecutor,
            );
            assert_eq!(reused, fresh, "{width}x{height}");
        }
    }
}
