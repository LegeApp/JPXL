//! The quantizer ladder: the enumerated set of *representable* quantizers the
//! encoder's searches move over.
//!
//! Both controllers navigate the same discrete space. The byte-rate loop
//! ([`crate::rate`]) walks it for a size target; the perceptual loop
//! ([`crate::quality`]) walks it for a score target. Neither one owns it, so
//! it lives here (plan phase N0,
//! `JPXL/docs/scheduler-unification-and-multifidelity-plan.md`).
//!
//! # Why a ladder rather than a float
//!
//! I.2.1's `global_scale` is a `U32()` field with a largest expressible value,
//! `HfMul` is a Modular sample, and `quant_lf` has its own distribution. A
//! search that moved a float and rounded at the end would evaluate points it
//! cannot emit and report a size or a score for a plan that does not exist.
//! [`Rung`] is an index into the enumerated ladder of representable
//! quantizers, and it is the only thing a search ever moves.
//!
//! # The ladder, and the LF/HF coupling policy
//!
//! I.2.1 factors the quantizer: the HF step is `(1 << 16) / (global_scale *
//! HfMul)` and the LF step is `(1 << 16) * w / (global_scale * quant_lf)`. So
//! `global_scale` moves **both** planes together and the other two are ratios
//! against it. A search holds `quant_lf` at the request's value and moves
//! `global_scale` (then `HfMul` past the scale ceiling). That keeps the LF/HF
//! *balance* fixed across the ladder, so a search changes rate without
//! silently changing reconstruction character.
//!
//! `HfMul` is **not** an independent rate axis at fixed blocks: `HfMul = 2` at
//! `global_scale = g` is the same HF quantizer as `HfMul = 1` at
//! `global_scale = 2g`. It earns its place at the top of the ladder, where
//! `global_scale` has hit the largest value I.2.1 can express and `HfMul` is
//! the only way to go finer — and there the ladder interleaves it with
//! `global_scale` (see [`HF_MUL_RUNGS`]) and couples `quant_lf` to it, so
//! neither the HF step nor the LF/HF balance jumps at the ceiling.
//! Per-varblock `HfMul` is milestone 7.

use jpxl_encode::vardct::ids::{GlobalScale, HfMul, MAX_GLOBAL_SCALE, QuantLf};

use crate::error::Result;
use crate::request::EncodeRequest;

/// How many `HfMul` *octaves* the ladder extends above `global_scale`'s
/// ceiling: the top of the ladder is `HfMul = HF_MUL_RUNGS + 1` at the largest
/// `global_scale`, an HF step 65 times finer than the finest `global_scale`
/// alone can reach. Past that the coefficients are integers in the millions
/// and the LF plane is the accuracy floor anyway.
///
/// Above the ceiling the ladder does **not** step by whole `HfMul` multiples
/// (Phase Q3). `HfMul = 2` at the ceiling is a full octave finer than
/// `HfMul = 1` there, and on a smooth 4 MP photograph that one rung was a
/// 65% jump in bytes: every target between the two landed 35% under, "budget
/// spent, not at the ladder's limit". Instead each `HfMul = k` segment walks
/// `global_scale` from just above `(k - 1) / k` of the ceiling up to the
/// ceiling, so consecutive rungs differ by `k` units of effective scale and
/// the segments join without a gap: the last rung of segment `k` is
/// `k * MAX` and the first of `k + 1` is at most `k + 1` above it.
pub const HF_MUL_RUNGS: u32 = 64;

/// The largest frame-constant `HfMul` the ladder reaches.
const MAX_LADDER_MUL: u32 = HF_MUL_RUNGS + 1;

/// The `global_scale` a rung of segment `k` starts at: the smallest scale
/// whose product with `k` exceeds the previous segment's top `(k - 1) * MAX`.
const fn segment_first_scale(k: u32) -> u32 {
    // (k - 1) * MAX / k, floored, plus one — never below 1. The quotient is
    // below MAX_GLOBAL_SCALE by construction, so the narrowing is exact.
    let prev_top = (k as u64 - 1) * MAX_GLOBAL_SCALE as u64;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "(k - 1) * MAX / k < MAX fits u32"
    )]
    let floor = (prev_top / k as u64) as u32;
    if floor >= MAX_GLOBAL_SCALE {
        MAX_GLOBAL_SCALE
    } else {
        floor + 1
    }
}

/// How many rungs segment `k` holds.
const fn segment_len(k: u32) -> u32 {
    MAX_GLOBAL_SCALE - segment_first_scale(k) + 1
}

/// The rung count of every segment above the ceiling.
const fn upper_rungs() -> u32 {
    let mut total = 0u32;
    let mut k = 2u32;
    while k <= MAX_LADDER_MUL {
        total += segment_len(k);
        k += 1;
    }
    total
}

/// Above the ceiling, the LF quantiser is coupled to the segment: `quant_lf`
/// is multiplied by `k` while that keeps it at or below this bound, so as
/// `global_scale` halves across the first segments the LF step is unchanged
/// and the LF/HF balance the ladder holds below the ceiling carries through
/// it. The bound exists because jxl-oxide narrows `LfQuant` samples to signed
/// 16 bits (Phase 5F): the LF sample magnitude grows with
/// `global_scale * quant_lf`, and `MAX_GLOBAL_SCALE * 16` is the product the
/// fixed-quantizer defaults (`quant_lf` 16 at the ceiling) have always been
/// allowed to reach; a synthetic high-contrast oracle fixture wraps at twice
/// that. Beyond the bound `quant_lf` stays put and the LF step sawtooths by
/// at most `1 / (k + 1)` across a segment join, which for the target policy's
/// `quant_lf` 4 starts only above four octaves past the ceiling.
const LF_COUPLING_MAX_QUANT_LF: u32 = 16;

/// The coupled `quant_lf` for segment `k`: the largest `base * j`, `j <= k`,
/// that stays within [`LF_COUPLING_MAX_QUANT_LF`] — or `base` itself when
/// even `k = 1` would not (a request already finer than the bound is left
/// exactly as asked).
fn coupled_quant_lf(base: QuantLf, k: u32) -> QuantLf {
    let mut best = base;
    for j in 2..=k {
        match QuantLf::new(base.get().saturating_mul(j)) {
            Ok(candidate) if candidate.get() <= LF_COUPLING_MAX_QUANT_LF => best = candidate,
            _ => break,
        }
    }
    best
}

/// One past the last ladder index.
pub const LADDER_LEN: u32 = MAX_GLOBAL_SCALE + upper_rungs();

/// The `(global_scale, HfMul)` pair of a rung, on either side of the ceiling.
fn rung_fields(rung: Rung) -> (u32, u32) {
    if rung.get() < MAX_GLOBAL_SCALE {
        return (rung.get() + 1, 1);
    }
    let mut offset = rung.get() - MAX_GLOBAL_SCALE;
    let mut k = 2u32;
    while k < MAX_LADDER_MUL && offset >= segment_len(k) {
        offset -= segment_len(k);
        k += 1;
    }
    ((segment_first_scale(k) + offset).min(MAX_GLOBAL_SCALE), k)
}

/// A position on the quantizer ladder: **larger is finer, and bigger**.
///
/// Not a bare `u32` on purpose (`AGENTS.md`): a rung, a `global_scale` and a
/// byte count are three different things that are all small integers, and the
/// loop below mixes all three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rung(u32);

impl Rung {
    /// The coarsest quantizer the ladder holds.
    pub const FLOOR: Self = Self(0);
    /// The finest quantizer the ladder holds.
    pub const TOP: Self = Self(LADDER_LEN - 1);

    /// Wraps an index, clamped into the ladder.
    #[must_use]
    pub const fn new(index: u32) -> Self {
        if index >= LADDER_LEN {
            return Self::TOP;
        }
        Self(index)
    }

    /// The index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The rung whose `global_scale` is `scale` (and whose `HfMul` is one).
    #[must_use]
    pub const fn for_global_scale(scale: u32) -> Self {
        if scale == 0 {
            return Self::FLOOR;
        }
        Self::new(scale - 1)
    }
}

/// A representable quantizer: the three wire fields, already range-checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizerChoice {
    /// Where on the ladder this sits.
    pub rung: Rung,
    /// I.2.1's `global_scale`.
    pub global_scale: GlobalScale,
    /// The frame-constant `HfMul` (G.2.4).
    pub hf_mul: HfMul,
    /// I.2.1's `quant_lf`.
    pub quant_lf: QuantLf,
}

impl QuantizerChoice {
    /// The quantizer at `rung`, at the LF/HF ratio `quant_lf` names.
    ///
    /// # Errors
    ///
    /// [`PolicyError::Plan`] can only fire if [`LADDER_LEN`] and the fields'
    /// own ranges disagree, which the `every_rung_is_representable` test pins.
    pub fn at(rung: Rung, quant_lf: QuantLf) -> Result<Self> {
        let (scale, mul) = rung_fields(rung);
        Ok(Self {
            rung,
            global_scale: GlobalScale::new(scale)?,
            hf_mul: HfMul::new(mul)?,
            quant_lf: coupled_quant_lf(quant_lf, mul),
        })
    }

    /// The quantizer a request names when it sets no target.
    ///
    /// The rung is the one whose `global_scale` matches; a request with a
    /// non-unit `HfMul` has no exact rung, and gets the `global_scale` one,
    /// which is only ever used as the search's starting point.
    #[must_use]
    pub fn from_request(request: &EncodeRequest) -> Self {
        Self {
            rung: Rung::for_global_scale(request.global_scale.get()),
            global_scale: request.global_scale,
            hf_mul: request.hf_mul,
            quant_lf: request.quant_lf,
        }
    }
}

/// The effective quantizer fineness at a rung: `global_scale * HfMul`.
///
/// The rung *index* is not proportional to the quantizer. Below
/// `MAX_GLOBAL_SCALE` a rung is a `global_scale` step; above it `global_scale`
/// is pinned and each rung is an `HfMul` step worth `MAX_GLOBAL_SCALE` of the
/// lower segment. Interpolating on the index therefore aims badly across that
/// kink — which is exactly where high-rate targets live. This quantity is
/// smooth and strictly increasing across the whole ladder.
pub fn effective_scale(rung: Rung) -> u64 {
    let (scale, mul) = rung_fields(rung);
    u64::from(scale) * u64::from(mul)
}

/// The inverse of [`effective_scale`], rounded down to the finest
/// representable rung whose effective scale does not exceed `scale`.
pub fn rung_for_effective_scale(scale: u64) -> Rung {
    let max = u64::from(MAX_GLOBAL_SCALE);
    if scale <= max {
        return Rung::new(u32::try_from(scale.saturating_sub(1)).unwrap_or(u32::MAX));
    }
    if scale >= max * u64::from(MAX_LADDER_MUL) {
        return Rung::TOP;
    }
    // Segment k covers ((k - 1) * MAX, k * MAX].
    let k = u32::try_from(scale.div_ceil(max)).unwrap_or(MAX_LADDER_MUL);
    let k = k.clamp(2, MAX_LADDER_MUL);
    let gs = u32::try_from(scale / u64::from(k)).unwrap_or(MAX_GLOBAL_SCALE);
    let mut index = MAX_GLOBAL_SCALE;
    let mut j = 2u32;
    while j < k {
        index += segment_len(j);
        j += 1;
    }
    if gs < segment_first_scale(k) {
        // Below this segment's first rung: the previous segment's top is the
        // finest rung not above `scale`.
        return Rung::new(index.saturating_sub(1));
    }
    Rung::new(index + (gs - segment_first_scale(k)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quant_lf() -> QuantLf {
        QuantLf::new(16).expect("legal")
    }

    #[test]
    fn every_rung_is_representable_and_the_ladder_is_ordered() {
        let lf = quant_lf();
        for index in [
            0u32,
            1,
            1000,
            MAX_GLOBAL_SCALE - 1,
            MAX_GLOBAL_SCALE,
            LADDER_LEN - 1,
        ] {
            let q = QuantizerChoice::at(Rung::new(index), lf).expect("representable");
            assert!(q.global_scale.get() >= 1 && q.global_scale.get() <= MAX_GLOBAL_SCALE);
            assert!(q.hf_mul.get() >= 1);
        }
        // The HF step is `1 / (global_scale * HfMul)`: strictly finer with the
        // rung, across the join between the two segments.
        let mut previous = 0u64;
        for index in [
            0u32,
            1,
            MAX_GLOBAL_SCALE - 2,
            MAX_GLOBAL_SCALE - 1,
            MAX_GLOBAL_SCALE,
            MAX_GLOBAL_SCALE + 1,
            LADDER_LEN - 1,
        ] {
            let q = QuantizerChoice::at(Rung::new(index), lf).expect("representable");
            let product = u64::from(q.global_scale.get()) * u64::from(q.hf_mul.get());
            assert!(product > previous, "rung {index} did not go finer");
            previous = product;
        }
    }

    #[test]
    fn lf_coupling_follows_the_segment_up_to_the_bound_and_never_below_the_request() {
        let four = QuantLf::new(4).expect("legal");
        assert_eq!(coupled_quant_lf(four, 1).get(), 4);
        assert_eq!(coupled_quant_lf(four, 2).get(), 8);
        assert_eq!(coupled_quant_lf(four, 4).get(), 16);
        assert_eq!(coupled_quant_lf(four, 5).get(), 16, "the bound holds");
        assert_eq!(coupled_quant_lf(four, 65).get(), 16);
        let sixteen = QuantLf::new(16).expect("legal");
        assert_eq!(
            coupled_quant_lf(sixteen, 3).get(),
            16,
            "already at the bound"
        );
        let coarse_lf = QuantLf::new(3).expect("legal");
        assert_eq!(
            coupled_quant_lf(coarse_lf, 6).get(),
            15,
            "largest multiple within the bound"
        );
        let finer_than_bound = QuantLf::new(64).expect("legal");
        assert_eq!(
            coupled_quant_lf(finer_than_bound, 4).get(),
            64,
            "left as asked"
        );
    }

    /// Phase Q3: above the ceiling every rung is a distinct, strictly finer
    /// effective scale, the inverse map is exact, the segments join without a
    /// gap, and `quant_lf` follows the segment up to the coupling cap.
    #[test]
    fn the_upper_ladder_is_dense_invertible_and_lf_coupled() {
        let lf = quant_lf();
        let mut previous = effective_scale(Rung::new(MAX_GLOBAL_SCALE - 1));
        assert_eq!(previous, u64::from(MAX_GLOBAL_SCALE));
        for index in MAX_GLOBAL_SCALE..LADDER_LEN {
            let rung = Rung::new(index);
            let scale = effective_scale(rung);
            assert!(scale > previous, "rung {index} did not go finer");
            // Adjacent rungs of segment k differ by exactly k; the first rung
            // of a segment sits at most k above the previous segment's top.
            let (gs, mul) = rung_fields(rung);
            assert!((1..=MAX_GLOBAL_SCALE).contains(&gs));
            if gs == segment_first_scale(mul) {
                assert!(scale - previous <= u64::from(mul), "rung {index}");
            } else {
                assert_eq!(scale - previous, u64::from(mul), "rung {index}");
            }
            assert_eq!(rung_for_effective_scale(scale), rung, "rung {index}");
            // Any scale strictly between two rungs rounds down to the lower.
            if scale - previous > 1 {
                assert_eq!(rung_for_effective_scale(scale - 1), Rung::new(index - 1));
            }
            let q = QuantizerChoice::at(rung, lf).expect("representable");
            assert_eq!(q.hf_mul.get(), mul);
            assert_eq!(q.quant_lf, coupled_quant_lf(lf, mul));
            assert!(q.quant_lf.get() >= lf.get());
            assert!(q.quant_lf.get() <= LF_COUPLING_MAX_QUANT_LF.max(lf.get()));
            previous = scale;
        }
        // The top is unchanged from the whole-multiple ladder: 65 x MAX.
        assert_eq!(
            effective_scale(Rung::TOP),
            u64::from(MAX_GLOBAL_SCALE) * u64::from(MAX_LADDER_MUL)
        );
        assert_eq!(rung_fields(Rung::TOP), (MAX_GLOBAL_SCALE, MAX_LADDER_MUL));
        // Below the ceiling nothing moved: rung == global_scale - 1, HfMul 1,
        // the request's own quant_lf.
        for index in [0u32, 4_096, MAX_GLOBAL_SCALE - 1] {
            let q = QuantizerChoice::at(Rung::new(index), lf).expect("representable");
            assert_eq!(q.global_scale.get(), index + 1);
            assert_eq!(q.hf_mul.get(), 1);
            assert_eq!(q.quant_lf, lf);
        }
        // For the target policy's `quant_lf` 4 the LF step
        // (1 / (global_scale * quant_lf)) is continuous across the ceiling
        // and the first coupled segments: the product at the first rung above
        // the ceiling is at least the ceiling's own, and it keeps rising
        // through the segments the bound admits.
        let four = QuantLf::new(4).expect("legal");
        let lf_product =
            |q: QuantizerChoice| u64::from(q.global_scale.get()) * u64::from(q.quant_lf.get());
        let ceiling = QuantizerChoice::at(Rung::new(MAX_GLOBAL_SCALE - 1), four).expect("ok");
        let above = QuantizerChoice::at(Rung::new(MAX_GLOBAL_SCALE), four).expect("ok");
        assert!(lf_product(above) >= lf_product(ceiling));
        assert_eq!(above.quant_lf.get(), 8);
        // A request already at the bound is left alone above the ceiling.
        let sixteen = QuantizerChoice::at(Rung::new(MAX_GLOBAL_SCALE), lf).expect("ok");
        assert_eq!(sixteen.quant_lf, lf);
    }

    #[test]
    fn a_rung_out_of_range_saturates_rather_than_wrapping() {
        assert_eq!(Rung::new(u32::MAX), Rung::TOP);
        assert_eq!(Rung::for_global_scale(0), Rung::FLOOR);
        assert_eq!(Rung::for_global_scale(1), Rung::FLOOR);
        assert_eq!(Rung::for_global_scale(32_768).get(), 32_767);
    }
}
