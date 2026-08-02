//! The LZ77 layer of the symbol decoder.
//!
//! ISO/IEC 18181-1 Table C.1 (the `LZ77Params` bundle) and the LZ77 machinery
//! inside `DecodeHybridVarLenUint` of C.3.3.
//!
//! Every symbol a stream produces is appended to a circular window of the last
//! `1 << 20` symbols. A token at or above `lz77.min_symbol` is not a value but
//! a back-reference: it carries a copy length, and a following symbol from a
//! dedicated context carries a distance. Subsequent reads replay the window
//! until the copy is exhausted.
//!
//! Every path here is **bit-exact** (see `docs/PLAN.md`).

use jpxl_bitstream::{BitReader, U32Dist, U32Spec, read_u32};
use jpxl_core::limits::AllocGuard;

use crate::error::{Result, malformed};

/// Log2 of the LZ77 window length (18181-1 C.1).
pub const LOG_WINDOW_SIZE: u32 = 20;
/// Number of symbols retained in the LZ77 window.
pub const WINDOW_SIZE: usize = 1 << LOG_WINDOW_SIZE;
/// Index mask for the circular window.
const WINDOW_MASK: u64 = (WINDOW_SIZE as u64) - 1;

/// `U32(224, 512, 4096, 8 + u(15))` — 18181-1 Table C.1, `min_symbol`.
const MIN_SYMBOL: U32Spec = U32Spec::new([
    U32Dist::Val(224),
    U32Dist::Val(512),
    U32Dist::Val(4096),
    U32Dist::BitsOffset {
        bits: 15,
        offset: 8,
    },
]);

/// `U32(3, 4, 5 + u(2), 9 + u(8))` — 18181-1 Table C.1, `min_length`.
const MIN_LENGTH: U32Spec = U32Spec::new([
    U32Dist::Val(3),
    U32Dist::Val(4),
    U32Dist::BitsOffset { bits: 2, offset: 5 },
    U32Dist::BitsOffset { bits: 8, offset: 9 },
]);

/// Distance shorthands for two-dimensional back-references (18181-1 C.3.3).
///
/// Entry `d` is `(dx, dy)`: a distance of `d` below 120 means the sample
/// `dx + dist_multiplier * dy` positions back, where `dist_multiplier` is the
/// row stride supplied by the calling clause.
///
/// The scanned tables in both available transcriptions of C.3.3 are heavily
/// corrupted (digit/letter substitutions such as `O` for `0` and `l` for `1`,
/// and the markdown conversion drops whole rows). This table was reconstructed
/// from the LaTeX transcription, which retains all 120 entries, and then
/// validated structurally: `dx*dx + dy*dy` is non-decreasing across the whole
/// table, which the `special_distances_are_ordered_by_radius` test asserts.
/// A stray digit would break that ordering.
pub const SPECIAL_DISTANCES: [(i32, i32); 120] = [
    (0, 1),
    (1, 0),
    (1, 1),
    (-1, 1),
    (0, 2),
    (2, 0),
    (1, 2),
    (-1, 2),
    (2, 1),
    (-2, 1),
    (2, 2),
    (-2, 2),
    (0, 3),
    (3, 0),
    (1, 3),
    (-1, 3),
    (3, 1),
    (-3, 1),
    (2, 3),
    (-2, 3),
    (3, 2),
    (-3, 2),
    (0, 4),
    (4, 0),
    (1, 4),
    (-1, 4),
    (4, 1),
    (-4, 1),
    (3, 3),
    (-3, 3),
    (2, 4),
    (-2, 4),
    (4, 2),
    (-4, 2),
    (0, 5),
    (3, 4),
    (-3, 4),
    (4, 3),
    (-4, 3),
    (5, 0),
    (1, 5),
    (-1, 5),
    (5, 1),
    (-5, 1),
    (2, 5),
    (-2, 5),
    (5, 2),
    (-5, 2),
    (4, 4),
    (-4, 4),
    (3, 5),
    (-3, 5),
    (5, 3),
    (-5, 3),
    (0, 6),
    (6, 0),
    (1, 6),
    (-1, 6),
    (6, 1),
    (-6, 1),
    (2, 6),
    (-2, 6),
    (6, 2),
    (-6, 2),
    (4, 5),
    (-4, 5),
    (5, 4),
    (-5, 4),
    (3, 6),
    (-3, 6),
    (6, 3),
    (-6, 3),
    (0, 7),
    (7, 0),
    (1, 7),
    (-1, 7),
    (5, 5),
    (-5, 5),
    (7, 1),
    (-7, 1),
    (4, 6),
    (-4, 6),
    (6, 4),
    (-6, 4),
    (2, 7),
    (-2, 7),
    (7, 2),
    (-7, 2),
    (3, 7),
    (-3, 7),
    (7, 3),
    (-7, 3),
    (5, 6),
    (-5, 6),
    (6, 5),
    (-6, 5),
    (8, 0),
    (4, 7),
    (-4, 7),
    (7, 4),
    (-7, 4),
    (8, 1),
    (8, 2),
    (6, 6),
    (-6, 6),
    (8, 3),
    (5, 7),
    (-5, 7),
    (7, 5),
    (-7, 5),
    (8, 4),
    (6, 7),
    (-6, 7),
    (7, 6),
    (-7, 6),
    (8, 5),
    (7, 7),
    (-7, 7),
    (8, 6),
    (8, 7),
];

/// The `LZ77Params` bundle of 18181-1 Table C.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Lz77Params {
    /// Whether back-references are in use.
    pub enabled: bool,
    /// First token value that denotes a back-reference rather than a value.
    pub min_symbol: u32,
    /// Constant added to every decoded copy length.
    pub min_length: u32,
}

impl Lz77Params {
    /// Reads the bundle (18181-1 Table C.1).
    ///
    /// # Errors
    ///
    /// A bitstream error if the input is exhausted.
    pub fn read(reader: &mut BitReader<'_>) -> Result<Self> {
        let enabled = reader.read_bool()?;
        if !enabled {
            return Ok(Self::default());
        }
        Ok(Self {
            enabled,
            min_symbol: read_u32(reader, &MIN_SYMBOL)?,
            min_length: read_u32(reader, &MIN_LENGTH)?,
        })
    }
}

/// The circular window of previously decoded symbols (18181-1 C.1, C.3.3).
///
/// Allocated only when `lz77.enabled`, as the clause note permits.
#[derive(Debug)]
pub struct Lz77Window {
    /// The last [`WINDOW_SIZE`] symbols, indexed modulo the window size.
    window: Vec<u32>,
    /// Total symbols emitted so far; the clause's `num_decoded`.
    num_decoded: u64,
    /// Symbols still to be replayed from the window.
    num_to_copy: u32,
    /// Read cursor for an in-progress copy.
    copy_pos: u64,
}

impl Lz77Window {
    /// Allocates a zero-filled window, charging it to `guard` first.
    ///
    /// # Errors
    ///
    /// A limit error if the 4 MiB window would exceed the allocation budget.
    pub fn new(guard: &mut AllocGuard) -> Result<Self> {
        guard.charge(WINDOW_SIZE as u64 * 4)?;
        Ok(Self {
            window: vec![0u32; WINDOW_SIZE],
            num_decoded: 0,
            num_to_copy: 0,
            copy_pos: 0,
        })
    }

    /// Number of symbols emitted so far.
    #[must_use]
    pub const fn num_decoded(&self) -> u64 {
        self.num_decoded
    }

    /// Whether a copy is in progress.
    #[must_use]
    pub const fn copying(&self) -> bool {
        self.num_to_copy > 0
    }

    /// Records a freshly decoded symbol, per the tail of
    /// `DecodeHybridVarLenUint`.
    pub fn push(&mut self, value: u32) {
        let index = (self.num_decoded & WINDOW_MASK) as usize;
        if let Some(slot) = self.window.get_mut(index) {
            *slot = value;
        }
        self.num_decoded = self.num_decoded.wrapping_add(1);
    }

    /// Takes the next symbol of an in-progress copy and records it again.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Malformed`](crate::EntropyError::Malformed) if no copy
    /// is in progress.
    pub fn next_copied(&mut self) -> Result<u32> {
        if self.num_to_copy == 0 {
            return Err(malformed!("C.3.3: no LZ77 copy is in progress"));
        }
        let index = (self.copy_pos & WINDOW_MASK) as usize;
        let value = self
            .window
            .get(index)
            .copied()
            .ok_or_else(|| malformed!("C.3.3: window index {index} out of range"))?;
        self.copy_pos = self.copy_pos.wrapping_add(1);
        self.num_to_copy -= 1;
        self.push(value);
        Ok(value)
    }

    /// Starts a copy of `length` symbols from `distance` positions back.
    ///
    /// `distance` is clamped as C.3.3 requires: to the number of symbols
    /// decoded so far and to the window size.
    pub fn start_copy(&mut self, length: u32, distance: u64) {
        let distance = distance.min(self.num_decoded).min(WINDOW_SIZE as u64);
        self.num_to_copy = length;
        self.copy_pos = self.num_decoded - distance;
    }
}

/// Applies the distance transform of 18181-1 C.3.3.
///
/// With `dist_multiplier == 0` the distance is simply incremented. Otherwise
/// values below 120 index [`SPECIAL_DISTANCES`] to express a two-dimensional
/// offset, and larger values are shifted down past that table.
///
/// # Errors
///
/// [`EntropyError::Malformed`](crate::EntropyError::Malformed) if the raw
/// distance is out of range for the transform.
pub fn resolve_distance(raw: u32, dist_multiplier: u32) -> Result<u64> {
    if dist_multiplier == 0 {
        return Ok(u64::from(raw) + 1);
    }
    if raw < 120 {
        let (dx, dy) = *SPECIAL_DISTANCES
            .get(raw as usize)
            .ok_or_else(|| malformed!("C.3.3: special distance {raw} out of range"))?;
        let distance = i64::from(dx) + i64::from(dist_multiplier) * i64::from(dy);
        return Ok(u64::try_from(distance.max(1)).unwrap_or(1));
    }
    Ok(u64::from(raw - 119))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jpxl_core::limits::Limits;

    /// Structural validation of the reconstructed table: the entries are
    /// ordered by increasing Euclidean radius, so a mis-OCRed digit anywhere
    /// would show up as an inversion.
    #[test]
    fn special_distances_are_ordered_by_radius() {
        let mut previous = 0i32;
        for (i, &(dx, dy)) in SPECIAL_DISTANCES.iter().enumerate() {
            let radius = dx * dx + dy * dy;
            assert!(
                radius >= previous,
                "entry {i} = ({dx}, {dy}) has radius {radius} below the previous {previous}"
            );
            previous = radius;
        }
    }

    /// The vertical component is a row count and is never negative; the
    /// horizontal component is a signed column offset.
    #[test]
    fn special_distance_components_have_the_expected_signs() {
        assert_eq!(SPECIAL_DISTANCES.len(), 120);
        for &(dx, dy) in &SPECIAL_DISTANCES {
            assert!(dy >= 0, "row offset ({dx}, {dy}) must not be negative");
            assert!((-8..=8).contains(&dx));
            assert!((0..=8).contains(&dy));
        }
        assert_eq!(
            SPECIAL_DISTANCES[0],
            (0, 1),
            "nearest neighbour is one row up"
        );
        assert_eq!(SPECIAL_DISTANCES[1], (1, 0), "then the previous sample");
    }

    #[test]
    fn distance_without_multiplier_is_one_based() {
        assert_eq!(resolve_distance(0, 0).expect("ok"), 1);
        assert_eq!(resolve_distance(41, 0).expect("ok"), 42);
    }

    #[test]
    fn distance_with_multiplier_uses_the_special_table() {
        // Entry 0 is (0, 1): one row back, i.e. exactly dist_multiplier.
        assert_eq!(resolve_distance(0, 64).expect("ok"), 64);
        // Entry 1 is (1, 0): the immediately preceding sample.
        assert_eq!(resolve_distance(1, 64).expect("ok"), 1);
        // Entry 3 is (-1, 1): one row back and one column forward.
        assert_eq!(resolve_distance(3, 64).expect("ok"), 63);
        // Values past the table are shifted down by 119.
        assert_eq!(resolve_distance(120, 64).expect("ok"), 1);
        assert_eq!(resolve_distance(200, 64).expect("ok"), 81);
    }

    #[test]
    fn distance_is_clamped_to_at_least_one() {
        // Entry 41 is (-1, 5); with a multiplier of 0 the clause would give a
        // negative distance, so the clamp to 1 applies. Multiplier 1 gives 4.
        assert_eq!(resolve_distance(41, 1).expect("ok"), 4);
        // A large negative dx against a small multiplier clamps to 1.
        // Entry 79 is (-7, 1): -7 + 1 * 1 = -6 -> 1.
        assert_eq!(SPECIAL_DISTANCES[79], (-7, 1));
        assert_eq!(resolve_distance(79, 1).expect("ok"), 1);
    }

    #[test]
    fn overlapping_copy_repeats_the_recent_pattern() {
        // The classic LZ77 property: a copy whose length exceeds its distance
        // replays the bytes it is itself producing.
        let mut w = Lz77Window::new(&mut AllocGuard::new(&Limits::relaxed())).expect("window");
        w.push(7);
        w.push(9);
        assert_eq!(w.num_decoded(), 2);
        // Distance 2, length 5 -> 7, 9, 7, 9, 7
        w.start_copy(5, 2);
        let got: Vec<u32> = (0..5)
            .map(|_| w.next_copied().expect("copy in progress"))
            .collect();
        assert_eq!(got, vec![7, 9, 7, 9, 7]);
        assert!(!w.copying());
        assert_eq!(w.num_decoded(), 7);
        assert!(w.next_copied().is_err(), "copy is exhausted");
    }

    #[test]
    fn distance_one_copy_repeats_a_single_symbol() {
        let mut w = Lz77Window::new(&mut AllocGuard::new(&Limits::relaxed())).expect("window");
        w.push(42);
        w.start_copy(4, 1);
        let got: Vec<u32> = (0..4)
            .map(|_| w.next_copied().expect("copy in progress"))
            .collect();
        assert_eq!(got, vec![42, 42, 42, 42]);
    }

    #[test]
    fn copy_from_an_empty_window_yields_zeroes() {
        // C.3.3 clamps the distance to num_decoded, which is 0 here, so the
        // copy reads the zero-initialized window rather than failing.
        let mut w = Lz77Window::new(&mut AllocGuard::new(&Limits::relaxed())).expect("window");
        w.start_copy(3, 5);
        for _ in 0..3 {
            assert_eq!(w.next_copied().expect("copy"), 0);
        }
    }

    #[test]
    fn disabled_params_read_a_single_bit() {
        let data = [0x00u8];
        let mut r = BitReader::new(&data);
        let params = Lz77Params::read(&mut r).expect("params");
        assert!(!params.enabled);
        assert_eq!(r.total_bits_read(), 1);
    }

    #[test]
    fn enabled_params_read_both_u32_fields() {
        // b0 = 1 (enabled)
        // min_symbol selector u(2) = 0 -> 224   (b1, b2 = 0, 0)
        // min_length selector u(2) = 1 -> 4     (b3, b4 = 1, 0)
        // byte = b0 | b3 = 1 + 8 = 9
        let data = [0b0000_1001u8];
        let mut r = BitReader::new(&data);
        let params = Lz77Params::read(&mut r).expect("params");
        assert_eq!(
            params,
            Lz77Params {
                enabled: true,
                min_symbol: 224,
                min_length: 4
            }
        );
        assert_eq!(r.total_bits_read(), 5);
    }
}
