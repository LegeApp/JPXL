//! Asymmetric numeral systems: distributions, alias mapping, decoder state.
//!
//! ISO/IEC 18181-1 C.2.5 (reading a probability distribution), C.2.6 (building
//! the alias mapping) and C.3.2 (the rANS state machine).
//!
//! Probabilities are 12-bit: a distribution is an array of `1 <<
//! log_alphabet_size` counts that sum to exactly `1 << 12`. The alias mapping
//! turns the 12 low bits of the decoder state into a (symbol, offset) pair in
//! constant time by splitting the 4096-slot probability space into
//! `1 << log_alphabet_size` equal buckets, each shared by at most two symbols.
//!
//! Every path here is **bit-exact** (see `docs/PLAN.md`).

use jpxl_bitstream::BitReader;
use jpxl_core::limits::AllocGuard;

use crate::error::{Result, malformed};

/// Total probability mass of a distribution: `1 << 12` (18181-1 C.2.5).
pub const PROBABILITY_TOTAL: u32 = 1 << 12;

/// The state value an ANS stream must hold once fully consumed (18181-1 C.3.2).
pub const FINAL_ANS_STATE: u32 = 0x0013_0000;

/// The fixed prefix code used for `logcounts` in 18181-1 C.2.5.
///
/// The clause prints each code in stream order and notes that the bits are
/// parsed right to left, so the value read most-significant-bit-first is the
/// printed string reversed. Entries are `(bit length, MSB-first value)` indexed
/// by the number they encode. This code is complete but *not* canonical, so it
/// is transcribed rather than derived.
///
/// Reversal check, printed -> MSB-first: `0`: 10001 -> 10001, `1`: 1011 ->
/// 1101, `2`: 1111 -> 1111, `3`: 0011 -> 1100, `4`: 1001 -> 1001, `5`: 0111 ->
/// 1110, `6`: 100 -> 001, `7`: 010 -> 010, `8`: 101 -> 101, `9`: 110 -> 011,
/// `10`: 000 -> 000, `11`: 100001 -> 100001, `12`: 0000001 -> 1000000,
/// `13`: 1000001 -> 1000001.
const LOGCOUNT_CODE: [(u32, u32); 14] = [
    (5, 0b10001),
    (4, 0b1101),
    (4, 0b1111),
    (4, 0b1100),
    (4, 0b1001),
    (4, 0b1110),
    (3, 0b001),
    (3, 0b010),
    (3, 0b101),
    (3, 0b011),
    (3, 0b000),
    (6, 0b100001),
    (7, 0b1000000),
    (7, 0b1000001),
];

/// Longest code in [`LOGCOUNT_CODE`].
const LOGCOUNT_MAX_BITS: u32 = 7;

/// Reads one `logcounts` value using the fixed code of 18181-1 C.2.5.
fn read_logcount(reader: &mut BitReader<'_>) -> Result<u32> {
    let mut code = 0u32;
    for len in 1..=LOGCOUNT_MAX_BITS {
        code = (code << 1) | reader.read_bits(1)?;
        for (number, &(code_len, code_value)) in LOGCOUNT_CODE.iter().enumerate() {
            if code_len == len && code_value == code {
                return u32::try_from(number)
                    .map_err(|_| malformed!("C.2.5: logcount index out of range"));
            }
        }
    }
    Err(malformed!(
        "C.2.5: no logcount code matched within {LOGCOUNT_MAX_BITS} bits"
    ))
}

/// The `U8()` helper of 18181-1 C.2.5: a value in `[0, 256)` in 1 to 11 bits.
fn read_u8(reader: &mut BitReader<'_>) -> Result<u32> {
    if reader.read_bits(1)? == 0 {
        return Ok(0);
    }
    let n = reader.read_bits(3)?;
    Ok(reader.read_bits(n)? + (1 << n))
}

/// A decoded ANS probability distribution with its alias mapping.
///
/// Built by [`AnsDistribution::read`], which performs C.2.5 followed by C.2.6.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnsDistribution {
    /// `12 - log_alphabet_size`.
    log_bucket_size: u32,
    /// `1 << log_bucket_size`.
    bucket_size: u32,
    /// Per-symbol probabilities; `1 << log_alphabet_size` entries summing to
    /// [`PROBABILITY_TOTAL`].
    probabilities: Vec<u32>,
    /// Alias table: the second symbol sharing each bucket.
    symbols: Vec<u32>,
    /// Alias table: offset added for the aliased half of each bucket.
    offsets: Vec<u32>,
    /// Alias table: split point within each bucket.
    cutoffs: Vec<u32>,
}

impl AnsDistribution {
    /// Reads a distribution (18181-1 C.2.5) and builds its alias mapping
    /// (C.2.6).
    ///
    /// # Errors
    ///
    /// [`EntropyError::Malformed`](crate::EntropyError::Malformed) if the
    /// counts do not sum to [`PROBABILITY_TOTAL`], if a symbol index falls
    /// outside the table, or if any other clause invariant fails.
    pub fn read(
        reader: &mut BitReader<'_>,
        log_alphabet_size: u32,
        guard: &mut AllocGuard,
    ) -> Result<Self> {
        let (probabilities, alphabet_size) = read_probabilities(reader, log_alphabet_size, guard)?;
        Self::from_probabilities(probabilities, alphabet_size, log_alphabet_size)
    }

    /// Builds the alias mapping of 18181-1 C.2.6 over an existing
    /// distribution.
    ///
    /// `probabilities` must have `1 << log_alphabet_size` entries summing to
    /// [`PROBABILITY_TOTAL`]; `alphabet_size` is the count of leading entries
    /// that may be nonzero, as established by C.2.5.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Malformed`](crate::EntropyError::Malformed) if those
    /// preconditions do not hold or the construction produces an
    /// inconsistent table.
    pub fn from_probabilities(
        probabilities: Vec<u32>,
        alphabet_size: usize,
        log_alphabet_size: u32,
    ) -> Result<Self> {
        if log_alphabet_size > 12 {
            return Err(malformed!(
                "C.2.6: log_alphabet_size {log_alphabet_size} exceeds 12"
            ));
        }
        let table_size = 1usize << log_alphabet_size;
        if probabilities.len() != table_size {
            return Err(malformed!(
                "C.2.6: distribution has {} entries, expected {table_size}",
                probabilities.len()
            ));
        }
        if alphabet_size > table_size {
            return Err(malformed!(
                "C.2.5: alphabet size {alphabet_size} exceeds the table size {table_size}"
            ));
        }
        let total: u32 = probabilities.iter().copied().sum();
        if total != PROBABILITY_TOTAL {
            return Err(malformed!(
                "C.2.5: probabilities sum to {total}, expected {PROBABILITY_TOTAL}"
            ));
        }

        let log_bucket_size = 12 - log_alphabet_size;
        let bucket_size = 1u32 << log_bucket_size;

        let mut symbols = vec![0u32; table_size];
        let mut offsets = vec![0i64; table_size];
        let mut cutoffs = vec![0i64; table_size];

        // C.2.6 single-symbol shortcut: one symbol owns the whole table.
        let mut nonzero = probabilities.iter().filter(|&&p| p != 0);
        let first_nonzero = nonzero.next();
        if let Some(_) = first_nonzero
            && nonzero.next().is_none()
        {
            let s = probabilities
                .iter()
                .position(|&p| p != 0)
                .ok_or_else(|| malformed!("C.2.6: no nonzero probability"))?;
            let s = u32::try_from(s).map_err(|_| malformed!("C.2.6: symbol out of range"))?;
            for (i, (sym, off)) in symbols.iter_mut().zip(offsets.iter_mut()).enumerate() {
                *sym = s;
                *off = i64::from(bucket_size) * i as i64;
            }
            return Ok(Self {
                log_bucket_size,
                bucket_size,
                probabilities,
                symbols,
                offsets: to_u32_vec(offsets)?,
                cutoffs: to_u32_vec(cutoffs)?,
            });
        }

        let mut overfull: Vec<usize> = Vec::new();
        let mut underfull: Vec<usize> = Vec::new();
        let bucket = i64::from(bucket_size);

        for i in 0..alphabet_size {
            let p = i64::from(
                *probabilities
                    .get(i)
                    .ok_or_else(|| malformed!("C.2.6: probability index {i} out of range"))?,
            );
            set(&mut cutoffs, i, p)?;
            set(&mut symbols, i, u32::try_from(i).unwrap_or(u32::MAX))?;
            if p > bucket {
                overfull.push(i);
            } else if p < bucket {
                underfull.push(i);
            }
        }
        for i in alphabet_size..table_size {
            set(&mut cutoffs, i, 0)?;
            underfull.push(i);
        }

        while let Some(o) = overfull.pop() {
            let u = underfull.pop().ok_or_else(|| {
                malformed!("C.2.6: overfull bucket {o} with no underfull bucket to pair with")
            })?;
            let by = bucket - get(&cutoffs, u)?;
            let new_o = get(&cutoffs, o)? - by;
            set(&mut cutoffs, o, new_o)?;
            // The LaTeX transcription renders this as `symbols[u] = 0`; the
            // markdown OCR has `symbols[u] = o`, which is the only reading
            // consistent with the alias construction (bucket `u` is completed
            // with the surplus of symbol `o`).
            set(
                &mut symbols,
                u,
                u32::try_from(o).map_err(|_| malformed!("C.2.6: symbol {o} out of range"))?,
            )?;
            set(&mut offsets, u, new_o)?;
            if new_o < bucket {
                underfull.push(o);
            } else if new_o > bucket {
                overfull.push(o);
            }
        }
        if !underfull.is_empty() {
            return Err(malformed!(
                "C.2.6: {} underfull buckets remain after balancing",
                underfull.len()
            ));
        }

        for i in 0..table_size {
            if get(&cutoffs, i)? == bucket {
                set(&mut symbols, i, u32::try_from(i).unwrap_or(u32::MAX))?;
                set(&mut offsets, i, 0)?;
                set(&mut cutoffs, i, 0)?;
            } else {
                let v = get(&offsets, i)? - get(&cutoffs, i)?;
                set(&mut offsets, i, v)?;
            }
        }

        Ok(Self {
            log_bucket_size,
            bucket_size,
            probabilities,
            symbols,
            offsets: to_u32_vec(offsets)?,
            cutoffs: to_u32_vec(cutoffs)?,
        })
    }

    /// Number of entries in the alias table, i.e. `1 << log_alphabet_size`.
    #[must_use]
    pub fn table_size(&self) -> usize {
        self.probabilities.len()
    }

    /// Probability mass of `symbol`; out-of-range indices read as 0, as
    /// 18181-1 C.2.5 requires.
    #[must_use]
    pub fn probability(&self, symbol: u32) -> u32 {
        self.probabilities
            .get(symbol as usize)
            .copied()
            .unwrap_or(0)
    }

    /// `AliasMapping(x)` of 18181-1 C.2.6: maps a 12-bit slot to its symbol
    /// and the offset within that symbol's mass.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Malformed`](crate::EntropyError::Malformed) if `x` is
    /// outside `[0, 1 << 12)`.
    pub fn alias_mapping(&self, x: u32) -> Result<(u32, u32)> {
        let i = (x >> self.log_bucket_size) as usize;
        let pos = x & (self.bucket_size - 1);
        let cutoff = *self
            .cutoffs
            .get(i)
            .ok_or_else(|| malformed!("C.2.6: alias index {i} out of range"))?;
        if pos >= cutoff {
            let symbol = *self
                .symbols
                .get(i)
                .ok_or_else(|| malformed!("C.2.6: alias index {i} out of range"))?;
            let offset = self
                .offsets
                .get(i)
                .copied()
                .ok_or_else(|| malformed!("C.2.6: alias index {i} out of range"))?
                + pos;
            Ok((symbol, offset))
        } else {
            let symbol =
                u32::try_from(i).map_err(|_| malformed!("C.2.6: alias index {i} out of range"))?;
            Ok((symbol, pos))
        }
    }
}

/// Reads element `i`, reporting an out-of-range index rather than panicking.
fn get(v: &[i64], i: usize) -> Result<i64> {
    v.get(i)
        .copied()
        .ok_or_else(|| malformed!("C.2.6: index {i} out of range"))
}

/// Writes element `i`, reporting an out-of-range index rather than panicking.
fn set<T>(v: &mut [T], i: usize, value: T) -> Result<()> {
    *v.get_mut(i)
        .ok_or_else(|| malformed!("C.2.6: index {i} out of range"))? = value;
    Ok(())
}

/// Narrows a signed working vector, rejecting the negative values that a
/// malformed distribution could otherwise smuggle through.
fn to_u32_vec(v: Vec<i64>) -> Result<Vec<u32>> {
    v.into_iter()
        .map(|x| {
            u32::try_from(x).map_err(|_| malformed!("C.2.6: alias table entry {x} is out of range"))
        })
        .collect()
}

/// Reads the raw probability array of 18181-1 C.2.5.
///
/// Returns the `1 << log_alphabet_size` probabilities and the alphabet size
/// the clause established for them.
fn read_probabilities(
    reader: &mut BitReader<'_>,
    log_alphabet_size: u32,
    guard: &mut AllocGuard,
) -> Result<(Vec<u32>, usize)> {
    if log_alphabet_size > 12 {
        return Err(malformed!(
            "C.2.5: log_alphabet_size {log_alphabet_size} exceeds 12"
        ));
    }
    let table_size = 1usize << log_alphabet_size;
    guard.charge(table_size as u64 * 4)?;
    let mut d = vec![0u32; table_size];

    let bounded = |v: u32| -> Result<usize> {
        let v = v as usize;
        if v >= table_size {
            return Err(malformed!(
                "C.2.5: symbol {v} is outside the table of size {table_size}"
            ));
        }
        Ok(v)
    };

    // Two shortcut encodings: one or two symbols carrying the whole mass.
    if reader.read_bool()? {
        if reader.read_bool()? {
            let v1 = bounded(read_u8(reader)?)?;
            let v2 = bounded(read_u8(reader)?)?;
            if v1 == v2 {
                return Err(malformed!("C.2.5: the two-symbol form requires v1 != v2"));
            }
            let p = reader.read_bits(12)?;
            set_u32(&mut d, v1, p)?;
            set_u32(&mut d, v2, PROBABILITY_TOTAL - p)?;
            return Ok((d, 1 + v1.max(v2)));
        }
        let x = bounded(read_u8(reader)?)?;
        set_u32(&mut d, x, PROBABILITY_TOTAL)?;
        return Ok((d, 1 + x));
    }

    // Flat distribution over the first `alphabet_size` symbols.
    if reader.read_bool()? {
        let alphabet_size = read_u8(reader)? as usize + 1;
        if alphabet_size > table_size {
            return Err(malformed!(
                "C.2.5: alphabet size {alphabet_size} exceeds the table size {table_size}"
            ));
        }
        let n = u32::try_from(alphabet_size)
            .map_err(|_| malformed!("C.2.5: alphabet size {alphabet_size} out of range"))?;
        let share = PROBABILITY_TOTAL / n;
        let remainder = PROBABILITY_TOTAL % n;
        for i in 0..alphabet_size {
            let extra = u32::try_from(i).is_ok_and(|i| i < remainder);
            set_u32(&mut d, i, share + u32::from(extra))?;
        }
        return Ok((d, alphabet_size));
    }

    // General form: per-symbol logarithmic counts plus refinement bits.
    let mut len = 0u32;
    while len < 3 {
        if reader.read_bool()? {
            len += 1;
        } else {
            break;
        }
    }
    let shift = i32::try_from(reader.read_bits(len)? + (1 << len) - 1)
        .map_err(|_| malformed!("C.2.5: shift out of range"))?;
    if shift > 13 {
        return Err(malformed!("C.2.5: shift {shift} exceeds 13"));
    }
    let alphabet_size = read_u8(reader)? as usize + 3;
    if alphabet_size > table_size {
        return Err(malformed!(
            "C.2.5: alphabet size {alphabet_size} exceeds the table size {table_size}"
        ));
    }

    guard.charge(alphabet_size as u64 * 8)?;
    let mut logcounts = vec![0u32; alphabet_size];
    let mut same = vec![0u32; alphabet_size];
    let mut omit_log: i32 = -1;
    let mut omit_pos: i32 = -1;

    let mut i = 0usize;
    while i < alphabet_size {
        let lc = read_logcount(reader)?;
        set_u32(&mut logcounts, i, lc)?;
        if lc == 13 {
            let rle = read_u8(reader)?;
            set_u32(&mut same, i, rle + 5)?;
            // The clause advances by `rle + 3` and then the loop increments,
            // so the next logcount lands `rle + 4` entries along.
            i = i
                .checked_add(rle as usize + 4)
                .ok_or_else(|| malformed!("C.2.5: run-length overflow"))?;
            continue;
        }
        let lc_signed =
            i32::try_from(lc).map_err(|_| malformed!("C.2.5: logcount out of range"))?;
        if lc_signed > omit_log {
            omit_log = lc_signed;
            omit_pos = i32::try_from(i).map_err(|_| malformed!("C.2.5: position out of range"))?;
        }
        i += 1;
    }

    let omit_pos = usize::try_from(omit_pos)
        .map_err(|_| malformed!("C.2.5: every entry is a repeat, so omit_pos is undefined"))?;

    let mut total: u32 = 0;
    let mut numsame: u32 = 0;
    let mut prev: u32 = 0;
    for i in 0..alphabet_size {
        if get_u32(&same, i)? != 0 {
            numsame = get_u32(&same, i)? - 1;
            prev = if i > 0 { get_u32(&d, i - 1)? } else { 0 };
        }
        if numsame > 0 {
            set_u32(&mut d, i, prev)?;
            numsame -= 1;
        } else {
            let code = get_u32(&logcounts, i)?;
            if i == omit_pos || code == 0 {
                continue;
            } else if code == 1 {
                set_u32(&mut d, i, 1)?;
            } else {
                let code_i32 =
                    i32::try_from(code).map_err(|_| malformed!("C.2.5: logcount overflow"))?;
                let bitcount = (shift - ((12 - code_i32 + 1) >> 1))
                    .max(0)
                    .min(code_i32 - 1);
                let bitcount = u32::try_from(bitcount)
                    .map_err(|_| malformed!("C.2.5: bit count out of range"))?;
                let extra = reader.read_bits(bitcount)?;
                let value = (1u32 << (code - 1)) + (extra << (code - 1 - bitcount));
                set_u32(&mut d, i, value)?;
            }
        }
        total = total
            .checked_add(get_u32(&d, i)?)
            .ok_or_else(|| malformed!("C.2.5: probability total overflows"))?;
    }

    if total > PROBABILITY_TOTAL {
        return Err(malformed!(
            "C.2.5: probabilities total {total}, over the budget of {PROBABILITY_TOTAL}"
        ));
    }
    set_u32(&mut d, omit_pos, PROBABILITY_TOTAL - total)?;
    Ok((d, alphabet_size))
}

fn get_u32(v: &[u32], i: usize) -> Result<u32> {
    v.get(i)
        .copied()
        .ok_or_else(|| malformed!("C.2.5: index {i} out of range"))
}

fn set_u32(v: &mut [u32], i: usize, value: u32) -> Result<()> {
    *v.get_mut(i)
        .ok_or_else(|| malformed!("C.2.5: index {i} out of range"))? = value;
    Ok(())
}

/// The rANS decoder state of 18181-1 C.3.2.
///
/// A single state is shared by every context in a stream. It is seeded with a
/// raw `u(32)` when the stream opens and must equal [`FINAL_ANS_STATE`] once
/// the stream has been fully consumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnsState {
    state: u32,
}

impl AnsState {
    /// Seeds the state from the stream (18181-1 C.3.2).
    ///
    /// # Errors
    ///
    /// A bitstream error if fewer than 32 bits remain.
    pub fn init(reader: &mut BitReader<'_>) -> Result<Self> {
        Ok(Self {
            state: reader.read_bits(32)?,
        })
    }

    /// Builds a state with an explicit value, for tests and for encoders.
    #[must_use]
    pub const fn from_raw(state: u32) -> Self {
        Self { state }
    }

    /// The current 32-bit state.
    #[must_use]
    pub const fn raw(&self) -> u32 {
        self.state
    }

    /// Whether the state matches [`FINAL_ANS_STATE`].
    #[must_use]
    pub const fn is_final(&self) -> bool {
        self.state == FINAL_ANS_STATE
    }

    /// Decodes one symbol from `dist`, renormalizing as 18181-1 C.3.2
    /// specifies.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Malformed`](crate::EntropyError::Malformed) if the
    /// state update leaves the 32-bit range, or a bitstream error if the
    /// renormalization has no bits to read.
    pub fn decode(&mut self, reader: &mut BitReader<'_>, dist: &AnsDistribution) -> Result<u32> {
        let index = self.state & 0xFFF;
        let (symbol, offset) = dist.alias_mapping(index)?;
        let next =
            u64::from(dist.probability(symbol)) * u64::from(self.state >> 12) + u64::from(offset);
        let mut next = u32::try_from(next)
            .map_err(|_| malformed!("C.3.2: ANS state update {next} leaves the 32-bit range"))?;
        if next < (1 << 16) {
            next = (next << 16) | reader.read_bits(16)?;
        }
        self.state = next;
        Ok(symbol)
    }
}

#[cfg(test)]
// In test code an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves. The lints
// exist for the decode paths above.
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use jpxl_core::limits::Limits;

    fn guard() -> AllocGuard {
        AllocGuard::new(&Limits::relaxed())
    }

    #[test]
    fn logcount_code_is_a_complete_prefix_code() {
        // Kraft sum over the 14 codes must be exactly 1.
        let denom: u64 = 1 << LOGCOUNT_MAX_BITS;
        let sum: u64 = LOGCOUNT_CODE
            .iter()
            .map(|&(len, _)| 1u64 << (LOGCOUNT_MAX_BITS - len))
            .sum();
        assert_eq!(sum, denom, "logcount code must be complete");

        // And prefix-free: no code is a prefix of another.
        for &(len_a, val_a) in &LOGCOUNT_CODE {
            for &(len_b, val_b) in &LOGCOUNT_CODE {
                if (len_a, val_a) == (len_b, val_b) || len_a >= len_b {
                    continue;
                }
                assert_ne!(
                    val_a,
                    val_b >> (len_b - len_a),
                    "code {val_a:b} is a prefix of {val_b:b}"
                );
            }
        }
    }

    #[test]
    fn alias_table_covers_each_symbol_exactly_its_probability() {
        // The defining invariant of C.2.6: over all 4096 slots, symbol s must
        // appear exactly D[s] times, and the offsets within each symbol must
        // be a permutation of 0..D[s].
        let log_alphabet_size = 5;
        let table_size = 1usize << log_alphabet_size;
        let mut probs = vec![0u32; table_size];
        // A deliberately lumpy distribution: some buckets overfull, some empty.
        probs[0] = 2000;
        probs[1] = 1000;
        probs[2] = 500;
        probs[3] = 400;
        probs[4] = 100;
        probs[5] = 96;
        let dist = AnsDistribution::from_probabilities(probs.clone(), 6, log_alphabet_size)
            .expect("valid distribution");

        let mut seen: Vec<Vec<u32>> = vec![Vec::new(); table_size];
        for x in 0..PROBABILITY_TOTAL {
            let (symbol, offset) = dist.alias_mapping(x).expect("in range");
            seen[symbol as usize].push(offset);
        }
        for (symbol, offsets) in seen.iter().enumerate() {
            assert_eq!(
                offsets.len() as u32,
                probs[symbol],
                "symbol {symbol} must occupy exactly its probability mass"
            );
            let mut sorted = offsets.clone();
            sorted.sort_unstable();
            let expected: Vec<u32> = (0..probs[symbol]).collect();
            assert_eq!(sorted, expected, "offsets for symbol {symbol}");
        }
    }

    #[test]
    fn single_symbol_distribution_maps_every_slot() {
        let log_alphabet_size = 5;
        let table_size = 1usize << log_alphabet_size;
        let mut probs = vec![0u32; table_size];
        probs[7] = PROBABILITY_TOTAL;
        let dist = AnsDistribution::from_probabilities(probs, 8, log_alphabet_size).expect("valid");
        let mut offsets = Vec::new();
        for x in 0..PROBABILITY_TOTAL {
            let (symbol, offset) = dist.alias_mapping(x).expect("in range");
            assert_eq!(symbol, 7);
            offsets.push(offset);
        }
        offsets.sort_unstable();
        assert_eq!(offsets, (0..PROBABILITY_TOTAL).collect::<Vec<_>>());
    }

    #[test]
    fn distributions_must_sum_to_the_total() {
        let mut probs = vec![0u32; 32];
        probs[0] = 4095;
        assert!(AnsDistribution::from_probabilities(probs, 1, 5).is_err());
    }

    #[test]
    fn u8_helper_covers_its_stated_range() {
        // C.2.5 U8(): a leading 0 bit means the value 0.
        let mut r = BitReader::new(&[0x00]);
        assert_eq!(read_u8(&mut r).expect("zero"), 0);
        assert_eq!(r.total_bits_read(), 1);

        // 1, n=u(3)=0, u(0)=0 -> 0 + (1<<0) = 1.
        // bits: b0=1, b1..b3=0 -> byte = 1
        let mut r = BitReader::new(&[0b0000_0001]);
        assert_eq!(read_u8(&mut r).expect("one"), 1);
        assert_eq!(r.total_bits_read(), 4);

        // 1, n=u(3)=7, u(7)=127 -> 127 + 128 = 255 (the maximum).
        // bits: b0=1, b1..b3=1,1,1 (n=7), b4..b10=1 (127)  -> 11 ones
        let mut r = BitReader::new(&[0xFF, 0x07]);
        assert_eq!(read_u8(&mut r).expect("max"), 255);
        assert_eq!(r.total_bits_read(), 11);
    }

    #[test]
    fn flat_distribution_form_round_trips() {
        // Encoding: Bool()=0, Bool()=1, U8() = 3 -> alphabet_size = 4.
        // U8() for 3: bit 1, n=u(3)=1, u(1)=1 -> 1 + 2 = 3.
        // Bit sequence (in read order): 0, 1, 1, 1,0,0, 1
        //   b0=0 (not the shortcut form)
        //   b1=1 (flat form)
        //   b2=1 (U8 nonzero)
        //   b3,b4,b5 = 1,0,0 -> n = 1
        //   b6 = 1 -> u(1) = 1 -> value 3
        // byte = b1|b2|b3|b6 = 2 + 4 + 8 + 64 = 78
        let mut g = guard();
        let data = [78u8];
        let mut r = BitReader::new(&data);
        let (probs, alphabet_size) = read_probabilities(&mut r, 5, &mut g).expect("flat");
        assert_eq!(alphabet_size, 4);
        assert_eq!(r.total_bits_read(), 7);
        // 4096 / 4 = 1024 exactly, no remainder.
        assert_eq!(&probs[..4], &[1024, 1024, 1024, 1024]);
        assert_eq!(probs[4..].iter().sum::<u32>(), 0);
    }

    #[test]
    fn single_symbol_shortcut_form() {
        // Bool()=1, Bool()=0, U8() = 0 -> D[0] = 4096, alphabet_size = 1.
        // bits: b0=1, b1=0, b2=0 -> byte = 1
        let mut g = guard();
        let data = [0b0000_0001u8];
        let mut r = BitReader::new(&data);
        let (probs, alphabet_size) = read_probabilities(&mut r, 5, &mut g).expect("single");
        assert_eq!(alphabet_size, 1);
        assert_eq!(probs[0], PROBABILITY_TOTAL);
        assert_eq!(r.total_bits_read(), 3);
    }

    #[test]
    fn ans_state_renormalizes_below_2_16() {
        let log_alphabet_size = 5;
        let mut probs = vec![0u32; 1 << log_alphabet_size];
        probs[3] = PROBABILITY_TOTAL;
        let dist = AnsDistribution::from_probabilities(probs, 4, log_alphabet_size).expect("ok");

        // With a single symbol of full mass, AliasMapping(x) = (3, x) and the
        // update is state = 4096 * (state >> 12) + (state & 0xFFF) = state.
        let mut state = AnsState::from_raw(0x0001_2345);
        let mut r = BitReader::new(&[]);
        assert_eq!(state.decode(&mut r, &dist).expect("symbol"), 3);
        assert_eq!(
            state.raw(),
            0x0001_2345,
            "identity update for a sole symbol"
        );
        assert_eq!(r.total_bits_read(), 0, "no renormalization above 2^16");

        // Below 2^16 the state pulls in 16 fresh bits.
        let mut state = AnsState::from_raw(0x0000_1234);
        let data = [0xCDu8, 0xAB];
        let mut r = BitReader::new(&data);
        assert_eq!(state.decode(&mut r, &dist).expect("symbol"), 3);
        assert_eq!(state.raw(), 0x1234_ABCD);
        assert_eq!(r.total_bits_read(), 16);
    }
}
