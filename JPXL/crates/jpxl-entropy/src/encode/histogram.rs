//! ANS histogram building and serialization — the write side of 18181-1 C.2.5.
//!
//! A [`Histogram`] is a normalized probability distribution: `1 <<
//! log_alphabet_size` counts summing to exactly `1 << 12`, every symbol that
//! the stream will actually use carrying at least 1.
//!
//! # Choosing an encoding
//!
//! C.2.5 offers four forms, and which is shortest depends entirely on the
//! distribution:
//!
//! | Form | Shape | Cost |
//! | --- | --- | --- |
//! | one symbol | all mass on one symbol | 2 bits + `U8` |
//! | two symbols | mass split between two | 2 bits + 2 `U8` + `u(12)` |
//! | flat | `4096 / n` each, remainder to the front | 2 bits + `U8` |
//! | general | per-symbol logarithmic counts + refinement bits | varies |
//!
//! Rather than reason about the crossovers, [`Histogram::write`] encodes into a
//! scratch writer with each applicable form, keeps the shortest, and writes
//! that one for real. Every form reproduces the same probabilities, so the
//! choice is purely a density decision and cannot change what the decoder sees.
//!
//! # The general form
//!
//! Each symbol's logarithmic count is `ceil(log2(D[i] + 1))`, i.e. the bit
//! width of its probability, and the refinement bits carry the value below the
//! leading one at a granularity fixed by `shift`. The encoder picks the
//! *smallest* `shift` that expresses every probability exactly, because a
//! larger `shift` only buys precision this histogram does not need. The symbol
//! with the largest logarithmic count is omitted — the decoder recovers it as
//! the residue of the 4096 total — and C.2.5's tie-break (strictly greater
//! wins) means that is the *first* symbol attaining the maximum.
//!
//! Run-length compression of equal logarithmic counts (the `logcounts == 13`
//! escape) is **not emitted**; it is a density optimization, and the decoder
//! reads it either way. See the module note in `docs/` and slice 19.
//!
//! Every path here is **bit-exact** (see `docs/PLAN.md`).

use jpxl_bitstream::BitWriter;

use crate::ans::{AnsDistribution, LOGCOUNT_CODE, PROBABILITY_TOTAL};
use crate::error::{Result, encode_error};
use crate::hybrid::bit_width;

/// Largest `shift` C.2.5 permits.
const MAX_SHIFT: u32 = 13;

/// The four encodings C.2.5 defines for a distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    /// One symbol carries the whole mass.
    Single,
    /// Two symbols split the mass.
    TwoSymbol,
    /// `4096 / alphabet_size` each, remainder spread over the front.
    Flat,
    /// Per-symbol logarithmic counts plus refinement bits.
    General,
}

/// Every form, in the order the candidates are tried.
const FORMS: [Form; 4] = [Form::Single, Form::TwoSymbol, Form::Flat, Form::General];

/// A normalized ANS probability distribution ready to be written (18181-1
/// C.2.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Histogram {
    /// `1 << log_alphabet_size` probabilities summing to [`PROBABILITY_TOTAL`].
    probabilities: Vec<u32>,
    /// One past the highest symbol with nonzero probability, which is what
    /// C.2.5 reports as `alphabet_size` for every form this module emits.
    alphabet_size: usize,
    log_alphabet_size: u32,
}

impl Histogram {
    /// Normalizes raw symbol counts into a distribution.
    ///
    /// Every symbol with a nonzero count gets at least one unit of mass, so a
    /// symbol that occurs can always be encoded. Symbols with no occurrences
    /// get none. An all-zero census yields the degenerate distribution that
    /// puts the whole mass on symbol 0.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if
    /// `log_alphabet_size` is out of range or `counts` is longer than the
    /// table.
    pub fn from_counts(counts: &[u64], log_alphabet_size: u32) -> Result<Self> {
        let table_size = table_size(log_alphabet_size)?;
        if counts.len() > table_size {
            return Err(encode_error!(
                "C.2.5: {} counts exceed the table size {table_size}",
                counts.len()
            ));
        }
        Self::from_probabilities(normalize(counts, table_size)?, log_alphabet_size)
    }

    /// Wraps probabilities that already sum to [`PROBABILITY_TOTAL`].
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the array is
    /// the wrong length, does not sum to the total, or is entirely zero.
    pub fn from_probabilities(probabilities: Vec<u32>, log_alphabet_size: u32) -> Result<Self> {
        let table_size = table_size(log_alphabet_size)?;
        if probabilities.len() != table_size {
            return Err(encode_error!(
                "C.2.5: distribution has {} entries, expected {table_size}",
                probabilities.len()
            ));
        }
        let total: u64 = probabilities.iter().map(|&p| u64::from(p)).sum();
        if total != u64::from(PROBABILITY_TOTAL) {
            return Err(encode_error!(
                "C.2.5: probabilities sum to {total}, expected {PROBABILITY_TOTAL}"
            ));
        }
        let alphabet_size = probabilities
            .iter()
            .rposition(|&p| p != 0)
            .ok_or_else(|| encode_error!("C.2.5: distribution has no nonzero probability"))?
            + 1;
        Ok(Self {
            probabilities,
            alphabet_size,
            log_alphabet_size,
        })
    }

    /// The normalized probabilities, one per table slot.
    #[must_use]
    pub fn probabilities(&self) -> &[u32] {
        &self.probabilities
    }

    /// One past the highest symbol carrying mass — the `alphabet_size` C.2.5
    /// will report to the decoder.
    #[must_use]
    pub const fn alphabet_size(&self) -> usize {
        self.alphabet_size
    }

    /// The `log_alphabet_size` this histogram is written at.
    #[must_use]
    pub const fn log_alphabet_size(&self) -> u32 {
        self.log_alphabet_size
    }

    /// Probability mass of `symbol`, 0 outside the table.
    #[must_use]
    pub fn probability(&self, symbol: u32) -> u32 {
        self.probabilities
            .get(symbol as usize)
            .copied()
            .unwrap_or(0)
    }

    /// Builds the alias mapping the ANS coder needs (18181-1 C.2.6).
    ///
    /// The alias table is *shared* with the decoder rather than reimplemented:
    /// it is a table, not control flow, and it is the one part of Annex C that
    /// the corpus has already proved bit-exact.
    /// [`AnsEncodeTable`](super::ans::AnsEncodeTable) re-derives the inverse
    /// mapping from it and independently asserts the bijection, so a corrupted
    /// table cannot pass silently.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Malformed`](crate::EntropyError::Malformed) if the
    /// construction of C.2.6 fails, which a validated histogram cannot cause.
    pub fn distribution(&self) -> Result<AnsDistribution> {
        AnsDistribution::from_probabilities(
            self.probabilities.clone(),
            self.alphabet_size,
            self.log_alphabet_size,
        )
    }

    /// Writes the distribution (18181-1 C.2.5), choosing the shortest legal
    /// form.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if no form can
    /// express this distribution, or a bitstream error.
    pub fn write(&self, w: &mut BitWriter) -> Result<()> {
        let mut best: Option<(Form, u64)> = None;
        for form in FORMS {
            let mut probe = BitWriter::new();
            if self.write_form(&mut probe, form).is_ok() {
                let len = probe.bit_len();
                if best.is_none_or(|(_, best_len)| len < best_len) {
                    best = Some((form, len));
                }
            }
        }
        let (form, _) = best.ok_or_else(|| {
            encode_error!("C.2.5: no distribution encoding can express this histogram")
        })?;
        self.write_form(w, form)
    }

    /// Writes one specific form, or fails if it does not apply.
    fn write_form(&self, w: &mut BitWriter, form: Form) -> Result<()> {
        match form {
            Form::Single => self.write_single(w),
            Form::TwoSymbol => self.write_two_symbol(w),
            Form::Flat => self.write_flat(w),
            Form::General => self.write_general(w),
        }
    }

    /// Indices of the symbols carrying mass.
    fn nonzero(&self) -> Vec<usize> {
        self.probabilities
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| (p != 0).then_some(i))
            .collect()
    }

    fn write_single(&self, w: &mut BitWriter) -> Result<()> {
        let nonzero = self.nonzero();
        let [only] = nonzero[..] else {
            return Err(encode_error!(
                "C.2.5: the one-symbol form needs exactly one nonzero probability"
            ));
        };
        w.write_bool(true);
        w.write_bool(false);
        write_u8(w, only)
    }

    fn write_two_symbol(&self, w: &mut BitWriter) -> Result<()> {
        let nonzero = self.nonzero();
        let [v1, v2] = nonzero[..] else {
            return Err(encode_error!(
                "C.2.5: the two-symbol form needs exactly two nonzero probabilities"
            ));
        };
        let p1 = self.probability(u32::try_from(v1).unwrap_or(u32::MAX));
        if p1 >= PROBABILITY_TOTAL {
            return Err(encode_error!(
                "C.2.5: the two-symbol form needs D[v1] < 4096"
            ));
        }
        w.write_bool(true);
        w.write_bool(true);
        write_u8(w, v1)?;
        write_u8(w, v2)?;
        w.write_bits(12, p1)?;
        Ok(())
    }

    fn write_flat(&self, w: &mut BitWriter) -> Result<()> {
        let n = u32::try_from(self.alphabet_size)
            .map_err(|_| encode_error!("C.2.5: alphabet size out of range"))?;
        let share = PROBABILITY_TOTAL / n;
        let remainder = PROBABILITY_TOTAL % n;
        for (i, &p) in self.probabilities.iter().enumerate() {
            let expected = if i >= self.alphabet_size {
                0
            } else {
                let extra = u32::try_from(i).is_ok_and(|i| i < remainder);
                share + u32::from(extra)
            };
            if p != expected {
                return Err(encode_error!(
                    "C.2.5: the flat form needs D[{i}] == {expected}, not {p}"
                ));
            }
        }
        w.write_bool(false);
        w.write_bool(true);
        write_u8(w, self.alphabet_size - 1)
    }

    fn write_general(&self, w: &mut BitWriter) -> Result<()> {
        // C.2.5 codes the alphabet size as `U8() + 3`.
        if self.alphabet_size < 3 {
            return Err(encode_error!(
                "C.2.5: the general form needs an alphabet size of at least 3"
            ));
        }
        let logcounts = self.logcounts()?;
        let omit_pos = omit_pos(&logcounts)?;
        let shift = self.general_shift(&logcounts, omit_pos)?;

        w.write_bool(false);
        w.write_bool(false);
        write_shift(w, shift)?;
        write_u8(w, self.alphabet_size - 3)?;

        // C.2.5 reads every logarithmic count first, then every refinement
        // field, so the two loops must stay separate.
        for &code in &logcounts {
            write_logcount(w, code)?;
        }
        for (i, &code) in logcounts.iter().enumerate() {
            if i == omit_pos || code < 2 {
                continue;
            }
            let p = self.probability(u32::try_from(i).unwrap_or(u32::MAX));
            let bitcount = refinement_bits(shift, code);
            let granularity = 1u32 << (code - 1 - bitcount);
            let residue = p - (1u32 << (code - 1));
            if !residue.is_multiple_of(granularity) {
                return Err(encode_error!(
                    "C.2.5: D[{i}] = {p} is not representable at shift {shift}"
                ));
            }
            w.write_bits(bitcount, residue / granularity)?;
        }
        Ok(())
    }

    /// The logarithmic count of every symbol below `alphabet_size`.
    fn logcounts(&self) -> Result<Vec<u32>> {
        let head = self
            .probabilities
            .get(..self.alphabet_size)
            .ok_or_else(|| encode_error!("C.2.5: alphabet size exceeds the table"))?;
        head.iter()
            .map(|&p| {
                let code = bit_width(p);
                // 13 is the run-length escape, so a probability of 4096 cannot
                // be written by the general form at all. It is always the sole
                // nonzero probability, which the one-symbol form covers.
                if code >= 13 {
                    return Err(encode_error!(
                        "C.2.5: a probability of {p} collides with the run-length escape"
                    ));
                }
                Ok(code)
            })
            .collect()
    }

    /// The smallest `shift` at which every refinement field is exact.
    fn general_shift(&self, logcounts: &[u32], omit_pos: usize) -> Result<u32> {
        for shift in 0..=MAX_SHIFT {
            let exact = logcounts.iter().enumerate().all(|(i, &code)| {
                if i == omit_pos || code < 2 {
                    return true;
                }
                let bitcount = refinement_bits(shift, code);
                let granularity = 1u32 << (code - 1 - bitcount);
                let p = self.probability(u32::try_from(i).unwrap_or(u32::MAX));
                (p - (1u32 << (code - 1))).is_multiple_of(granularity)
            });
            if exact {
                return Ok(shift);
            }
        }
        Err(encode_error!(
            "C.2.5: no shift in 0..={MAX_SHIFT} expresses this histogram exactly"
        ))
    }
}

/// `bitcount` of C.2.5's refinement field.
fn refinement_bits(shift: u32, code: u32) -> u32 {
    let shift = i64::from(shift);
    let code = i64::from(code);
    let bits = (shift - ((12 - code + 1) >> 1)).clamp(0, code - 1);
    u32::try_from(bits).unwrap_or(0)
}

/// The symbol C.2.5 omits: the *first* one attaining the largest logarithmic
/// count, because the clause updates `omit_pos` only on a strict increase.
fn omit_pos(logcounts: &[u32]) -> Result<usize> {
    let mut best: Option<(usize, u32)> = None;
    for (i, &code) in logcounts.iter().enumerate() {
        if best.is_none_or(|(_, max)| code > max) {
            best = Some((i, code));
        }
    }
    best.map(|(i, _)| i)
        .ok_or_else(|| encode_error!("C.2.5: an empty alphabet has no omitted symbol"))
}

/// Writes `shift` as C.2.5's up-to-three-ones prefix plus `u(len)`.
fn write_shift(w: &mut BitWriter, shift: u32) -> Result<()> {
    for len in 0..=3u32 {
        let base = (1u32 << len) - 1;
        let span = 1u32 << len;
        if shift < base + span || len == 3 {
            if shift < base || shift - base >= span {
                return Err(encode_error!("C.2.5: shift {shift} is out of range"));
            }
            for _ in 0..len {
                w.write_bool(true);
            }
            // The loop of C.2.5 stops either on a zero bit or at len == 3.
            if len < 3 {
                w.write_bool(false);
            }
            return w.write_bits(len, shift - base).map_err(Into::into);
        }
    }
    Err(encode_error!("C.2.5: shift {shift} is out of range"))
}

/// Writes one logarithmic count with the fixed prefix code of C.2.5.
fn write_logcount(w: &mut BitWriter, number: u32) -> Result<()> {
    let &(len, value) = LOGCOUNT_CODE
        .get(number as usize)
        .ok_or_else(|| encode_error!("C.2.5: logcount {number} is outside the fixed code"))?;
    // C.2.4 consumes a prefix code most-significant bit first.
    for shift in (0..len).rev() {
        w.write_bool((value >> shift) & 1 == 1);
    }
    Ok(())
}

/// Writes C.2.5's `U8()`: a value in `[0, 256)` in 1 to 11 bits.
fn write_u8(w: &mut BitWriter, value: usize) -> Result<()> {
    let value = u32::try_from(value)
        .ok()
        .filter(|&v| v < 256)
        .ok_or_else(|| encode_error!("C.2.5: U8 value {value} is out of range"))?;
    if value == 0 {
        w.write_bool(false);
        return Ok(());
    }
    w.write_bool(true);
    let n = bit_width(value) - 1;
    w.write_bits(3, n)?;
    w.write_bits(n, value - (1 << n))?;
    Ok(())
}

/// `1 << log_alphabet_size`, rejecting the widths C.2.5 cannot express.
fn table_size(log_alphabet_size: u32) -> Result<usize> {
    if log_alphabet_size > 12 {
        return Err(encode_error!(
            "C.2.5: log_alphabet_size {log_alphabet_size} exceeds 12"
        ));
    }
    Ok(1usize << log_alphabet_size)
}

/// Turns raw counts into probabilities summing to exactly [`PROBABILITY_TOTAL`].
///
/// Proportional allocation with a floor of 1 for every symbol that occurs,
/// then largest-remainder for the shortfall and a repeated shave of the largest
/// entries for the excess. Deterministic: ties resolve to the lowest index.
fn normalize(counts: &[u64], table_size: usize) -> Result<Vec<u32>> {
    let mut probabilities = vec![0u32; table_size];
    let total: u128 = counts.iter().map(|&c| u128::from(c)).sum();
    if total == 0 {
        set(&mut probabilities, 0, PROBABILITY_TOTAL)?;
        return Ok(probabilities);
    }
    let used = counts.iter().filter(|&&c| c != 0).count();
    if used > table_size || used > PROBABILITY_TOTAL as usize {
        return Err(encode_error!(
            "C.2.5: {used} symbols cannot each hold probability mass"
        ));
    }

    let mut remainders: Vec<(u128, usize)> = Vec::with_capacity(used);
    let mut sum: u32 = 0;
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let scaled = u128::from(count) * u128::from(PROBABILITY_TOTAL);
        let share = u32::try_from(scaled / total)
            .map_err(|_| encode_error!("C.2.5: normalization overflowed"))?;
        let p = share.max(1);
        set(&mut probabilities, i, p)?;
        sum = sum
            .checked_add(p)
            .ok_or_else(|| encode_error!("C.2.5: normalization overflowed"))?;
        remainders.push((scaled % total, i));
    }

    // Shortfall: hand the spare mass to the largest fractional parts first.
    if sum < PROBABILITY_TOTAL {
        remainders.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut cursor = remainders.iter().map(|&(_, i)| i).cycle();
        while sum < PROBABILITY_TOTAL {
            let i = cursor
                .next()
                .ok_or_else(|| encode_error!("C.2.5: no symbol to receive spare mass"))?;
            let p = get(&probabilities, i)?;
            set(&mut probabilities, i, p + 1)?;
            sum += 1;
        }
    }

    // Excess: only the floor of 1 can cause it, so it is bounded by the symbol
    // count. Shave the heaviest entries, never below 1.
    while sum > PROBABILITY_TOTAL {
        let mut heaviest: Option<(u32, usize)> = None;
        for (i, &p) in probabilities.iter().enumerate() {
            if p > 1 && heaviest.is_none_or(|(best, _)| p > best) {
                heaviest = Some((p, i));
            }
        }
        let (p, i) = heaviest
            .ok_or_else(|| encode_error!("C.2.5: cannot fit the alphabet into 4096 units"))?;
        set(&mut probabilities, i, p - 1)?;
        sum -= 1;
    }

    Ok(probabilities)
}

fn get(v: &[u32], i: usize) -> Result<u32> {
    v.get(i)
        .copied()
        .ok_or_else(|| encode_error!("C.2.5: index {i} out of range"))
}

fn set(v: &mut [u32], i: usize, value: u32) -> Result<()> {
    *v.get_mut(i)
        .ok_or_else(|| encode_error!("C.2.5: index {i} out of range"))? = value;
    Ok(())
}

#[cfg(test)]
// Test code: an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves.
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use jpxl_bitstream::BitReader;
    use jpxl_core::limits::{AllocGuard, Limits};

    /// Writes a histogram and reads it back with the decoder, asserting the
    /// probabilities survive exactly. This is the only thing that proves a
    /// form is legal.
    fn round_trip(histogram: &Histogram) -> u64 {
        let mut w = BitWriter::new();
        histogram.write(&mut w).expect("writes");
        let bits = w.bit_len();
        let bytes = w.into_bytes();
        let mut guard = AllocGuard::new(&Limits::relaxed());
        let mut r = BitReader::new(&bytes);
        let read = AnsDistribution::read(&mut r, histogram.log_alphabet_size(), &mut guard)
            .expect("decoder reads it back");
        assert_eq!(
            read,
            histogram.distribution().expect("alias table"),
            "the decoded distribution must equal the encoded one"
        );
        assert_eq!(r.total_bits_read(), bits, "no bits left unread");
        bits
    }

    fn histogram(probabilities: &[u32], log_alphabet_size: u32) -> Histogram {
        let mut probs = vec![0u32; 1 << log_alphabet_size];
        probs[..probabilities.len()].copy_from_slice(probabilities);
        Histogram::from_probabilities(probs, log_alphabet_size).expect("valid")
    }

    #[test]
    fn single_symbol_form_round_trips() {
        for symbol in [0usize, 1, 7, 31] {
            let mut probs = vec![0u32; 32];
            probs[symbol] = PROBABILITY_TOTAL;
            let h = Histogram::from_probabilities(probs, 5).expect("valid");
            round_trip(&h);
        }
    }

    #[test]
    fn two_symbol_form_round_trips() {
        for split in [1u32, 2048, 4095] {
            let mut probs = vec![0u32; 64];
            probs[3] = split;
            probs[40] = PROBABILITY_TOTAL - split;
            let h = Histogram::from_probabilities(probs, 6).expect("valid");
            round_trip(&h);
        }
    }

    #[test]
    fn flat_form_is_chosen_when_it_wins() {
        // Three equal shares: 1366, 1365, 1365 (the remainder goes to the front).
        let h = histogram(&[1366, 1365, 1365], 5);
        let bits = round_trip(&h);
        assert!(bits <= 12, "the flat form should be tiny, got {bits} bits");
    }

    #[test]
    fn general_form_round_trips_at_every_needed_shift() {
        // Probabilities of assorted precisions: exact powers of two need shift
        // 0, a mid-precision value needs more, and an odd value needs 13.
        let cases: [&[u32]; 4] = [
            &[2048, 1024, 1024],
            &[2048, 1536, 512],
            &[2049, 1023, 1024],
            &[1000, 1096, 2000],
        ];
        for probs in cases {
            let h = histogram(probs, 5);
            round_trip(&h);
        }
    }

    #[test]
    fn normalization_gives_every_used_symbol_mass() {
        let counts = vec![1_000_000u64, 1, 0, 3, 0, 0, 0, 7];
        let h = Histogram::from_counts(&counts, 5).expect("normalizes");
        assert_eq!(h.probabilities().iter().sum::<u32>(), PROBABILITY_TOTAL);
        for (i, &c) in counts.iter().enumerate() {
            let p = h.probability(i as u32);
            assert_eq!(c == 0, p == 0, "symbol {i}: count {c}, probability {p}");
        }
        round_trip(&h);
    }

    #[test]
    fn an_empty_census_still_yields_a_legal_histogram() {
        let h = Histogram::from_counts(&[0, 0, 0], 5).expect("degenerate");
        assert_eq!(h.probability(0), PROBABILITY_TOTAL);
        round_trip(&h);
    }

    #[test]
    fn a_full_alphabet_of_rare_symbols_normalizes() {
        // 256 symbols, all used, hugely skewed: the floor of 1 forces the
        // excess path.
        let counts: Vec<u64> = (0..256).map(|i| if i == 0 { 1 << 40 } else { 1 }).collect();
        let h = Histogram::from_counts(&counts, 8).expect("normalizes");
        assert_eq!(h.probabilities().iter().sum::<u32>(), PROBABILITY_TOTAL);
        assert!(h.probabilities().iter().all(|&p| p >= 1));
        round_trip(&h);
    }

    #[test]
    fn the_omitted_symbol_is_the_first_maximum() {
        assert_eq!(omit_pos(&[3, 5, 5, 2]).expect("some"), 1);
        assert_eq!(omit_pos(&[9, 1, 1]).expect("some"), 0);
    }

    #[test]
    fn shift_encoding_covers_its_whole_range() {
        for shift in 0..=MAX_SHIFT {
            let mut w = BitWriter::new();
            write_shift(&mut w, shift).expect("in range");
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            let mut len = 0u32;
            while len < 3 {
                if r.read_bool().expect("bit") {
                    len += 1;
                } else {
                    break;
                }
            }
            let read = r.read_bits(len).expect("payload") + (1 << len) - 1;
            assert_eq!(read, shift);
        }
        assert!(write_shift(&mut BitWriter::new(), 15).is_err());
    }

    #[test]
    fn u8_encoding_covers_its_whole_range() {
        for value in 0..256usize {
            let mut w = BitWriter::new();
            write_u8(&mut w, value).expect("in range");
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            let read = if r.read_bool().expect("flag") {
                let n = r.read_bits(3).expect("width");
                r.read_bits(n).expect("payload") + (1 << n)
            } else {
                0
            };
            assert_eq!(read as usize, value);
        }
        assert!(write_u8(&mut BitWriter::new(), 256).is_err());
    }
}
