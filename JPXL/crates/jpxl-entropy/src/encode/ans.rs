//! rANS emission — the inverse of the state machine of 18181-1 C.3.2.
//!
//! # Why encoding runs backwards
//!
//! C.3.2's decode step is
//!
//! ```text
//! index    = state & 0xFFF
//! (s, off) = AliasMapping(index)
//! state    = D[s] * (state >> 12) + off
//! if state < (1 << 16): state = (state << 16) | u(16)
//! ```
//!
//! which maps one state to *(symbol, next state)*. Inverting it means starting
//! from the next state and recovering the previous one, so the encoder walks
//! the symbol sequence from the last symbol to the first. The clause fixes both
//! ends of that walk: the state after the final symbol is `0x00130000`, and the
//! state before the first is what the decoder reads as its `u(32)` seed. So the
//! backward pass starts at the terminal value and finishes holding the seed.
//!
//! # The inverse step
//!
//! Write `y` for the state after decoding symbol `s`, `p = D[s]`, and
//! `slot(s, off)` for the inverse alias mapping. Then
//!
//! ```text
//! x = ((y / p) << 12) | slot(s, y % p)
//! ```
//!
//! reproduces `y` exactly, because `p * (x >> 12) + off = p * (y / p) + y % p`.
//! The decoder's renormalization is inverted by splitting `y` first: if
//! `y >= p << 20` the low 16 bits of `y` are the word the decoder will read
//! back, and the step continues with `y >> 16`.
//!
//! # The state invariant
//!
//! Every state is in `[1 << 16, 1 << 32)`. Decoding preserves it, and so does
//! this inverse:
//!
//! * the upper bound needs `y / p < 1 << 20`, which is exactly the
//!   renormalization condition, and one 16-bit split always suffices because
//!   `y >> 16 < 1 << 16 <= p << 20`;
//! * the lower bound needs `y / p >= 16`, which holds for `y >= 1 << 16`
//!   because `p <= 1 << 12`, and after a split because `y >= p << 20` implies
//!   `y >> 16 >= p << 4`.
//!
//! That is why the emitted word count is decided per symbol and never loops.
//!
//! Every path here is **bit-exact** (see `docs/PLAN.md`).

use crate::ans::{AnsDistribution, FINAL_ANS_STATE, PROBABILITY_TOTAL};
use crate::error::{Result, encode_error};

/// Number of alias slots, i.e. the 12-bit probability space of C.2.5.
const SLOT_COUNT: u32 = PROBABILITY_TOTAL;

/// The inverse of C.2.6's alias mapping: `(symbol, offset)` back to the slot
/// that produces it.
///
/// Built by enumerating all 4096 slots of a decoded [`AnsDistribution`]. The
/// alias table is a table and is shared deliberately; the *bijection* it must
/// satisfy — each symbol occupying exactly its probability mass, with offsets
/// forming a permutation of `0..D[s]` — is re-checked here from scratch, so a
/// broken table fails construction rather than silently producing a stream
/// only this encoder can read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnsEncodeTable {
    probabilities: Vec<u32>,
    /// Start of each symbol's offset range within [`slots`](Self::slots).
    starts: Vec<u32>,
    /// `slots[starts[s] + offset]` is the state slot decoding to `(s, offset)`.
    slots: Vec<u16>,
}

impl AnsEncodeTable {
    /// Inverts the alias mapping of `dist`.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the alias
    /// mapping is not the bijection C.2.6 requires.
    pub fn new(dist: &AnsDistribution) -> Result<Self> {
        let table_size = dist.table_size();
        let mut probabilities = Vec::with_capacity(table_size);
        let mut starts = Vec::with_capacity(table_size + 1);
        let mut running = 0u32;
        for symbol in 0..table_size {
            let p = dist.probability(u32::try_from(symbol).unwrap_or(u32::MAX));
            starts.push(running);
            running = running
                .checked_add(p)
                .ok_or_else(|| encode_error!("C.2.6: probabilities overflow the 12-bit space"))?;
            probabilities.push(p);
        }
        starts.push(running);
        if running != SLOT_COUNT {
            return Err(encode_error!(
                "C.2.6: probabilities sum to {running}, expected {SLOT_COUNT}"
            ));
        }

        let mut slots = vec![u16::MAX; SLOT_COUNT as usize];
        let mut filled = vec![false; SLOT_COUNT as usize];
        for x in 0..SLOT_COUNT {
            let (symbol, offset) = dist.alias_mapping(x)?;
            let p = probabilities
                .get(symbol as usize)
                .copied()
                .ok_or_else(|| encode_error!("C.2.6: alias symbol {symbol} out of range"))?;
            if offset >= p {
                return Err(encode_error!(
                    "C.2.6: slot {x} maps to offset {offset} of symbol {symbol}, whose mass is {p}"
                ));
            }
            let index = starts
                .get(symbol as usize)
                .copied()
                .ok_or_else(|| encode_error!("C.2.6: alias symbol {symbol} out of range"))?
                + offset;
            let slot = slots
                .get_mut(index as usize)
                .ok_or_else(|| encode_error!("C.2.6: alias index {index} out of range"))?;
            let seen = filled
                .get_mut(index as usize)
                .ok_or_else(|| encode_error!("C.2.6: alias index {index} out of range"))?;
            if *seen {
                return Err(encode_error!(
                    "C.2.6: two slots decode to symbol {symbol} offset {offset}"
                ));
            }
            *seen = true;
            *slot = u16::try_from(x)
                .map_err(|_| encode_error!("C.2.6: slot {x} does not fit in 12 bits"))?;
        }
        if let Some(missing) = filled.iter().position(|&f| !f) {
            return Err(encode_error!(
                "C.2.6: alias entry {missing} is never produced, so the mapping is not a bijection"
            ));
        }

        Ok(Self {
            probabilities,
            starts,
            slots,
        })
    }

    /// Probability mass of `symbol`, 0 outside the table.
    #[must_use]
    pub fn probability(&self, symbol: u32) -> u32 {
        self.probabilities
            .get(symbol as usize)
            .copied()
            .unwrap_or(0)
    }

    /// Number of table entries, i.e. `1 << log_alphabet_size`.
    #[must_use]
    pub fn table_size(&self) -> usize {
        self.probabilities.len()
    }

    /// The state slot whose alias mapping is `(symbol, offset)`.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the pair is not
    /// in the table.
    pub fn slot(&self, symbol: u32, offset: u32) -> Result<u32> {
        let start = self
            .starts
            .get(symbol as usize)
            .copied()
            .ok_or_else(|| encode_error!("C.2.6: symbol {symbol} is outside the table"))?;
        if offset >= self.probability(symbol) {
            return Err(encode_error!(
                "C.2.6: offset {offset} is outside the mass of symbol {symbol}"
            ));
        }
        self.slots
            .get((start + offset) as usize)
            .map(|&s| u32::from(s))
            .ok_or_else(|| encode_error!("C.2.6: alias index for symbol {symbol} out of range"))
    }

    /// The state slot whose alias mapping is `(symbol, offset)`, validated form.
    ///
    /// [`new`](Self::new) re-checks the C.2.6 bijection at construction, so
    /// once the caller has rejected a zero-mass symbol and holds
    /// `offset < probability(symbol)` — exactly the backward pass's
    /// `state % p` — both `starts[symbol]` and `slots[starts[symbol] + offset]`
    /// exist by construction. This removes three checked lookups and the
    /// `Result` machinery from the once-per-symbol rANS step (the checked
    /// [`slot`](Self::slot) was 3.99% of matched-rate cycles); every other
    /// caller keeps the checked form.
    #[inline]
    #[expect(
        clippy::indexing_slicing,
        clippy::cast_possible_truncation,
        reason = "unreachable by construction: `new` validated the alias bijection, \
                  the caller's zero-mass rejection bounds `symbol`, and \
                  `offset < p <= u32::MAX`"
    )]
    fn slot_prevalidated(&self, symbol: u32, offset: u64) -> u32 {
        debug_assert!(
            offset < u64::from(self.probabilities[symbol as usize]),
            "alias offset above the symbol's mass"
        );
        let index = self.starts[symbol as usize] + offset as u32;
        u32::from(self.slots[index as usize])
    }
}

/// One entropy-coded symbol: which cluster's distribution codes it, and the
/// token itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnsSymbol {
    /// Index into the encoder's per-cluster tables.
    pub cluster: usize,
    /// The token, i.e. the symbol of the cluster's alphabet.
    pub token: u32,
}

/// The result of the backward pass: everything the forward writer needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnsPayload {
    /// The `u(32)` seed C.3.2 reads before the first symbol.
    initial_state: u32,
    /// One entry per symbol, in *decode* order: the 16-bit word the decoder
    /// pulls in while decoding that symbol, if it renormalizes there.
    renormalizations: Vec<Option<u16>>,
}

impl AnsPayload {
    /// The `u(32)` seed of C.3.2.
    #[must_use]
    pub const fn initial_state(&self) -> u32 {
        self.initial_state
    }

    /// The renormalization word for symbol `index` in decode order, if the
    /// decoder reads one there.
    #[must_use]
    pub fn renormalization(&self, index: usize) -> Option<u16> {
        self.renormalizations.get(index).copied().flatten()
    }

    /// Number of symbols this payload covers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.renormalizations.len()
    }

    /// Whether the stream carries no symbols at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.renormalizations.is_empty()
    }
}

/// Runs the backward rANS pass over `symbols` (18181-1 C.3.2, inverted).
///
/// `symbols` is in decode order; the pass walks it in reverse. The returned
/// payload's terminal state is [`FINAL_ANS_STATE`] by construction, which is
/// what makes [`SymbolDecoder::finish`](crate::SymbolDecoder::finish) succeed.
///
/// # Errors
///
/// [`EntropyError::Encode`](crate::EntropyError::Encode) if a symbol names a
/// cluster with no table, or a token with no probability mass in it.
pub fn encode_symbols(tables: &[AnsEncodeTable], symbols: &[AnsSymbol]) -> Result<AnsPayload> {
    encode_symbols_with(tables, symbols.len(), |index| {
        symbols.get(index).copied().unwrap_or(AnsSymbol {
            cluster: usize::MAX,
            token: u32::MAX,
        })
    })
}

/// [`encode_symbols`] over `count` symbols produced by `symbol_at(index)`,
/// so a caller holding its symbols in another layout (a token tape) need not
/// materialise a `Vec<AnsSymbol>` first. Same backward pass, same words.
///
/// # Errors
///
/// As [`encode_symbols`].
pub fn encode_symbols_with(
    tables: &[AnsEncodeTable],
    count: usize,
    symbol_at: impl Fn(usize) -> AnsSymbol,
) -> Result<AnsPayload> {
    let mut renormalizations = vec![None; count];
    let mut state = u64::from(FINAL_ANS_STATE);

    for index in (0..count).rev() {
        let symbol = &symbol_at(index);
        let table = tables.get(symbol.cluster).ok_or_else(|| {
            encode_error!("C.2.1: cluster {} has no distribution", symbol.cluster)
        })?;
        let p = u64::from(table.probability(symbol.token));
        if p == 0 {
            return Err(encode_error!(
                "C.3.2: token {} has no probability mass in cluster {}",
                symbol.token,
                symbol.cluster
            ));
        }

        // The decoder renormalizes exactly when the pre-renormalization state
        // is below 1 << 16, which is exactly when the state we are carrying
        // does not fit the interval this symbol's mass allows.
        if state >= (p << 20) {
            let word = u16::try_from(state & 0xFFFF)
                .map_err(|_| encode_error!("C.3.2: renormalization word out of range"))?;
            let slot = renormalizations
                .get_mut(index)
                .ok_or_else(|| encode_error!("C.3.2: symbol index {index} out of range"))?;
            *slot = Some(word);
            state >>= 16;
        }

        let quotient = state / p;
        let offset = state % p;
        // `offset = state % p` is below the mass the `p == 0` gate just
        // validated, so the prevalidated form is exact here.
        let slot = u64::from(table.slot_prevalidated(symbol.token, offset));
        state = (quotient << 12) | slot;
        if state >= 1 << 32 {
            return Err(encode_error!(
                "C.3.2: ANS state {state} leaves the 32-bit range"
            ));
        }
    }

    let initial_state = u32::try_from(state)
        .map_err(|_| encode_error!("C.3.2: initial ANS state {state} leaves the 32-bit range"))?;
    Ok(AnsPayload {
        initial_state,
        renormalizations,
    })
}

#[cfg(test)]
// Test code: an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves.
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::ans::AnsState;
    use jpxl_bitstream::{BitReader, BitWriter};

    fn distribution(probabilities: &[u32], log_alphabet_size: u32) -> AnsDistribution {
        let mut probs = vec![0u32; 1 << log_alphabet_size];
        probs[..probabilities.len()].copy_from_slice(probabilities);
        let alphabet = probabilities.len();
        AnsDistribution::from_probabilities(probs, alphabet, log_alphabet_size).expect("valid")
    }

    /// Encodes a token sequence and replays it through the decoder's state
    /// machine, asserting both the symbols and the terminal state of C.3.2.
    fn round_trip(dist: &AnsDistribution, tokens: &[u32]) {
        let table = AnsEncodeTable::new(dist).expect("invertible");
        let symbols: Vec<AnsSymbol> = tokens
            .iter()
            .map(|&token| AnsSymbol { cluster: 0, token })
            .collect();
        let payload = encode_symbols(std::slice::from_ref(&table), &symbols).expect("encodes");

        let mut w = BitWriter::new();
        w.write_bits(32, payload.initial_state()).expect("seed");
        for index in 0..symbols.len() {
            if let Some(word) = payload.renormalization(index) {
                w.write_bits(16, u32::from(word)).expect("word");
            }
        }
        let bytes = w.into_bytes();

        let mut r = BitReader::new(&bytes);
        let mut state = AnsState::init(&mut r).expect("seed");
        for (i, &token) in tokens.iter().enumerate() {
            assert_eq!(
                state.decode(&mut r, dist).expect("symbol"),
                token,
                "token {i}"
            );
        }
        assert!(
            state.is_final(),
            "C.3.2 terminal state: got {:#x}",
            state.raw()
        );
        assert_eq!(
            r.total_bits_read() as usize,
            bytes.len() * 8 - (bytes.len() * 8 - w_bits(&payload)),
            "every emitted bit is consumed"
        );
    }

    fn w_bits(payload: &AnsPayload) -> usize {
        32 + 16
            * (0..payload.len())
                .filter(|&i| payload.renormalization(i).is_some())
                .count()
    }

    #[test]
    fn a_single_symbol_distribution_needs_no_renormalization() {
        let dist = distribution(&[PROBABILITY_TOTAL], 5);
        let tokens = vec![0u32; 50];
        round_trip(&dist, &tokens);

        let table = AnsEncodeTable::new(&dist).expect("invertible");
        let symbols: Vec<AnsSymbol> = tokens
            .iter()
            .map(|&token| AnsSymbol { cluster: 0, token })
            .collect();
        let payload = encode_symbols(&[table], &symbols).expect("encodes");
        assert_eq!(payload.initial_state(), FINAL_ANS_STATE);
        assert!((0..payload.len()).all(|i| payload.renormalization(i).is_none()));
    }

    #[test]
    fn an_empty_stream_is_just_the_terminal_state() {
        let dist = distribution(&[2048, 2048], 5);
        let table = AnsEncodeTable::new(&dist).expect("invertible");
        let payload = encode_symbols(&[table], &[]).expect("encodes");
        assert_eq!(payload.initial_state(), FINAL_ANS_STATE);
        assert!(payload.is_empty());
        round_trip(&dist, &[]);
    }

    #[test]
    fn skewed_and_uniform_distributions_round_trip() {
        for probs in [
            vec![2048u32, 2048],
            vec![4095, 1],
            vec![1, 4095],
            vec![1024, 1024, 1024, 1024],
            vec![1, 1, 4094],
        ] {
            let dist = distribution(&probs, 5);
            let mut seed = 0x1234_5678u32;
            let tokens: Vec<u32> = (0..500)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let mut pick = (seed >> 16) % probs.iter().sum::<u32>();
                    let mut token = 0u32;
                    for (i, &p) in probs.iter().enumerate() {
                        if pick < p {
                            token = i as u32;
                            break;
                        }
                        pick -= p;
                    }
                    token
                })
                .collect();
            round_trip(&dist, &tokens);
        }
    }

    #[test]
    fn every_log_alphabet_size_round_trips() {
        for log_alphabet_size in 5..=8u32 {
            let table_size = 1usize << log_alphabet_size;
            let share = PROBABILITY_TOTAL / table_size as u32;
            let mut probs = vec![share; table_size];
            probs[0] += PROBABILITY_TOTAL - share * table_size as u32;
            let dist = AnsDistribution::from_probabilities(probs, table_size, log_alphabet_size)
                .expect("valid");
            let tokens: Vec<u32> = (0..table_size as u32).cycle().take(1000).collect();
            round_trip(&dist, &tokens);
        }
    }

    #[test]
    fn tokens_without_mass_are_rejected() {
        let dist = distribution(&[4096], 5);
        let table = AnsEncodeTable::new(&dist).expect("invertible");
        let symbols = [AnsSymbol {
            cluster: 0,
            token: 1,
        }];
        assert!(encode_symbols(&[table], &symbols).is_err());
    }

    #[test]
    fn the_inverse_alias_table_is_a_bijection() {
        let dist = distribution(&[2000, 1000, 500, 400, 100, 96], 5);
        let table = AnsEncodeTable::new(&dist).expect("invertible");
        for symbol in 0..6u32 {
            for offset in 0..table.probability(symbol) {
                let slot = table.slot(symbol, offset).expect("in table");
                assert_eq!(
                    dist.alias_mapping(slot).expect("in range"),
                    (symbol, offset),
                    "slot {slot}"
                );
            }
            assert!(table.slot(symbol, table.probability(symbol)).is_err());
        }
    }
}
