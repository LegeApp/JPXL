//! Canonical prefix code construction and emission — the write side of
//! 18181-1 C.2.4, i.e. of IETF RFC 7932:2016 sections 3.2, 3.4 and 3.5.
//!
//! # What is emitted
//!
//! * an alphabet with one used symbol becomes a **simple** code with `NSYM = 1`,
//!   which costs no bits per symbol;
//! * two, three or four used symbols become a **simple** code, whose shapes
//!   (`[1,1]`, `[1,2,2]`, `[2,2,2,2]` and `[1,2,3,3]`) are exactly the optimal
//!   code shapes for those alphabet sizes;
//! * anything larger becomes a **complex** code with `HSKIP = 0`: the lengths
//!   of the code-length code, then one code-length symbol per alphabet symbol.
//!
//! Run-length codes 16 and 17 of the code-length alphabet are **not emitted**;
//! they are a density optimization and the decoder reads them either way. The
//! emitter still tracks the RFC's `space` accounting exactly, because the
//! decoder stops reading the moment the code is complete, so a trailing length
//! written past that point would be consumed as something else entirely.
//!
//! # Length limiting
//!
//! Huffman code lengths can exceed the 15-bit ceiling of RFC 7932 section 3.2
//! on skewed alphabets. [`huffman_lengths`] therefore clamps and then repairs
//! the Kraft sum: lengthen the least frequent symbols until the code is no
//! longer over-subscribed, then shorten the most frequent ones until it is
//! complete again. Both loops are shown to terminate in their comments. The
//! result is a legal, complete, length-limited code; it is not proved optimal,
//! and squeezing the last fraction of a bit out of it belongs to the density
//! slice, not here.
//!
//! Every path here is **bit-exact** (see `docs/PLAN.md`).

use jpxl_bitstream::BitWriter;

use crate::error::{Result, encode_error};
use crate::hybrid::bit_width;
use crate::prefix::{
    CODE_LENGTH_ALPHABET, CODE_LENGTH_CODE_LENGTHS, CODE_LENGTH_ORDER, MAX_CODE_LENGTH,
};

/// Longest code the code-length code itself may use (RFC 7932 section 3.5: its
/// lengths are read with a six-symbol fixed code).
const MAX_CODE_LENGTH_CODE_LENGTH: u8 = 5;

/// Largest alphabet a prefix-coded distribution may declare (18181-1 C.2.1).
pub const MAX_PREFIX_ALPHABET: usize = 1 << 15;

/// A canonical prefix code ready to be written and to write symbols with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixEncoder {
    /// Code length of every symbol; 0 for an unused one.
    lengths: Vec<u8>,
    /// Canonical code value of every symbol, most-significant bit first.
    codes: Vec<u32>,
    /// Set when the whole alphabet collapses to one symbol coded in zero bits.
    constant: Option<u32>,
}

impl PrefixEncoder {
    /// Builds the code for an alphabet of `counts.len()` symbols.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the alphabet is
    /// empty or larger than [`MAX_PREFIX_ALPHABET`].
    pub fn from_counts(counts: &[u64]) -> Result<Self> {
        if counts.is_empty() {
            return Err(encode_error!("C.2.1: a prefix alphabet needs a symbol"));
        }
        if counts.len() > MAX_PREFIX_ALPHABET {
            return Err(encode_error!(
                "C.2.1: alphabet size {} exceeds {MAX_PREFIX_ALPHABET}",
                counts.len()
            ));
        }
        let used: Vec<usize> = counts
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| (c != 0).then_some(i))
            .collect();

        // C.2.4: a one-symbol alphabet has no histogram at all, and RFC 7932
        // section 3.4's NSYM = 1 has the same effect for a wider alphabet whose
        // symbols all collapse onto one.
        if counts.len() == 1 || used.len() <= 1 {
            let symbol = used.first().copied().unwrap_or(0);
            return Ok(Self {
                lengths: vec![0; counts.len()],
                codes: vec![0; counts.len()],
                constant: Some(
                    u32::try_from(symbol)
                        .map_err(|_| encode_error!("C.2.4: symbol {symbol} out of range"))?,
                ),
            });
        }

        let lengths = huffman_lengths(counts, MAX_CODE_LENGTH_U8)?;
        let codes = canonical_codes(&lengths)?;
        Ok(Self {
            lengths,
            codes,
            constant: None,
        })
    }

    /// Number of symbols the alphabet declares (18181-1 C.2.1 `count[i]`).
    #[must_use]
    pub fn alphabet_size(&self) -> usize {
        self.lengths.len()
    }

    /// The symbol every read yields, when the code costs no bits.
    #[must_use]
    pub const fn constant_symbol(&self) -> Option<u32> {
        self.constant
    }

    /// Code length of a symbol; 0 means it is not in the code.
    #[must_use]
    pub fn code_length(&self, symbol: u32) -> u8 {
        self.lengths.get(symbol as usize).copied().unwrap_or(0)
    }

    /// Writes the code itself (18181-1 C.2.4).
    ///
    /// The `count[i]` field that precedes it belongs to C.2.1 and is written by
    /// the bundle writer. An alphabet of size 1 writes nothing at all.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the code cannot
    /// be expressed, or a bitstream error.
    pub fn write_code(&self, w: &mut BitWriter) -> Result<()> {
        if self.alphabet_size() <= 1 {
            return Ok(());
        }
        if let Some(symbol) = self.constant {
            return self.write_simple(w, &[symbol]);
        }
        let used: Vec<u32> = sorted_used_symbols(&self.lengths)?;
        if used.len() <= 4 {
            return self.write_simple(w, &used);
        }
        self.write_complex(w)
    }

    /// Writes one symbol's code word, most-significant bit first (C.2.4).
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the symbol is
    /// not in the code, or a bitstream error.
    pub fn write_symbol(&self, w: &mut BitWriter, symbol: u32) -> Result<()> {
        if let Some(constant) = self.constant {
            if symbol != constant {
                return Err(encode_error!(
                    "C.2.4: this code only carries symbol {constant}, not {symbol}"
                ));
            }
            return Ok(());
        }
        let length = self.code_length(symbol);
        if length == 0 {
            return Err(encode_error!(
                "C.2.4: symbol {symbol} is not in this prefix code"
            ));
        }
        let code = self
            .codes
            .get(symbol as usize)
            .copied()
            .ok_or_else(|| encode_error!("C.2.4: symbol {symbol} is outside the alphabet"))?;
        write_code_word(w, code, u32::from(length));
        Ok(())
    }

    /// RFC 7932 section 3.4. `symbols` is in the order the lengths
    /// `[]`/`[1,1]`/`[1,2,2]`/`[1,2,3,3]` or `[2,2,2,2]` apply.
    fn write_simple(&self, w: &mut BitWriter, symbols: &[u32]) -> Result<()> {
        let alphabet_bits = bit_width(
            u32::try_from(self.alphabet_size() - 1)
                .map_err(|_| encode_error!("RFC 7932 3.4: alphabet size out of range"))?,
        );
        let nsym = symbols.len();
        if nsym == 0 || nsym > 4 {
            return Err(encode_error!("RFC 7932 3.4: NSYM {nsym} is out of range"));
        }
        // The 2-bit selector value 1 marks a simple code.
        w.write_bits(2, 1)?;
        w.write_bits(
            2,
            u32::try_from(nsym - 1).map_err(|_| encode_error!("RFC 7932 3.4: NSYM overflow"))?,
        )?;
        for &symbol in symbols {
            w.write_bits(alphabet_bits, symbol)?;
        }
        if nsym == 4 {
            // tree_select: true for [1,2,3,3], false for [2,2,2,2].
            let skewed = symbols.first().is_some_and(|&s| self.code_length(s) == 1);
            w.write_bool(skewed);
        }
        Ok(())
    }

    /// RFC 7932 section 3.5, with `HSKIP = 0` and no run-length codes.
    fn write_complex(&self, w: &mut BitWriter) -> Result<()> {
        let last = self
            .lengths
            .iter()
            .rposition(|&l| l != 0)
            .ok_or_else(|| encode_error!("RFC 7932 3.5: the code has no symbol"))?;
        let transmitted = self
            .lengths
            .get(..=last)
            .ok_or_else(|| encode_error!("RFC 7932 3.5: length index out of range"))?;

        // Frequencies of the code-length symbols that will be transmitted.
        let mut clc_counts = [0u64; CODE_LENGTH_ALPHABET];
        for &length in transmitted {
            let slot = clc_counts
                .get_mut(usize::from(length))
                .ok_or_else(|| encode_error!("RFC 7932 3.5: code length {length} out of range"))?;
            *slot += 1;
        }
        let distinct = clc_counts.iter().filter(|&&c| c != 0).count();

        let clc_lengths: Vec<u8> = if distinct == 1 {
            // One length value for the whole transmitted range. RFC 7932 3.5
            // turns a code-length code with a single nonzero length into a
            // zero-bit code, so every alphabet length then costs nothing.
            let only = clc_counts
                .iter()
                .position(|&c| c != 0)
                .ok_or_else(|| encode_error!("RFC 7932 3.5: no code length is used"))?;
            let mut lengths = vec![0u8; CODE_LENGTH_ALPHABET];
            let slot = lengths
                .get_mut(only)
                .ok_or_else(|| encode_error!("RFC 7932 3.5: code length {only} out of range"))?;
            *slot = 1;
            lengths
        } else {
            huffman_lengths(&clc_counts, MAX_CODE_LENGTH_CODE_LENGTH)?
        };

        // The 2-bit selector doubles as HSKIP; 1 would mean a simple code, so 0
        // both selects the complex form and skips nothing.
        w.write_bits(2, 0)?;

        // Step 1: the lengths of the code-length code, in the RFC's order and
        // through its fixed code. The decoder stops as soon as that code is
        // complete, so this must stop at the same place.
        let fixed = canonical_codes(&CODE_LENGTH_CODE_LENGTHS)?;
        let mut space = 32i32;
        for &symbol in &CODE_LENGTH_ORDER {
            let length = clc_lengths
                .get(symbol)
                .copied()
                .ok_or_else(|| encode_error!("RFC 7932 3.5: symbol {symbol} out of range"))?;
            let code = fixed
                .get(usize::from(length))
                .copied()
                .ok_or_else(|| encode_error!("RFC 7932 3.5: code length {length} out of range"))?;
            let code_length = CODE_LENGTH_CODE_LENGTHS
                .get(usize::from(length))
                .copied()
                .ok_or_else(|| encode_error!("RFC 7932 3.5: code length {length} out of range"))?;
            write_code_word(w, code, u32::from(code_length));
            if length != 0 {
                space -= 32 >> length;
                if space <= 0 {
                    break;
                }
            }
        }

        // Step 2: one code-length symbol per alphabet symbol, again stopping
        // where the decoder does.
        let clc = canonical_codes(&clc_lengths)?;
        let mut space = 32_768i64;
        for &length in transmitted {
            if distinct > 1 {
                let code = clc
                    .get(usize::from(length))
                    .copied()
                    .ok_or_else(|| encode_error!("RFC 7932 3.5: length {length} out of range"))?;
                let code_length = clc_lengths
                    .get(usize::from(length))
                    .copied()
                    .ok_or_else(|| encode_error!("RFC 7932 3.5: length {length} out of range"))?;
                write_code_word(w, code, u32::from(code_length));
            }
            if length != 0 {
                space -= 32_768i64 >> length;
                if space <= 0 {
                    break;
                }
            }
        }
        if space != 0 {
            return Err(encode_error!(
                "RFC 7932 3.5: the emitted code is not complete (residual space {space})"
            ));
        }
        Ok(())
    }
}

/// [`MAX_CODE_LENGTH`] as the `u8` the length arrays use.
const MAX_CODE_LENGTH_U8: u8 = 15;
const _: () = assert!(MAX_CODE_LENGTH == 15);

/// Writes a canonical code word, most-significant bit first (18181-1 C.2.4).
fn write_code_word(w: &mut BitWriter, code: u32, length: u32) {
    for shift in (0..length).rev() {
        w.write_bool((code >> shift) & 1 == 1);
    }
}

/// Symbols of a code in canonical order: by length, then by symbol.
fn sorted_used_symbols(lengths: &[u8]) -> Result<Vec<u32>> {
    let mut used: Vec<(u8, u32)> = Vec::new();
    for (symbol, &length) in lengths.iter().enumerate() {
        if length != 0 {
            used.push((
                length,
                u32::try_from(symbol)
                    .map_err(|_| encode_error!("C.2.4: symbol {symbol} out of range"))?,
            ));
        }
    }
    used.sort_unstable();
    Ok(used.into_iter().map(|(_, symbol)| symbol).collect())
}

/// Assigns canonical code values from code lengths (RFC 7932 section 3.2).
///
/// Codes are handed out in increasing numeric order within each length, and
/// lengths are processed shortest first, which is the order the decoder walks
/// when it matches bits.
///
/// # Errors
///
/// [`EntropyError::Encode`](crate::EntropyError::Encode) if a length exceeds
/// [`MAX_CODE_LENGTH`].
pub fn canonical_codes(lengths: &[u8]) -> Result<Vec<u32>> {
    let mut counts = [0u32; MAX_CODE_LENGTH + 1];
    for &length in lengths {
        if usize::from(length) > MAX_CODE_LENGTH {
            return Err(encode_error!(
                "RFC 7932 3.2: code length {length} exceeds {MAX_CODE_LENGTH}"
            ));
        }
        if length == 0 {
            continue;
        }
        let slot = counts
            .get_mut(usize::from(length))
            .ok_or_else(|| encode_error!("RFC 7932 3.2: code length {length} out of range"))?;
        *slot += 1;
    }

    let mut next = [0u32; MAX_CODE_LENGTH + 2];
    let mut code = 0u32;
    for length in 1..=MAX_CODE_LENGTH {
        let previous = counts
            .get(length - 1)
            .copied()
            .ok_or_else(|| encode_error!("RFC 7932 3.2: length {length} out of range"))?;
        code = (code + previous) << 1;
        let slot = next
            .get_mut(length)
            .ok_or_else(|| encode_error!("RFC 7932 3.2: length {length} out of range"))?;
        *slot = code;
    }

    let mut codes = vec![0u32; lengths.len()];
    for (symbol, &length) in lengths.iter().enumerate() {
        if length == 0 {
            continue;
        }
        let slot = next
            .get_mut(usize::from(length))
            .ok_or_else(|| encode_error!("RFC 7932 3.2: length {length} out of range"))?;
        let value = *slot;
        *slot += 1;
        let out = codes
            .get_mut(symbol)
            .ok_or_else(|| encode_error!("RFC 7932 3.2: symbol {symbol} out of range"))?;
        *out = value;
    }
    Ok(codes)
}

/// Builds length-limited Huffman code lengths from symbol counts.
///
/// Symbols with a zero count get length 0. The result is a complete code over
/// the rest: the Kraft sum is exactly 1.
///
/// # Errors
///
/// [`EntropyError::Encode`](crate::EntropyError::Encode) if fewer than two
/// symbols are used (which needs a zero-bit code, not this) or if the alphabet
/// cannot fit under `limit`.
pub fn huffman_lengths(counts: &[u64], limit: u8) -> Result<Vec<u8>> {
    let used: Vec<usize> = counts
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| (c != 0).then_some(i))
        .collect();
    if used.len() < 2 {
        return Err(encode_error!(
            "RFC 7932 3.2: a length-limited code needs at least two symbols"
        ));
    }
    if limit == 0 || limit > MAX_CODE_LENGTH_U8 || used.len() > 1usize << limit {
        return Err(encode_error!(
            "RFC 7932 3.2: {} symbols do not fit under a code length of {limit}",
            used.len()
        ));
    }

    let mut lengths = vec![0u8; counts.len()];
    for (symbol, depth) in huffman_depths(counts, &used)? {
        let slot = lengths
            .get_mut(symbol)
            .ok_or_else(|| encode_error!("RFC 7932 3.2: symbol {symbol} out of range"))?;
        *slot = depth;
    }
    limit_lengths(&mut lengths, counts, limit)?;
    Ok(lengths)
}

/// Optimal Huffman depths, by the two-queue construction.
///
/// Leaves are consumed in `(count, symbol)` order and internal nodes in the
/// order they are created, which is non-decreasing in weight; taking the
/// smaller head of the two queues therefore always takes a global minimum, and
/// ties resolve deterministically towards the leaf queue.
fn huffman_depths(counts: &[u64], used: &[usize]) -> Result<Vec<(usize, u8)>> {
    let mut leaves: Vec<(u64, usize)> = used
        .iter()
        .map(|&symbol| {
            counts
                .get(symbol)
                .copied()
                .map(|count| (count, symbol))
                .ok_or_else(|| encode_error!("RFC 7932 3.2: symbol {symbol} out of range"))
        })
        .collect::<Result<_>>()?;
    leaves.sort_unstable();

    // Node 0..leaves.len() are the leaves; the rest are internal nodes storing
    // their two children.
    let leaf_count = leaves.len();
    let mut children: Vec<(usize, usize)> = Vec::with_capacity(leaf_count.saturating_sub(1));
    let mut internal: Vec<u64> = Vec::with_capacity(leaf_count.saturating_sub(1));

    let mut leaf_head = 0usize;
    let mut internal_head = 0usize;
    let take = |leaf_head: &mut usize,
                internal_head: &mut usize,
                leaves: &[(u64, usize)],
                internal: &[u64]|
     -> Result<(u64, usize)> {
        let leaf = leaves.get(*leaf_head).copied();
        let node = internal.get(*internal_head).copied();
        match (leaf, node) {
            (Some((weight, _)), Some(node_weight)) if weight <= node_weight => {
                *leaf_head += 1;
                Ok((weight, *leaf_head - 1))
            }
            (_, Some(node_weight)) => {
                *internal_head += 1;
                Ok((node_weight, leaf_count + *internal_head - 1))
            }
            (Some((weight, _)), None) => {
                *leaf_head += 1;
                Ok((weight, *leaf_head - 1))
            }
            (None, None) => Err(encode_error!("RFC 7932 3.2: Huffman queues ran dry")),
        }
    };

    while (leaf_count - leaf_head) + (children.len() - internal_head) > 1 {
        let (w1, n1) = take(&mut leaf_head, &mut internal_head, &leaves, &internal)?;
        let (w2, n2) = take(&mut leaf_head, &mut internal_head, &leaves, &internal)?;
        children.push((n1, n2));
        internal.push(
            w1.checked_add(w2)
                .ok_or_else(|| encode_error!("RFC 7932 3.2: symbol counts overflow"))?,
        );
    }

    // Walk down from the root accumulating depth.
    let mut depths = vec![0u8; leaf_count];
    let root = leaf_count + children.len() - 1;
    let mut stack = vec![(root, 0u32)];
    while let Some((node, depth)) = stack.pop() {
        if node < leaf_count {
            let slot = depths
                .get_mut(node)
                .ok_or_else(|| encode_error!("RFC 7932 3.2: leaf {node} out of range"))?;
            *slot = u8::try_from(depth.max(1))
                .map_err(|_| encode_error!("RFC 7932 3.2: code length {depth} out of range"))?;
            continue;
        }
        let &(left, right) = children
            .get(node - leaf_count)
            .ok_or_else(|| encode_error!("RFC 7932 3.2: node {node} out of range"))?;
        let next = depth
            .checked_add(1)
            .ok_or_else(|| encode_error!("RFC 7932 3.2: tree is too deep"))?;
        stack.push((left, next));
        stack.push((right, next));
    }

    leaves
        .iter()
        .enumerate()
        .map(|(index, &(_, symbol))| {
            depths
                .get(index)
                .copied()
                .map(|depth| (symbol, depth))
                .ok_or_else(|| encode_error!("RFC 7932 3.2: leaf {index} out of range"))
        })
        .collect()
}

/// Clamps code lengths to `limit` and repairs the Kraft sum.
///
/// The sum is tracked in units of `2^-limit`, so a complete code is exactly
/// `1 << limit`.
///
/// *Lengthening* terminates: each step strictly reduces the sum, and it cannot
/// run out of candidates, because if every symbol sat at `limit` the sum would
/// be the symbol count, which is at most `1 << limit`.
///
/// *Shortening* terminates: each step strictly reduces the deficit, and a
/// candidate always exists — if every symbol's gain exceeded the deficit `D`,
/// then every term of the sum would be a power of two above `D`, so the
/// smallest such term would divide both the sum and `1 << limit`, forcing
/// `D = 0`.
fn limit_lengths(lengths: &mut [u8], counts: &[u64], limit: u8) -> Result<()> {
    for length in lengths.iter_mut() {
        if *length > limit {
            *length = limit;
        }
    }

    let full = 1u64 << limit;
    let weight = |length: u8| -> u64 { 1u64 << (limit - length) };
    let mut kraft: u64 = lengths
        .iter()
        .filter(|&&l| l != 0)
        .map(|&l| weight(l))
        .sum();

    while kraft > full {
        // Lengthen the least frequent symbol that is not already at the limit;
        // ties go to the longer code, then to the later symbol.
        // Ties: the longer code first, then the earlier symbol.
        let mut choice: Option<(u64, u8, usize)> = None;
        for (symbol, &length) in lengths.iter().enumerate() {
            if length == 0 || length >= limit {
                continue;
            }
            let count = counts.get(symbol).copied().unwrap_or(0);
            let better = choice.is_none_or(|(best_count, best_length, _)| {
                (count, u8::MAX - length) < (best_count, u8::MAX - best_length)
            });
            if better {
                choice = Some((count, length, symbol));
            }
        }
        let (_, _, symbol) = choice.ok_or_else(|| {
            encode_error!("RFC 7932 3.2: cannot fit the code under a length of {limit}")
        })?;
        let slot = lengths
            .get_mut(symbol)
            .ok_or_else(|| encode_error!("RFC 7932 3.2: symbol {symbol} out of range"))?;
        *slot += 1;
        kraft -= weight(*slot);
    }

    while kraft < full {
        let deficit = full - kraft;
        let mut choice: Option<(u64, usize)> = None;
        for (symbol, &length) in lengths.iter().enumerate() {
            if length <= 1 || weight(length) > deficit {
                continue;
            }
            let count = counts.get(symbol).copied().unwrap_or(0);
            if choice.is_none_or(|(best, _)| count > best) {
                choice = Some((count, symbol));
            }
        }
        let (_, symbol) = choice
            .ok_or_else(|| encode_error!("RFC 7932 3.2: the code cannot be made complete"))?;
        let slot = lengths
            .get_mut(symbol)
            .ok_or_else(|| encode_error!("RFC 7932 3.2: symbol {symbol} out of range"))?;
        kraft += weight(*slot);
        *slot -= 1;
    }

    Ok(())
}

#[cfg(test)]
// Test code: an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves.
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::prefix::read_prefix_code;
    use jpxl_bitstream::BitReader;
    use jpxl_core::limits::{AllocGuard, Limits};

    /// Writes the code, reads it back with the decoder, and checks that every
    /// used symbol survives a write/read round trip.
    fn round_trip(counts: &[u64]) {
        let encoder = PrefixEncoder::from_counts(counts).expect("builds");
        let mut w = BitWriter::new();
        encoder.write_code(&mut w).expect("writes");
        let used: Vec<u32> = counts
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| (c != 0).then_some(i as u32))
            .collect();
        for &symbol in &used {
            encoder.write_symbol(&mut w, symbol).expect("symbol");
        }
        let bits = w.bit_len();
        let bytes = w.into_bytes();

        let mut guard = AllocGuard::new(&Limits::relaxed());
        let mut r = BitReader::new(&bytes);
        let code = read_prefix_code(&mut r, counts.len(), &mut guard).expect("decoder reads it");
        for &symbol in &used {
            assert_eq!(code.decode(&mut r).expect("symbol"), symbol, "symbol");
        }
        assert_eq!(r.total_bits_read(), bits, "no bits left unread");
    }

    #[test]
    fn kraft_sum_is_exact_for_every_shape() {
        let cases: Vec<Vec<u64>> = vec![
            vec![1, 1],
            vec![5, 1],
            vec![3, 2, 1],
            vec![1, 1, 1, 1],
            vec![100, 10, 5, 1],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            (0..200u64).map(|i| i * i + 1).collect(),
        ];
        for counts in cases {
            let lengths = huffman_lengths(&counts, 15).expect("lengths");
            let kraft: u64 = lengths
                .iter()
                .filter(|&&l| l != 0)
                .map(|&l| 1u64 << (15 - l))
                .sum();
            assert_eq!(kraft, 1 << 15, "counts {counts:?} -> lengths {lengths:?}");
        }
    }

    /// A distribution whose optimal Huffman code is deeper than the limit: the
    /// Fibonacci weights force depth 15+ over 20 symbols.
    #[test]
    fn deep_codes_are_limited_and_stay_complete() {
        let mut counts = vec![1u64, 1];
        for i in 2..25 {
            let next = counts[i - 1] + counts[i - 2];
            counts.push(next);
        }
        counts.reverse();
        for limit in [5u8, 8, 15] {
            if counts.len() > 1usize << limit {
                continue;
            }
            let lengths = huffman_lengths(&counts, limit).expect("lengths");
            assert!(lengths.iter().all(|&l| l <= limit), "{lengths:?}");
            let kraft: u64 = lengths
                .iter()
                .filter(|&&l| l != 0)
                .map(|&l| 1u64 << (limit - l))
                .sum();
            assert_eq!(kraft, 1u64 << limit, "limit {limit}");
        }
        round_trip(&counts);
    }

    #[test]
    fn canonical_assignment_matches_the_rfc_example() {
        // RFC 7932 section 3.2: lengths (2,1,3,3) -> A=10, B=0, C=110, D=111.
        let codes = canonical_codes(&[2, 1, 3, 3]).expect("codes");
        assert_eq!(codes, vec![0b10, 0b0, 0b110, 0b111]);
    }

    #[test]
    fn the_fixed_code_length_code_matches_the_decoders() {
        // Derived from the same lengths the decoder canonicalizes, and checked
        // against the values the RFC prints (reversed, see the decoder note).
        let codes = canonical_codes(&CODE_LENGTH_CODE_LENGTHS).expect("codes");
        assert_eq!(codes, vec![0b00, 0b1110, 0b110, 0b01, 0b10, 0b1111]);
    }

    #[test]
    fn simple_codes_round_trip_at_every_nsym() {
        round_trip(&[1, 0, 0, 0]); // NSYM = 1
        round_trip(&[1, 1, 0, 0]); // NSYM = 2
        round_trip(&[3, 2, 1, 0]); // NSYM = 3, shape [1, 2, 2]
        round_trip(&[1, 1, 1, 1]); // NSYM = 4, shape [2, 2, 2, 2]
        round_trip(&[9, 3, 2, 1]); // NSYM = 4, shape [1, 2, 3, 3]
    }

    #[test]
    fn complex_codes_round_trip() {
        round_trip(&[10, 9, 8, 7, 6, 5]);
        round_trip(&[1, 0, 5, 0, 9, 0, 0, 2, 30]);
        round_trip(&(1..=300u64).rev().collect::<Vec<_>>());
    }

    #[test]
    fn a_flat_code_uses_the_zero_bit_length_code() {
        // 16 equally likely symbols: every length is 4, so the code-length code
        // collapses to a single symbol and costs nothing per length.
        let counts = vec![1u64; 16];
        let encoder = PrefixEncoder::from_counts(&counts).expect("builds");
        assert!(encoder.lengths.iter().all(|&l| l == 4));
        round_trip(&counts);
    }

    #[test]
    fn a_one_symbol_alphabet_writes_nothing() {
        let encoder = PrefixEncoder::from_counts(&[7]).expect("builds");
        let mut w = BitWriter::new();
        encoder.write_code(&mut w).expect("writes");
        assert_eq!(w.bit_len(), 0);
        encoder.write_symbol(&mut w, 0).expect("symbol");
        assert_eq!(w.bit_len(), 0);
        assert!(encoder.write_symbol(&mut w, 1).is_err());
    }

    #[test]
    fn a_single_used_symbol_of_a_wide_alphabet_costs_no_symbol_bits() {
        let counts = vec![0u64, 0, 5, 0];
        let encoder = PrefixEncoder::from_counts(&counts).expect("builds");
        assert_eq!(encoder.constant_symbol(), Some(2));
        round_trip(&counts);
    }
}
