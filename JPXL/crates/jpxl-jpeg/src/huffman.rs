//! Huffman tables (10918-1 B.2.4.2 storage; Annex C code assignment; Annex F
//! decode).
//!
//! A `DHT` segment stores, per table, a class/id byte, sixteen count bytes
//! `L_i` (codes of each length 1..=16) and `sum(L_i)` value bytes `V`. The
//! canonical code assignment of Annex C is a pure function of the counts, so
//! the table is re-emitted byte-for-byte from `(class, id, counts, values)`
//! alone. This module derives both the decode tables (Annex F.2.2.1) and the
//! per-symbol encode table from that stored form.

// Indexing here is into fixed `[_; 16]` / `[_; 17]` / `[_; 256]` tables and
// into `Vec`s whose length is established just above the access, always by an
// index the surrounding loop bounds to range (code length `1..=16`, symbol
// value `0..=255`). The `as` narrowings are range-checked first (a code of
// length `si <= 16` is `< 2^si`, and `l + 1 <= 16`).
#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

use crate::error::{JpegError, Result};

/// A single Huffman table, as stored in a `DHT` segment plus the decode/encode
/// lookups derived from it.
#[derive(Clone, Debug)]
pub struct HuffmanTable {
    /// Table class `Tc`: 0 = DC / lossless, 1 = AC.
    pub class: u8,
    /// Table destination identifier `Th`, `0..=3`.
    pub id: u8,
    /// `L_i`: the number of codes of each length `1..=16`.
    pub counts: [u8; 16],
    /// `V`: the symbol values, in canonical order.
    pub values: Vec<u8>,
    /// Per length `l` (indexed `1..=16`): smallest code of that length.
    mincode: [i32; 17],
    /// Per length `l`: largest code of that length, or `-1` if none.
    maxcode: [i32; 17],
    /// Per length `l`: index into `values` of the first symbol of that length.
    valptr: [usize; 17],
    /// Per symbol value: its code length (0 = value not present).
    enc_size: [u8; 256],
    /// Per symbol value: its code word, right-aligned.
    enc_code: [u16; 256],
}

impl HuffmanTable {
    /// Builds a table from its stored form, deriving the lookups.
    ///
    /// Rejects tables whose counts describe more codes than fit in their length
    /// (an over-subscribed / non-prefix code, 10918-1 C.2).
    pub fn new(class: u8, id: u8, counts: [u8; 16], values: Vec<u8>) -> Result<Self> {
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        if total != values.len() {
            return Err(JpegError::Malformed(format!(
                "B.2.4.2: Huffman table sum(L_i)={total} but {} values present",
                values.len()
            )));
        }
        if total == 0 {
            return Err(JpegError::Malformed(
                "B.2.4.2: empty Huffman table (sum(L_i)=0)".into(),
            ));
        }
        if total > 256 {
            return Err(JpegError::Malformed(format!(
                "B.2.4.2: Huffman table has {total} codes (>256)"
            )));
        }

        // Annex C: HUFFSIZE, then HUFFCODE (canonical, monotone by length).
        let mut huffsize: Vec<u8> = Vec::with_capacity(total);
        for (l, &n) in counts.iter().enumerate() {
            for _ in 0..n {
                huffsize.push((l + 1) as u8);
            }
        }
        let mut huffcode = vec![0u16; total];
        let mut code: u32 = 0;
        let mut si = huffsize[0];
        let mut k = 0usize;
        loop {
            while k < total && huffsize[k] == si {
                if code >= (1u32 << si) {
                    return Err(JpegError::Malformed(format!(
                        "C.2: Huffman code of length {si} overflows (over-subscribed table)"
                    )));
                }
                huffcode[k] = code as u16;
                code += 1;
                k += 1;
            }
            if k >= total {
                break;
            }
            // Advance to the next used length, shifting the code left each step.
            while k < total && huffsize[k] != si {
                code <<= 1;
                si += 1;
            }
        }

        // Annex F.2.2.1 decode tables.
        let mut mincode = [0i32; 17];
        let mut maxcode = [-1i32; 17];
        let mut valptr = [0usize; 17];
        let mut p = 0usize;
        for l in 1..=16usize {
            let n = counts[l - 1] as usize;
            if n == 0 {
                maxcode[l] = -1;
                continue;
            }
            valptr[l] = p;
            mincode[l] = i32::from(huffcode[p]);
            p += n;
            maxcode[l] = i32::from(huffcode[p - 1]);
        }

        // Encode lookup, keyed by symbol value.
        let mut enc_size = [0u8; 256];
        let mut enc_code = [0u16; 256];
        for (idx, &v) in values.iter().enumerate() {
            enc_size[v as usize] = huffsize[idx];
            enc_code[v as usize] = huffcode[idx];
        }

        Ok(Self {
            class,
            id,
            counts,
            values,
            mincode,
            maxcode,
            valptr,
            enc_size,
            enc_code,
        })
    }

    /// Decodes one symbol value from `reader` (Annex F.2.2.3, `DECODE`).
    pub fn decode(&self, reader: &mut crate::bitio::EntropyReader<'_>) -> Result<u8> {
        let mut code: i32 = reader.read_bit()? as i32;
        let mut l = 1usize;
        while code > self.maxcode[l] {
            code = (code << 1) | (reader.read_bit()? as i32);
            l += 1;
            if l > 16 {
                return Err(JpegError::Malformed(
                    "F.2.2.3: no Huffman code of length <= 16 matched".into(),
                ));
            }
        }
        let idx = self.valptr[l] + (code - self.mincode[l]) as usize;
        self.values.get(idx).copied().ok_or_else(|| {
            JpegError::Malformed(format!("F.2.2.3: decoded Huffman index {idx} out of range"))
        })
    }

    /// The code length in bits for symbol `value`, or 0 if it is not in the
    /// table.
    #[must_use]
    pub fn code_len(&self, value: u8) -> u8 {
        self.enc_size[value as usize]
    }

    /// Emits the code word for symbol `value` (Annex F.1.2.2, `ENCODE`).
    pub fn encode(&self, writer: &mut crate::bitio::EntropyWriter<'_>, value: u8) -> Result<()> {
        let size = self.enc_size[value as usize];
        if size == 0 {
            return Err(JpegError::Encode(format!(
                "symbol 0x{value:02X} is absent from Huffman table (class {}, id {})",
                self.class, self.id
            )));
        }
        writer.put_bits(u32::from(self.enc_code[value as usize]), u32::from(size));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitio::{EntropyReader, EntropyWriter};

    #[test]
    fn rejects_over_subscribed_table() {
        // Three codes of length 1 is impossible (only two 1-bit codes exist).
        let mut counts = [0u8; 16];
        counts[0] = 3;
        assert!(HuffmanTable::new(0, 0, counts, vec![0, 1, 2]).is_err());
    }

    #[test]
    fn encode_decode_symbol_stream_roundtrips() {
        // One code each of lengths 1..=4: canonical codes 0, 10, 110, 1110.
        let mut counts = [0u8; 16];
        counts[0] = 1;
        counts[1] = 1;
        counts[2] = 1;
        counts[3] = 1;
        let table =
            HuffmanTable::new(1, 0, counts, vec![0x05, 0x11, 0xF0, 0x00]).expect("valid table");
        let symbols = [0x05u8, 0x11, 0x00, 0xF0, 0x05, 0xF0, 0x11, 0x00];

        let mut bytes = Vec::new();
        let mut total_bits = 0u32;
        {
            let mut w = EntropyWriter::new(&mut bytes);
            for &s in &symbols {
                table.encode(&mut w, s).expect("symbol in table");
                total_bits += u32::from(table.code_len(s));
            }
            let pad = u8::try_from((8 - (total_bits % 8)) % 8).expect("pad < 8");
            let bits = if pad == 0 { 0 } else { (1u8 << pad) - 1 };
            w.flush_padding(crate::bitio::Padding { nbits: pad, bits })
                .expect("aligned");
        }
        // Terminate the entropy region with a marker so the reader has a bound.
        bytes.push(0xFF);
        bytes.push(0xD9);

        let mut r = EntropyReader::new(&bytes, 0);
        for &expected in &symbols {
            assert_eq!(table.decode(&mut r).expect("decode"), expected);
        }
    }
}
