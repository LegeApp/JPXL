//! Bit-level I/O for entropy-coded JPEG data (10918-1 F.1.2.1 / E.1.3).
//!
//! Entropy-coded segments are read most-significant-bit first. Two wire
//! conventions live here:
//!
//! * **Byte stuffing** — a `0xFF` data byte is written as `0xFF 0x00`, so a
//!   `0xFF` in the entropy stream can only be the start of a marker.
//! * **Marker termination** — a `0xFF` followed by any non-zero code ends the
//!   current entropy segment; the reader stops without consuming the marker.
//!
//! Both the reader and writer track just enough state to reproduce the
//! stream's padding bits exactly, which is what makes a byte-for-byte
//! round-trip possible.

// Every `as u32`/`as u8` narrowing in this module is immediately masked to the
// bits actually requested (`read_bits`/`put_bits` take `n <= 32`), so the
// truncation is intentional and lossless within the requested width.
#![allow(clippy::cast_possible_truncation)]

use crate::error::{JpegError, Result};

/// The padding written after the real bits of an entropy segment to reach a
/// byte boundary (10918-1 F.1.2.3): `nbits` bits with value `bits`
/// (right-aligned). Standard encoders pad with 1-bits; capturing the exact
/// value keeps the round-trip faithful to non-standard encoders too.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Padding {
    /// Number of padding bits (`0..=7`).
    pub nbits: u8,
    /// The padding bit pattern, right-aligned in the low `nbits` bits.
    pub bits: u8,
}

/// Reads bits MSB-first from an entropy-coded region, unstuffing `0xFF 0x00`
/// and stopping at the first marker.
pub struct EntropyReader<'a> {
    data: &'a [u8],
    /// Index of the next byte to consider.
    pos: usize,
    /// Pending bits, right-aligned: the next bit to emit is bit `nbits - 1`.
    acc: u64,
    /// Count of valid pending bits in `acc`.
    nbits: u32,
    /// Set once a marker terminated the segment; holds the marker code byte.
    marker: Option<u8>,
}

impl<'a> EntropyReader<'a> {
    /// Creates a reader over `data`, starting at byte `start`.
    #[must_use]
    pub fn new(data: &'a [u8], start: usize) -> Self {
        Self {
            data,
            pos: start,
            acc: 0,
            nbits: 0,
            marker: None,
        }
    }

    /// The marker code that terminated the segment, if one has been reached.
    #[must_use]
    pub fn marker(&self) -> Option<u8> {
        self.marker
    }

    /// The byte offset of the next unconsumed byte (points at the terminating
    /// marker's `0xFF` once a marker has been reached).
    #[must_use]
    pub fn byte_pos(&self) -> usize {
        self.pos
    }

    /// Pulls one raw entropy byte, handling stuffing and marker detection.
    ///
    /// Returns `Ok(None)` when a marker is reached (recording its code);
    /// otherwise the unstuffed data byte.
    fn next_data_byte(&mut self) -> Result<Option<u8>> {
        if self.marker.is_some() {
            return Ok(None);
        }
        let Some(&b) = self.data.get(self.pos) else {
            return Err(JpegError::UnexpectedEof {
                while_reading: "entropy-coded data",
            });
        };
        if b != 0xFF {
            self.pos += 1;
            return Ok(Some(b));
        }
        // b == 0xFF: skip any run of fill 0xFF bytes to find the code byte.
        let mut j = self.pos + 1;
        while self.data.get(j) == Some(&0xFF) {
            j += 1;
        }
        let Some(&code) = self.data.get(j) else {
            return Err(JpegError::UnexpectedEof {
                while_reading: "marker code after 0xFF",
            });
        };
        if code == 0x00 {
            // Stuffed 0xFF: consume `FF 00`, yield a literal 0xFF byte.
            self.pos = j + 1;
            return Ok(Some(0xFF));
        }
        // A real marker. Leave `pos` on the 0xFF immediately preceding the
        // code so the caller sees a contiguous `FF code`, and record it.
        self.pos = j - 1;
        self.marker = Some(code);
        Ok(None)
    }

    /// Refills `acc` up to at least `need` valid bits, or until a marker.
    fn fill(&mut self, need: u32) -> Result<()> {
        while self.nbits < need {
            match self.next_data_byte()? {
                Some(byte) => {
                    self.acc = (self.acc << 8) | u64::from(byte);
                    self.nbits += 8;
                }
                None => break,
            }
        }
        Ok(())
    }

    /// Reads `n` bits (`0..=32`) MSB-first as an unsigned value.
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        self.fill(n)?;
        if self.nbits < n {
            return Err(JpegError::Malformed(format!(
                "F.1.2.1: entropy data exhausted needing {n} bits (only {} available before marker)",
                self.nbits
            )));
        }
        self.nbits -= n;
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        Ok(((self.acc >> self.nbits) as u32) & mask)
    }

    /// Reads a single bit.
    pub fn read_bit(&mut self) -> Result<u32> {
        self.read_bits(1)
    }

    /// Consumes the remaining sub-byte bits as segment padding and returns
    /// them (10918-1 F.1.2.3). Leaves the reader byte-aligned.
    pub fn take_padding(&mut self) -> Result<Padding> {
        // Any whole bytes still buffered would mean the decoder under-read the
        // segment — a bug worth surfacing rather than silently dropping.
        if self.nbits >= 8 {
            return Err(JpegError::Malformed(format!(
                "F.1.2.3: {} bits buffered at a byte-alignment point (expected < 8)",
                self.nbits
            )));
        }
        let nbits = self.nbits;
        let bits = (self.acc & ((1u64 << nbits) - 1)) as u8;
        self.nbits = 0;
        self.acc = 0;
        Ok(Padding {
            nbits: nbits as u8,
            bits,
        })
    }

    /// Ensures the terminating marker is detected while byte-aligned, and
    /// returns its code. Call only after [`Self::take_padding`] (i.e. no
    /// pending bits): the next bytes must be a marker.
    pub fn probe_marker(&mut self) -> Result<u8> {
        if let Some(c) = self.marker {
            return Ok(c);
        }
        let Some(&b) = self.data.get(self.pos) else {
            return Err(JpegError::UnexpectedEof {
                while_reading: "marker after entropy segment",
            });
        };
        if b != crate::marker::MARKER_PREFIX {
            return Err(JpegError::Malformed(format!(
                "expected a marker prefix 0xFF after an entropy segment, found 0x{b:02X}"
            )));
        }
        let mut j = self.pos + 1;
        while self.data.get(j) == Some(&0xFF) {
            j += 1;
        }
        let Some(&code) = self.data.get(j) else {
            return Err(JpegError::UnexpectedEof {
                while_reading: "marker code after entropy segment",
            });
        };
        self.pos = j - 1;
        self.marker = Some(code);
        Ok(code)
    }

    /// After a restart marker, advances past its two bytes and resumes reading
    /// a fresh (byte-aligned) entropy segment.
    pub fn resume_after_restart(&mut self) -> Result<()> {
        let code = self.marker.ok_or_else(|| {
            JpegError::Malformed("resume_after_restart called with no marker pending".into())
        })?;
        if !crate::marker::is_restart(code) {
            return Err(JpegError::Malformed(format!(
                "expected a restart marker to resume past, found 0xFF{code:02X}"
            )));
        }
        // `pos` is on the marker's 0xFF; skip the two marker bytes.
        self.pos += 2;
        self.marker = None;
        self.acc = 0;
        self.nbits = 0;
        Ok(())
    }
}

/// Writes bits MSB-first into a byte buffer, stuffing `0xFF` data bytes as
/// `0xFF 0x00`.
pub struct EntropyWriter<'a> {
    out: &'a mut Vec<u8>,
    /// Pending bits, right-aligned in `acc`.
    acc: u64,
    nbits: u32,
}

impl<'a> EntropyWriter<'a> {
    /// Creates a writer appending to `out`.
    pub fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            acc: 0,
            nbits: 0,
        }
    }

    /// Writes the low `n` bits (`0..=32`) of `value`, MSB-first.
    pub fn put_bits(&mut self, value: u32, n: u32) {
        if n == 0 {
            return;
        }
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        self.acc = (self.acc << n) | u64::from(value & mask);
        self.nbits += n;
        while self.nbits >= 8 {
            self.nbits -= 8;
            let byte = ((self.acc >> self.nbits) & 0xFF) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0x00);
            }
        }
    }

    /// Writes a single bit.
    pub fn put_bit(&mut self, bit: u32) {
        self.put_bits(bit & 1, 1);
    }

    /// Writes raw bytes directly (e.g. a restart marker) while byte-aligned.
    ///
    /// Must be called only when no partial byte is pending — i.e. right after
    /// [`Self::flush_padding`]. Errors otherwise rather than corrupting the
    /// bit stream.
    pub fn write_aligned_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        if self.nbits != 0 {
            return Err(JpegError::Encode(
                "write_aligned_bytes called with a partial byte pending".into(),
            ));
        }
        self.out.extend_from_slice(bytes);
        Ok(())
    }

    /// Emits the captured segment [`Padding`] to reach a byte boundary.
    ///
    /// Returns an error if the writer is not byte-aligned afterwards, which can
    /// only happen if the re-encoded real bits diverged from the original.
    pub fn flush_padding(&mut self, padding: Padding) -> Result<()> {
        self.put_bits(u32::from(padding.bits), u32::from(padding.nbits));
        if self.nbits != 0 {
            return Err(JpegError::Encode(format!(
                "F.1.2.3: {} bits pending after padding — re-encoded bit count diverged",
                self.nbits
            )));
        }
        Ok(())
    }
}
