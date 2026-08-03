//! The `SectionStore`: the answer to "the TOC comes first but needs lengths".
//!
//! 18181-1 F.3.1 puts the TOC — an array of section byte sizes — on the wire
//! *before* the sections it measures. An encoder therefore cannot stream a
//! frame in one pass: it must know every length before it may write the first
//! one.
//!
//! There are three ways out and only one of them is acceptable. Encoding the
//! whole frame twice doubles the work and risks the two passes disagreeing.
//! Reserving a fixed-width TOC and back-patching is impossible, because F.3.3
//! entries are `U32()` fields whose width depends on the value. What is left
//! is what this module does: **encode each section exactly once into its own
//! buffer, keep the buffers, then write the TOC from their lengths and append
//! them in order.** The peak cost is one encoded frame in memory, which is
//! bounded by the image, and no section is ever encoded twice.
//!
//! A spill-to-disk variant is the obvious extension for very large images and
//! is deliberately not here: it would change nothing about the ordering, which
//! is the part that is easy to get wrong.

use jpxl_bitstream::BitWriter;

use crate::error::Result;
use crate::frame::write_toc;

/// Section bodies held in emission order, each already byte-aligned.
#[derive(Debug, Clone, Default)]
pub struct SectionStore {
    sections: Vec<Vec<u8>>,
}

impl SectionStore {
    /// An empty store.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sections: Vec::new(),
        }
    }

    /// Appends one section body.
    pub fn push(&mut self, body: Vec<u8>) {
        self.sections.push(body);
    }

    /// Appends an empty section, i.e. a zero-length TOC entry.
    ///
    /// F.3.1 NOTE 1 says this is normal in modular mode: the LF-group and
    /// `HfGlobal` sections exist in the table and carry nothing.
    pub fn push_empty(&mut self) {
        self.sections.push(Vec::new());
    }

    /// How many sections are stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sections.len()
    }

    /// Whether no section has been stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// The section lengths, in order.
    #[must_use]
    pub fn lengths(&self) -> Vec<usize> {
        self.sections.iter().map(Vec::len).collect()
    }

    /// Total size of every section body.
    #[must_use]
    pub fn total_len(&self) -> usize {
        self.sections.iter().map(Vec::len).sum()
    }

    /// Writes the TOC and then every section body.
    ///
    /// `w` must be positioned immediately after the `FrameHeader`; on return
    /// it is byte-aligned at the end of the frame.
    ///
    /// # Errors
    ///
    /// [`EncodeError::ValueOutOfRange`](crate::EncodeError::ValueOutOfRange)
    /// if a section is too large for the F.3.3 entry distribution, or a bit
    /// writer error.
    pub fn write(self, w: &mut BitWriter) -> Result<()> {
        write_toc(w, &self.lengths())?;
        for body in self.sections {
            w.write_bytes(&body)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jpxl_bitstream::BitReader;
    use jpxl_core::limits::{AllocGuard, Limits};
    use jpxl_decode::frame::read_toc;

    #[test]
    fn the_toc_describes_exactly_where_each_body_lands() {
        let mut store = SectionStore::new();
        store.push(vec![1u8; 40]);
        store.push_empty();
        store.push_empty();
        store.push(vec![2u8; 1500]);
        store.push(vec![3u8; 7]);
        let lengths = store.lengths();
        assert_eq!(lengths, vec![40, 0, 0, 1500, 7]);
        let total = store.total_len();

        let mut w = BitWriter::new();
        store.write(&mut w).expect("writes");
        let bytes = w.into_bytes();

        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&bytes);
        let toc = read_toc(&mut r, lengths.len() as u64, &limits, &mut guard).expect("valid toc");
        let base = usize::try_from(r.total_bits_read() / 8).expect("byte offset");
        assert_eq!(base + total, bytes.len(), "no gap between TOC and bodies");

        // Each body must be byte-identical at the offset the TOC implies.
        for (index, &len) in lengths.iter().enumerate() {
            let offset = usize::try_from(toc.offset_of(index).expect("entry")).expect("offset");
            let slice = bytes
                .get(base + offset..base + offset + len)
                .expect("in range");
            let expected = match index {
                0 => vec![1u8; 40],
                3 => vec![2u8; 1500],
                4 => vec![3u8; 7],
                _ => Vec::new(),
            };
            assert_eq!(slice, expected.as_slice(), "section {index}");
        }
    }

    #[test]
    fn an_empty_store_is_reported_as_empty() {
        let store = SectionStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.total_len(), 0);
    }
}
