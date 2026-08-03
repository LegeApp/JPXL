//! The JPEG XL container: box framing and box types (ISO/IEC 18181-2
//! clauses 8 and 9).
//!
//! # Two entry points, deliberately
//!
//! [`extract_codestream`] is the fast path every other module uses: give it a
//! file, get the codestream. It parses the box structure and reassembles
//! `jxlp` fragments, and it does **not** enforce the clause-9 file-structure
//! rules, because a decoder that refuses to show you an image over a missing
//! `ftyp` box is not more correct, only less useful.
//!
//! [`BoxTree::parse`] is the full parser: every box, its type, its byte range
//! and its payload. [`BoxTree::validate`] then checks the "shall" requirements
//! of clause 9 separately, so a caller can ask "is this a conforming file?"
//! without that question being entangled with "can I decode it?".
//!
//! # Box framing (clause 8)
//!
//! Every box is `LBox` (big-endian `u32`), `TBox` (four type bytes), then the
//! payload. `LBox` counts the whole box including the header. Three values are
//! special: `0` means the box runs to the end of the file, `1` means the real
//! length is an `XLBox` `u64` between `TBox` and the payload, and anything
//! else shall be at least 8. `XLBox` shall be at least 16.
//!
//! # Box coverage (clause 9)
//!
//! | Box | Clause | Handling |
//! |---|---|---|
//! | signature | 9.1 | framed, contents validated |
//! | `ftyp` | 9.2 | framed, contents validated |
//! | `jxll` | 9.3 | parsed to a level; at most one, third box |
//! | `jumb` | 9.4 | raw payload exposed; 19566-5 is not parsed |
//! | `Exif` | 9.5 | `tiff_header_offset` + raw payload |
//! | `xml ` | 9.6 | raw payload; several are permitted |
//! | `brob` | 9.7 | target type + compressed payload; see below |
//! | `jxli` | 9.8 | raw payload; Table 9 is not parsed |
//! | `jxlc` | 9.9 | the codestream |
//! | `jxlp` | 9.10 | index-validated, order-checked reassembly |
//! | `jbrd` | 9.11 | recognised and skipped; see below |
//! | anything else | 8 | framed and skipped, as clause 8 requires |
//!
//! **`brob` is recognised, not decompressed.** 9.7 defines the box as "treat
//! this as a box of the payload type, with Brotli-decompressed contents", and
//! RFC 7932 decompression is a dependency this workspace does not take. The
//! box is therefore surfaced with its target type and its compressed bytes,
//! and [`BrobBox::decompressed`] is a typed [`DecodeError::Unsupported`]
//! rather than a guess. The 9.7 constraints on the target type *are* checked,
//! because those are structure, not content.
//!
//! **`jbrd` is recognised and skipped.** JPEG bitstream reconstruction is its
//! own future slice, and Table 11 is known-garbled in the available OCR (see
//! `docs/HANDOFF.md`); implementing the bundle from that table would encode
//! the garbling. The raw payload is exposed so that slice has something to
//! start from.
//!
//! # Untrusted input
//!
//! Box lengths are attacker-controlled and 64 bits wide. Every arithmetic step
//! here is checked, every range is validated against the actual file length
//! before it is taken, and both the box count and the reassembled codestream
//! are charged to an [`AllocGuard`] before they allocate.

use jpxl_core::JpxlError;
use jpxl_core::limits::AllocGuard;

use crate::error::{DecodeError, Result};

/// The 12-byte JPEG XL container signature box (18181-2 9.1).
pub const CONTAINER_SIGNATURE: [u8; 12] = [
    0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
];

/// The payload the file type box shall carry (18181-2 9.2): the major brand
/// `jxl `, minor version 0, and `jxl ` as the only compatible brand.
pub const FILE_TYPE_PAYLOAD: [u8; 12] = [
    b'j', b'x', b'l', b' ', 0x00, 0x00, 0x00, 0x00, b'j', b'x', b'l', b' ',
];

/// The level assumed when no level box is present (18181-2 9.3).
pub const DEFAULT_LEVEL: u8 = 5;

/// Bytes charged to the guard per parsed box, covering the bookkeeping entry.
///
/// A file made of nothing but eight-byte empty boxes would otherwise let an
/// attacker turn `n` bytes of input into `n / 8` heap entries for free.
const BOX_BOOKKEEPING_BYTES: u64 = 64;

/// The high bit of a `jxlp` index, marking the final fragment (18181-2 9.10).
const JXLP_LAST_FLAG: u32 = 0x8000_0000;

/// Which box type this is (18181-2 clause 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxKind {
    /// 9.1 JPEG XL signature box.
    Signature,
    /// 9.2 file type box.
    FileType,
    /// 9.3 level box.
    Level,
    /// 9.4 JUMBF box.
    Jumbf,
    /// 9.5 Exif box.
    Exif,
    /// 9.6 XML box.
    Xml,
    /// 9.7 Brotli-compressed box.
    Brob,
    /// 9.8 frame index box.
    FrameIndex,
    /// 9.9 JPEG XL codestream box.
    Codestream,
    /// 9.10 JPEG XL partial codestream box.
    PartialCodestream,
    /// 9.11 JPEG bitstream reconstruction data box.
    JpegReconstruction,
    /// A type clause 9 does not define. Clause 8 says to skip it.
    Unknown,
}

impl BoxKind {
    /// Classifies a four-byte `TBox`.
    #[must_use]
    pub const fn from_type_code(code: &[u8; 4]) -> Self {
        match code {
            b"JXL " => Self::Signature,
            b"ftyp" => Self::FileType,
            b"jxll" => Self::Level,
            b"jumb" => Self::Jumbf,
            b"Exif" => Self::Exif,
            b"xml " => Self::Xml,
            b"brob" => Self::Brob,
            b"jxli" => Self::FrameIndex,
            b"jxlc" => Self::Codestream,
            b"jxlp" => Self::PartialCodestream,
            b"jbrd" => Self::JpegReconstruction,
            _ => Self::Unknown,
        }
    }

    /// The clause defining this box type, for listings and error messages.
    #[must_use]
    pub const fn clause(self) -> &'static str {
        match self {
            Self::Signature => "18181-2 9.1",
            Self::FileType => "18181-2 9.2",
            Self::Level => "18181-2 9.3",
            Self::Jumbf => "18181-2 9.4",
            Self::Exif => "18181-2 9.5",
            Self::Xml => "18181-2 9.6",
            Self::Brob => "18181-2 9.7",
            Self::FrameIndex => "18181-2 9.8",
            Self::Codestream => "18181-2 9.9",
            Self::PartialCodestream => "18181-2 9.10",
            Self::JpegReconstruction => "18181-2 9.11",
            Self::Unknown => "18181-2 clause 8",
        }
    }
}

/// One box, as framed by clause 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContainerBox<'a> {
    /// The classified type.
    pub kind: BoxKind,
    /// The raw four `TBox` bytes, which `Unknown` needs and the others keep
    /// so a listing can print what was actually there.
    pub type_code: [u8; 4],
    /// Byte offset of `LBox` within the file.
    pub offset: usize,
    /// Header length: 8 normally, 16 for the `XLBox` form.
    pub header_len: usize,
    /// The box content (`DBox`).
    pub payload: &'a [u8],
    /// Whether `LBox` was 0, i.e. the box runs to the end of the file.
    pub extends_to_end: bool,
}

impl ContainerBox<'_> {
    /// Total size of the box in bytes, header included.
    #[must_use]
    pub const fn total_len(&self) -> usize {
        self.header_len + self.payload.len()
    }
}

/// An `Exif` box (18181-2 9.5), split into its two fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExifBox<'a> {
    /// Bytes from the first payload byte to the first TIFF header.
    pub tiff_header_offset: u32,
    /// The Exif payload, as JEITA CP-3451E / CP-3461B defines it.
    ///
    /// Carried verbatim. 9.5 says the codestream takes precedence wherever
    /// the two overlap, so nothing here is ever merged into the image
    /// metadata; the caller decides.
    pub payload: &'a [u8],
}

/// A `brob` box (18181-2 9.7): a Brotli-compressed box of another type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrobBox<'a> {
    /// The four `TBox` bytes the decompressed content should be treated as.
    pub target_type_code: [u8; 4],
    /// The classified target type.
    pub target_kind: BoxKind,
    /// The Brotli stream, exactly as stored.
    pub compressed: &'a [u8],
}

impl BrobBox<'_> {
    /// The decompressed box content.
    ///
    /// # Errors
    ///
    /// Always [`DecodeError::Unsupported`]: RFC 7932 decompression would be a
    /// third-party dependency, which `AGENTS.md` forbids without a recorded
    /// decision. The compressed bytes and the target type are available on
    /// this struct so a caller with its own Brotli implementation can finish
    /// the job.
    pub fn decompressed(&self) -> Result<Vec<u8>> {
        Err(DecodeError::Unsupported {
            feature: "transparent Brotli decompression of a brob box",
            clause: "18181-2 9.7",
        })
    }
}

/// Every box of a container file, in file order.
#[derive(Debug, Clone, Default)]
pub struct BoxTree<'a> {
    boxes: Vec<ContainerBox<'a>>,
}

impl<'a> BoxTree<'a> {
    /// Parses the box structure of `data` (18181-2 clause 8).
    ///
    /// Framing errors — a truncated header, an `LBox` below 8, an `XLBox`
    /// below 16, a length that runs past the end of the file, trailing bytes
    /// too short to be a box — are hard errors. Clause-9 structure is *not*
    /// checked here; see [`BoxTree::validate`].
    ///
    /// # Errors
    ///
    /// [`DecodeError::Core`] with an `InvalidHeader` message naming the clause
    /// for a framing violation, or a guard rejection for an absurd box count.
    pub fn parse(data: &'a [u8], guard: &mut AllocGuard) -> Result<Self> {
        let mut boxes = Vec::new();
        let mut pos = 0usize;

        while pos < data.len() {
            let header = data.get(pos..pos.saturating_add(8)).ok_or_else(|| {
                malformed(
                    "clause 8",
                    "a trailing fragment too short to be a box header",
                )
            })?;
            let raw_len = be_u32(header.get(..4).unwrap_or_default());
            let mut type_code = [0u8; 4];
            if let (Some(dst), Some(src)) = (type_code.get_mut(..), header.get(4..8)) {
                dst.copy_from_slice(src);
            }

            let (header_len, end, extends_to_end) = match raw_len {
                // 8.1: LBox 0 means the box is the last one and runs to EOF.
                0 => (8usize, data.len(), true),
                // 8.1: LBox 1 means a 64-bit XLBox follows TBox.
                1 => {
                    let ext = data
                        .get(pos + 8..pos + 16)
                        .ok_or_else(|| malformed("clause 8", "a truncated XLBox header"))?;
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(ext);
                    let len = u64::from_be_bytes(bytes);
                    if len < 16 {
                        return Err(malformed(
                            "clause 8",
                            "an XLBox below the 16-byte minimum the clause states",
                        ));
                    }
                    let end = usize::try_from(len)
                        .ok()
                        .and_then(|l| pos.checked_add(l))
                        .filter(|&e| e <= data.len())
                        .ok_or_else(|| {
                            malformed(
                                "clause 8",
                                "an XLBox length that runs past the end of the file",
                            )
                        })?;
                    (16usize, end, false)
                }
                len if len < 8 => {
                    return Err(malformed(
                        "clause 8",
                        "an LBox below the 8-byte minimum the clause states",
                    ));
                }
                len => {
                    let end = usize::try_from(len)
                        .ok()
                        .and_then(|l| pos.checked_add(l))
                        .filter(|&e| e <= data.len())
                        .ok_or_else(|| {
                            malformed(
                                "clause 8",
                                "an LBox length that runs past the end of the file",
                            )
                        })?;
                    (8usize, end, false)
                }
            };

            let payload_start = pos
                .checked_add(header_len)
                .filter(|&s| s <= end)
                .ok_or_else(|| malformed("clause 8", "a box shorter than its own header"))?;
            let payload = data
                .get(payload_start..end)
                .ok_or_else(|| malformed("clause 8", "a box payload outside the file"))?;

            guard
                .charge(BOX_BOOKKEEPING_BYTES)
                .map_err(DecodeError::Core)?;
            boxes.push(ContainerBox {
                kind: BoxKind::from_type_code(&type_code),
                type_code,
                offset: pos,
                header_len,
                payload,
                extends_to_end,
            });

            if extends_to_end {
                break;
            }
            // `end > pos` is guaranteed: end >= pos + header_len >= pos + 8.
            pos = end;
        }

        Ok(Self { boxes })
    }

    /// Every box, in file order.
    #[must_use]
    pub fn boxes(&self) -> &[ContainerBox<'a>] {
        &self.boxes
    }

    /// The boxes of one kind, in file order.
    fn of_kind(&self, kind: BoxKind) -> impl Iterator<Item = &ContainerBox<'a>> {
        self.boxes.iter().filter(move |b| b.kind == kind)
    }

    /// Checks the clause-9 file-structure requirements.
    ///
    /// This is separate from [`BoxTree::parse`] on purpose: a file can be
    /// perfectly decodable and still break a "shall" (a missing `ftyp`, say),
    /// and conflating the two would make the decoder refuse images it can
    /// read. Nothing in the decode path calls this; tests and tools do.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Core`] naming the clause and the violated requirement.
    pub fn validate(&self) -> Result<()> {
        // 9.1: exactly one signature box, and it is the first box.
        if self.of_kind(BoxKind::Signature).count() != 1 {
            return Err(malformed("9.1", "a file without exactly one signature box"));
        }
        let first = self
            .boxes
            .first()
            .ok_or_else(|| malformed("9.1", "an empty file with no signature box"))?;
        if first.kind != BoxKind::Signature {
            return Err(malformed(
                "9.1",
                "a first box that is not the signature box",
            ));
        }
        if first.total_len() != CONTAINER_SIGNATURE.len()
            || first.payload != CONTAINER_SIGNATURE.get(8..).unwrap_or_default()
        {
            return Err(malformed(
                "9.1",
                "a signature box that is not the twelve bytes the clause fixes",
            ));
        }

        // 9.2: exactly one file type box, and it is the second box.
        if self.of_kind(BoxKind::FileType).count() != 1 {
            return Err(malformed("9.2", "a file without exactly one file type box"));
        }
        let second = self
            .boxes
            .get(1)
            .ok_or_else(|| malformed("9.2", "a file with no second box to hold ftyp"))?;
        if second.kind != BoxKind::FileType {
            return Err(malformed(
                "9.2",
                "a second box that is not the file type box",
            ));
        }
        if second.total_len() != 20 || second.payload != FILE_TYPE_PAYLOAD {
            return Err(malformed(
                "9.2",
                "a file type box that is not the twenty bytes the clause fixes",
            ));
        }

        // 9.3: at most one level box, and if present it is the third box.
        let levels = self.of_kind(BoxKind::Level).count();
        if levels > 1 {
            return Err(malformed("9.3", "more than one level box"));
        }
        if levels == 1 {
            let third = self.boxes.get(2);
            if third.map(|b| b.kind) != Some(BoxKind::Level) {
                return Err(malformed(
                    "9.3",
                    "a level box that is not immediately after the file type box",
                ));
            }
            if third.map(|b| b.payload.len()) != Some(1) {
                return Err(malformed(
                    "9.3",
                    "a level box whose content is not one byte",
                ));
            }
        }

        // 9.8: zero or one frame index boxes.
        if self.of_kind(BoxKind::FrameIndex).count() > 1 {
            return Err(malformed("9.8", "more than one frame index box"));
        }

        // 9.9 / 9.10: one jxlc, or one or more jxlp, but not both.
        self.check_codestream_boxes()?;
        // 9.10 in full, including the final-fragment marker.
        let _ = self.jxlp_fragments(true)?;
        // 9.7 constrains what a brob may claim to be.
        let _ = self.brob()?;
        // 9.5's fixed-width field has to be there to be read.
        let _ = self.exif()?;
        Ok(())
    }

    /// The declared level, or [`DEFAULT_LEVEL`] when there is no level box
    /// (18181-2 9.3).
    ///
    /// # Errors
    ///
    /// [`DecodeError::Core`] if a level box is present but does not hold
    /// exactly one byte.
    pub fn level(&self) -> Result<u8> {
        let Some(level_box) = self.of_kind(BoxKind::Level).next() else {
            return Ok(DEFAULT_LEVEL);
        };
        match level_box.payload {
            [level] => Ok(*level),
            _ => Err(malformed(
                "9.3",
                "a level box whose content is not one byte",
            )),
        }
    }

    /// Every `Exif` box (18181-2 9.5).
    ///
    /// # Errors
    ///
    /// [`DecodeError::Core`] if a payload is too short to hold the `u32`
    /// TIFF header offset the clause puts first.
    pub fn exif(&self) -> Result<Vec<ExifBox<'a>>> {
        let mut out = Vec::new();
        for b in self.of_kind(BoxKind::Exif) {
            let offset = b
                .payload
                .get(..4)
                .ok_or_else(|| malformed("9.5", "an Exif box with no tiff_header_offset field"))?;
            out.push(ExifBox {
                tiff_header_offset: be_u32(offset),
                payload: b.payload.get(4..).unwrap_or_default(),
            });
        }
        Ok(out)
    }

    /// Every XML box payload (18181-2 9.6). Several are permitted.
    #[must_use]
    pub fn xml(&self) -> Vec<&'a [u8]> {
        self.of_kind(BoxKind::Xml).map(|b| b.payload).collect()
    }

    /// Every JUMBF box payload (18181-2 9.4), unparsed: the content is
    /// ISO/IEC 19566-5's, not this standard's.
    #[must_use]
    pub fn jumbf(&self) -> Vec<&'a [u8]> {
        self.of_kind(BoxKind::Jumbf).map(|b| b.payload).collect()
    }

    /// Every `brob` box (18181-2 9.7), with the target type validated.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Core`] if a payload has no room for the four-byte target
    /// type, or if that type is one 9.7 forbids: `brob` itself, anything
    /// starting `jxl`, or `jbrd`.
    pub fn brob(&self) -> Result<Vec<BrobBox<'a>>> {
        let mut out = Vec::new();
        for b in self.of_kind(BoxKind::Brob) {
            let head = b
                .payload
                .get(..4)
                .ok_or_else(|| malformed("9.7", "a brob box with no payload box type"))?;
            let mut target = [0u8; 4];
            if let Some(dst) = target.get_mut(..) {
                dst.copy_from_slice(head);
            }
            if &target == b"brob" || target.starts_with(b"jxl") || &target == b"jbrd" {
                return Err(malformed(
                    "9.7",
                    "a brob box claiming a payload type the clause forbids \
                     (brob, jbrd, or anything starting jxl)",
                ));
            }
            out.push(BrobBox {
                target_kind: BoxKind::from_type_code(&target),
                target_type_code: target,
                compressed: b.payload.get(4..).unwrap_or_default(),
            });
        }
        Ok(out)
    }

    /// The frame index box payload (18181-2 9.8), unparsed.
    ///
    /// Table 9 is not decoded: the index is an optional seeking aid, nothing
    /// in a full decode consults it, and its `Varint()` fields are only worth
    /// parsing once something needs them.
    #[must_use]
    pub fn frame_index(&self) -> Option<&'a [u8]> {
        self.of_kind(BoxKind::FrameIndex).next().map(|b| b.payload)
    }

    /// The JPEG bitstream reconstruction payload (18181-2 9.11), unparsed.
    ///
    /// Recognised so it is skipped deliberately rather than as an unknown
    /// box. Table 11 is badly OCR-garbled in the sources available here, so
    /// parsing it would encode the garbling; that is a separate slice with a
    /// scan cross-check.
    #[must_use]
    pub fn jpeg_reconstruction(&self) -> Option<&'a [u8]> {
        self.of_kind(BoxKind::JpegReconstruction)
            .next()
            .map(|b| b.payload)
    }

    /// Whether the file carries a codestream in either form.
    #[must_use]
    pub fn has_codestream(&self) -> bool {
        self.of_kind(BoxKind::Codestream).next().is_some()
            || self.of_kind(BoxKind::PartialCodestream).next().is_some()
    }

    /// 9.9 / 9.10: a file carries one `jxlc`, or one or more `jxlp`, never
    /// both and never neither.
    fn check_codestream_boxes(&self) -> Result<()> {
        let whole = self.of_kind(BoxKind::Codestream).count();
        let partial = self.of_kind(BoxKind::PartialCodestream).count();
        match (whole, partial) {
            (1, 0) | (0, 1..) => Ok(()),
            (0, 0) => Err(malformed(
                "9.9",
                "a container with no jxlc or jxlp codestream box",
            )),
            (1.., 1..) => Err(malformed(
                "9.9",
                "a container with both a jxlc box and jxlp boxes",
            )),
            _ => Err(malformed("9.9", "a container with more than one jxlc box")),
        }
    }

    /// The `jxlp` fragment payloads in index order, validating 9.10's index
    /// rules.
    ///
    /// `check_final_marker` additionally requires the last fragment — and only
    /// the last — to carry the `2^31` marker. That is a well-formedness claim
    /// rather than something reassembly needs, so `codestream` leaves it off
    /// and `validate` turns it on.
    fn jxlp_fragments(&self, check_final_marker: bool) -> Result<Vec<&'a [u8]>> {
        let fragments: Vec<&ContainerBox<'a>> = self.of_kind(BoxKind::PartialCodestream).collect();
        let count = fragments.len();
        let mut out = Vec::with_capacity(count);
        for (position, b) in fragments.iter().enumerate() {
            let head = b
                .payload
                .get(..4)
                .ok_or_else(|| malformed("9.10", "a jxlp box with no fragment index"))?;
            let index = be_u32(head);
            let is_last = index & JXLP_LAST_FLAG != 0;
            // 9.10: the index modulo 2^31 is 0 for the first box and increments
            // by one for each next, and the boxes appear in increasing order.
            // So the low 31 bits must equal the box's position among the jxlp
            // boxes; anything else is a gap, a repeat or a reorder.
            let expected = u32::try_from(position).unwrap_or(u32::MAX);
            if index & !JXLP_LAST_FLAG != expected {
                return Err(malformed(
                    "9.10",
                    "a jxlp box whose index does not match its position in the file",
                ));
            }
            if check_final_marker && is_last != (position + 1 == count) {
                return Err(malformed(
                    "9.10",
                    "a jxlp final-fragment marker that is not on the last box",
                ));
            }
            out.push(b.payload.get(4..).unwrap_or_default());
        }
        Ok(out)
    }

    /// The complete codestream: the `jxlc` payload, or the `jxlp` fragments
    /// concatenated in index order (18181-2 9.9, 9.10).
    ///
    /// # Errors
    ///
    /// [`DecodeError::Core`] if the file has no codestream box, has both
    /// forms, or has `jxlp` indices that are not a contiguous increasing run;
    /// or a guard rejection if the reassembled codestream is too large.
    pub fn codestream(&self, guard: &mut AllocGuard) -> Result<Vec<u8>> {
        self.check_codestream_boxes()?;
        if let Some(whole) = self.of_kind(BoxKind::Codestream).next() {
            guard
                .charge(whole.payload.len() as u64)
                .map_err(DecodeError::Core)?;
            return Ok(whole.payload.to_vec());
        }

        let fragments = self.jxlp_fragments(false)?;
        let total: usize = fragments.iter().map(|f| f.len()).sum();
        guard.charge(total as u64).map_err(DecodeError::Core)?;
        let mut out = Vec::with_capacity(total);
        for fragment in fragments {
            out.extend_from_slice(fragment);
        }
        Ok(out)
    }
}

/// Whether `data` starts with the container signature box.
#[must_use]
pub fn is_container(data: &[u8]) -> bool {
    data.starts_with(&CONTAINER_SIGNATURE)
}

/// Extracts the naked codestream from a container (18181-2 9.9 and 9.10).
///
/// The decode-path shortcut for [`BoxTree::parse`] followed by
/// [`BoxTree::codestream`]. Clause-9 file structure is deliberately *not*
/// enforced: see the module documentation.
///
/// # Errors
///
/// Any framing error from [`BoxTree::parse`], or any codestream error from
/// [`BoxTree::codestream`].
pub fn extract_codestream(data: &[u8], guard: &mut AllocGuard) -> Result<Vec<u8>> {
    BoxTree::parse(data, guard)?.codestream(guard)
}

/// Reads a big-endian `u32` from the first four bytes of `bytes`, or 0.
fn be_u32(bytes: &[u8]) -> u32 {
    let mut buf = [0u8; 4];
    let n = bytes.len().min(4);
    if let (Some(dst), Some(src)) = (buf.get_mut(..n), bytes.get(..n)) {
        dst.copy_from_slice(src);
    }
    u32::from_be_bytes(buf)
}

/// A container structure violation.
///
/// `DecodeError` has no Part 2 structural variant and `error.rs` belongs to
/// another task, so this routes through `JpxlError::InvalidHeader`, whose
/// payload is a free-form string — which lets the clause number carry its own
/// part number instead of inheriting `FieldOutOfRange`'s hard-coded
/// `18181-1`.
fn malformed(clause: &str, what: &str) -> DecodeError {
    DecodeError::Core(JpxlError::InvalidHeader(format!(
        "18181-2 {clause}: {what}"
    )))
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
mod tests {
    use jpxl_core::limits::Limits;

    use super::*;

    fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&((payload.len() + 8) as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        out
    }

    /// The `LBox == 1` form: a 64-bit length between the type and the payload.
    fn boxed_xl(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(&((payload.len() + 16) as u64).to_be_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn ftyp() -> Vec<u8> {
        boxed(b"ftyp", &FILE_TYPE_PAYLOAD)
    }

    fn guard() -> AllocGuard {
        AllocGuard::new(&Limits::relaxed())
    }

    /// A minimal conforming file carrying `codestream` in a `jxlc` box.
    fn conforming(codestream: &[u8]) -> Vec<u8> {
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&ftyp());
        file.extend_from_slice(&boxed(b"jxlc", codestream));
        file
    }

    fn tree_of(file: &[u8]) -> BoxTree<'_> {
        BoxTree::parse(file, &mut guard()).expect("parses")
    }

    // ---- framing (clause 8) ---------------------------------------------

    #[test]
    fn framing_records_every_box_with_its_offset_and_length() {
        // Proves the walk consumes exactly `LBox` bytes per box: the offsets
        // must be the running sum, and the last box must end at EOF.
        let file = conforming(&[0xFF, 0x0A, 1, 2, 3]);
        let tree = tree_of(&file);
        let kinds: Vec<BoxKind> = tree.boxes().iter().map(|b| b.kind).collect();
        assert_eq!(
            kinds,
            vec![BoxKind::Signature, BoxKind::FileType, BoxKind::Codestream]
        );
        let mut offset = 0usize;
        for b in tree.boxes() {
            assert_eq!(b.offset, offset, "box {:?} offset", b.type_code);
            offset += b.total_len();
        }
        assert_eq!(offset, file.len(), "the boxes must tile the file exactly");
    }

    #[test]
    fn the_xlbox_form_frames_the_same_payload_as_the_short_form() {
        // Proves the 16-byte header is measured from the start of the box,
        // not from the end of TBox: an off-by-eight would shift the payload.
        let payload = b"0123456789abcdef";
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&ftyp());
        file.extend_from_slice(&boxed_xl(b"jxlc", payload));
        let tree = tree_of(&file);
        let last = tree.boxes().last().expect("three boxes");
        assert_eq!(last.header_len, 16);
        assert_eq!(last.payload, payload);
        assert_eq!(last.total_len(), payload.len() + 16);
        assert_eq!(
            extract_codestream(&file, &mut guard()).expect("jxlc"),
            payload
        );
    }

    #[test]
    fn a_zero_length_box_runs_to_the_end_of_the_file() {
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&ftyp());
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(b"jxlc");
        file.extend_from_slice(&[0xFF, 0x0A, 9]);
        let tree = tree_of(&file);
        let last = tree.boxes().last().expect("three boxes");
        assert!(last.extends_to_end);
        assert_eq!(
            extract_codestream(&file, &mut guard()).expect("jxlc"),
            vec![0xFF, 0x0A, 9]
        );
    }

    #[test]
    fn unknown_boxes_are_framed_and_skipped() {
        // Clause 8 says an unrecognised type is still a box; the codestream
        // after it must still be found, which is what proves it was skipped
        // by its length rather than scanned past.
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&ftyp());
        file.extend_from_slice(&boxed(b"zzzz", b"jxlc not really"));
        file.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A, 7]));
        let tree = tree_of(&file);
        assert_eq!(tree.boxes()[2].kind, BoxKind::Unknown);
        assert_eq!(&tree.boxes()[2].type_code, b"zzzz");
        assert_eq!(
            extract_codestream(&file, &mut guard()).expect("jxlc"),
            vec![0xFF, 0x0A, 7]
        );
        tree.validate()
            .expect("an unknown box does not break clause 9");
    }

    #[test]
    fn malformed_framing_errors_and_never_panics() {
        // Each case names the clause 8 rule it breaks.
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("LBox below the 8-byte minimum", {
                let mut f = CONTAINER_SIGNATURE.to_vec();
                f.extend_from_slice(&3u32.to_be_bytes());
                f.extend_from_slice(b"jxlc");
                f
            }),
            ("LBox past the end of the file", {
                let mut f = CONTAINER_SIGNATURE.to_vec();
                f.extend_from_slice(&0xFFFF_FFFFu32.to_be_bytes());
                f.extend_from_slice(b"jxlc");
                f
            }),
            ("XLBox below the 16-byte minimum", {
                let mut f = CONTAINER_SIGNATURE.to_vec();
                f.extend_from_slice(&1u32.to_be_bytes());
                f.extend_from_slice(b"jxlc");
                f.extend_from_slice(&15u64.to_be_bytes());
                f
            }),
            ("XLBox past the end of the file", {
                let mut f = CONTAINER_SIGNATURE.to_vec();
                f.extend_from_slice(&1u32.to_be_bytes());
                f.extend_from_slice(b"jxlc");
                f.extend_from_slice(&u64::MAX.to_be_bytes());
                f
            }),
            ("a truncated XLBox header", {
                let mut f = CONTAINER_SIGNATURE.to_vec();
                f.extend_from_slice(&1u32.to_be_bytes());
                f.extend_from_slice(b"jxlc");
                f.extend_from_slice(&[0, 0, 0]);
                f
            }),
            ("a trailing fragment shorter than a box header", {
                let mut f = conforming(&[0xFF, 0x0A]);
                f.extend_from_slice(&[0, 0, 0]);
                f
            }),
        ];
        for (what, file) in cases {
            let err = BoxTree::parse(&file, &mut guard()).expect_err(what);
            assert!(
                err.to_string().contains("18181-2"),
                "{what}: {err} should cite Part 2"
            );
        }
    }

    #[test]
    fn every_prefix_and_every_single_byte_corruption_is_handled() {
        // The whole point of the parser being attacker-facing: no panic, ever.
        let good = conforming(&[0xFF, 0x0A, 1, 2, 3, 4, 5, 6, 7, 8]);
        for cut in 0..=good.len() {
            if let Ok(tree) = BoxTree::parse(&good[..cut], &mut guard()) {
                let _ = tree.validate();
                let _ = tree.codestream(&mut guard());
            }
            let _ = extract_codestream(&good[..cut], &mut guard());
        }
        for index in 0..good.len() {
            for flip in [0x01u8, 0x80, 0xFF] {
                let mut bad = good.clone();
                bad[index] ^= flip;
                if let Ok(tree) = BoxTree::parse(&bad, &mut guard()) {
                    let _ = tree.validate();
                    let _ = tree.codestream(&mut guard());
                    let _ = tree.exif();
                    let _ = tree.brob();
                    let _ = tree.level();
                    let _ = tree.xml();
                    let _ = tree.jumbf();
                }
            }
        }
    }

    #[test]
    fn a_box_flood_is_charged_to_the_guard() {
        // 8-byte empty boxes are the cheapest way to make a parser allocate:
        // every 8 input bytes becomes a heap entry several times that size.
        // The guard is what bounds that, so it is metered per box and the
        // budget below is deliberately small enough to prove the charge is
        // real rather than nominal.
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&ftyp());
        for _ in 0..1_000 {
            file.extend_from_slice(&boxed(b"zzzz", &[]));
        }
        let budget = Limits {
            max_alloc_bytes: BOX_BOOKKEEPING_BYTES * 100,
            ..Limits::default()
        };
        let mut tight = AllocGuard::new(&budget);
        assert!(
            matches!(BoxTree::parse(&file, &mut tight), Err(DecodeError::Core(_))),
            "the guard must fire before the thousandth box is stored"
        );
        // The same file parses under a budget that can afford it, so the test
        // is measuring the meter and not a framing bug.
        assert_eq!(
            BoxTree::parse(&file, &mut guard())
                .expect("relaxed limits")
                .boxes()
                .len(),
            1_002
        );
    }

    // ---- clause 9 structure ---------------------------------------------

    #[test]
    fn a_minimal_conforming_file_validates() {
        let file = conforming(&[0xFF, 0x0A, 1, 2, 3]);
        let tree = tree_of(&file);
        tree.validate()
            .expect("this is the file clause 9 describes");
        assert_eq!(tree.level().expect("no level box"), DEFAULT_LEVEL);
        assert!(tree.has_codestream());
    }

    #[test]
    fn the_level_box_is_read_and_its_position_is_enforced() {
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&ftyp());
        file.extend_from_slice(&boxed(b"jxll", &[10]));
        file.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        let tree = tree_of(&file);
        tree.validate().expect("third box is where 9.3 puts it");
        assert_eq!(tree.level().expect("level box"), 10);

        // Two level boxes: 9.3 says at most one.
        let mut two = CONTAINER_SIGNATURE.to_vec();
        two.extend_from_slice(&ftyp());
        two.extend_from_slice(&boxed(b"jxll", &[10]));
        two.extend_from_slice(&boxed(b"jxll", &[5]));
        two.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        assert!(tree_of(&two).validate().is_err(), "double jxll must fail");

        // A level box after the codestream: 9.3 says third box.
        let mut late = CONTAINER_SIGNATURE.to_vec();
        late.extend_from_slice(&ftyp());
        late.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        late.extend_from_slice(&boxed(b"jxll", &[10]));
        assert!(
            tree_of(&late).validate().is_err(),
            "a misplaced jxll must fail"
        );

        // A two-byte level box: Table 5 is one u8.
        let mut wide = CONTAINER_SIGNATURE.to_vec();
        wide.extend_from_slice(&ftyp());
        wide.extend_from_slice(&boxed(b"jxll", &[10, 0]));
        wide.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        assert!(tree_of(&wide).level().is_err(), "a wide jxll must fail");
        assert!(tree_of(&wide).validate().is_err());
    }

    #[test]
    fn the_signature_and_ftyp_contents_are_checked_not_just_their_types() {
        // A file whose first twelve bytes are a JXL box with the wrong magic
        // still frames; only 9.1's content rule catches it.
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file[11] = 0x00;
        file.extend_from_slice(&ftyp());
        file.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        assert!(tree_of(&file).validate().is_err(), "bad signature content");

        let mut bad_ftyp = CONTAINER_SIGNATURE.to_vec();
        bad_ftyp.extend_from_slice(&boxed(b"ftyp", b"jpg \0\0\0\0jpg "));
        bad_ftyp.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        assert!(tree_of(&bad_ftyp).validate().is_err(), "bad ftyp brand");

        // ftyp missing entirely: framing is fine, clause 9 is not.
        let mut no_ftyp = CONTAINER_SIGNATURE.to_vec();
        no_ftyp.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A, 3]));
        let tree = tree_of(&no_ftyp);
        assert!(tree.validate().is_err(), "9.2 requires a file type box");
        // ... but the codestream is still extractable, which is the whole
        // reason validate() is separate from the decode path.
        assert_eq!(
            extract_codestream(&no_ftyp, &mut guard()).expect("still decodable"),
            vec![0xFF, 0x0A, 3]
        );
    }

    #[test]
    fn a_file_with_both_codestream_forms_is_rejected() {
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&ftyp());
        file.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        let mut fragment = JXLP_LAST_FLAG.to_be_bytes().to_vec();
        fragment.extend_from_slice(&[1, 2]);
        file.extend_from_slice(&boxed(b"jxlp", &fragment));
        let tree = tree_of(&file);
        assert!(tree.validate().is_err(), "9.9 forbids both forms");
        assert!(tree.codestream(&mut guard()).is_err());
    }

    #[test]
    fn a_container_without_a_codestream_box_is_reported() {
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&ftyp());
        file.extend_from_slice(&boxed(b"Exif", b"\0\0\0\0whatever"));
        let err = extract_codestream(&file, &mut guard()).expect_err("no codestream");
        assert!(err.to_string().contains("18181-2"));
        assert!(!tree_of(&file).has_codestream());
    }

    // ---- jxlp (9.10) -----------------------------------------------------

    /// Builds a `jxlp` file from `parts`, marking the last fragment unless
    /// `mark_last` is false.
    fn jxlp_file(parts: &[&[u8]], mark_last: bool) -> Vec<u8> {
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&ftyp());
        for (i, part) in parts.iter().enumerate() {
            let mut index = i as u32;
            if mark_last && i + 1 == parts.len() {
                index |= JXLP_LAST_FLAG;
            }
            let mut payload = index.to_be_bytes().to_vec();
            payload.extend_from_slice(part);
            file.extend_from_slice(&boxed(b"jxlp", &payload));
        }
        file
    }

    #[test]
    fn jxlp_fragments_reassemble_in_index_order() {
        let file = jxlp_file(&[&[0xFF, 0x0A], &[3, 4], &[], &[5]], true);
        let tree = tree_of(&file);
        tree.validate().expect("a conforming jxlp file");
        assert_eq!(
            tree.codestream(&mut guard()).expect("reassembles"),
            vec![0xFF, 0x0A, 3, 4, 5],
            "an empty fragment is legal (9.10 NOTE 2) and contributes nothing"
        );
    }

    #[test]
    fn a_single_jxlp_box_is_both_first_and_last() {
        let file = jxlp_file(&[&[0xFF, 0x0A, 1]], true);
        let tree = tree_of(&file);
        tree.validate().expect("index 0 | 2^31 is first and last");
        assert_eq!(
            tree.codestream(&mut guard()).expect("reassembles"),
            vec![0xFF, 0x0A, 1]
        );
    }

    #[test]
    fn out_of_order_repeated_and_gapped_jxlp_indices_are_rejected() {
        // 9.10 makes the index a position, not a sort key: the boxes shall
        // appear in increasing index order, so any deviation is malformed and
        // silently sorting it would hide a corrupt file.
        let build = |indices: &[u32]| {
            let mut file = CONTAINER_SIGNATURE.to_vec();
            file.extend_from_slice(&ftyp());
            for &index in indices {
                let mut payload = index.to_be_bytes().to_vec();
                payload.push(0xAA);
                file.extend_from_slice(&boxed(b"jxlp", &payload));
            }
            file
        };
        for indices in [
            vec![1u32, JXLP_LAST_FLAG],  // reordered
            vec![0, JXLP_LAST_FLAG],     // repeated index 0
            vec![0, 2 | JXLP_LAST_FLAG], // a gap
            vec![1, 2 | JXLP_LAST_FLAG], // does not start at 0
        ] {
            let file = build(&indices);
            let tree = tree_of(&file);
            assert!(
                tree.codestream(&mut guard()).is_err(),
                "indices {indices:?} must not reassemble"
            );
            assert!(tree.validate().is_err(), "indices {indices:?}");
        }
    }

    #[test]
    fn the_final_fragment_marker_is_a_validation_rule_not_a_reassembly_rule() {
        // Unmarked last fragment: reassembly still works (the indices are a
        // contiguous run), but 9.10's "shall" is broken and validate says so.
        let file = jxlp_file(&[&[0xFF, 0x0A], &[1, 2]], false);
        let tree = tree_of(&file);
        assert_eq!(
            tree.codestream(&mut guard()).expect("reassembles"),
            vec![0xFF, 0x0A, 1, 2]
        );
        assert!(tree.validate().is_err(), "9.10 requires the 2^31 marker");

        // The marker on a middle box is wrong in both directions at once.
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&ftyp());
        for index in [JXLP_LAST_FLAG, 1] {
            let mut payload = index.to_be_bytes().to_vec();
            payload.push(0xAA);
            file.extend_from_slice(&boxed(b"jxlp", &payload));
        }
        assert!(tree_of(&file).validate().is_err(), "marker on box 0 of 2");
    }

    #[test]
    fn a_jxlp_box_too_short_for_its_index_is_rejected() {
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&ftyp());
        file.extend_from_slice(&boxed(b"jxlp", &[0, 0, 0]));
        let tree = tree_of(&file);
        assert!(tree.codestream(&mut guard()).is_err());
    }

    // ---- metadata boxes --------------------------------------------------

    #[test]
    fn exif_splits_its_offset_field_from_its_payload() {
        let mut payload = 6u32.to_be_bytes().to_vec();
        payload.extend_from_slice(b"padding|MM\0*");
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&ftyp());
        file.extend_from_slice(&boxed(b"Exif", &payload));
        file.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        let tree = tree_of(&file);
        tree.validate().expect("conforming");
        let exif = tree.exif().expect("one Exif box");
        assert_eq!(exif.len(), 1);
        assert_eq!(exif[0].tiff_header_offset, 6);
        assert_eq!(exif[0].payload, b"padding|MM\0*");

        // A payload with no room for the u32 is malformed, not empty.
        let mut short = CONTAINER_SIGNATURE.to_vec();
        short.extend_from_slice(&ftyp());
        short.extend_from_slice(&boxed(b"Exif", &[0, 0]));
        short.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        assert!(tree_of(&short).exif().is_err());
    }

    #[test]
    fn several_xml_boxes_are_all_kept() {
        // 9.6 explicitly permits more than one, so returning only the first
        // would silently drop metadata.
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&ftyp());
        file.extend_from_slice(&boxed(b"xml ", b"<a/>"));
        file.extend_from_slice(&boxed(b"jumb", b"jumbf-bytes"));
        file.extend_from_slice(&boxed(b"xml ", b"<b/>"));
        file.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        let tree = tree_of(&file);
        tree.validate().expect("conforming");
        assert_eq!(tree.xml(), vec![&b"<a/>"[..], &b"<b/>"[..]]);
        assert_eq!(tree.jumbf(), vec![&b"jumbf-bytes"[..]]);
    }

    #[test]
    fn brob_exposes_its_target_and_refuses_to_guess_at_decompression() {
        let mut payload = b"xml ".to_vec();
        payload.extend_from_slice(&[0x1B, 0x00, 0x00]); // an opaque Brotli stream
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&ftyp());
        file.extend_from_slice(&boxed(b"brob", &payload));
        file.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        let tree = tree_of(&file);
        tree.validate().expect("conforming");
        let brob = tree.brob().expect("one brob box");
        assert_eq!(brob.len(), 1);
        assert_eq!(&brob[0].target_type_code, b"xml ");
        assert_eq!(brob[0].target_kind, BoxKind::Xml);
        assert_eq!(brob[0].compressed, &[0x1B, 0x00, 0x00]);
        // The target is *not* transparently substituted: an undecompressed
        // brob must not look like an XML box that happens to be empty.
        assert!(tree.xml().is_empty());
        let err = brob[0]
            .decompressed()
            .expect_err("no Brotli in this workspace");
        assert!(matches!(err, DecodeError::Unsupported { .. }), "{err}");
    }

    #[test]
    fn brob_target_types_the_clause_forbids_are_rejected() {
        for target in [&b"brob"[..], b"jbrd", b"jxlc", b"jxlp", b"jxll"] {
            let mut payload = target.to_vec();
            payload.push(0x1B);
            let mut file = CONTAINER_SIGNATURE.to_vec();
            file.extend_from_slice(&ftyp());
            file.extend_from_slice(&boxed(b"brob", &payload));
            file.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
            let tree = tree_of(&file);
            assert!(
                tree.brob().is_err(),
                "9.7 forbids a brob payload type of {:?}",
                core::str::from_utf8(target)
            );
        }
    }

    #[test]
    fn jbrd_and_jxli_are_recognised_and_carried_not_parsed() {
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend_from_slice(&ftyp());
        file.extend_from_slice(&boxed(b"jxli", b"index-bytes"));
        file.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        file.extend_from_slice(&boxed(b"jbrd", b"reconstruction-bytes"));
        let tree = tree_of(&file);
        tree.validate().expect("conforming");
        assert_eq!(tree.frame_index(), Some(&b"index-bytes"[..]));
        assert_eq!(
            tree.jpeg_reconstruction(),
            Some(&b"reconstruction-bytes"[..])
        );
        // Recognising them must not disturb codestream extraction.
        assert_eq!(
            tree.codestream(&mut guard()).expect("jxlc"),
            vec![0xFF, 0x0A]
        );

        // 9.8: zero or one frame index boxes.
        let mut two = CONTAINER_SIGNATURE.to_vec();
        two.extend_from_slice(&ftyp());
        two.extend_from_slice(&boxed(b"jxli", b"a"));
        two.extend_from_slice(&boxed(b"jxli", b"b"));
        two.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        assert!(tree_of(&two).validate().is_err(), "two jxli boxes");
    }

    #[test]
    fn recognises_the_signature_box() {
        assert!(is_container(&CONTAINER_SIGNATURE));
        assert!(!is_container(&[0xFF, 0x0A]));
        assert!(!is_container(&[]));
    }
}
