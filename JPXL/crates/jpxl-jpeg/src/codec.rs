//! Entropy decode and encode of scan data (10918-1 F.1.2 / F.2.2 baseline;
//! G.1.2 progressive).
//!
//! Decode and encode walk the *same* block enumeration ([`ScanGeom`]) in the
//! same order, so the encoder is a faithful inverse of the decoder rather than
//! a separate interpretation of the scan geometry. That shared layout is what
//! lets `serialize(parse(x)) == x` hold: the coefficients, the Huffman codes,
//! the restart cadence and the padding all line up bit-for-bit. The
//! progressive coder in [`crate::progressive`] reuses this same geometry.

// Coefficient indices come from `ZIGZAG_TO_NATURAL[k]` with `k` bounded to
// `1..=63`, component/prediction indices are bounded by the component count
// asserted when the vectors were sized, and every `as u8` narrows a value the
// code has just checked to `0..=15` (magnitude categories, run lengths). Bulk
// plane/block access still goes through checked `.get()`.
#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

use crate::bitio::{EntropyReader, EntropyWriter, Padding};
use crate::coeff::{extend, magnitude_category, mantissa_bits, to_coeff};
use crate::error::{JpegError, Result};
use crate::frame::{FrameGeometry, FrameHeader};
use crate::huffman::HuffmanTable;
use crate::marker::is_restart;
use crate::scan::ScanHeader;
use crate::segment::ComponentPlane;
use crate::units::ZIGZAG_TO_NATURAL;

/// Per-scan-component geometry and table bindings.
pub(crate) struct CompGeom<'a> {
    /// Index into `frame.components` / `planes`.
    pub frame_idx: usize,
    /// Horizontal sampling factor.
    pub h: usize,
    /// Vertical sampling factor.
    pub v: usize,
    /// DC Huffman table for this component (None for AC-only scans).
    pub dc: Option<&'a HuffmanTable>,
    /// AC Huffman table for this component (None for DC-only scans).
    pub ac: Option<&'a HuffmanTable>,
}

/// The block-visiting order for one scan.
pub(crate) struct ScanGeom<'a> {
    pub interleaved: bool,
    pub mcus_per_line: usize,
    /// Number of units (MCUs if interleaved, else blocks).
    pub num_units: usize,
    pub comps: Vec<CompGeom<'a>>,
    /// For non-interleaved scans, the single component's block-line width.
    pub ni_blocks_per_line: usize,
}

/// A single block reference: which plane and where in it.
pub(crate) struct BlockRef {
    pub frame_idx: usize,
    pub bx: usize,
    pub by: usize,
}

impl<'a> ScanGeom<'a> {
    /// Resolves the scan geometry and Huffman-table bindings.
    pub(crate) fn resolve(
        frame: &FrameHeader,
        geom: &FrameGeometry,
        header: &ScanHeader,
        dc_tables: &'a [Option<HuffmanTable>; 4],
        ac_tables: &'a [Option<HuffmanTable>; 4],
    ) -> Result<Self> {
        let interleaved = header.is_interleaved();
        let need_dc = header.spectral_start == 0;
        let need_ac = header.spectral_end != 0;
        let mut comps = Vec::with_capacity(header.components.len());
        for sc in &header.components {
            let frame_idx = frame
                .components
                .iter()
                .position(|fc| fc.id == sc.id)
                .ok_or_else(|| {
                    JpegError::Malformed(format!(
                        "B.2.3: scan component id {} not present in frame",
                        sc.id.0
                    ))
                })?;
            let fc = &frame.components[frame_idx];
            let dc = if need_dc {
                Some(pick_table(dc_tables, sc.dc_table, "DC")?)
            } else {
                None
            };
            let ac = if need_ac {
                Some(pick_table(ac_tables, sc.ac_table, "AC")?)
            } else {
                None
            };
            comps.push(CompGeom {
                frame_idx,
                h: fc.h as usize,
                v: fc.v as usize,
                dc,
                ac,
            });
        }

        let (num_units, ni_blocks_per_line) = if interleaved {
            (geom.mcus_per_line * geom.mcu_rows, 0)
        } else {
            let idx = comps
                .first()
                .ok_or_else(|| JpegError::Malformed("B.2.3: scan has no components".into()))?
                .frame_idx;
            let dims = frame.component_dims(idx, geom)?;
            (
                dims.blocks_per_line_noninterleaved * dims.block_rows_noninterleaved,
                dims.blocks_per_line_noninterleaved,
            )
        };

        Ok(Self {
            interleaved,
            mcus_per_line: geom.mcus_per_line,
            num_units,
            comps,
            ni_blocks_per_line,
        })
    }

    /// Appends the block references for unit `u` to `out` (in scan order).
    pub(crate) fn unit_blocks(&self, u: usize, out: &mut Vec<BlockRef>) {
        out.clear();
        if self.interleaved {
            let mx = u % self.mcus_per_line;
            let my = u / self.mcus_per_line;
            for c in &self.comps {
                for vv in 0..c.v {
                    for hh in 0..c.h {
                        out.push(BlockRef {
                            frame_idx: c.frame_idx,
                            bx: mx * c.h + hh,
                            by: my * c.v + vv,
                        });
                    }
                }
            }
        } else if let Some(c) = self.comps.first() {
            let bx = u % self.ni_blocks_per_line;
            let by = u / self.ni_blocks_per_line;
            out.push(BlockRef {
                frame_idx: c.frame_idx,
                bx,
                by,
            });
        }
    }

    /// The `CompGeom` a block belongs to (by matching `frame_idx`).
    pub(crate) fn comp_for(&self, frame_idx: usize) -> Result<&CompGeom<'a>> {
        self.comps
            .iter()
            .find(|c| c.frame_idx == frame_idx)
            .ok_or_else(|| JpegError::Malformed("no scan component for block".into()))
    }
}

fn pick_table<'a>(
    tables: &'a [Option<HuffmanTable>; 4],
    id: u8,
    which: &str,
) -> Result<&'a HuffmanTable> {
    tables
        .get(id as usize)
        .and_then(|t| t.as_ref())
        .ok_or_else(|| {
            JpegError::Malformed(format!(
                "B.2.3: scan selects undefined {which} Huffman table {id}"
            ))
        })
}

pub(crate) fn block_mut(
    planes: &mut [ComponentPlane],
    frame_idx: usize,
    bx: usize,
    by: usize,
) -> Result<&mut [i16; 64]> {
    let plane = planes
        .get_mut(frame_idx)
        .ok_or_else(|| JpegError::Malformed("plane index out of range".into()))?;
    let stride = plane.blocks_per_line;
    plane
        .blocks
        .get_mut(by * stride + bx)
        .ok_or_else(|| JpegError::Malformed("block position out of range".into()))
}

/// Whether this scan is baseline/sequential (a single full-band scan).
fn is_sequential(frame: &FrameHeader, header: &ScanHeader) -> bool {
    !frame.kind.is_progressive()
        && header.spectral_start == 0
        && header.spectral_end == 63
        && header.approx_high == 0
        && header.approx_low == 0
}

/// Decodes one scan's entropy data into `planes`, returning the per-segment
/// padding and (for a progressive AC scan) the observed EOB-run lengths.
#[allow(clippy::too_many_arguments)]
pub fn decode_scan(
    frame: &FrameHeader,
    geom: &FrameGeometry,
    planes: &mut [ComponentPlane],
    header: &ScanHeader,
    dc_tables: &[Option<HuffmanTable>; 4],
    ac_tables: &[Option<HuffmanTable>; 4],
    restart_interval: u16,
    reader: &mut EntropyReader<'_>,
) -> Result<(Vec<Padding>, Vec<u32>)> {
    if frame.kind.is_progressive() {
        return crate::progressive::decode_progressive_scan(
            frame,
            geom,
            planes,
            header,
            dc_tables,
            ac_tables,
            restart_interval,
            reader,
        );
    }
    if !is_sequential(frame, header) {
        return Err(JpegError::Malformed(format!(
            "B.2.3: sequential frame with progressive scan parameters (Ss={}, Se={}, Ah={}, Al={})",
            header.spectral_start, header.spectral_end, header.approx_high, header.approx_low
        )));
    }

    let sg = ScanGeom::resolve(frame, geom, header, dc_tables, ac_tables)?;
    let mut dc_pred = vec![0i32; frame.components.len()];
    let mut padding = Vec::new();
    let mut refs: Vec<BlockRef> = Vec::new();
    let ri = restart_interval as usize;

    for u in 0..sg.num_units {
        if ri != 0 && u != 0 && u % ri == 0 {
            padding.push(finish_restart_segment(reader)?);
            for p in dc_pred.iter_mut() {
                *p = 0;
            }
        }
        sg.unit_blocks(u, &mut refs);
        for r in &refs {
            let comp = sg.comp_for(r.frame_idx)?;
            let dc_tbl = comp.dc.ok_or_else(|| miss("DC"))?;
            let ac_tbl = comp.ac.ok_or_else(|| miss("AC"))?;
            let block = block_mut(planes, r.frame_idx, r.bx, r.by)?;
            decode_block_sequential(reader, dc_tbl, ac_tbl, &mut dc_pred[r.frame_idx], block)?;
        }
    }
    padding.push(reader.take_padding()?);
    Ok((padding, Vec::new()))
}

pub(crate) fn miss(which: &str) -> JpegError {
    JpegError::Malformed(format!("scan component lacks a {which} Huffman table"))
}

/// Consumes segment padding and the restart marker, resuming a fresh segment.
pub(crate) fn finish_restart_segment(reader: &mut EntropyReader<'_>) -> Result<Padding> {
    let pad = reader.take_padding()?;
    let code = reader.probe_marker()?;
    if !is_restart(code) {
        return Err(JpegError::Malformed(format!(
            "expected a restart marker between intervals, found 0xFF{code:02X}"
        )));
    }
    reader.resume_after_restart()?;
    Ok(pad)
}

fn decode_block_sequential(
    reader: &mut EntropyReader<'_>,
    dc: &HuffmanTable,
    ac: &HuffmanTable,
    dc_pred: &mut i32,
    block: &mut [i16; 64],
) -> Result<()> {
    let t = u32::from(dc.decode(reader)?);
    if t > 15 {
        return Err(JpegError::Malformed(format!(
            "F.1.2: DC magnitude category {t} > 15"
        )));
    }
    let diff = extend(reader.read_bits(t)?, t);
    *dc_pred = dc_pred.wrapping_add(diff);
    block[0] = to_coeff(*dc_pred)?;

    let mut k = 1usize;
    while k <= 63 {
        let rs = ac.decode(reader)?;
        let r = (rs >> 4) as usize;
        let s = u32::from(rs & 0x0F);
        if s == 0 {
            if r == 15 {
                k += 16;
                continue;
            }
            break; // EOB
        }
        k += r;
        if k > 63 {
            return Err(JpegError::Malformed(
                "F.1.2: AC run overruns the block (k > 63)".into(),
            ));
        }
        let val = extend(reader.read_bits(s)?, s);
        block[ZIGZAG_TO_NATURAL[k] as usize] = to_coeff(val)?;
        k += 1;
    }
    Ok(())
}

/// Encodes one scan's entropy data from `planes` into `out`, reproducing the
/// captured padding. `out` receives the raw entropy bytes and any restart
/// markers, but not the terminating marker.
#[allow(clippy::too_many_arguments)]
pub fn encode_scan(
    out: &mut Vec<u8>,
    frame: &FrameHeader,
    geom: &FrameGeometry,
    planes: &[ComponentPlane],
    header: &ScanHeader,
    dc_tables: &[Option<HuffmanTable>; 4],
    ac_tables: &[Option<HuffmanTable>; 4],
    restart_interval: u16,
    padding: &[Padding],
    eob_runs: &[u32],
) -> Result<()> {
    if frame.kind.is_progressive() {
        return crate::progressive::encode_progressive_scan(
            out,
            frame,
            geom,
            planes,
            header,
            dc_tables,
            ac_tables,
            restart_interval,
            padding,
            eob_runs,
        );
    }
    if !is_sequential(frame, header) {
        return Err(JpegError::Encode(
            "sequential frame with progressive scan parameters".into(),
        ));
    }

    let sg = ScanGeom::resolve(frame, geom, header, dc_tables, ac_tables)?;
    let mut dc_pred = vec![0i32; frame.components.len()];
    let ri = restart_interval as usize;
    let mut pad_iter = padding.iter();
    let mut refs: Vec<BlockRef> = Vec::new();
    let mut restart_counter = 0u8;

    let mut writer = EntropyWriter::new(out);
    for u in 0..sg.num_units {
        if ri != 0 && u != 0 && u % ri == 0 {
            let pad = *pad_iter
                .next()
                .ok_or_else(|| JpegError::Encode("missing padding for restart segment".into()))?;
            writer.flush_padding(pad)?;
            writer.write_aligned_bytes(&[0xFF, crate::marker::RST0 + (restart_counter & 7)])?;
            restart_counter = restart_counter.wrapping_add(1);
            for p in dc_pred.iter_mut() {
                *p = 0;
            }
        }
        sg.unit_blocks(u, &mut refs);
        for r in &refs {
            let comp = sg.comp_for(r.frame_idx)?;
            let dc_tbl = comp.dc.ok_or_else(|| miss("DC"))?;
            let ac_tbl = comp.ac.ok_or_else(|| miss("AC"))?;
            let plane = planes
                .get(r.frame_idx)
                .ok_or_else(|| JpegError::Encode("plane index out of range".into()))?;
            let block = plane
                .block(r.bx, r.by)
                .ok_or_else(|| JpegError::Encode("block position out of range".into()))?;
            encode_block_sequential(
                &mut writer,
                dc_tbl,
                ac_tbl,
                &mut dc_pred[r.frame_idx],
                block,
            )?;
        }
    }
    let pad = *pad_iter
        .next()
        .ok_or_else(|| JpegError::Encode("missing padding for final segment".into()))?;
    writer.flush_padding(pad)?;
    Ok(())
}

fn encode_block_sequential(
    writer: &mut EntropyWriter<'_>,
    dc: &HuffmanTable,
    ac: &HuffmanTable,
    dc_pred: &mut i32,
    block: &[i16; 64],
) -> Result<()> {
    let diff = i32::from(block[0]) - *dc_pred;
    *dc_pred = i32::from(block[0]);
    let s = magnitude_category(diff);
    if s > 15 {
        return Err(JpegError::Encode(format!(
            "DC diff {diff} needs category {s} > 15"
        )));
    }
    dc.encode(writer, s as u8)?;
    writer.put_bits(mantissa_bits(diff, s), s);

    let mut run = 0usize;
    for k in 1..=63usize {
        let coeff = i32::from(block[ZIGZAG_TO_NATURAL[k] as usize]);
        if coeff == 0 {
            run += 1;
            continue;
        }
        while run >= 16 {
            ac.encode(writer, 0xF0)?; // ZRL
            run -= 16;
        }
        let s = magnitude_category(coeff);
        if s == 0 || s > 15 {
            return Err(JpegError::Encode(format!(
                "AC coefficient {coeff} has category {s}"
            )));
        }
        let rs = ((run as u8) << 4) | (s as u8);
        ac.encode(writer, rs)?;
        writer.put_bits(mantissa_bits(coeff, s), s);
        run = 0;
    }
    if run > 0 {
        ac.encode(writer, 0x00)?; // EOB
    }
    Ok(())
}
