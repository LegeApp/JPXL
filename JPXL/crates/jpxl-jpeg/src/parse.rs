//! Parse a JPEG-1 bitstream into the typed [`Jpeg`] document model.
//!
//! The parser walks marker segments (B.1), builds the typed tables and
//! headers, and entropy-decodes every scan into coefficient planes. Everything
//! Annex-A reconstruction needs — segment order, table grouping, restart
//! cadence, padding bits, trailing bytes — is captured so the companion
//! [`crate::serialize`] can reproduce the input byte-for-byte.

use crate::bitio::EntropyReader;
use crate::codec::decode_scan;
use crate::error::{JpegError, Result, unsupported};
use crate::frame::{FrameComponent, FrameGeometry, FrameHeader};
use crate::huffman::HuffmanTable;
use crate::limits::Limits;
use crate::marker::{self, SofKind};
use crate::quant::QuantTable;
use crate::scan::{ScanComponent, ScanHeader};
use crate::segment::{AppSegment, ComponentPlane, Jpeg, OtherSegment, ScanSegment, Segment};
use crate::units::ComponentId;

/// A forward-only cursor over the byte stream, for reading segment headers.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn u8(&mut self, ctx: &'static str) -> Result<u8> {
        let b = self
            .data
            .get(self.pos)
            .copied()
            .ok_or(JpegError::UnexpectedEof { while_reading: ctx })?;
        self.pos += 1;
        Ok(b)
    }

    fn u16(&mut self, ctx: &'static str) -> Result<u16> {
        let hi = self.u8(ctx)?;
        let lo = self.u8(ctx)?;
        Ok((u16::from(hi) << 8) | u16::from(lo))
    }

    fn bytes(&mut self, n: usize, ctx: &'static str) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(JpegError::UnexpectedEof { while_reading: ctx })?;
        let slice = self
            .data
            .get(self.pos..end)
            .ok_or(JpegError::UnexpectedEof { while_reading: ctx })?;
        self.pos = end;
        Ok(slice)
    }
}

/// Parses `data` with default [`Limits`].
pub fn parse(data: &[u8]) -> Result<Jpeg> {
    parse_with_limits(data, &Limits::default())
}

/// Parses `data`, enforcing `limits` on all size-bearing fields.
pub fn parse_with_limits(data: &[u8], limits: &Limits) -> Result<Jpeg> {
    if data.len() > limits.max_input_len {
        return Err(JpegError::LimitExceeded(format!(
            "input is {} bytes (cap {})",
            data.len(),
            limits.max_input_len
        )));
    }
    let mut cur = Cursor { data, pos: 0 };
    // SOI.
    if cur.u8("SOI prefix")? != marker::MARKER_PREFIX || cur.u8("SOI code")? != marker::SOI {
        return Err(JpegError::Malformed(
            "stream does not start with SOI (0xFFD8)".into(),
        ));
    }

    let mut segments: Vec<Segment> = Vec::new();
    let mut frame: Option<FrameHeader> = None;
    let mut geom: Option<FrameGeometry> = None;
    let mut planes: Vec<ComponentPlane> = Vec::new();
    let mut dc_tables: [Option<HuffmanTable>; 4] = Default::default();
    let mut ac_tables: [Option<HuffmanTable>; 4] = Default::default();
    let mut restart_interval: u16 = 0;
    let tail;

    loop {
        let code = read_marker(&mut cur)?;
        match code {
            marker::EOI => {
                tail = data.get(cur.pos..).unwrap_or(&[]).to_vec();
                break;
            }
            marker::SOI => return Err(JpegError::Malformed("unexpected second SOI".into())),
            marker::TEM => {
                // Standalone, no payload.
                segments.push(Segment::Other(OtherSegment {
                    code,
                    payload: Vec::new(),
                }));
            }
            c if marker::is_restart(c) => {
                return Err(JpegError::Malformed(
                    "restart marker outside an entropy-coded segment".into(),
                ));
            }
            marker::DAC => {
                return Err(unsupported!(
                    "arithmetic coding (DAC, 0xFFCC) is out of scope"
                ));
            }
            marker::DHP | marker::EXP => {
                return Err(unsupported!("hierarchical mode (DHP/EXP) is out of scope"));
            }
            c if marker::is_sof(c) => {
                if frame.is_some() {
                    return Err(JpegError::Malformed("more than one SOF in a frame".into()));
                }
                let fh = parse_sof(&mut cur, c)?;
                let g = fh.geometry()?;
                planes = allocate_planes(&fh, &g, limits)?;
                geom = Some(g);
                segments.push(Segment::Sof(fh.clone()));
                frame = Some(fh);
            }
            marker::DHT => {
                let tables = parse_dht(&mut cur)?;
                for t in &tables {
                    let slot = if t.class == 0 {
                        &mut dc_tables
                    } else {
                        &mut ac_tables
                    };
                    // `id` is validated to 0..=3 in `parse_dht`, so this slot
                    // always exists.
                    if let Some(dst) = slot.get_mut(t.id as usize) {
                        *dst = Some(t.clone());
                    }
                }
                segments.push(Segment::Dht(tables));
            }
            marker::DQT => {
                let tables = parse_dqt(&mut cur)?;
                segments.push(Segment::Dqt(tables));
            }
            marker::DRI => {
                let len = cur.u16("DRI length")?;
                if len != 4 {
                    return Err(JpegError::Malformed(format!(
                        "B.2.4.4: DRI length {len} != 4"
                    )));
                }
                restart_interval = cur.u16("DRI interval")?;
                segments.push(Segment::Dri(restart_interval));
            }
            marker::SOS => {
                let header = parse_sos(&mut cur)?;
                let fh = frame
                    .as_ref()
                    .ok_or_else(|| JpegError::Malformed("SOS before SOF".into()))?;
                let g = geom
                    .as_ref()
                    .ok_or_else(|| JpegError::Malformed("SOS before SOF".into()))?;
                let mut reader = EntropyReader::new(data, cur.pos);
                let (padding, eob_runs) = decode_scan(
                    fh,
                    g,
                    &mut planes,
                    &header,
                    &dc_tables,
                    &ac_tables,
                    restart_interval,
                    &mut reader,
                )?;
                cur.pos = reader.byte_pos();
                segments.push(Segment::Sos(ScanSegment {
                    header,
                    padding,
                    eob_runs,
                }));
            }
            c @ (marker::APP0..=marker::APP15) => {
                let payload = read_length_payload(&mut cur, "APPn")?;
                segments.push(Segment::App(AppSegment { code: c, payload }));
            }
            marker::COM => {
                let payload = read_length_payload(&mut cur, "COM")?;
                segments.push(Segment::Com(payload));
            }
            other => {
                // Any other length-bearing marker (e.g. DNL): preserve verbatim.
                let payload = read_length_payload(&mut cur, "marker segment")?;
                segments.push(Segment::Other(OtherSegment {
                    code: other,
                    payload,
                }));
            }
        }
    }

    if frame.is_none() {
        return Err(JpegError::Malformed("no SOF segment before EOI".into()));
    }

    Ok(Jpeg {
        segments,
        frame,
        planes,
        tail,
    })
}

/// Reads the next marker code, skipping any leading `0xFF` fill bytes.
fn read_marker(cur: &mut Cursor<'_>) -> Result<u8> {
    let mut saw_ff = false;
    loop {
        let b = cur.u8("marker")?;
        if b == marker::MARKER_PREFIX {
            saw_ff = true;
            continue;
        }
        if !saw_ff {
            return Err(JpegError::Malformed(format!(
                "expected a marker prefix 0xFF, found 0x{b:02X}"
            )));
        }
        if b == 0x00 {
            return Err(JpegError::Malformed(
                "0xFF00 where a marker was expected".into(),
            ));
        }
        return Ok(b);
    }
}

/// Reads a `[length][payload]` segment body, returning the payload.
fn read_length_payload(cur: &mut Cursor<'_>, ctx: &'static str) -> Result<Vec<u8>> {
    let len = cur.u16(ctx)?;
    if len < 2 {
        return Err(JpegError::Malformed(format!(
            "{ctx} segment length {len} < 2"
        )));
    }
    Ok(cur.bytes(len as usize - 2, ctx)?.to_vec())
}

fn parse_sof(cur: &mut Cursor<'_>, code: u8) -> Result<FrameHeader> {
    let kind = SofKind::from_code(code)
        .ok_or_else(|| JpegError::Malformed(format!("0xFF{code:02X} is not a SOF marker")))?;
    match kind {
        SofKind::Arithmetic => {
            return Err(unsupported!("arithmetic-coded JPEG (SOF 0x{code:02X})"));
        }
        SofKind::Hierarchical => {
            return Err(unsupported!("hierarchical JPEG (SOF 0x{code:02X})"));
        }
        SofKind::Lossless => {
            return Err(unsupported!("lossless JPEG (SOF3)"));
        }
        _ => {}
    }
    let len = cur.u16("SOF length")?;
    let precision = cur.u8("SOF precision")?;
    if precision != 8 {
        return Err(unsupported!(
            "{precision}-bit samples (only 8-bit precision is in scope)"
        ));
    }
    let height = cur.u16("SOF Y")?;
    let width = cur.u16("SOF X")?;
    let nf = cur.u8("SOF Nf")?;
    if nf == 0 || nf > 4 {
        return Err(JpegError::Malformed(format!(
            "B.2.2: component count Nf={nf} not in 1..=4"
        )));
    }
    let expected_len = 8 + 3 * nf as u16;
    if len != expected_len {
        return Err(JpegError::Malformed(format!(
            "B.2.2: SOF length {len} != {expected_len} for Nf={nf}"
        )));
    }
    if width == 0 {
        return Err(JpegError::Malformed("B.2.2: SOF X (width) is 0".into()));
    }
    // Y == 0 is legal (lines defined later by DNL); reject it as out of scope
    // rather than mis-parse, since DNL handling is not implemented.
    if height == 0 {
        return Err(unsupported!("SOF Y (height) = 0 with deferred DNL height"));
    }
    let mut components = Vec::with_capacity(nf as usize);
    for _ in 0..nf {
        let id = ComponentId(cur.u8("SOF component id")?);
        let hv = cur.u8("SOF sampling")?;
        let h = hv >> 4;
        let v = hv & 0x0F;
        if !(1..=4).contains(&h) || !(1..=4).contains(&v) {
            return Err(JpegError::Malformed(format!(
                "B.2.2: sampling factors H={h} V={v} not in 1..=4"
            )));
        }
        let quant_id = cur.u8("SOF Tq")?;
        if quant_id > 3 {
            return Err(JpegError::Malformed(format!(
                "B.2.2: quant selector Tq={quant_id} > 3"
            )));
        }
        components.push(FrameComponent { id, h, v, quant_id });
    }
    // Component ids must be distinct (they are matched by SOS).
    let mut seen: Vec<u8> = Vec::with_capacity(components.len());
    for c in &components {
        if seen.contains(&c.id.0) {
            return Err(JpegError::Malformed(format!(
                "B.2.2: duplicate component id {}",
                c.id.0
            )));
        }
        seen.push(c.id.0);
    }
    Ok(FrameHeader {
        code,
        kind,
        precision,
        height,
        width,
        components,
    })
}

fn allocate_planes(
    fh: &FrameHeader,
    geom: &FrameGeometry,
    limits: &Limits,
) -> Result<Vec<ComponentPlane>> {
    let mut planes = Vec::with_capacity(fh.components.len());
    let mut total: u64 = 0;
    for (idx, fc) in fh.components.iter().enumerate() {
        let dims = fh.component_dims(idx, geom)?;
        let bpl = dims.blocks_per_line_interleaved;
        let bh = dims.block_rows_interleaved;
        let count = bpl
            .checked_mul(bh)
            .ok_or_else(|| JpegError::LimitExceeded("block count overflow".into()))?;
        total = total.saturating_add(count as u64);
        if total > limits.max_total_blocks {
            return Err(JpegError::LimitExceeded(format!(
                "total blocks {total} exceed cap {}",
                limits.max_total_blocks
            )));
        }
        planes.push(ComponentPlane {
            id: fc.id,
            h: fc.h,
            v: fc.v,
            blocks_per_line: bpl,
            block_rows: bh,
            blocks: vec![[0i16; 64]; count],
        });
    }
    Ok(planes)
}

fn parse_dht(cur: &mut Cursor<'_>) -> Result<Vec<HuffmanTable>> {
    let len = cur.u16("DHT length")?;
    if len < 2 {
        return Err(JpegError::Malformed("DHT length < 2".into()));
    }
    let end = cur.pos + len as usize - 2;
    let mut tables = Vec::new();
    while cur.pos < end {
        let tc_th = cur.u8("DHT Tc/Th")?;
        let class = tc_th >> 4;
        let id = tc_th & 0x0F;
        if class > 1 {
            return Err(JpegError::Malformed(format!(
                "B.2.4.2: Huffman table class Tc={class} > 1"
            )));
        }
        if id > 3 {
            return Err(JpegError::Malformed(format!(
                "B.2.4.2: Huffman table id Th={id} > 3"
            )));
        }
        let mut counts = [0u8; 16];
        let count_bytes = cur.bytes(16, "DHT counts")?;
        counts.copy_from_slice(count_bytes);
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        let values = cur.bytes(total, "DHT values")?.to_vec();
        tables.push(HuffmanTable::new(class, id, counts, values)?);
    }
    if cur.pos != end {
        return Err(JpegError::Malformed(
            "DHT segment length does not match its tables".into(),
        ));
    }
    Ok(tables)
}

fn parse_dqt(cur: &mut Cursor<'_>) -> Result<Vec<QuantTable>> {
    let len = cur.u16("DQT length")?;
    if len < 2 {
        return Err(JpegError::Malformed("DQT length < 2".into()));
    }
    let end = cur.pos + len as usize - 2;
    let mut tables = Vec::new();
    while cur.pos < end {
        let pq_tq = cur.u8("DQT Pq/Tq")?;
        let precision = pq_tq >> 4;
        let id = pq_tq & 0x0F;
        if precision > 1 {
            return Err(JpegError::Malformed(format!(
                "B.2.4.1: quant precision Pq={precision} > 1"
            )));
        }
        if id > 3 {
            return Err(JpegError::Malformed(format!(
                "B.2.4.1: quant table id Tq={id} > 3"
            )));
        }
        let mut values = [0u16; 64];
        if precision == 0 {
            let raw = cur.bytes(64, "DQT 8-bit values")?;
            for (dst, &src) in values.iter_mut().zip(raw) {
                *dst = u16::from(src);
            }
        } else {
            let raw = cur.bytes(128, "DQT 16-bit values")?;
            for (dst, pair) in values.iter_mut().zip(raw.chunks_exact(2)) {
                *dst = u16::from_be_bytes(pair.try_into().expect("chunks_exact(2)"));
            }
        }
        tables.push(QuantTable {
            precision,
            id,
            values,
        });
    }
    if cur.pos != end {
        return Err(JpegError::Malformed(
            "DQT segment length does not match its tables".into(),
        ));
    }
    Ok(tables)
}

fn parse_sos(cur: &mut Cursor<'_>) -> Result<ScanHeader> {
    let len = cur.u16("SOS length")?;
    let ns = cur.u8("SOS Ns")?;
    if ns == 0 || ns > 4 {
        return Err(JpegError::Malformed(format!(
            "B.2.3: scan component count Ns={ns} not in 1..=4"
        )));
    }
    let expected_len = 6 + 2 * ns as u16;
    if len != expected_len {
        return Err(JpegError::Malformed(format!(
            "B.2.3: SOS length {len} != {expected_len} for Ns={ns}"
        )));
    }
    let mut components = Vec::with_capacity(ns as usize);
    for _ in 0..ns {
        let id = ComponentId(cur.u8("SOS Cs")?);
        let td_ta = cur.u8("SOS Td/Ta")?;
        let dc_table = td_ta >> 4;
        let ac_table = td_ta & 0x0F;
        if dc_table > 3 || ac_table > 3 {
            return Err(JpegError::Malformed(format!(
                "B.2.3: table selectors Td={dc_table} Ta={ac_table} > 3"
            )));
        }
        components.push(ScanComponent {
            id,
            dc_table,
            ac_table,
        });
    }
    let spectral_start = cur.u8("SOS Ss")?;
    let spectral_end = cur.u8("SOS Se")?;
    let approx = cur.u8("SOS Ah/Al")?;
    let approx_high = approx >> 4;
    let approx_low = approx & 0x0F;
    if spectral_start > 63 || spectral_end > 63 {
        return Err(JpegError::Malformed(format!(
            "B.2.3: spectral selection Ss={spectral_start} Se={spectral_end} out of 0..=63"
        )));
    }
    Ok(ScanHeader {
        components,
        spectral_start,
        spectral_end,
        approx_high,
        approx_low,
    })
}
