//! Re-emit a [`Jpeg`] document model back to a byte stream.
//!
//! This is the exact inverse of [`crate::parse`]: it walks the ordered segment
//! list, re-emits every marker segment from its typed form, and re-encodes each
//! scan's coefficients through [`crate::codec::encode_scan`], reproducing the
//! captured padding and restart cadence. For a stream this codec parsed,
//! `serialize(parse(x)) == x` byte-for-byte.

use crate::codec::encode_scan;
use crate::error::{JpegError, Result};
use crate::huffman::HuffmanTable;
use crate::marker;
use crate::quant::QuantTable;
use crate::segment::{Jpeg, Segment};

/// Serializes `jpeg` to bytes.
pub fn serialize(jpeg: &Jpeg) -> Result<Vec<u8>> {
    let frame = jpeg
        .frame
        .as_ref()
        .ok_or_else(|| JpegError::Encode("cannot serialize a frame-less JPEG".into()))?;
    let geom = frame.geometry()?;

    let mut out = Vec::new();
    write_marker(&mut out, marker::SOI);

    let mut dc_tables: [Option<HuffmanTable>; 4] = Default::default();
    let mut ac_tables: [Option<HuffmanTable>; 4] = Default::default();
    let mut restart_interval: u16 = 0;

    for seg in &jpeg.segments {
        match seg {
            Segment::App(a) => {
                write_marker(&mut out, a.code);
                write_length_payload(&mut out, &a.payload)?;
            }
            Segment::Com(payload) => {
                write_marker(&mut out, marker::COM);
                write_length_payload(&mut out, payload)?;
            }
            Segment::Dqt(tables) => {
                write_marker(&mut out, marker::DQT);
                write_dqt(&mut out, tables)?;
            }
            Segment::Dht(tables) => {
                for t in tables {
                    let slot = if t.class == 0 {
                        &mut dc_tables
                    } else {
                        &mut ac_tables
                    };
                    if let Some(dst) = slot.get_mut(t.id as usize) {
                        *dst = Some(t.clone());
                    }
                }
                write_marker(&mut out, marker::DHT);
                write_dht(&mut out, tables)?;
            }
            Segment::Dri(ri) => {
                restart_interval = *ri;
                write_marker(&mut out, marker::DRI);
                write_u16(&mut out, 4);
                write_u16(&mut out, *ri);
            }
            Segment::Sof(fh) => {
                write_marker(&mut out, fh.code);
                write_sof(&mut out, fh)?;
            }
            Segment::Sos(scan) => {
                write_marker(&mut out, marker::SOS);
                write_sos(&mut out, &scan.header)?;
                encode_scan(
                    &mut out,
                    frame,
                    &geom,
                    &jpeg.planes,
                    &scan.header,
                    &dc_tables,
                    &ac_tables,
                    restart_interval,
                    &scan.padding,
                    &scan.eob_runs,
                )?;
            }
            Segment::Other(o) => {
                write_marker(&mut out, o.code);
                if o.code != marker::TEM {
                    write_length_payload(&mut out, &o.payload)?;
                }
            }
        }
    }

    write_marker(&mut out, marker::EOI);
    out.extend_from_slice(&jpeg.tail);
    Ok(out)
}

fn write_marker(out: &mut Vec<u8>, code: u8) {
    out.push(marker::MARKER_PREFIX);
    out.push(code);
}

fn write_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn segment_len(payload_bytes: usize, ctx: &'static str) -> Result<u16> {
    u16::try_from(payload_bytes + 2)
        .map_err(|_| JpegError::Encode(format!("{ctx} segment exceeds 65535 bytes")))
}

fn write_length_payload(out: &mut Vec<u8>, payload: &[u8]) -> Result<()> {
    write_u16(out, segment_len(payload.len(), "segment")?);
    out.extend_from_slice(payload);
    Ok(())
}

fn write_dqt(out: &mut Vec<u8>, tables: &[QuantTable]) -> Result<()> {
    let mut body = Vec::new();
    for t in tables {
        body.push((t.precision << 4) | t.id);
        if t.precision == 0 {
            for &v in &t.values {
                body.push(v.to_be_bytes()[1]);
            }
        } else {
            for &v in &t.values {
                body.extend_from_slice(&v.to_be_bytes());
            }
        }
    }
    write_u16(out, segment_len(body.len(), "DQT")?);
    out.extend_from_slice(&body);
    Ok(())
}

fn write_dht(out: &mut Vec<u8>, tables: &[HuffmanTable]) -> Result<()> {
    let mut body = Vec::new();
    for t in tables {
        body.push((t.class << 4) | t.id);
        body.extend_from_slice(&t.counts);
        body.extend_from_slice(&t.values);
    }
    write_u16(out, segment_len(body.len(), "DHT")?);
    out.extend_from_slice(&body);
    Ok(())
}

fn write_sof(out: &mut Vec<u8>, fh: &crate::frame::FrameHeader) -> Result<()> {
    let nf = u8::try_from(fh.components.len())
        .map_err(|_| JpegError::Encode("SOF component count > 255".into()))?;
    write_u16(out, 8 + 3 * u16::from(nf));
    out.push(fh.precision);
    write_u16(out, fh.height);
    write_u16(out, fh.width);
    out.push(nf);
    for c in &fh.components {
        out.push(c.id.0);
        out.push((c.h << 4) | c.v);
        out.push(c.quant_id);
    }
    Ok(())
}

fn write_sos(out: &mut Vec<u8>, header: &crate::scan::ScanHeader) -> Result<()> {
    let ns = u8::try_from(header.components.len())
        .map_err(|_| JpegError::Encode("SOS component count > 255".into()))?;
    write_u16(out, 6 + 2 * u16::from(ns));
    out.push(ns);
    for c in &header.components {
        out.push(c.id.0);
        out.push((c.dc_table << 4) | c.ac_table);
    }
    out.push(header.spectral_start);
    out.push(header.spectral_end);
    out.push((header.approx_high << 4) | header.approx_low);
    Ok(())
}
