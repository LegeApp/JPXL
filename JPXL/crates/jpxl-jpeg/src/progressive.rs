//! Progressive DCT scan coding (10918-1 G.1.2): DC and AC, first and
//! refinement scans, with spectral selection and successive approximation.
//!
//! Each coefficient is coded across several scans. The decoder accumulates the
//! full coefficient into the plane; the encoder re-derives every scan's symbols
//! from that final coefficient with the matching *point transform*: an
//! arithmetic (floor) shift for DC, a toward-zero shift for AC. Because the
//! plane holds the fully-refined value, `serialize(parse(x)) == x` reduces to
//! each scan's coder being an exact inverse of the other — no intermediate
//! per-scan state is stored.
//!
//! The EOB-run batching and the interleaving of correction bits with
//! zero-run-length (`ZRL`) codes follow the sequence the reference procedures
//! produce, so a stream from a standard encoder is reproduced bit-for-bit.

// Coefficient indices come from `nat(k)` (masked to `0..=63`); `run`/`rem` are
// bounded to `0..=15` before the `as u8` run-length narrowings, and EOB-run and
// correction-bit buffers are indexed by cursors the loops keep in range. Bulk
// plane/block access goes through checked `.get()`.
#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

use crate::bitio::{EntropyReader, EntropyWriter, Padding};
use crate::codec::{ScanGeom, block_mut, finish_restart_segment, miss};
use crate::coeff::{extend, magnitude_category, mantissa_bits, to_coeff};
use crate::error::{JpegError, Result};
use crate::frame::{FrameGeometry, FrameHeader};
use crate::huffman::HuffmanTable;
use crate::scan::ScanHeader;
use crate::segment::ComponentPlane;
use crate::units::ZIGZAG_TO_NATURAL;

#[inline]
fn nat(k: usize) -> usize {
    ZIGZAG_TO_NATURAL[k & 63] as usize
}

/// DC point transform (encoder side): arithmetic floor shift by `al`.
#[inline]
fn dc_point(v: i32, al: u8) -> i32 {
    v >> al
}

/// AC point transform (encoder side): division by `2^al` rounded toward zero.
#[inline]
fn ac_point(v: i32, al: u8) -> i32 {
    if v >= 0 { v >> al } else { -((-v) >> al) }
}

/// Decodes one progressive scan into `planes`, returning per-segment padding
/// and the observed EOB-run lengths (empty for DC scans).
#[allow(clippy::too_many_arguments)]
pub fn decode_progressive_scan(
    frame: &FrameHeader,
    geom: &FrameGeometry,
    planes: &mut [ComponentPlane],
    header: &ScanHeader,
    dc_tables: &[Option<HuffmanTable>; 4],
    ac_tables: &[Option<HuffmanTable>; 4],
    restart_interval: u16,
    reader: &mut EntropyReader<'_>,
) -> Result<(Vec<Padding>, Vec<u32>)> {
    let sg = ScanGeom::resolve(frame, geom, header, dc_tables, ac_tables)?;
    let ss = header.spectral_start as usize;
    let se = header.spectral_end as usize;
    let ah = header.approx_high;
    let al = header.approx_low;
    let ri = restart_interval as usize;
    let mut padding = Vec::new();
    let mut eob_runs: Vec<u32> = Vec::new();
    let mut refs = Vec::new();

    if ss == 0 {
        // DC scan (may be interleaved).
        let mut dc_pred = vec![0i32; frame.components.len()];
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
                let block = block_mut(planes, r.frame_idx, r.bx, r.by)?;
                if ah == 0 {
                    let t = u32::from(dc_tbl.decode(reader)?);
                    if t > 15 {
                        return Err(JpegError::Malformed(format!(
                            "G.1.2.1: DC magnitude category {t} > 15"
                        )));
                    }
                    let diff = extend(reader.read_bits(t)?, t);
                    let pred = &mut dc_pred[r.frame_idx];
                    *pred = pred.wrapping_add(diff);
                    block[0] = to_coeff(*pred << al)?;
                } else if reader.read_bit()? == 1 {
                    let v = i32::from(block[0]) | (1i32 << al);
                    block[0] = to_coeff(v)?;
                }
            }
        }
    } else {
        // AC scan (single component, non-interleaved).
        let mut eobrun: u32 = 0;
        for u in 0..sg.num_units {
            if ri != 0 && u != 0 && u % ri == 0 {
                padding.push(finish_restart_segment(reader)?);
                eobrun = 0;
            }
            sg.unit_blocks(u, &mut refs);
            let r = &refs[0];
            let comp = sg.comp_for(r.frame_idx)?;
            let ac_tbl = comp.ac.ok_or_else(|| miss("AC"))?;
            let block = block_mut(planes, r.frame_idx, r.bx, r.by)?;
            if ah == 0 {
                decode_ac_first(
                    reader,
                    ac_tbl,
                    block,
                    ss,
                    se,
                    al,
                    &mut eobrun,
                    &mut eob_runs,
                )?;
            } else {
                decode_ac_refine(
                    reader,
                    ac_tbl,
                    block,
                    ss,
                    se,
                    al,
                    &mut eobrun,
                    &mut eob_runs,
                )?;
            }
        }
    }
    padding.push(reader.take_padding()?);
    Ok((padding, eob_runs))
}

#[allow(clippy::too_many_arguments)]
fn decode_ac_first(
    reader: &mut EntropyReader<'_>,
    ac: &HuffmanTable,
    block: &mut [i16; 64],
    ss: usize,
    se: usize,
    al: u8,
    eobrun: &mut u32,
    eob_runs: &mut Vec<u32>,
) -> Result<()> {
    if *eobrun > 0 {
        *eobrun -= 1;
        return Ok(());
    }
    let mut k = ss;
    while k <= se {
        let rs = ac.decode(reader)?;
        let r = (rs >> 4) as usize;
        let s = u32::from(rs & 0x0F);
        if s == 0 {
            if r != 15 {
                let mut run = 1u32 << r;
                if r > 0 {
                    run += reader.read_bits(r as u32)?;
                }
                eob_runs.push(run);
                *eobrun = run - 1;
                break;
            }
            k += 16; // ZRL: 16 zero coefficients
        } else {
            k += r;
            if k > se {
                return Err(JpegError::Malformed(
                    "G.1.2.2: AC-first run overruns the band".into(),
                ));
            }
            let val = extend(reader.read_bits(s)?, s);
            block[nat(k)] = to_coeff(val << al)?;
            k += 1;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_ac_refine(
    reader: &mut EntropyReader<'_>,
    ac: &HuffmanTable,
    block: &mut [i16; 64],
    ss: usize,
    se: usize,
    al: u8,
    eobrun: &mut u32,
    eob_runs: &mut Vec<u32>,
) -> Result<()> {
    let p1: i32 = 1 << al;
    let m1: i32 = -(1 << al);
    let mut k = ss;

    if *eobrun == 0 {
        while k <= se {
            let rs = ac.decode(reader)?;
            let mut r = (rs >> 4) as i32;
            let s = u32::from(rs & 0x0F);
            let mut newval: i32 = 0;
            if s == 0 {
                if r != 15 {
                    let mut run = 1u32 << r;
                    if r > 0 {
                        run += reader.read_bits(r as u32)?;
                    }
                    eob_runs.push(run);
                    *eobrun = run;
                    break;
                }
                // r == 15: ZRL, skip 16 zero-history coefficients.
            } else {
                if s != 1 {
                    return Err(JpegError::Malformed(
                        "G.1.2.3: refinement coefficient size != 1".into(),
                    ));
                }
                let bit = reader.read_bit()?;
                newval = if bit == 1 { p1 } else { m1 };
            }
            // Advance over zero-history coefficients, correcting nonzero ones.
            loop {
                let coef = i32::from(block[nat(k)]);
                if coef != 0 {
                    let cb = reader.read_bit()?;
                    if cb == 1 && (coef & p1) == 0 {
                        block[nat(k)] = to_coeff(coef + if coef >= 0 { p1 } else { m1 })?;
                    }
                } else {
                    r -= 1;
                    if r < 0 {
                        break;
                    }
                }
                k += 1;
                if k > se {
                    break;
                }
            }
            if s != 0 {
                if k > se {
                    return Err(JpegError::Malformed(
                        "G.1.2.3: refinement coefficient position past band end".into(),
                    ));
                }
                block[nat(k)] = to_coeff(newval)?;
            }
            k += 1;
        }
    }

    if *eobrun > 0 {
        while k <= se {
            let coef = i32::from(block[nat(k)]);
            if coef != 0 {
                let cb = reader.read_bit()?;
                if cb == 1 && (coef & p1) == 0 {
                    block[nat(k)] = to_coeff(coef + if coef >= 0 { p1 } else { m1 })?;
                }
            }
            k += 1;
        }
        *eobrun -= 1;
    }
    Ok(())
}

/// Encodes one progressive scan from `planes` into `out`.
#[allow(clippy::too_many_arguments)]
pub fn encode_progressive_scan(
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
    let sg = ScanGeom::resolve(frame, geom, header, dc_tables, ac_tables)?;
    let ss = header.spectral_start as usize;
    let se = header.spectral_end as usize;
    let ah = header.approx_high;
    let al = header.approx_low;
    let ri = restart_interval as usize;
    let mut pad_iter = padding.iter();
    let mut refs = Vec::new();
    let mut restart_counter = 0u8;
    let mut writer = EntropyWriter::new(out);

    /// Closes the current entropy segment at a restart boundary: emits padding
    /// and the cycling `RSTn` marker.
    fn emit_restart(
        writer: &mut EntropyWriter<'_>,
        pad_iter: &mut std::slice::Iter<'_, Padding>,
        restart_counter: &mut u8,
    ) -> Result<()> {
        let pad = *pad_iter
            .next()
            .ok_or_else(|| JpegError::Encode("missing padding for restart segment".into()))?;
        writer.flush_padding(pad)?;
        writer.write_aligned_bytes(&[0xFF, crate::marker::RST0 + (*restart_counter & 7)])?;
        *restart_counter = restart_counter.wrapping_add(1);
        Ok(())
    }

    if ss == 0 {
        let mut dc_pred = vec![0i32; frame.components.len()];
        for u in 0..sg.num_units {
            if ri != 0 && u != 0 && u % ri == 0 {
                emit_restart(&mut writer, &mut pad_iter, &mut restart_counter)?;
                for p in dc_pred.iter_mut() {
                    *p = 0;
                }
            }
            sg.unit_blocks(u, &mut refs);
            for r in &refs {
                let comp = sg.comp_for(r.frame_idx)?;
                let plane = planes
                    .get(r.frame_idx)
                    .ok_or_else(|| JpegError::Encode("plane index out of range".into()))?;
                let block = plane
                    .block(r.bx, r.by)
                    .ok_or_else(|| JpegError::Encode("block position out of range".into()))?;
                if ah == 0 {
                    let dc_tbl = comp.dc.ok_or_else(|| miss("DC"))?;
                    let v = dc_point(i32::from(block[0]), al);
                    let diff = v - dc_pred[r.frame_idx];
                    dc_pred[r.frame_idx] = v;
                    let s = magnitude_category(diff);
                    if s > 15 {
                        return Err(JpegError::Encode(format!(
                            "DC diff {diff} needs category {s} > 15"
                        )));
                    }
                    dc_tbl.encode(&mut writer, s as u8)?;
                    writer.put_bits(mantissa_bits(diff, s), s);
                } else {
                    let bit = (i32::from(block[0]) >> al) & 1;
                    writer.put_bit(bit as u32);
                }
            }
        }
    } else {
        let ac_tbl = sg
            .comps
            .first()
            .and_then(|c| c.ac)
            .ok_or_else(|| miss("AC"))?;
        // EOB runs are replayed at exactly the lengths the original encoder
        // used (`eob_runs`, recorded at decode), rather than re-derived: an
        // encoder may split a run into several `EOBn` codes, and that split is
        // not recoverable from the coefficients alone.
        let mut eobrun: u32 = 0;
        let mut be: Vec<u8> = Vec::new();
        let mut ei = 0usize;
        for u in 0..sg.num_units {
            if ri != 0 && u != 0 && u % ri == 0 {
                // A restart resets the decoder's EOB run, so any pending run was
                // flushed before the marker; `eobrun` is already 0 here after the
                // match-flush below, but guard defensively.
                if eobrun > 0 {
                    emit_eobrun(&mut writer, ac_tbl, &mut eobrun, &mut be)?;
                    ei += 1;
                }
                emit_restart(&mut writer, &mut pad_iter, &mut restart_counter)?;
            }
            sg.unit_blocks(u, &mut refs);
            let r = &refs[0];
            let plane = planes
                .get(r.frame_idx)
                .ok_or_else(|| JpegError::Encode("plane index out of range".into()))?;
            let block = plane
                .block(r.bx, r.by)
                .ok_or_else(|| JpegError::Encode("block position out of range".into()))?;
            let contributes = if ah == 0 {
                encode_ac_first(&mut writer, ac_tbl, block, ss, se, al)?
            } else {
                encode_ac_refine(&mut writer, ac_tbl, block, ss, se, al, &mut be)?
            };
            if contributes {
                eobrun += 1;
                // Flush at exactly the recorded run length (handles both natural
                // ends at the next significant block and mid-run encoder splits).
                if ei < eob_runs.len() && eobrun == eob_runs[ei] {
                    emit_eobrun(&mut writer, ac_tbl, &mut eobrun, &mut be)?;
                    ei += 1;
                }
            }
        }
        // Flush any final run (its length is the last recorded value).
        if eobrun > 0 {
            emit_eobrun(&mut writer, ac_tbl, &mut eobrun, &mut be)?;
        }
    }

    let pad = *pad_iter
        .next()
        .ok_or_else(|| JpegError::Encode("missing padding for final segment".into()))?;
    writer.flush_padding(pad)?;
    Ok(())
}

/// Emits a pending EOB run code (if any) and any buffered correction bits.
fn emit_eobrun(
    writer: &mut EntropyWriter<'_>,
    ac: &HuffmanTable,
    eobrun: &mut u32,
    be: &mut Vec<u8>,
) -> Result<()> {
    if *eobrun > 0 {
        let e = *eobrun;
        let n = 31 - e.leading_zeros(); // floor(log2(e))
        ac.encode(writer, (n as u8) << 4)?;
        if n > 0 {
            writer.put_bits(e - (1 << n), n);
        }
        *eobrun = 0;
    }
    for &b in be.iter() {
        writer.put_bit(u32::from(b));
    }
    be.clear();
    Ok(())
}

/// Encodes one AC-first block. Returns whether the band ends in a run of zeros
/// (so the caller extends the pending EOB run). EOB-run flushing is the
/// caller's job (it replays the recorded run lengths).
fn encode_ac_first(
    writer: &mut EntropyWriter<'_>,
    ac: &HuffmanTable,
    block: &[i16; 64],
    ss: usize,
    se: usize,
    al: u8,
) -> Result<bool> {
    let mut run = 0usize;
    for k in ss..=se {
        let coef = ac_point(i32::from(block[nat(k)]), al);
        if coef == 0 {
            run += 1;
            continue;
        }
        while run >= 16 {
            ac.encode(writer, 0xF0)?; // ZRL
            run -= 16;
        }
        let s = magnitude_category(coef);
        if s == 0 || s > 15 {
            return Err(JpegError::Encode(format!(
                "AC-first coefficient {coef} has category {s}"
            )));
        }
        ac.encode(writer, ((run as u8) << 4) | (s as u8))?;
        writer.put_bits(mantissa_bits(coef, s), s);
        run = 0;
    }
    // Trailing zeros (or a wholly-zero band) begin/extend an EOB run.
    Ok(run > 0)
}

/// Encodes one AC-refinement block. Returns whether the band ends in a run
/// (trailing zeros or trailing already-significant corrections), so the caller
/// extends the pending EOB run; trailing correction bits are appended to `be`
/// for the caller to emit when it flushes that run.
///
/// `run` counts consecutive zero-history (still-insignificant) coefficients.
/// Already-significant coefficients are transparent to the run; their correction
/// bits are buffered with the zero-count that preceded them, so that when a
/// newly-significant coefficient forces `ZRL` codes, the corrections replay in
/// the same 16-zero spans the decoder walks (10918-1 G.1.2.3). Only
/// newly-significant coefficients emit `ZRL`s; a block with none is folded into
/// the EOB run.
fn encode_ac_refine(
    writer: &mut EntropyWriter<'_>,
    ac: &HuffmanTable,
    block: &[i16; 64],
    ss: usize,
    se: usize,
    al: u8,
    be: &mut Vec<u8>,
) -> Result<bool> {
    let mut run = 0usize;
    let mut br: Vec<(usize, u8)> = Vec::new(); // (zeros seen before this correction, bit)
    for k in ss..=se {
        let coef = i32::from(block[nat(k)]);
        let mag = coef.abs() >> al; // magnitude at this scan
        if mag == 0 {
            run += 1;
            continue;
        }
        if mag > 1 {
            // Already significant: buffer its correction bit; the run continues.
            br.push((run, (mag & 1) as u8));
            continue;
        }
        // Newly significant (mag == 1).
        let num_zrl = run / 16;
        let rem = run % 16;
        let mut bi = 0usize;
        for j in 0..num_zrl {
            ac.encode(writer, 0xF0)?; // ZRL
            while bi < br.len() && br[bi].0 / 16 == j {
                writer.put_bit(u32::from(br[bi].1));
                bi += 1;
            }
        }
        ac.encode(writer, ((rem as u8) << 4) | 1)?;
        let sign = if coef < 0 { 0 } else { 1 };
        writer.put_bit(sign);
        // Corrections in the final (remainder) span follow the coefficient code.
        while bi < br.len() {
            writer.put_bit(u32::from(br[bi].1));
            bi += 1;
        }
        br.clear();
        run = 0;
    }
    if run > 0 || !br.is_empty() {
        // Trailing zeros / already-significant corrections extend an EOB run;
        // their correction bits are emitted when that run is flushed.
        for &(_, bit) in &br {
            be.push(bit);
        }
        Ok(true)
    } else {
        Ok(false)
    }
}
