//! Archive roundtrip sweep: `serialize(parse(x)) == x` over a directory of
//! JPEG files, with deterministic ordering and a coverage/refusal summary.
//!
//! ```text
//! cargo run --release -p jpxl-jpeg --example roundtrip_sweep -- <dir> [stride]
//! ```
//!
//! Files are collected recursively, sorted by path, and (optionally) sampled
//! by the given stride, so the sample is a pure function of the archive
//! contents. Every file is classified as byte-identical, refused (typed
//! `JpegError::Unsupported`), parse error, or mismatch; the exit code is
//! nonzero if any mismatch occurs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn collect(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect(&path, out)?;
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"))
        {
            out.push(path);
        }
    }
    Ok(())
}

/// Coverage attributes of one parsed JPEG, for the summary table.
fn classify(jpeg: &jpxl_jpeg::Jpeg, tally: &mut BTreeMap<&'static str, usize>) {
    let mut bump = |key: &'static str| *tally.entry(key).or_insert(0) += 1;
    let mut sof_code = None;
    for segment in &jpeg.segments {
        match segment {
            jpxl_jpeg::Segment::Sof(frame) => {
                sof_code = Some(frame.code);
                match frame.components.len() {
                    1 => bump("grayscale"),
                    3 => {
                        let factors: Vec<(u8, u8)> =
                            frame.components.iter().map(|c| (c.h, c.v)).collect();
                        match factors.first() {
                            Some(&(2, 2)) => bump("chroma-420"),
                            Some(&(2, 1)) => bump("chroma-422"),
                            Some(&(1, 2)) => bump("chroma-440"),
                            Some(&(1, 1)) => bump("chroma-444"),
                            _ => bump("chroma-other"),
                        }
                    }
                    _ => bump("components-other"),
                }
            }
            jpxl_jpeg::Segment::Dri(interval) if *interval > 0 => bump("restart-interval"),
            jpxl_jpeg::Segment::App(app) => {
                if app.code == 0xE1 && app.payload.starts_with(b"Exif\0") {
                    bump("exif");
                } else if app.code == 0xE1 && app.payload.starts_with(b"http://ns.adobe.com/xap/") {
                    bump("xmp");
                } else if app.code == 0xE2 && app.payload.starts_with(b"ICC_PROFILE\0") {
                    bump("icc");
                }
            }
            _ => {}
        }
    }
    match sof_code {
        Some(0xC0) => bump("baseline"),
        Some(0xC1) => bump("extended-sequential"),
        Some(0xC2) => bump("progressive"),
        Some(_) => bump("sof-other"),
        None => bump("no-sof"),
    }
    if !jpeg.tail.is_empty() {
        bump("trailing-data");
    }
}

fn main() -> std::process::ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(dir) = args.next() else {
        eprintln!("usage: roundtrip_sweep <dir> [stride]");
        return std::process::ExitCode::FAILURE;
    };
    let stride: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1).max(1);
    let mut files = Vec::new();
    if let Err(e) = collect(Path::new(&dir), &mut files) {
        eprintln!("walk failed: {e}");
        return std::process::ExitCode::FAILURE;
    }
    files.sort();
    let sample: Vec<&PathBuf> = files.iter().step_by(stride).collect();
    let (mut identical, mut mismatches, mut parse_errors) = (0usize, 0usize, 0usize);
    let mut refusals: BTreeMap<String, usize> = BTreeMap::new();
    let mut coverage: BTreeMap<&'static str, usize> = BTreeMap::new();
    for path in &sample {
        let data = match std::fs::read(path) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("READ-ERROR {} {e}", path.display());
                parse_errors += 1;
                continue;
            }
        };
        match jpxl_jpeg::parse(&data) {
            Ok(jpeg) => match jpxl_jpeg::serialize(&jpeg) {
                Ok(bytes) if bytes == data => {
                    identical += 1;
                    classify(&jpeg, &mut coverage);
                }
                Ok(_) => {
                    mismatches += 1;
                    eprintln!("MISMATCH {}", path.display());
                }
                Err(e) => {
                    mismatches += 1;
                    eprintln!("SERIALIZE-ERROR {} {e}", path.display());
                }
            },
            Err(jpxl_jpeg::JpegError::Unsupported(reason)) => {
                *refusals.entry(reason).or_insert(0) += 1;
            }
            Err(e) => {
                parse_errors += 1;
                eprintln!("PARSE-ERROR {} {e}", path.display());
            }
        }
    }
    println!(
        "swept {} of {} files (stride {stride}): {identical} byte-identical, \
         {mismatches} mismatches, {parse_errors} parse errors, {} refused",
        sample.len(),
        files.len(),
        refusals.values().sum::<usize>()
    );
    for (reason, count) in &refusals {
        println!("  refused[{count}]: {reason}");
    }
    println!("coverage of the byte-identical set:");
    for (key, count) in &coverage {
        println!("  {key}: {count}");
    }
    if mismatches == 0 {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::FAILURE
    }
}
