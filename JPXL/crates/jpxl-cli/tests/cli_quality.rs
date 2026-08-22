//! PR 1 CLI surface for the perceptual quality controller.
//!
//! Drives the built `jpxl` binary on a small generated PPM through the
//! perceptual path (the printed report line carries the resolved target) and
//! the score-100 lossless route.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A P6 PPM gradient written to a fresh temp directory for one test.
fn fixture(name: &str, width: u32, height: u32) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("jpxl_cli_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let input = dir.join("in.ppm");
    let output = dir.join("out.jxl");

    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
    for y in 0..height {
        for x in 0..width {
            let r = u8::try_from((x * 3) % 256).unwrap_or(0);
            let g = u8::try_from((y * 5) % 256).unwrap_or(0);
            let b = u8::try_from((x + y) % 256).unwrap_or(0);
            ppm.extend_from_slice(&[r, g, b]);
        }
    }
    std::fs::write(&input, &ppm).expect("write ppm");
    (input, output)
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_jpxl"))
        .args(args)
        .output()
        .expect("run jpxl binary")
}

fn encode(input: &Path, output: &Path, extra: &[&str]) -> Output {
    let mut args = vec!["encode"];
    args.extend_from_slice(extra);
    let input = input.to_str().expect("utf8 input path");
    let output = output.to_str().expect("utf8 output path");
    args.push(input);
    args.push(output);
    run(&args)
}

#[test]
fn quality_default_is_70_fast_85_balanced() {
    let (input, output) = fixture("quality_default", 64, 64);

    // Balanced is the default effort, so `--quality` with no number resolves
    // to 85; the controller's report line carries the resolved target.
    let balanced = encode(&input, &output, &["--quality", "--threads", "1"]);
    assert_eq!(
        balanced.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&balanced.stderr)
    );
    let stdout = String::from_utf8_lossy(&balanced.stdout);
    assert!(
        stdout.contains("quality_target=85.0000") && stdout.contains("effort=balanced"),
        "balanced default score should be 85: {stdout}"
    );

    // `--effort fast` lowers the default to 70.
    let fast = encode(
        &input,
        &output,
        &["--effort", "fast", "--quality", "--threads", "1"],
    );
    assert_eq!(
        fast.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&fast.stderr)
    );
    let stdout = String::from_utf8_lossy(&fast.stdout);
    assert!(
        stdout.contains("quality_target=70.0000") && stdout.contains("effort=fast"),
        "fast default score should be 70: {stdout}"
    );
}

#[test]
fn quality_100_routes_to_lossless() {
    let (input, output) = fixture("quality_100", 64, 64);
    let out = encode(&input, &output, &["--quality", "100"]);
    assert_eq!(out.status.code(), Some(0), "score 100 encodes");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("quality_target=100.0000"),
        "expected the perceptual line: {stdout}"
    );
    assert!(
        stdout.contains("status=routed_to_lossless"),
        "score 100 routes to lossless: {stdout}"
    );
    assert!(
        stdout.contains("metric=ssimulacra2-jpxl-1"),
        "the metric version is reported: {stdout}"
    );
    assert!(output.exists() && output.metadata().map(|m| m.len()).unwrap_or(0) > 0);
}

#[test]
fn conflicting_targets_rejected() {
    let (input, output) = fixture("conflict", 64, 64);

    let quality_then_bpp = encode(&input, &output, &["--quality", "100", "--bpp", "1.0"]);
    assert_eq!(quality_then_bpp.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&quality_then_bpp.stderr);
    assert!(
        stderr.contains("one lossy target"),
        "conflicting targets rejected: {stderr}"
    );

    let bpp_then_global = encode(&input, &output, &["--bpp", "1.0", "--global-scale", "1000"]);
    assert_eq!(bpp_then_global.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&bpp_then_global.stderr);
    assert!(stderr.contains("one lossy target"), "{stderr}");
}

#[test]
fn global_scale_encodes() {
    let (input, output) = fixture("global_scale", 64, 64);
    let out = encode(&input, &output, &["--global-scale", "32768"]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "fixed-quantizer encode succeeds: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(output.exists());
    assert!(output.metadata().map(|m| m.len()).unwrap_or(0) > 0);

    // The emitted stream is recognised as JPEG XL.
    let info = run(&["info", output.to_str().expect("utf8 path")]);
    assert_eq!(info.status.code(), Some(0), "output is a JPEG XL stream");
}
