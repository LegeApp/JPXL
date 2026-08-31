#!/usr/bin/env bash
# Build an external ZenJXL benchmark adapter without adding an AGPL dependency
# to the JPXL workspace. Generated files live in gitignored scratch by default.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
adapter_dir="${1:-${repo_root}/.agent/scratch/zenjxl-bench-adapter}"

mkdir -p "${adapter_dir}/src"

cat > "${adapter_dir}/Cargo.toml" <<'EOF'
[package]
name = "zenjxl-bench-adapter"
version = "0.1.0"
edition = "2024"
publish = false

[dependencies]
zenjxl = { version = "=0.2.1", default-features = false, features = ["encode", "parallel"] }
EOF

cat > "${adapter_dir}/src/main.rs" <<'EOF'
use std::{env, fs, process};
use zenjxl::{LossyConfig, PixelLayout};

const VERSION: &str = "zenjxl-bench-adapter 0.1.0 (zenjxl 0.2.1)";

fn fail(message: impl AsRef<str>) -> ! {
    eprintln!("error: {}", message.as_ref());
    process::exit(2);
}

fn ppm_token<'a>(bytes: &'a [u8], cursor: &mut usize) -> Option<&'a [u8]> {
    loop {
        while bytes.get(*cursor).is_some_and(u8::is_ascii_whitespace) {
            *cursor += 1;
        }
        if bytes.get(*cursor) != Some(&b'#') {
            break;
        }
        while bytes.get(*cursor).is_some_and(|byte| *byte != b'\n') {
            *cursor += 1;
        }
    }
    let start = *cursor;
    while bytes.get(*cursor).is_some_and(|byte| !byte.is_ascii_whitespace()) {
        *cursor += 1;
    }
    (start != *cursor).then_some(&bytes[start..*cursor])
}

fn parse_usize(token: Option<&[u8]>, label: &str) -> usize {
    let text = token
        .and_then(|value| std::str::from_utf8(value).ok())
        .unwrap_or_else(|| fail(format!("missing or invalid PPM {label}")));
    text.parse()
        .unwrap_or_else(|_| fail(format!("invalid PPM {label}: {text}")))
}

fn read_ppm(path: &str) -> (Vec<u8>, u32, u32) {
    let bytes = fs::read(path).unwrap_or_else(|error| fail(format!("read {path}: {error}")));
    let mut cursor = 0;
    if ppm_token(&bytes, &mut cursor) != Some(b"P6") {
        fail("input must be a binary P6 PPM");
    }
    let width = parse_usize(ppm_token(&bytes, &mut cursor), "width");
    let height = parse_usize(ppm_token(&bytes, &mut cursor), "height");
    let maxval = parse_usize(ppm_token(&bytes, &mut cursor), "maxval");
    if maxval != 255 {
        fail("only 8-bit P6 PPM input (maxval 255) is supported");
    }
    if !bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
        fail("missing whitespace after PPM header");
    }
    cursor += 1;
    let expected = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .unwrap_or_else(|| fail("PPM dimensions overflow"));
    let pixels = bytes
        .get(cursor..)
        .filter(|pixels| pixels.len() == expected)
        .unwrap_or_else(|| fail("PPM pixel payload length does not match dimensions"));
    let width = u32::try_from(width).unwrap_or_else(|_| fail("PPM width exceeds u32"));
    let height = u32::try_from(height).unwrap_or_else(|_| fail("PPM height exceeds u32"));
    (pixels.to_vec(), width, height)
}

fn usage() -> ! {
    fail("usage: zenjxl-bench-adapter --distance D --effort N input.ppm output.jxl");
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.as_slice() == ["--version"] {
        println!("{VERSION}");
        return;
    }
    if args.len() != 6 || args[0] != "--distance" || args[2] != "--effort" {
        usage();
    }
    let distance: f32 = args[1]
        .parse()
        .unwrap_or_else(|_| fail("distance must be a number"));
    let effort: u8 = args[3]
        .parse()
        .unwrap_or_else(|_| fail("effort must be an integer"));
    let (pixels, width, height) = read_ppm(&args[4]);
    let encoded = LossyConfig::new(distance)
        .with_effort(effort)
        .encode(&pixels, width, height, PixelLayout::Rgb8)
        .unwrap_or_else(|error| fail(format!("ZenJXL encode failed: {error}")));
    fs::write(&args[5], encoded)
        .unwrap_or_else(|error| fail(format!("write {}: {error}", args[5])));
}
EOF

printf '%s\n' \
  'ZenJXL is AGPL-3.0-or-commercial. This adapter is for external black-box benchmarking only.' \
  'It is not part of the JPXL workspace or distributed binaries.'
cargo build --manifest-path "${adapter_dir}/Cargo.toml" --release
printf 'adapter: %s\n' "${adapter_dir}/target/release/zenjxl-bench-adapter"
