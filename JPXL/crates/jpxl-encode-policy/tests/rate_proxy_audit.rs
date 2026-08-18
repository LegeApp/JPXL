//! Phase Q4 measurement harness: how well does the cover objective's rate
//! proxy price what the writer actually spends, per transform size?
//!
//! The cover search charges a candidate `bitlen(|q|) + 1` per nonzero HF
//! coefficient, nothing per zero, `PER_VARBLOCK_BITS` per varblock and
//! `NON_DCT8X8_SIGNAL_BITS` per merged transform. The writer spends
//! `-log2 p(token | cluster)` plus extra bits per I.4 event under the plan's
//! own trained histograms, and interior zeros are events too. This walks the
//! chosen plan of a real target-rate encode with a costing sink and reports,
//! per transform, actual bits against the proxy and the split of the actual
//! bits into `non_zeros` symbols, nonzero-coefficient tokens and interior-zero
//! tokens — the numbers a better proxy would be fitted to.
//!
//! Ignored by default (a measurement, not a gate); set `JPXL_RATE_AUDIT_PPM`
//! to one or more `;`-separated P6 paths, and optionally `JPXL_RATE_AUDIT_BPP`.

// A measurement harness over trusted plan data: plain indexing and narrowing
// casts read better than the guarded forms the decode paths need.
#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

use std::collections::BTreeMap;

use jpxl_core::varblock::TransformType;
use jpxl_encode::vardct::ids::PreContextId;
use jpxl_encode::vardct::{HfEventSink, walk_frame};
use jpxl_encode_policy::{EncodeRequest, RateSearchPreset, RateTarget};

/// Ideal token cost under the plan's trained clusters.
struct Coster {
    /// Cluster per pre-context.
    context_map: Vec<usize>,
    /// Per cluster: log2 of the total count and per-symbol log2 counts.
    log_totals: Vec<f64>,
    log_counts: Vec<Vec<f64>>,
    hybrid: Vec<jpxl_entropy::HybridUintConfig>,
}

impl Coster {
    fn bits(&self, ctx: PreContextId, value: u32) -> f64 {
        let cluster = self.context_map[ctx.get() as usize];
        let split = self.hybrid[cluster].tokenize(value).expect("tokenizable");
        let token = split.token as usize;
        let log_count = self.log_counts[cluster].get(token).copied().unwrap_or(0.0);
        (self.log_totals[cluster] - log_count).max(0.0) + f64::from(split.extra_bits)
    }
}

#[derive(Default, Clone, Copy)]
struct Bucket {
    varblocks: u64,
    nnz_symbols: u64,
    nnz_bits: f64,
    nonzero_tokens: u64,
    nonzero_bits: f64,
    zero_tokens: u64,
    zero_bits: f64,
    proxy_bits: f64,
    /// Sum over channels of the coding-order index of the last nonzero (+1).
    walk_len: u64,
}

struct AuditSink<'a> {
    coster: &'a Coster,
    current: Option<TransformType>,
    per_transform: BTreeMap<u32, Bucket>,
    /// (side, proxy, actual, interior zeros) per varblock, for the fits.
    per_varblock: Vec<(u32, f64, f64, u64)>,
    running_proxy: f64,
    running_actual: f64,
    running_walk: u64,
    running_zeros: u64,
}

impl AuditSink<'_> {
    fn flush(&mut self) {
        if let Some(t) = self.current {
            let side = t.sample_cols() as u32;
            let signal = if side == 8 { 0.0 } else { 32.0 };
            let proxy = self.running_proxy + 2.0 + signal;
            let b = self.per_transform.entry(side).or_default();
            b.varblocks += 1;
            b.proxy_bits += proxy;
            b.walk_len += self.running_walk;
            self.per_varblock
                .push((side, proxy, self.running_actual, self.running_zeros));
        }
        self.running_proxy = 0.0;
        self.running_actual = 0.0;
        self.running_walk = 0;
        self.running_zeros = 0;
    }
}

impl HfEventSink for AuditSink<'_> {
    fn varblock(&mut self, transform: TransformType, _hf_mul: u32) {
        self.flush();
        self.current = Some(transform);
    }
    fn nonzeros(&mut self, context: PreContextId, value: u32) {
        let bits = self.coster.bits(context, value);
        let side = self.current.map_or(0, |t| t.sample_cols() as u32);
        let b = self.per_transform.entry(side).or_default();
        b.nnz_symbols += 1;
        b.nnz_bits += bits;
        self.running_actual += bits;
    }
    fn coefficient(&mut self, context: PreContextId, value: u32) {
        let bits = self.coster.bits(context, value);
        let side = self.current.map_or(0, |t| t.sample_cols() as u32);
        let b = self.per_transform.entry(side).or_default();
        self.running_walk += 1;
        if value == 0 {
            b.zero_tokens += 1;
            b.zero_bits += bits;
            self.running_zeros += 1;
        } else {
            b.nonzero_tokens += 1;
            b.nonzero_bits += bits;
            // PackSigned: |q| = ceil(value / 2).
            let magnitude = value.div_ceil(2);
            let proxy = f64::from(32 - magnitude.leading_zeros()) + 1.0;
            self.running_proxy += proxy;
        }
        self.running_actual += bits;
    }
}

fn read_ppm(path: &str) -> (u32, u32, Vec<u8>) {
    let bytes = std::fs::read(path).expect("read ppm");
    let image = jpxl_conformance::metrics::Image::from_ppm(&bytes).expect("P6");
    assert_eq!(image.channels, 3);
    let rgb: Vec<u8> = image
        .samples
        .iter()
        .map(|&v| u8::try_from(v.min(255)).unwrap_or(255))
        .collect();
    (image.w, image.h, rgb)
}

#[test]
#[ignore = "Phase Q4 measurement harness; set JPXL_RATE_AUDIT_PPM"]
fn rate_proxy_audit() {
    let Ok(paths) = std::env::var("JPXL_RATE_AUDIT_PPM") else {
        eprintln!("RATE AUDIT skipped: set JPXL_RATE_AUDIT_PPM to semicolon-separated P6 paths");
        return;
    };
    let bpp: f64 = std::env::var("JPXL_RATE_AUDIT_BPP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    for path in paths.split(';').filter(|p| !p.is_empty()) {
        let (w, h, rgb) = read_ppm(path);
        let target = RateTarget::BitsPerPixel(bpp);
        let mut request = EncodeRequest::for_target(target);
        request.rate_preset = RateSearchPreset::Balanced;
        let outcome = jpxl_encode_policy::encode_srgb8_to_target(w, h, &rgb, &request, target)
            .expect("encode");
        let plan = outcome.plan.plan();
        let geometry = outcome.plan.geometry().expect("geometry");
        let pass = plan.entropy.passes.first().expect("a pass");
        let dist = &pass.distributions;
        let coster = Coster {
            context_map: dist.context_map.iter().map(|c| c.get() as usize).collect(),
            log_totals: dist
                .histograms
                .iter()
                .map(|h| f64::from(h.counts().iter().sum::<u32>().max(1)).log2())
                .collect(),
            log_counts: dist
                .histograms
                .iter()
                .map(|h| {
                    h.counts()
                        .iter()
                        .map(|&c| f64::from(c.max(1)).log2())
                        .collect()
                })
                .collect(),
            hybrid: dist
                .hybrid_uint
                .iter()
                .map(|u| {
                    jpxl_entropy::HybridUintConfig::new(
                        u32::from(u.split_exponent),
                        u32::from(u.msb_in_token),
                        u32::from(u.lsb_in_token),
                    )
                    .expect("valid config")
                })
                .collect(),
        };
        let mut sink = AuditSink {
            coster: &coster,
            current: None,
            per_transform: BTreeMap::new(),
            per_varblock: Vec::new(),
            running_proxy: 0.0,
            running_actual: 0.0,
            running_walk: 0,
            running_zeros: 0,
        };
        walk_frame(plan, &geometry, &mut sink).expect("walk");
        sink.flush();

        println!(
            "== {path}: {}x{} at {bpp} bpp Balanced -> {} bytes (rung {}, hf_mul {})",
            w,
            h,
            outcome.codestream.len(),
            outcome.chosen.rung.get(),
            outcome.chosen.hf_mul.get()
        );
        println!(
            "{:>5} {:>8} {:>12} {:>12} {:>7} | {:>10} {:>10} {:>10} | {:>8} {:>8} {:>8} | {:>9}",
            "side",
            "vblocks",
            "actual bits",
            "proxy bits",
            "act/prx",
            "nnz b/sym",
            "nz b/tok",
            "zero b/tok",
            "nz/vb",
            "zero/vb",
            "walk/vb",
            "act b/vb"
        );
        for (side, b) in &sink.per_transform {
            let actual = b.nnz_bits + b.nonzero_bits + b.zero_bits;
            let n = b.varblocks.max(1) as f64;
            println!(
                "{:>5} {:>8} {:>12.0} {:>12.0} {:>7.3} | {:>10.2} {:>10.2} {:>10.2} | {:>8.1} {:>8.1} {:>8.1} | {:>9.1}",
                side,
                b.varblocks,
                actual,
                b.proxy_bits,
                actual / b.proxy_bits.max(1.0),
                b.nnz_bits / b.nnz_symbols.max(1) as f64,
                b.nonzero_bits / b.nonzero_tokens.max(1) as f64,
                b.zero_bits / b.zero_tokens.max(1) as f64,
                b.nonzero_tokens as f64 / n,
                b.zero_tokens as f64 / n,
                b.walk_len as f64 / n,
                actual / n
            );
        }
        // Per-varblock fit: actual ~ a * proxy, by side, least squares through the origin,
        // and the residual spread.
        for side in [8u32, 16, 32] {
            let pts: Vec<(f64, f64, f64)> = sink
                .per_varblock
                .iter()
                .filter(|(s, _, _, _)| *s == side)
                .map(|(_, p, a, z)| (*p, *a, *z as f64))
                .collect();
            if pts.is_empty() {
                continue;
            }
            let sxy: f64 = pts.iter().map(|(p, a, _)| p * a).sum();
            let sxx: f64 = pts.iter().map(|(p, _, _)| p * p).sum();
            let slope = sxy / sxx.max(1.0);
            let spread = |pred: &dyn Fn(&(f64, f64, f64)) -> f64| {
                let mut rel: Vec<f64> = pts
                    .iter()
                    .map(|pt| (pt.1 - pred(pt)) / pred(pt).max(1.0))
                    .collect();
                rel.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
                let q = |f: f64| rel[((rel.len() - 1) as f64 * f) as usize];
                (q(0.1), q(0.5), q(0.9))
            };
            let (p10, p50, p90) = spread(&|pt| slope * pt.0);
            println!(
                "   side {side}: actual ~ {slope:.3} * proxy; relative residual p10 {p10:+.2} p50 {p50:+.2} p90 {p90:+.2}"
            );
            // Two-parameter fit through the origin: actual ~ s * proxy + z * zeros
            // (normal equations), to see how much a zero-run term explains.
            let (mut spp, mut spz, mut szz, mut spa, mut sza) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for (p, a, z) in &pts {
                spp += p * p;
                spz += p * z;
                szz += z * z;
                spa += p * a;
                sza += z * a;
            }
            let det = spp * szz - spz * spz;
            if det.abs() > 1e-9 {
                let s2 = (spa * szz - sza * spz) / det;
                let z2 = (sza * spp - spa * spz) / det;
                let (p10, p50, p90) = spread(&|pt| s2 * pt.0 + z2 * pt.2);
                println!(
                    "   side {side}: actual ~ {s2:.3} * proxy + {z2:.3} * zeros; residual p10 {p10:+.2} p50 {p50:+.2} p90 {p90:+.2}"
                );
            }
        }
    }
}
