# Speed-parity reconciliation

## Question

Why did the public 2026-08-18 comparison report JPXL far slower than `cjxl`
when Phase 42 had already recorded a speed-parity window?

## Method

Compared the frozen Phase 42 command, its raw output, and the committed public
comparison harness. Ran the current release binary again with `Balanced` on
the same 4 MP and 12 MP PPMs used by the public report.

## Findings

Phase 42 was an equal-resource, matched-SSIMULACRA2 speed screen: JPXL used
`--lossy-preset balanced` or `fast`, `--threads 4`, and four pinned P-cores;
`cjxl` used `-e 7 --num_threads=4` on the same cores. It used `-d 2.25` for
the 4 MP input and `-d 1.25` for the 12 MP input, selected to match the
Balanced SSIMULACRA2 results. Its raw three-run window was:

| Image | JPXL Balanced | JPXL Fast | cjxl `-e 7` | SSIMULACRA2 (Balanced / cjxl) |
| --- | ---: | ---: | ---: | ---: |
| 2400×1800 | 0.42–0.43 s | 0.31–0.33 s | 0.49–0.51 s | 72.38 / 72.20 |
| 4000×3000 | 1.05–1.43 s | 0.76–0.97 s | 1.51–1.66 s | 83.63 / 83.70 |

The superseded report instead used JPXL `Quality` at 1 bpp, while `cjxl -d 1`
received its default thread count. JPXL was explicitly limited to four
workers. On the 4 MP image, that pair also emitted 0.994 versus 1.490 bpp, so
it was not matched on rate or quality. `Quality` intentionally spends far
more work on rate search and entropy alternatives than the performance paths.

A corrected current warm-process run at 1 bpp, `Balanced`, four workers each,
and `cjxl -e 7 -d 2.25` measured 340.120 ms for JPXL and 434.153 ms for
`cjxl` on the 4 MP anchor. It is diagnostic rather than a new matched-quality
claim: current JPXL scored 77.1846 SSIMULACRA2 while that `cjxl` point scored
72.2068, reflecting quality-track changes after the Phase 42 match.

## Conclusion

The Phase 42 speed-parity result was not invalidated. The public report
combined a deliberately exhaustive JPXL preset, unequal CPU allocation, and
unmatched operating points. Speed parity is limited to the specified
Balanced/Fast, equal-resource, matched-SSIMULACRA2 window; it does not assert
quality, density, Butteraugli, or general codec parity.

## Consequences

`compare-libjxl.ps1` now defaults to `Balanced`, passes the configured thread
count to both encoders, and pins `cjxl -e 7`. Use an explicit `Quality` preset
only for a quality/density curve, and label it as such.
