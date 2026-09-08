# jpxl-cli

Command-line front end for [JPXL](https://github.com/LegeApp/JPXL), a
clean-room Rust implementation of JPEG XL (ISO/IEC 18181). No C dependencies.

```sh
cargo install jpxl-cli   # installs the `jpxl` binary
```

```sh
jpxl encode photo.png                    # lossless -> photo.jxl
jpxl encode --quality photo.jpg          # lossy at the effort's default score
jpxl encode --quality 90 photo.jpg out.jxl
jpxl encode --quality --effort fast photo.jpg
jpxl decode photo.jxl                    # -> photo.png
jpxl info photo.jxl
```

The output path is optional: omitted, `encode` writes `<in>.jxl` and `decode`
writes `<in>.png` beside the input, and refuses rather than overwrite a file
you did not name. `-` is stdin/stdout for pipelines.

`--quality N` is a minimum SSIMULACRA2 score (`0..=100`, 100 = lossless), not a
distance — the encoder returns the smallest stream whose *reconstructed pixels*
score at least `N`. `cjxl -d` targets Butteraugli, a different and inverted
scale.

Raster formats in and out: PNG, JPEG, WebP, TIFF, BMP, GIF, ICO, TGA, QOI, PGM,
PPM. PNG, TIFF and PNM keep 16-bit samples.

`jpxl --help` covers ordinary encoding; `jpxl --help-advanced` covers the
research, calibration and tuning surface.

## License

MIT.
