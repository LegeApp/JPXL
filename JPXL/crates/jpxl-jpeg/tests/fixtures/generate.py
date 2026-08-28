#!/usr/bin/env python3
"""Deterministically generate the jpxl-jpeg round-trip fixtures.

All source pixels are synthesised here from a fixed seed (no third-party image
is used), so the fixtures are original content, CC0 / public domain, and
regenerable byte-for-byte. Encoders: libjpeg-turbo `cjpeg`/`jpegtran` and
Pillow (libjpeg-turbo) for the Exif/ICC cases.

Run from this directory:  python3 generate.py
It writes each `*.jpg` plus a `*.jpg.prov` provenance sidecar.
"""

import hashlib
import math
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def synth_rgb(width, height, seed):
    """A rich synthetic RGB image: gradients, sinusoids, and block noise, so the
    DCT coefficients span many run/size categories rather than being trivially
    sparse."""
    # xorshift32 for a dependency-free deterministic PRNG.
    state = seed & 0xFFFFFFFF or 1

    def rnd():
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return state & 0xFFFFFFFF

    data = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            i = (y * width + x) * 3
            r = (x * 255) // (width - 1)
            g = (y * 255) // (height - 1)
            b = int(127 + 100 * math.sin(x / 11.0) * math.cos(y / 13.0))
            # Block-structured noise every 16 px to exercise sharp edges.
            if ((x // 16) + (y // 16)) % 3 == 0:
                n = rnd() % 64
                r = (r + n) & 0xFF
                g = (g ^ (n << 1)) & 0xFF
                b = (b + (rnd() % 48)) & 0xFF
            data[i] = r & 0xFF
            data[i + 1] = g & 0xFF
            data[i + 2] = max(0, min(255, b)) & 0xFF
    return bytes(data)


def write_ppm(path, width, height, rgb):
    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        f.write(rgb)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def provenance(jpg, how):
    with open(jpg + ".prov", "w") as f:
        f.write(
            "fixture: {name}\n"
            "origin: synthesised by tests/fixtures/generate.py (original content)\n"
            "license: CC0-1.0 / public domain (no third-party image used)\n"
            "sha256: {digest}\n"
            "regenerate: {how}\n".format(
                name=os.path.basename(jpg), digest=sha256(jpg), how=how
            )
        )


def cjpeg(ppm, out, args, how):
    subprocess.run(["cjpeg", "-outfile", out] + args + [ppm], check=True)
    provenance(out, how)


def main():
    # Odd dimensions (not multiples of 8/16) to exercise edge / padding blocks
    # and the interleaved-vs-non-interleaved block-count subtlety.
    W, H = 385, 259
    color = synth_rgb(W, H, 0xC0FFEE)
    ppm = os.path.join(HERE, "_src_color.ppm")
    write_ppm(ppm, W, H, color)

    j = lambda n: os.path.join(HERE, n)

    # Baseline sequential, all chroma subsamplings.
    cjpeg(ppm, j("base_444.jpg"), ["-quality", "85", "-sample", "1x1"],
          "cjpeg -quality 85 -sample 1x1")
    cjpeg(ppm, j("base_422.jpg"), ["-quality", "85", "-sample", "2x1"],
          "cjpeg -quality 85 -sample 2x1")
    cjpeg(ppm, j("base_420.jpg"), ["-quality", "85", "-sample", "2x2"],
          "cjpeg -quality 85 -sample 2x2")
    cjpeg(ppm, j("base_440.jpg"), ["-quality", "85", "-sample", "1x2"],
          "cjpeg -quality 85 -sample 1x2")
    cjpeg(ppm, j("base_gray.jpg"), ["-quality", "85", "-grayscale"],
          "cjpeg -quality 85 -grayscale")
    # Restart intervals (MCU count, and MCU-row form).
    cjpeg(ppm, j("base_restart.jpg"),
          ["-quality", "90", "-sample", "2x2", "-restart", "5"],
          "cjpeg -quality 90 -sample 2x2 -restart 5")
    cjpeg(ppm, j("base_restart_rows.jpg"),
          ["-quality", "90", "-sample", "1x1", "-restart", "2B"],
          "cjpeg -quality 90 -sample 1x1 -restart 2B")
    # High quality (dense coefficients) and low quality (sparse, long EOBs).
    cjpeg(ppm, j("base_q98.jpg"), ["-quality", "98", "-sample", "1x1"],
          "cjpeg -quality 98 -sample 1x1")
    cjpeg(ppm, j("base_q20.jpg"), ["-quality", "20", "-sample", "2x2"],
          "cjpeg -quality 20 -sample 2x2")

    # Progressive.
    cjpeg(ppm, j("prog_444.jpg"), ["-quality", "85", "-progressive", "-sample", "1x1"],
          "cjpeg -quality 85 -progressive -sample 1x1")
    cjpeg(ppm, j("prog_420.jpg"), ["-quality", "85", "-progressive", "-sample", "2x2"],
          "cjpeg -quality 85 -progressive -sample 2x2")
    cjpeg(ppm, j("prog_gray.jpg"), ["-quality", "85", "-progressive", "-grayscale"],
          "cjpeg -quality 85 -progressive -grayscale")
    cjpeg(ppm, j("prog_restart.jpg"),
          ["-quality", "90", "-progressive", "-sample", "2x2", "-restart", "4"],
          "cjpeg -quality 90 -progressive -sample 2x2 -restart 4")
    cjpeg(ppm, j("prog_q30.jpg"), ["-quality", "30", "-progressive", "-sample", "2x2"],
          "cjpeg -quality 30 -progressive -sample 2x2")

    # Large, detailed progressive image: its AC refinement scans have long EOB
    # runs that libjpeg splits across several EOBn codes (correction-bit buffer
    # fills), which the re-encoder must replay from recorded run lengths rather
    # than re-derive. Regression for the archive-sweep EOB-run failures.
    bw, bh = 1024, 768
    bstate = 0x1234567

    def brnd():
        nonlocal bstate
        bstate ^= (bstate << 13) & 0xFFFFFFFF
        bstate ^= bstate >> 17
        bstate ^= (bstate << 5) & 0xFFFFFFFF
        return bstate & 0xFFFFFFFF

    big = bytearray(bw * bh * 3)
    for y in range(bh):
        for x in range(bw):
            i = (y * bw + x) * 3
            r = int(127 + 90 * math.sin(x / 7.0) * math.cos(y / 9.0))
            g = (x ^ y) & 0xFF
            b = int(127 + 80 * math.sin((x + y) / 5.0))
            if ((x // 8) + (y // 8)) % 2 == 0:
                n = brnd() % 96
                r = (r + n) & 0xFF
                g = (g ^ n) & 0xFF
                b = (b + (brnd() % 64)) & 0xFF
            big[i] = r & 0xFF
            big[i + 1] = max(0, min(255, g)) & 0xFF
            big[i + 2] = max(0, min(255, b)) & 0xFF
    bigppm = os.path.join(HERE, "_big.ppm")
    write_ppm(bigppm, bw, bh, bytes(big))
    cjpeg(bigppm, j("prog_large.jpg"),
          ["-quality", "92", "-progressive", "-sample", "2x2"],
          "cjpeg -quality 92 -progressive -sample 2x2 (1024x768 synthetic source)")
    os.remove(bigppm)

    # Trailing garbage after EOI (Annex A "tail data").
    with open(j("base_444.jpg"), "rb") as f:
        base = f.read()
    with open(j("trailing_garbage.jpg"), "wb") as f:
        f.write(base + b"\x00\x01\x02trailing-bytes-after-EOI\xff\xd9\xde\xad")
    provenance(j("trailing_garbage.jpg"),
               "base_444.jpg with literal trailing bytes appended after EOI")

    # Exif + ICC via Pillow (baseline). Kept optional so the core set does not
    # depend on Pillow being importable.
    try:
        from PIL import Image
        img = Image.frombytes("RGB", (W, H), color)
        # A minimal but valid ICC profile and Exif blob.
        exif = img.getexif()
        exif[0x010E] = "jpxl-jpeg synthetic fixture"  # ImageDescription
        exif[0x0131] = "generate.py"                   # Software
        # A tiny sRGB-ish ICC stub is not a valid profile; instead let Pillow
        # attach a real one if littlecms is available, else skip ICC.
        icc = b""
        try:
            from PIL import ImageCms
            prof = ImageCms.createProfile("sRGB")
            icc = ImageCms.ImageCmsProfile(prof).tobytes()
        except Exception:
            icc = b""
        save_kwargs = dict(format="JPEG", quality=88, subsampling=0,
                           exif=exif.tobytes())
        if icc:
            save_kwargs["icc_profile"] = icc
        img.save(j("meta_exif_icc.jpg"), **save_kwargs)
        provenance(j("meta_exif_icc.jpg"),
                   "Pillow img.save(quality=88, subsampling=0, exif=..., icc_profile=sRGB)")
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"skipping Pillow Exif/ICC fixture: {e}\n")

    # XMP metadata (APP1 with the XMP namespace) via Pillow (baseline).
    try:
        from PIL import Image
        img = Image.frombytes("RGB", (W, H), color)
        xmp = (
            b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
            b'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
            b"<dc:title>jpxl-jpeg synthetic fixture</dc:title>"
            b"</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>"
        )
        img.save(j("meta_xmp.jpg"), format="JPEG", quality=88, subsampling=2, xmp=xmp)
        provenance(j("meta_xmp.jpg"),
                   "Pillow img.save(quality=88, subsampling=2, xmp=<XMP packet>)")
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"skipping XMP fixture: {e}\n")

    # Arithmetic-coded JPEG — a *refusal* fixture (must be rejected, not decoded).
    try:
        cjpeg(ppm, j("refuse_arithmetic.jpg"),
              ["-quality", "85", "-arithmetic", "-sample", "2x2"],
              "cjpeg -quality 85 -arithmetic -sample 2x2")
    except subprocess.CalledProcessError as e:  # pragma: no cover
        sys.stderr.write(f"skipping arithmetic fixture: {e}\n")

    os.remove(ppm)
    print("fixtures generated in", HERE)


if __name__ == "__main__":
    main()
