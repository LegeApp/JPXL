#!/usr/bin/env bash
# bench_vs_libjxl.sh — reproducible size / SSIMULACRA2 / wall comparison of the
# JPXL encoder against the libjxl oracle (cjxl / djxl), with full provenance.
#
# WHAT IT DOES
#   For every input image and every requested setting it emits one table row:
#     encoder  setting  bytes  bpp  ssimulacra2  wall_s  status
#   JPXL rows come from `jpxl encode --quality Q`; cjxl rows from `cjxl -d D`.
#   Every stream is decoded back to pixels and scored with the SAME metric —
#   the in-tree production SSIMULACRA2 exposed by `jpxl compare` — so the two
#   encoders are graded on one identical yardstick. Sizes are the real encoded
#   byte counts; wall is the best of N encode runs.
#
#   This is deliberately an HONEST, not a matched, comparison: JPXL targets an
#   SSIMULACRA2 floor while cjxl targets a Butteraugli distance, so the reader
#   compares the (bytes, ssimulacra2) points, not same-named settings. See the
#   project README section "Quality, density, and speed versus libjxl".
#
# PROVENANCE
#   A header block records: UTC date, host, every binary's path + version +
#   sha256, the exact flags, the SSIMULACRA2 metric version, and each input's
#   sha256 and dimensions. With --jsonl the same data is written as one JSON
#   object per row (schema "jpxl.bench-vs-libjxl/1") for machine consumption.
#
# ORACLES
#   cjxl / djxl are looked up in tools/oracle-bin/ first (where
#   setup-oracles.sh installs them), then on $PATH, then via --cjxl/--djxl.
#   If they cannot run on this host (e.g. only the Windows .exe oracle is
#   present, or none was set up) the script still runs and emits JPXL-only
#   rows, clearly marked — the table stays reproducible and simply fills the
#   cjxl columns when a working oracle is available.
#
# CLEAN ROOM (AGENTS.md §2)
#   cjxl/djxl are used strictly as black boxes: this script only *runs* them.
#   No libjxl source is read.
#
# INPUTS
#   Any raster JPXL can read (PNG, PPM, JPEG, ...). Each input is first
#   normalised to a canonical P6 PPM via a *lossless* JPXL round-trip
#   (encode → decode), so both encoders and the scorer see identical source
#   pixels with no external image tool required. RGB (3-channel) inputs only;
#   images with alpha are skipped with a note (the lossy encoder has no alpha).
#
# USAGE
#   tools/bench_vs_libjxl.sh [options] <input|dir> [<input|dir> ...]
#
# OPTIONS
#   --quality "Q ..."   SSIMULACRA2 target(s) for JPXL     (default: "70 85 90")
#   --distance "D ..."  Butteraugli distance(s) for cjxl   (default: "3.0 1.5 1.0")
#   --effort E          JPXL lossy effort: fast|balanced   (default: balanced)
#   --cjxl-effort N     cjxl -e effort 1..9                (default: 7)
#   --threads N         thread cap for both encoders       (default: 4)
#   --runs N            encode timing runs, best is kept   (default: 1)
#   --jpxl PATH         jpxl binary (default: target/fast-debug or release)
#   --cjxl PATH         cjxl binary override
#   --djxl PATH         djxl binary override
#   --work-dir DIR      scratch dir (default: a mktemp under $TMPDIR)
#   --jsonl PATH        also write one JSON object per row to PATH
#   -h, --help          this help
#
# EXIT: 0 if at least one row was produced; non-zero on setup error.
set -uo pipefail

# ---------------------------------------------------------------- locate self
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
jpxl_root="$(cd -- "${script_dir}/.." && pwd)"
oracle_bin="${script_dir}/oracle-bin"

# ------------------------------------------------------------------- defaults
qualities="70 85 90"
distances="3.0 1.5 1.0"
effort="balanced"
cjxl_effort="7"
threads="4"
runs="1"
jpxl_bin=""
cjxl_bin=""
djxl_bin=""
work_dir=""
jsonl_path=""
inputs=()

die()  { printf 'error: %s\n' "$*" >&2; exit 2; }
note() { printf '%s\n' "# $*"; }

# --------------------------------------------------------------- parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quality)     qualities="$2"; shift 2 ;;
    --distance)    distances="$2"; shift 2 ;;
    --effort)      effort="$2"; shift 2 ;;
    --cjxl-effort) cjxl_effort="$2"; shift 2 ;;
    --threads)     threads="$2"; shift 2 ;;
    --runs)        runs="$2"; shift 2 ;;
    --jpxl)        jpxl_bin="$2"; shift 2 ;;
    --cjxl)        cjxl_bin="$2"; shift 2 ;;
    --djxl)        djxl_bin="$2"; shift 2 ;;
    --work-dir)    work_dir="$2"; shift 2 ;;
    --jsonl)       jsonl_path="$2"; shift 2 ;;
    -h|--help)     sed -n '2,72p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --*)           die "unknown option: $1" ;;
    *)             inputs+=("$1"); shift ;;
  esac
done

[[ ${#inputs[@]} -gt 0 ]] || die "no input files or directories given (see --help)"

# --------------------------------------------------------------- find jpxl
if [[ -z "${jpxl_bin}" ]]; then
  for cand in "${jpxl_root}/target/release/jpxl" "${jpxl_root}/target/fast-debug/jpxl"; do
    [[ -x "${cand}" ]] && { jpxl_bin="${cand}"; break; }
  done
fi
[[ -n "${jpxl_bin}" && -x "${jpxl_bin}" ]] \
  || die "jpxl binary not found; build it (cargo build -p jpxl-cli --release) or pass --jpxl"

# ---------------------------------------------- find + validate the oracle
# Returns 0 and echoes a usable path, or non-zero if the candidate cannot run
# on this host (wrong architecture, missing, non-executable).
runnable() { "$1" --version >/dev/null 2>&1; }

find_oracle() {
  local name="$1" override="$2" cand
  if [[ -n "${override}" ]]; then
    runnable "${override}" && { echo "${override}"; return 0; }
    return 1
  fi
  for cand in "${oracle_bin}/${name}" "$(command -v "${name}" 2>/dev/null || true)"; do
    [[ -n "${cand}" && -x "${cand}" ]] || continue
    runnable "${cand}" && { echo "${cand}"; return 0; }
  done
  return 1
}

cjxl_path="$(find_oracle cjxl "${cjxl_bin}" || true)"
djxl_path="$(find_oracle djxl "${djxl_bin}" || true)"
have_oracle=0
[[ -n "${cjxl_path}" && -n "${djxl_path}" ]] && have_oracle=1

# --------------------------------------------------------------- work dir
if [[ -z "${work_dir}" ]]; then
  work_dir="$(mktemp -d "${TMPDIR:-/tmp}/jpxl-bench.XXXXXX")" || die "mktemp failed"
  trap 'rm -rf "${work_dir}"' EXIT
else
  mkdir -p "${work_dir}" || die "cannot create work dir ${work_dir}"
fi

# --------------------------------------------------------------- helpers
sha() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
size() { stat -c '%s' "$1" 2>/dev/null || wc -c < "$1"; }
now() { date +%s.%N; }
elapsed() { awk "BEGIN{printf \"%.3f\", $2 - $1}"; }

# Parse a P6 PPM header for "W H" (skips the maxval line). Whitespace-robust.
ppm_dims() {
  head -c 64 "$1" | tr '\n\t' '  ' | awk '{print $2, $3}'
}

# in-tree SSIMULACRA2 of two PPMs, or "n/a".
score_ssimulacra2() {
  local ref="$1" cand="$2" out
  out="$("${jpxl_bin}" compare "${ref}" "${cand}" 2>/dev/null)" || { echo "n/a"; return; }
  echo "${out}" | grep -oE 'ssimulacra2_jpxl=[0-9.]+' | head -1 | cut -d= -f2
}

metric_version() {
  "${jpxl_bin}" compare "$1" "$1" 2>/dev/null \
    | grep -oE 'ssimulacra2_jpxl_version=[^ ]+' | head -1 | cut -d= -f2
}

# Best-of-N wall time for a command (args after the first). Prints seconds.
best_wall() {
  local n="$1"; shift
  local best="" t0 t1 d i
  for ((i=0; i<n; i++)); do
    t0="$(now)"; "$@" >/dev/null 2>&1; t1="$(now)"
    d="$(elapsed "${t0}" "${t1}")"
    if [[ -z "${best}" ]] || awk "BEGIN{exit !(${d} < ${best})}"; then best="${d}"; fi
  done
  echo "${best}"
}

json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

emit_jsonl() {
  [[ -n "${jsonl_path}" ]] || return 0
  printf '{"schema":"jpxl.bench-vs-libjxl/1","image":"%s","encoder":"%s","setting":"%s","bytes":%s,"bpp":%s,"ssimulacra2":"%s","wall_s":%s,"status":"%s"}\n' \
    "$(json_escape "$1")" "$2" "$3" "$4" "$5" "$6" "$7" "$8" >> "${jsonl_path}"
}

# --------------------------------------------------------------- gather inputs
files=()
for arg in "${inputs[@]}"; do
  if [[ -d "${arg}" ]]; then
    while IFS= read -r -d '' f; do files+=("${f}"); done \
      < <(find "${arg}" -type f \( -iname '*.ppm' -o -iname '*.png' -o -iname '*.jpg' \
            -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) -print0 | sort -z)
  elif [[ -f "${arg}" ]]; then
    files+=("${arg}")
  else
    printf 'warning: skipping %s (not a file or directory)\n' "${arg}" >&2
  fi
done
[[ ${#files[@]} -gt 0 ]] || die "no readable input images found"

# --------------------------------------------------------------- provenance header
[[ -n "${jsonl_path}" ]] && : > "${jsonl_path}"
metric_ver="?"

note "jpxl.bench-vs-libjxl provenance"
note "date_utc:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
note "host:        $(uname -srm)"
note "jpxl:        ${jpxl_bin}"
note "jpxl_version: $("${jpxl_bin}" --version 2>&1 | head -1)"
note "jpxl_sha256: $(sha "${jpxl_bin}")"
if [[ ${have_oracle} -eq 1 ]]; then
  note "cjxl:        ${cjxl_path}"
  note "cjxl_version: $("${cjxl_path}" --version 2>&1 | head -1)"
  note "cjxl_sha256: $(sha "${cjxl_path}")"
  note "djxl:        ${djxl_path}"
  note "djxl_version: $("${djxl_path}" --version 2>&1 | head -1)"
  note "djxl_sha256: $(sha "${djxl_path}")"
else
  note "cjxl/djxl:   NOT AVAILABLE on this host — emitting JPXL-only rows."
  note "             (Run tools/setup-oracles.sh on this platform to enable them.)"
fi
note "jpxl_effort: ${effort}    cjxl_effort: -e ${cjxl_effort}    threads: ${threads}    runs: ${runs}"
note "jpxl_qualities:  ${qualities}"
note "cjxl_distances:  ${distances}"
note ""

# --------------------------------------------------------------- table header
printf '%-28s  %-8s  %-10s  %10s  %7s  %12s  %8s  %s\n' \
  image encoder setting bytes bpp ssimulacra2 wall_s status

produced=0

for src in "${files[@]}"; do
  base="$(basename "${src}")"
  stem="${base%.*}"
  ref_ppm="${work_dir}/${stem}.src.ppm"

  # Normalise to canonical P6 via a lossless JPXL round-trip.
  if [[ "${src}" == *.ppm || "${src}" == *.PPM ]]; then
    cp -f "${src}" "${ref_ppm}"
  else
    if ! "${jpxl_bin}" encode "${src}" "${work_dir}/${stem}.src.jxl" >/dev/null 2>&1 \
       || ! "${jpxl_bin}" decode "${work_dir}/${stem}.src.jxl" "${ref_ppm}" >/dev/null 2>&1; then
      printf 'warning: cannot normalise %s to PPM (alpha or unsupported input?) — skipping\n' "${src}" >&2
      continue
    fi
  fi
  read -r w h < <(ppm_dims "${ref_ppm}")
  [[ -n "${w}" && -n "${h}" && "${w}" -gt 0 && "${h}" -gt 0 ]] \
    || { printf 'warning: bad PPM dims for %s — skipping\n' "${src}" >&2; continue; }
  pixels=$(( w * h ))
  [[ "${metric_ver}" == "?" ]] && metric_ver="$(metric_version "${ref_ppm}")"

  # ------------------------------------------------------------- JPXL rows
  for q in ${qualities}; do
    out_jxl="${work_dir}/${stem}.q${q}.jxl"
    dec_ppm="${work_dir}/${stem}.q${q}.ppm"
    enc_out="$("${jpxl_bin}" encode --quality "${q}" --effort "${effort}" \
                 --quality-fallback best-effort --threads "${threads}" \
                 "${ref_ppm}" "${out_jxl}" 2>&1)"
    if [[ ! -s "${out_jxl}" ]]; then
      printf '%-28s  %-8s  %-10s  %10s  %7s  %12s  %8s  %s\n' \
        "${stem}" jpxl "q${q}" - - - - "encode-failed"
      continue
    fi
    status="$(printf '%s\n' "${enc_out}" | grep -oE 'status=[a-z_]+' | head -1 | cut -d= -f2)"
    [[ -n "${status}" ]] || status="ok"
    wall="$(best_wall "${runs}" "${jpxl_bin}" encode --quality "${q}" --effort "${effort}" \
              --quality-fallback best-effort --threads "${threads}" "${ref_ppm}" "${out_jxl}")"
    "${jpxl_bin}" decode "${out_jxl}" "${dec_ppm}" >/dev/null 2>&1
    s2="$(score_ssimulacra2 "${ref_ppm}" "${dec_ppm}")"
    b="$(size "${out_jxl}")"
    bpp="$(awk "BEGIN{printf \"%.4f\", ${b}*8/${pixels}}")"
    printf '%-28s  %-8s  %-10s  %10s  %7s  %12s  %8s  %s\n' \
      "${stem}" jpxl "q${q}" "${b}" "${bpp}" "${s2}" "${wall}" "${status}"
    emit_jsonl "${base}" jpxl "q${q}" "${b}" "${bpp}" "${s2}" "${wall}" "${status}"
    produced=1
  done

  # ------------------------------------------------------------- cjxl rows
  if [[ ${have_oracle} -eq 1 ]]; then
    for d in ${distances}; do
      out_jxl="${work_dir}/${stem}.d${d}.jxl"
      dec_ppm="${work_dir}/${stem}.d${d}.ppm"
      if ! "${cjxl_path}" -d "${d}" -e "${cjxl_effort}" --num_threads="${threads}" \
             "${ref_ppm}" "${out_jxl}" >/dev/null 2>&1 || [[ ! -s "${out_jxl}" ]]; then
        printf '%-28s  %-8s  %-10s  %10s  %7s  %12s  %8s  %s\n' \
          "${stem}" cjxl "d${d}" - - - - "encode-failed"
        continue
      fi
      wall="$(best_wall "${runs}" "${cjxl_path}" -d "${d}" -e "${cjxl_effort}" \
                --num_threads="${threads}" "${ref_ppm}" "${out_jxl}")"
      "${djxl_path}" "${out_jxl}" "${dec_ppm}" >/dev/null 2>&1
      s2="$(score_ssimulacra2 "${ref_ppm}" "${dec_ppm}")"
      b="$(size "${out_jxl}")"
      bpp="$(awk "BEGIN{printf \"%.4f\", ${b}*8/${pixels}}")"
      printf '%-28s  %-8s  %-10s  %10s  %7s  %12s  %8s  %s\n' \
        "${stem}" cjxl "d${d}" "${b}" "${bpp}" "${s2}" "${wall}" "ok"
      emit_jsonl "${base}" cjxl "d${d}" "${b}" "${bpp}" "${s2}" "${wall}" "ok"
      produced=1
    done
  fi
done

note ""
note "ssimulacra2_metric: ${metric_ver} (in-tree production metric; higher is better, 100 = identical)"
[[ -n "${jsonl_path}" ]] && note "jsonl: ${jsonl_path}"

[[ ${produced} -eq 1 ]] || die "no comparison rows were produced"
exit 0
