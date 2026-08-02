#!/usr/bin/env bash
# Build and install the black-box reference decoders JPXL grades itself against.
#
# Two oracles are set up:
#
#   djxl / cjxl   built from the sibling libjxl checkout with CMake
#   jxl-oxide     installed with `cargo install jxl-oxide-cli --locked`
#
# The binaries are copied into tools/oracle-bin/, which
# jpxl_conformance::oracle::discover() searches first. Their exact provenance is
# recorded in tools/oracle-bin/PINNED_REVISIONS.txt so a decode mismatch can
# always be attributed to a specific oracle build.
#
# IMPORTANT (clean room): libjxl is used as a BLACK BOX. This script builds it;
# nobody reads its source. Do not consult libjxl sources while implementing the
# codec.
#
# The script is idempotent: each step is skipped when its artifact already
# exists. Force a rebuild by deleting tools/oracle-bin/ (or the specific
# binary) and re-running. Nothing here touches the JPXL crates.
#
# Usage:  tools/setup-oracles.sh
# Env:    JPXL_LIBJXL_DIR   override the libjxl checkout location
#         JPXL_JOBS         override the parallel build job count

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
jpxl_root="$(cd -- "${script_dir}/.." && pwd)"
repo_root="$(cd -- "${jpxl_root}/.." && pwd)"

libjxl_dir="${JPXL_LIBJXL_DIR:-${repo_root}/libjxl}"
build_dir="${libjxl_dir}/build"
oracle_bin="${script_dir}/oracle-bin"
revisions_file="${oracle_bin}/PINNED_REVISIONS.txt"

# CMake flags are listed once so they can be echoed verbatim into the
# provenance record below.
cmake_flags=(
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_TESTING=OFF
  -DJPEGXL_ENABLE_BENCHMARK=OFF
  -DJPEGXL_ENABLE_EXAMPLES=OFF
  -DJPEGXL_ENABLE_MANPAGES=OFF
  -DJPEGXL_ENABLE_PLUGINS=OFF
  -DJPEGXL_ENABLE_DOXYGEN=OFF
  -DJPEGXL_ENABLE_JNI=OFF
  -DJPEGXL_ENABLE_SJPEG=OFF
  -DJPEGXL_ENABLE_OPENEXR=OFF
  -DBUILD_SHARED_LIBS=OFF
)

log()  { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# Ninja is materially faster than Make on a tree the size of libjxl; fall back
# to CMake's default generator when it is absent.
if have ninja; then
  cmake_flags+=(-G Ninja)
fi

jobs="${JPXL_JOBS:-}"
if [[ -z "${jobs}" ]]; then
  if have nproc; then
    jobs="$(nproc)"
  else
    jobs=4
  fi
fi

mkdir -p "${oracle_bin}"

# ---------------------------------------------------------------- libjxl ----

build_libjxl() {
  if [[ -x "${oracle_bin}/djxl" && -x "${oracle_bin}/cjxl" ]]; then
    log "djxl and cjxl already present in tools/oracle-bin -- skipping libjxl build"
    return 0
  fi

  if [[ ! -d "${libjxl_dir}" ]]; then
    warn "no libjxl checkout at ${libjxl_dir}; skipping djxl/cjxl"
    warn "clone it there, or set JPXL_LIBJXL_DIR, then re-run"
    return 1
  fi

  have cmake || { warn "cmake not found; skipping djxl/cjxl"; return 1; }

  # libjxl vendors Highway, Brotli, skcms and friends as git submodules and
  # will not configure without them. A fresh clone has them uninitialised, so
  # initialise `third_party/` here -- but NOT `testdata/`, which is hundreds of
  # megabytes of images we have no use for. Requires network on first run.
  if [[ -d "${libjxl_dir}/.git" ]] && have git; then
    if [[ ! -e "${libjxl_dir}/third_party/highway/CMakeLists.txt" ]]; then
      log "initialising libjxl third_party submodules (network required)"
      git -C "${libjxl_dir}" submodule update --init --recursive third_party \
        || { warn "could not initialise libjxl submodules"; return 1; }
    else
      log "libjxl third_party submodules already present"
    fi
  fi

  log "configuring libjxl in ${build_dir}"
  # Note: this function is invoked from a `||` list, which disables `set -e`
  # inside it. Every failure must therefore be handled explicitly, or a broken
  # configure silently proceeds to a build that cannot work.
  cmake -S "${libjxl_dir}" -B "${build_dir}" "${cmake_flags[@]}" \
    || { warn "cmake configure failed; see the output above"; return 1; }

  log "building djxl and cjxl with ${jobs} jobs (this takes a while)"
  cmake --build "${build_dir}" --target djxl cjxl -j "${jobs}" \
    || { warn "cmake build failed; see the output above"; return 1; }

  local found=0
  local tool
  for tool in djxl cjxl; do
    # The binaries land in tools/ under the build tree; search rather than
    # hard-coding a layout that upstream may change.
    local src
    src="$(find "${build_dir}" -type f -name "${tool}" -perm -u+x -print -quit)"
    if [[ -n "${src}" ]]; then
      install -m 0755 "${src}" "${oracle_bin}/${tool}"
      log "installed ${tool} -> tools/oracle-bin/${tool}"
      found=$((found + 1))
    else
      warn "built libjxl but could not find a ${tool} binary under ${build_dir}"
    fi
  done

  [[ "${found}" -eq 2 ]]
}

# ------------------------------------------------------------- jxl-oxide ----

install_jxl_oxide() {
  if have jxl-oxide; then
    log "jxl-oxide already on PATH ($(command -v jxl-oxide)) -- skipping install"
    return 0
  fi

  have cargo || { warn "cargo not found; skipping jxl-oxide"; return 1; }

  log "installing jxl-oxide-cli (network required)"
  cargo install jxl-oxide-cli --locked
}

# ------------------------------------------------------------ provenance ----

record_revisions() {
  log "writing ${revisions_file}"
  {
    echo "# JPXL oracle provenance"
    echo "# Generated by tools/setup-oracles.sh -- do not edit by hand."
    echo "# Oracles are black boxes: their source is never read."
    echo
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host: $(uname -srm)"
    echo

    echo "[libjxl]"
    echo "path: ${libjxl_dir}"
    if [[ -d "${libjxl_dir}/.git" ]] && have git; then
      echo "revision: $(git -C "${libjxl_dir}" rev-parse HEAD)"
      echo "describe: $(git -C "${libjxl_dir}" describe --tags --always --dirty 2>/dev/null || echo unknown)"
    else
      echo "revision: unknown (not a git checkout)"
    fi
    echo "cmake_flags: ${cmake_flags[*]}"
    if [[ -x "${oracle_bin}/djxl" ]]; then
      echo "djxl_version: $("${oracle_bin}/djxl" --version 2>&1 | head -n 1 || echo unknown)"
    else
      echo "djxl_version: not installed"
    fi
    if [[ -x "${oracle_bin}/cjxl" ]]; then
      echo "cjxl_version: $("${oracle_bin}/cjxl" --version 2>&1 | head -n 1 || echo unknown)"
    else
      echo "cjxl_version: not installed"
    fi
    echo

    echo "[jxl-oxide]"
    if have jxl-oxide; then
      echo "path: $(command -v jxl-oxide)"
      echo "version: $(jxl-oxide --version 2>&1 | head -n 1 || echo unknown)"
    else
      echo "path: not installed"
      echo "version: not installed"
    fi
  } > "${revisions_file}"
}

# ------------------------------------------------------------------ main ----

log "JPXL root:   ${jpxl_root}"
log "oracle bin:  ${oracle_bin}"

libjxl_ok=0
build_libjxl || libjxl_ok=1

oxide_ok=0
install_jxl_oxide || oxide_ok=1

record_revisions

if [[ "${libjxl_ok}" -ne 0 && "${oxide_ok}" -ne 0 ]]; then
  die "no oracle could be set up; see the warnings above"
fi

log "done. Conformance tests will discover these automatically."
log "run them with: cargo test -p jpxl-conformance -- --ignored"
