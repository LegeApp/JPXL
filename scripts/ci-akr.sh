#!/usr/bin/env bash
# AKR + format gates. Run on a clean working tree (freshness is committed-history only;
# uncommitted changes to watched paths raise AKR-G004 under --strict).
#
# Requires `akr` on PATH:
#   cargo install --git https://github.com/LegeApp/AKR.git akr-cli
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
akr check
akr check --views-current
( cd JPXL && cargo fmt --all --check )
echo "AKR gates passed."