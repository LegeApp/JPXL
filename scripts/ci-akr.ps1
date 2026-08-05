# AKR + format gates. Run on a clean working tree (freshness is computed from committed
# history; uncommitted changes to watched paths raise AKR-G004 under --strict).
#
# Requires `akr` on PATH:
#   cargo install --git https://github.com/LegeApp/AKR.git akr-cli
#   (or scripts/setup-akr-mcp.ps1 from the AKR repo)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    akr check
    if ($LASTEXITCODE -ne 0) { throw "akr check failed" }
    akr check --views-current
    if ($LASTEXITCODE -ne 0) { throw "akr check --views-current failed" }
    Push-Location JPXL
    try {
        cargo fmt --all --check
        if ($LASTEXITCODE -ne 0) { throw "cargo fmt --check failed" }
    } finally { Pop-Location }
    Write-Host "AKR gates passed."
} finally { Pop-Location }