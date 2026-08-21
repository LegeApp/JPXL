<#
.SYNOPSIS
    Compatibility wrapper for the versioned JPXL/libjxl comparison harness.

.DESCRIPTION
    Builds untimed rate-distortion curves, derives equivalent-quality settings,
    and then times only those frozen settings through codec_compare.py. Raw
    JSONL is authoritative; the TSV is a derived timing summary.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)] [Alias('Input')] [string[]]$Source,
    [double[]]$JpxlBpp = @(0.5, 1.0, 2.0),
    [double[]]$CjxlDistance = @(0.5, 1.0, 2.0),
    [ValidateRange(1, 99)] [int]$Runs = 9,
    [ValidateRange(1, 256)] [int]$Threads = [Environment]::ProcessorCount,
    [ValidateSet('quality', 'balanced', 'fast')] [string]$JpxlPreset = 'balanced',
    [ValidateRange(1, 256)] [int]$CjxlThreads = $Threads,
    [ValidateRange(1, 9)] [int]$CjxlEffort = 7,
    [string]$PythonPath, [string]$JpxlPath, [string]$CjxlPath, [string]$DjxlPath,
    [string]$OutputDir, [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$scriptRoot = Split-Path -Parent $PSCommandPath
$jpxlRoot = Split-Path -Parent $scriptRoot
$repoRoot = Split-Path -Parent $jpxlRoot
$core = Join-Path $scriptRoot 'codec_compare.py'
if (-not $PythonPath) {
    $python = Get-Command python3 -ErrorAction SilentlyContinue
    if (-not $python) { $python = Get-Command python -ErrorAction SilentlyContinue }
    if (-not $python) { throw 'Python 3 is required for codec_compare.py.' }
    $PythonPath = $python.Source
}
if (-not $JpxlPath) { $JpxlPath = Join-Path $jpxlRoot 'target/release/jpxl.exe' }
if (-not $CjxlPath) { $CjxlPath = Join-Path $jpxlRoot 'tools/oracle-bin/cjxl.exe' }
if (-not $DjxlPath) { $DjxlPath = Join-Path $jpxlRoot 'tools/oracle-bin/djxl.exe' }
if (-not $OutputDir) {
    $OutputDir = Join-Path $repoRoot ('.agent/scratch/libjxl-compare-' + (Get-Date -Format 'yyyyMMddTHHmmssZ'))
}

function Invoke-Checked([string[]]$Arguments) {
    & $PythonPath $core @Arguments
    if ($LASTEXITCODE -ne 0) { throw "codec_compare.py failed with exit code $LASTEXITCODE" }
}

if (-not $SkipBuild) {
    Push-Location $jpxlRoot
    try {
        & cargo build --release -p jpxl-cli --features perceptual
        if ($LASTEXITCODE -ne 0) { throw 'Metric-enabled jpxl build failed.' }
    } finally { Pop-Location }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$images = @()
foreach ($item in $Source) {
    $path = (Resolve-Path -LiteralPath $item).Path
    $images += @{
        id = [IO.Path]::GetFileNameWithoutExtension($path)
        path = $path
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
        strata = @('caller-supplied')
        provenance = 'caller-supplied; retain provenance beside the source corpus'
    }
}
$manifest = Join-Path $OutputDir 'corpus.json'
@{ schema = 'jpxl.codec-corpus/1'; images = $images } |
    ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifest -Encoding utf8

$curveRaw = Join-Path $OutputDir 'curve-unenriched.jsonl'
$curve = Join-Path $OutputDir 'curve.jsonl'
$summary = Join-Path $OutputDir 'summary.json'
$plan = Join-Path $OutputDir 'timing-plan.json'
$timing = Join-Path $OutputDir 'timing.jsonl'
$timingSummary = Join-Path $OutputDir 'timing-summary.json'
$tsv = Join-Path $OutputDir 'timing.tsv'
$work = Join-Path $OutputDir 'artifacts'
$common = @(
    '--jpxl', $JpxlPath, '--cjxl', $CjxlPath,
    '--threads', "$Threads", '--cjxl-threads', "$CjxlThreads",
    '--preset', $JpxlPreset, '--effort', "$CjxlEffort"
)

Invoke-Checked @('manifest-check', $manifest)
Invoke-Checked (@(
    'curve', '--manifest', $manifest, '--output', $curveRaw, '--work-dir', $work,
    '--bpp', ($JpxlBpp -join ','), '--distance', ($CjxlDistance -join ','),
    '--djxl', $DjxlPath
) + $common)
Invoke-Checked @('enrich-work', '--input', $curveRaw, '--output', $curve, '--jpxl', $JpxlPath)
Invoke-Checked @('summarize', '--input', $curve, '--output', $summary, '--timing-plan', $plan)
Invoke-Checked (@(
    'time', '--plan', $plan, '--output', $timing, '--work-dir', $work,
    '--runs', "$Runs"
) + $common)
Invoke-Checked @('timing-report', '--input', $timing, '--output', $timingSummary, '--tsv', $tsv)

Write-Host "Comparison complete: $summary"
Write-Host "Raw curves: $curve"
Write-Host "Raw timing: $timing"
Get-Content -LiteralPath $tsv
