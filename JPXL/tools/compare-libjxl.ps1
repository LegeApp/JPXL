<#
.SYNOPSIS
    Reproducible JPXL VarDCT versus libjxl encoder comparison.

.DESCRIPTION
    Times the two encoders in alternating order on the same P6 PPM inputs,
    decodes each result with djxl, and writes a TSV with provenance, stream
    hashes, rate, signal quality, and optional perceptual metrics. JPXL targets
    a byte rate while cjxl targets Butteraugli distance, so this is a
    rate/distortion curve comparison, not a same-setting quality claim.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)] [Alias('Input')] [string[]]$Source,
    [double[]]$JpxlBpp = @(0.5, 1.0, 2.0),
    [double[]]$CjxlDistance = @(0.5, 1.0, 2.0),
    [ValidateRange(3, 99)] [int]$Runs = 3,
    [ValidateRange(1, 256)] [int]$Threads = [Environment]::ProcessorCount,
    [ValidateSet('quality', 'balanced', 'fast')] [string]$JpxlPreset = 'balanced',
    [ValidateRange(1, 256)] [int]$CjxlThreads = $Threads,
    [ValidateRange(1, 9)] [int]$CjxlEffort = 7,
    [string]$JpxlPath, [string]$CjxlPath, [string]$DjxlPath,
    [string]$OutputDir, [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($JpxlBpp.Count -ne $CjxlDistance.Count) { throw 'JpxlBpp and CjxlDistance need equal-length curves.' }
$scriptRoot = Split-Path -Parent $PSCommandPath
$jpxlRoot = Split-Path -Parent $scriptRoot
$repoRoot = Split-Path -Parent $jpxlRoot
if (-not $JpxlPath) { $JpxlPath = Join-Path $jpxlRoot 'target/release/jpxl.exe' }
if (-not $CjxlPath) { $CjxlPath = Join-Path $jpxlRoot 'tools/oracle-bin/cjxl.exe' }
if (-not $DjxlPath) { $DjxlPath = Join-Path $jpxlRoot 'tools/oracle-bin/djxl.exe' }
if (-not $OutputDir) { $OutputDir = Join-Path $repoRoot ('.agent/scratch/libjxl-compare-' + (Get-Date -Format 'yyyyMMddTHHmmssZ')) }

function Require-File([string]$Path, [string]$What) { if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$What not found: $Path" } }
function Sha256([string]$Path) { (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant() }
function Median([double[]]$Values) { $v = @($Values | Sort-Object); $m = [int]($v.Count / 2); if (($v.Count % 2) -eq 1) { $v[$m] } else { ($v[$m - 1] + $v[$m]) / 2.0 } }
function Read-PpmDimensions([string]$Path) {
    $bytes = [IO.File]::ReadAllBytes($Path); $tokens = New-Object 'System.Collections.Generic.List[string]'; $i = 0
    while ($tokens.Count -lt 4 -and $i -lt $bytes.Length) {
        while ($i -lt $bytes.Length -and [char]$bytes[$i] -match '\s') { $i++ }
        if ($i -lt $bytes.Length -and $bytes[$i] -eq 35) { while ($i -lt $bytes.Length -and $bytes[$i] -ne 10) { $i++ }; continue }
        $start = $i; while ($i -lt $bytes.Length -and [char]$bytes[$i] -notmatch '\s') { $i++ }
        if ($start -lt $i) { $tokens.Add([Text.Encoding]::ASCII.GetString($bytes, $start, $i - $start)) }
    }
    if ($tokens.Count -ne 4 -or $tokens[0] -ne 'P6' -or $tokens[3] -ne '255') { throw "Expected 8-bit binary RGB PPM (P6): $Path" }
    @{ Width = [int]$tokens[1]; Height = [int]$tokens[2] }
}
function Invoke-Timed([string]$Program, [string[]]$Arguments) {
    $timer = [Diagnostics.Stopwatch]::StartNew(); & $Program @Arguments *> $null; $exitCode = $LASTEXITCODE; $timer.Stop()
    if ($exitCode -ne 0) { throw "Command failed ($exitCode): $Program $($Arguments -join ' ')" }; $timer.Elapsed.TotalMilliseconds
}
function Get-Metrics([string]$Reference, [string]$Decoded) {
    $line = (& $JpxlPath compare $Reference $Decoded 2>&1 | Select-Object -Last 1)
    if ($LASTEXITCODE -ne 0) { throw "jpxl compare failed for ${Decoded}: $line" }
    $r = @{}; foreach ($n in @('psnr_db', 'ssimulacra2', 'butteraugli', 'butteraugli_pnorm3')) { if ($line -match "$n=([-+0-9.eE]+|inf)") { $r[$n] = $Matches[1] } else { $r[$n] = 'n/a' } }; $r
}

Require-File $CjxlPath 'cjxl'; Require-File $DjxlPath 'djxl'
if (-not $SkipBuild) { Push-Location $jpxlRoot; try { & cargo build --release -p jpxl-cli --features perceptual; if ($LASTEXITCODE -ne 0) { throw 'Metric-enabled jpxl build failed.' } } finally { Pop-Location } }
Require-File $JpxlPath 'jpxl'; New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$tsvPath = Join-Path $OutputDir 'results.tsv'; $metaPath = Join-Path $OutputDir 'provenance.txt'
$header = "input`tinput_sha256`twidth`theight`tcodec`tsetting`titerations`tthreads`tpreset`twall_ms_min`twall_ms_median`twall_ms_max`toutput_bytes`tbpp`toutput_sha256`tpsnr_db`tssimulacra2`tbutteraugli`tbutteraugli_pnorm3"; Set-Content -LiteralPath $tsvPath -Value $header -NoNewline
$jpxlVersion = (& $JpxlPath --version) -join ' '
$cjxlVersion = (& $CjxlPath --version 2>&1 | Select-Object -First 1)
$djxlVersion = (& $DjxlPath --version 2>&1 | Select-Object -First 1)
@("created_utc=$((Get-Date).ToUniversalTime().ToString('o'))", "host=$env:COMPUTERNAME", "os=$([Environment]::OSVersion.VersionString)", "cpu=$((Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name).Trim())", "logical_processors=$([Environment]::ProcessorCount)", "runs=$Runs (one warm-up per codec/point, then alternating JPXL/cjxl timed runs)", 'cache_regime=warm-process; encode timing excludes decode and metrics', "jpxl_threads=$Threads", "cjxl_threads=$CjxlThreads", "cjxl_effort=$CjxlEffort", "jpxl_path=$JpxlPath", "jpxl_sha256=$(Sha256 $JpxlPath)", "jpxl_version=$jpxlVersion", "cjxl_path=$CjxlPath", "cjxl_sha256=$(Sha256 $CjxlPath)", "cjxl_version=$cjxlVersion", "djxl_path=$DjxlPath", "djxl_sha256=$(Sha256 $DjxlPath)", "djxl_version=$djxlVersion") | Set-Content -LiteralPath $metaPath

foreach ($source in $Source) {
    $sourcePath = (Resolve-Path -LiteralPath $source).Path; Require-File $sourcePath 'Input PPM'; $d = Read-PpmDimensions $sourcePath
    $pixels = [int64]$d.Width * $d.Height; $sourceHash = Sha256 $sourcePath; $stem = [IO.Path]::GetFileNameWithoutExtension($sourcePath)
    for ($point = 0; $point -lt $JpxlBpp.Count; $point++) {
        $jb = $JpxlBpp[$point]; $cd = $CjxlDistance[$point]; $jo = Join-Path $OutputDir "$stem-jpxl-$jb.jxl"; $co = Join-Path $OutputDir "$stem-cjxl-d$cd.jxl"
        [void](Invoke-Timed $JpxlPath @('encode', '--bpp', "$jb", '--threads', "$Threads", '--lossy-preset', $JpxlPreset, $sourcePath, $jo)); [void](Invoke-Timed $CjxlPath @($sourcePath, $co, '-d', "$cd", '-e', "$CjxlEffort", '--num_threads', "$CjxlThreads"))
        $jt = New-Object 'System.Collections.Generic.List[double]'; $ct = New-Object 'System.Collections.Generic.List[double]'
        for ($run = 0; $run -lt $Runs; $run++) { $jt.Add((Invoke-Timed $JpxlPath @('encode', '--bpp', "$jb", '--threads', "$Threads", '--lossy-preset', $JpxlPreset, $sourcePath, $jo))); $ct.Add((Invoke-Timed $CjxlPath @($sourcePath, $co, '-d', "$cd", '-e', "$CjxlEffort", '--num_threads', "$CjxlThreads"))) }
        foreach ($entry in @(@{ Codec = 'jpxl'; Setting = "bpp=$jb"; Output = $jo; Times = $jt; Decode = (Join-Path $OutputDir "$stem-jpxl-$jb.ppm") }, @{ Codec = 'cjxl'; Setting = "distance=$cd"; Output = $co; Times = $ct; Decode = (Join-Path $OutputDir "$stem-cjxl-d$cd.ppm") })) {
            & $DjxlPath $entry.Output $entry.Decode *> $null; if ($LASTEXITCODE -ne 0) { throw "djxl rejected $($entry.Output)" }; $m = Get-Metrics $sourcePath $entry.Decode
            $values = @([IO.Path]::GetFileName($sourcePath), $sourceHash, $d.Width, $d.Height, $entry.Codec, $entry.Setting, $Runs, $Threads, $JpxlPreset, ('{0:F3}' -f (($entry.Times | Measure-Object -Minimum).Minimum)), ('{0:F3}' -f (Median @($entry.Times))), ('{0:F3}' -f (($entry.Times | Measure-Object -Maximum).Maximum)), (Get-Item -LiteralPath $entry.Output).Length, ('{0:F6}' -f (((Get-Item -LiteralPath $entry.Output).Length * 8.0) / $pixels)), (Sha256 $entry.Output), $m.psnr_db, $m.ssimulacra2, $m.butteraugli, $m.butteraugli_pnorm3)
            Add-Content -LiteralPath $tsvPath -Value ("`n" + ($values -join "`t")) -NoNewline
        }
    }
}
Write-Host "Comparison complete: $tsvPath"; Write-Host "Provenance: $metaPath"; Get-Content -LiteralPath $tsvPath
