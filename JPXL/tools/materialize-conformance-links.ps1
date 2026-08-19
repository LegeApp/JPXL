<#
.SYNOPSIS
    Repairs downloaded conformance references that Windows materialized as
    small IntxLNK placeholder files.

.DESCRIPTION
    The upstream conformance download script stores payloads by SHA-256 and
    symlinks each testcase filename to that object. Some Windows environments
    preserve the link as a small `IntxLNK` file instead of a usable filesystem
    link. This script reads every test.json sha256sums entry, finds an already
    present same-hash testcase file or `.objects/<sha>` payload, and replaces
    missing/placeholding targets with NTFS hard links (falling back to copies).

    It performs no network access. Run the upstream download script first when
    a required SHA is not already present somewhere in the corpus.
#>
[CmdletBinding()]
param(
    [string]$CorpusDir = (
        Join-Path (Split-Path -Parent $PSScriptRoot) 'tests/fixtures/conformance'
    )
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $CorpusDir).Path)
$rootPrefix = $root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
$testcases = Join-Path $root 'testcases'
if (-not (Test-Path -LiteralPath $testcases -PathType Container)) {
    throw "conformance testcases directory not found: $testcases"
}

function Assert-In-Corpus([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "refusing to modify a path outside the conformance corpus: $full"
    }
    $full
}

function Is-IntxLink([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    $stream = [IO.File]::OpenRead($Path)
    try {
        if ($stream.Length -lt 8) {
            return $false
        }
        $magic = New-Object byte[] 8
        [void]$stream.Read($magic, 0, 8)
        [Text.Encoding]::ASCII.GetString($magic) -eq "IntxLNK$([char]1)"
    } finally {
        $stream.Dispose()
    }
}

function Verified-Hash([string]$Path, [string]$Expected) {
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        throw "SHA-256 mismatch for ${Path}: expected $Expected, got $actual"
    }
    $actual
}

$records = New-Object 'System.Collections.Generic.List[object]'
Get-ChildItem -LiteralPath $testcases -Recurse -Filter test.json -File |
    Sort-Object FullName |
    ForEach-Object {
        $json = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
        if ($null -eq $json.sha256sums) {
            return
        }
        foreach ($property in $json.sha256sums.PSObject.Properties) {
            $sha = ([string]$property.Value).ToLowerInvariant()
            if ($sha -notmatch '^[0-9a-f]{64}$') {
                throw "invalid SHA-256 in $($_.FullName): $sha"
            }
            $target = Assert-In-Corpus (Join-Path $_.DirectoryName $property.Name)
            $records.Add([pscustomobject]@{
                Target = $target
                Sha256 = $sha
            })
        }
    }

$sources = @{}
$objects = Join-Path $root '.objects'
foreach ($record in $records) {
    if ((Test-Path -LiteralPath $record.Target -PathType Leaf) -and
        -not (Is-IntxLink $record.Target)) {
        [void](Verified-Hash $record.Target $record.Sha256)
        if (-not $sources.ContainsKey($record.Sha256)) {
            $sources[$record.Sha256] = $record.Target
        }
    }

    $object = Join-Path $objects $record.Sha256
    if ((Test-Path -LiteralPath $object -PathType Leaf) -and
        -not $sources.ContainsKey($record.Sha256)) {
        [void](Verified-Hash $object $record.Sha256)
        $sources[$record.Sha256] = $object
    }
}

$repaired = 0
$missing = New-Object 'System.Collections.Generic.List[string]'
foreach ($record in $records) {
    if ((Test-Path -LiteralPath $record.Target -PathType Leaf) -and
        -not (Is-IntxLink $record.Target)) {
        continue
    }
    if (-not $sources.ContainsKey($record.Sha256)) {
        $missing.Add("$($record.Target) [$($record.Sha256)]")
        continue
    }

    if (Test-Path -LiteralPath $record.Target) {
        Remove-Item -LiteralPath (Assert-In-Corpus $record.Target) -Force
    }
    $source = [string]$sources[$record.Sha256]
    try {
        $null = New-Item -ItemType HardLink -Path $record.Target -Target $source
        Write-Output "hardlinked $($record.Target) -> $source"
    } catch {
        Copy-Item -LiteralPath $source -Destination $record.Target
        Write-Output "copied $($record.Target) <- $source (hard link unavailable)"
    }
    [void](Verified-Hash $record.Target $record.Sha256)
    $repaired++
}

if ($missing.Count -ne 0) {
    throw (
        "missing $($missing.Count) conformance object(s); run the upstream " +
        "download_and_symlink_using_curl.sh once, then re-run this script:`n" +
        ($missing -join "`n")
    )
}

Write-Output "CONFORMANCE_LINKS_READY records=$($records.Count) repaired=$repaired"
