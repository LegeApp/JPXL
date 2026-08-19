<#
.SYNOPSIS
    Native-Windows equivalent of fetch-conformance.sh: clone the official
    JPEG XL conformance suite at a pinned commit, without WSL.

.DESCRIPTION
    Clones https://github.com/libjxl/conformance at PINNED_COMMIT into
    JPXL/tests/fixtures/conformance/ (gitignored -- a third-party tree with its
    own licence and history, fetched, never vendored). NETWORK ACCESS IS
    REQUIRED and no build or test invokes this: run it by hand.

    It is safe to re-run. If the checkout already sits at PINNED_COMMIT nothing
    happens; if it sits elsewhere the script fetches and checks out the pin.
    Keep this pin identical to fetch-conformance.sh's PINNED_COMMIT so the two
    entry points are interchangeable and conformance numbers stay reproducible.

    After a clone on a Windows checkout, if a downloaded reference materialises
    as a small `IntxLNK` placeholder rather than a usable link, run
    tools/materialize-conformance-links.ps1.

.PARAMETER Destination
    Override the destination directory (default:
    JPXL/tests/fixtures/conformance, or $env:JPXL_CONFORMANCE_DIR).
#>
[CmdletBinding()]
param(
    [string]$Destination
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# --------------------------------------------------------------------------
# The pinned suite revision. Keep in lockstep with fetch-conformance.sh.
# To bump it: git ls-remote https://github.com/libjxl/conformance HEAD, replace
# the SHA in BOTH scripts, commit on its own with the date and reason, and
# re-baseline any recorded pass rates in the same commit.
# --------------------------------------------------------------------------
$PinnedCommit = '4bf053529c7cefd2951be453475bb3dccc7e7be8'
$RepoUrl = 'https://github.com/libjxl/conformance'

$scriptDir = Split-Path -Parent $PSCommandPath
$jpxlRoot = Split-Path -Parent $scriptDir
if (-not $Destination) {
    $Destination = if ($env:JPXL_CONFORMANCE_DIR) {
        $env:JPXL_CONFORMANCE_DIR
    } else {
        Join-Path $jpxlRoot 'tests/fixtures/conformance'
    }
}

function Write-Log([string]$Message) { Write-Host "==> $Message" }

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $output = & git @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed ($LASTEXITCODE):`n$output"
    }
    return @($output)
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'git is required'
}
if ($PinnedCommit -notmatch '^[0-9a-f]{40}$') {
    throw "PINNED_COMMIT must be a full 40-character commit SHA, got: $PinnedCommit"
}

Write-Log "This script requires network access; it clones $RepoUrl."
Write-Log "destination: $Destination"

$gitDir = Join-Path $Destination '.git'
if (Test-Path -LiteralPath $gitDir) {
    $current = & git -C $Destination rev-parse HEAD 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $current) { $current = 'none' }
    $current = "$current".Trim()
    if ($current -eq $PinnedCommit) {
        Write-Log "already at $PinnedCommit -- nothing to do"
        return
    }
    Write-Log "checkout is at $current; moving to $PinnedCommit"
    $null = Invoke-Git @('-C', $Destination, 'fetch', '--depth', '1', 'origin', $PinnedCommit)
    $null = Invoke-Git @('-C', $Destination, 'checkout', '--detach', 'FETCH_HEAD')
} else {
    if (Test-Path -LiteralPath $Destination) {
        throw "$Destination exists but is not a git checkout; remove it and re-run"
    }
    $parent = Split-Path -Parent $Destination
    if ($parent) { $null = New-Item -ItemType Directory -Force $parent }
    Write-Log 'cloning (shallow, single commit)'
    $null = Invoke-Git @('init', '--quiet', $Destination)
    $null = Invoke-Git @('-C', $Destination, 'remote', 'add', 'origin', $RepoUrl)
    $null = Invoke-Git @('-C', $Destination, 'fetch', '--depth', '1', 'origin', $PinnedCommit)
    $null = Invoke-Git @('-C', $Destination, 'checkout', '--detach', 'FETCH_HEAD')
}

Write-Log "conformance suite ready at $Destination"
Write-Log "pinned commit: $PinnedCommit"
Write-Log 'note: this tree is gitignored and is NOT part of the JPXL repository.'
