<#
.SYNOPSIS
Build and install the pgvector extension into the local PostgreSQL 18 installation.

.DESCRIPTION
pgvector is not shipped as a binary for Windows, so it is built from source. This
script does the whole job in one elevated pass:

  1. adds the MSVC x64 toolset and Windows SDK to Visual Studio Build Tools
  2. builds pgvector from the pinned upstream tag with nmake
  3. installs the built DLL, control file, SQL and headers into PostgreSQL
  4. prints the resulting extension version

The PostgreSQL server itself does not need to be restarted: the extension is
loaded per-session by CREATE EXTENSION.

.PARAMETER SkipVSInstall
Skip step 1. Use on re-runs once cl.exe is present.

.PARAMETER PGVECTOR_TAG
Upstream git tag to build. Pinned deliberately so a re-run is reproducible.

.EXAMPLE
powershell -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File scripts\install-pgvector.ps1'"
#>

[CmdletBinding()]
param(
    [switch]$SkipVSInstall,
    [string]$PGVECTOR_TAG = 'v0.8.6'
)

$ErrorActionPreference = 'Stop'

# ── paths ────────────────────────────────────────────────────────────────────
$PGROOT     = 'C:\Program Files\PostgreSQL\18'
$VSPath     = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools'
$SetupExe   = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe'
$SrcDir     = Join-Path $env:LOCALAPPDATA 'pgvector-src'
$RepoDir    = Join-Path $SrcDir 'pgvector'
$LogFile    = Join-Path $SrcDir 'install-pgvector.log'
$ResultFile = Join-Path $SrcDir 'install-pgvector.result'

# The two components that provide cl.exe, nmake.exe and the SDK headers/libs.
# Deliberately narrower than the 4.2 GB "Desktop development with C++" workload.
$VSComponents = @(
    'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
    'Microsoft.VisualStudio.Component.Windows11SDK.26100'
)

function Write-Log {
    param([string]$Message)
    $line = '{0}  {1}' -f (Get-Date -Format 'HH:mm:ss'), $Message
    Write-Host $line
    Add-Content -LiteralPath $LogFile -Value $line
}

function Fail {
    param([string]$Message, [int]$Code = 1)
    Write-Log "FAILED: $Message"
    Set-Content -LiteralPath $ResultFile -Value "FAILED: $Message"
    Write-Host ''
    Write-Host "  FAILED: $Message" -ForegroundColor Red
    exit $Code
}

function Find-VcVars {
    <# Locate vcvars64.bat, which sets up the MSVC + SDK environment for nmake. #>
    $candidates = @(
        (Join-Path $VSPath 'VC\Auxiliary\Build\vcvars64.bat')
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    # Fall back to a search in case the layout differs.
    $found = Get-ChildItem -Path $VSPath -Recurse -Filter 'vcvars64.bat' -ErrorAction SilentlyContinue |
             Select-Object -First 1 -ExpandProperty FullName
    return $found
}

# ── 0. preflight ─────────────────────────────────────────────────────────────
New-Item -ItemType Directory -Path $SrcDir -Force | Out-Null
Set-Content -LiteralPath $LogFile -Value ''
Remove-Item -LiteralPath $ResultFile -ErrorAction SilentlyContinue

$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent()
)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Fail 'This script must run elevated. Right-click PowerShell, "Run as administrator", then re-run.' 2
}
Write-Log 'running elevated'

foreach ($p in @("$PGROOT\bin\postgres.exe", "$PGROOT\include\server\postgres.h", "$PGROOT\lib\postgres.lib")) {
    if (-not (Test-Path $p)) { Fail "PostgreSQL 18 not found at $PGROOT (missing: $p)" }
}
Write-Log "PostgreSQL 18 present at $PGROOT"

# ── 1. MSVC toolset + Windows SDK ────────────────────────────────────────────
function Get-ClExe {
    Get-ChildItem -Path (Join-Path $VSPath 'VC\Tools\MSVC') -Recurse -Filter 'cl.exe' -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match 'Hostx64\\x64' } | Select-Object -First 1 -ExpandProperty FullName
}
function Get-SdkIncludeDir {
    Get-ChildItem 'C:\Program Files (x86)\Windows Kits\10\Include' -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName 'um\windows.h') } |
        Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
}

$clPath = Get-ClExe
$haveSdk = [bool](Get-SdkIncludeDir)

if (($clPath -and $haveSdk) -or $SkipVSInstall) {
    Write-Log 'MSVC toolset and Windows SDK already present, skipping component install'
} else {
    Write-Log 'adding MSVC x64 toolset and Windows SDK to Visual Studio Build Tools'
    Write-Log 'this downloads roughly 600 MB and can take 10-40 minutes'

    # Build one argument string. Start-Process joins an -ArgumentList array with
    # plain spaces, which silently splits a path containing spaces, so the
    # quotes have to be part of the string we hand over.
    $argString = 'modify --installPath "{0}" --installWhileDownloading --quiet --norestart' -f $VSPath
    foreach ($c in $VSComponents) { $argString += " --add $c" }
    Write-Log "setup.exe $argString"

    $proc = Start-Process -FilePath $SetupExe -ArgumentList $argString -Wait -PassThru -NoNewWindow
    # The installer relaunches itself as a detached elevated child, so -Wait on
    # the parent can return before the packages have actually landed. Poll for
    # the result instead of trusting the exit code alone.
    $deadline = (Get-Date).AddMinutes(60)
    while ((Get-Date) -lt $deadline) {
        if ((Get-ClExe) -and (Get-SdkIncludeDir)) { break }
        if ($proc.HasExited -and $proc.ExitCode -notin @(0, 3010)) {
            # Give the elevated child a moment to surface a real failure before
            # giving up, then check whether it is still working.
            Start-Sleep -Seconds 10
            if ((Get-ClExe) -and (Get-SdkIncludeDir)) { break }
            $running = Get-Process -Name setup, vs_installer, vs_installer.windows -ErrorAction SilentlyContinue
            if ($running) {
                Write-Log 'installer child still running after parent exit; waiting'
                $deadline = (Get-Date).AddMinutes(60)
            } else {
                $installerLog = Get-ChildItem $env:TEMP -Filter 'dd_installer_elevated_*.log' -ErrorAction SilentlyContinue |
                                Sort-Object LastWriteTime -Descending | Select-Object -First 1
                $hint = if ($installerLog) { " See $($installerLog.FullName)" } else { '' }
                Fail "Visual Studio installer exited with code $($proc.ExitCode) and the toolset is still missing.$hint"
            }
        }
        Start-Sleep -Seconds 15
    }

    $clPath = Get-ClExe
    if (-not $clPath) { Fail 'cl.exe not found after installing the toolset' }
    $sdkDir = Get-SdkIncludeDir
    if (-not $sdkDir) { Fail 'Windows SDK headers not found after installing the SDK' }
    Write-Log "Visual Studio installer finished (parent exit code $($proc.ExitCode))"
}
Write-Log "cl.exe: $clPath"
$sdkDir = Get-SdkIncludeDir
if ($sdkDir) { Write-Log "Windows SDK: $sdkDir" } else { Fail 'Windows SDK headers not found' }

$vcvars = Find-VcVars
if (-not $vcvars) { Fail 'vcvars64.bat not found; the MSVC environment cannot be set up' }
Write-Log "vcvars64: $vcvars"

# ── 2. source ────────────────────────────────────────────────────────────────
if (Test-Path (Join-Path $RepoDir '.git')) {
    Write-Log "reusing existing clone at $RepoDir"
} else {
    Write-Log "cloning pgvector $PGVECTOR_TAG"
    & git clone --depth 1 --branch $PGVECTOR_TAG https://github.com/pgvector/pgvector.git $RepoDir 2>&1 |
        ForEach-Object { Write-Log "  git: $_" }
    if ($LASTEXITCODE -ne 0) { Fail "git clone failed with exit code $LASTEXITCODE" }
}
if (-not (Test-Path (Join-Path $RepoDir 'Makefile.win'))) { Fail 'Makefile.win missing from the clone' }

# ── 3. build and install ─────────────────────────────────────────────────────
# Run under cmd so the Makefile's `for %f in (...)` install loop works, with the
# MSVC environment sourced first. PGROOT is what the Makefile reads.
Write-Log 'building with nmake'
$cmdLine = 'call "{0}" >nul && set "PGROOT={1}" && nmake /NOLOGO /F Makefile.win && nmake /NOLOGO /F Makefile.win install' -f $vcvars, $PGROOT

$buildLog = Join-Path $SrcDir 'nmake.log'
$proc = Start-Process -FilePath 'cmd.exe' `
                      -ArgumentList '/c', $cmdLine `
                      -WorkingDirectory $RepoDir `
                      -RedirectStandardOutput $buildLog `
                      -RedirectStandardError (Join-Path $SrcDir 'nmake.err.log') `
                      -Wait -PassThru -NoNewWindow
Get-Content -LiteralPath $buildLog -ErrorAction SilentlyContinue | ForEach-Object { Write-Log "  $_" }
if ($proc.ExitCode -ne 0) {
    Get-Content -LiteralPath (Join-Path $SrcDir 'nmake.err.log') -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Log "  stderr: $_" }
    Fail "nmake exited with code $($proc.ExitCode); see $buildLog"
}
Write-Log 'nmake build and install succeeded'

# ── 4. verify ────────────────────────────────────────────────────────────────
$installed = @(
    "$PGROOT\lib\vector.dll"
    "$PGROOT\share\extension\vector.control"
)
foreach ($p in $installed) {
    if (-not (Test-Path $p)) { Fail "expected file missing after install: $p" }
    Write-Log "installed: $p"
}

$sqlFiles = Get-ChildItem "$PGROOT\share\extension\vector--*.sql" -ErrorAction SilentlyContinue
if (-not $sqlFiles) { Fail 'no vector--*.sql installed' }
Write-Log "installed: $($sqlFiles.Name -join ', ')"

$tagVersion = $PGVECTOR_TAG.TrimStart('v')
$expected = (Select-String -LiteralPath (Join-Path $RepoDir 'vector.control') -Pattern "default_version\s*=\s*'([^']+)'").Matches[0].Groups[1].Value
$makeVersion = (Select-String -LiteralPath (Join-Path $RepoDir 'Makefile.win') -Pattern '^EXTVERSION\s*=\s*(\S+)').Matches[0].Groups[1].Value
Write-Log "vector.control default_version = $expected"
Write-Log "Makefile.win EXTVERSION       = $makeVersion"
Write-Log "upstream tag                   = $PGVECTOR_TAG"

if ($expected -ne $tagVersion) { Fail "version mismatch: vector.control says $expected, tag $PGVECTOR_TAG says $tagVersion" }
if ($makeVersion -ne $tagVersion) { Fail "version mismatch: Makefile.win says $makeVersion, tag $PGVECTOR_TAG says $tagVersion" }

# The extension is not activated here: CREATE EXTENSION needs no admin and is
# run separately against a database, so this script stays free of credentials.
$psql = Join-Path $PGROOT 'bin\psql.exe'
$available = (& $psql --version) 2>&1
Write-Log "psql reports: $available"

$summary = @"
OK
pgvector_version=$expected
pgvector_tag=$PGVECTOR_TAG
postgresql_root=$PGROOT
msvc_cl=$clPath
"@
Set-Content -LiteralPath $ResultFile -Value $summary

Write-Host ''
Write-Host '  Build and install complete.' -ForegroundColor Green
Write-Host "  pgvector $expected is installed into $PGROOT"
Write-Host ''
Write-Host '  Next: activate it per database and run the round-trip test.'
Write-Host '    CREATE EXTENSION vector;'
Write-Host ''
exit 0
