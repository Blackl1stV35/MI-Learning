#Requires -Version 5.1
<#
.SYNOPSIS
    Start the XAUUSD Meta-Policy live signal engine (signal_server.exe in serve mode).

.DESCRIPTION
    Launches Rust TCP server on 127.0.0.1:5555 with the BC-trained actor.
    MT5 connects via AutoTradeEA.mq5 (auto-executes) or XAUUSD_Meta.mq5 (indicator-only).

.PARAMETER Port
    TCP port the server listens on (default 5555, must match MT5 EA input).

.PARAMETER Actor
    Path to actor_weights.json.  Includes obs_mean/obs_std for scatter normalisation.

.EXAMPLE
    .\start_live.ps1
    .\start_live.ps1 -Port 5555 -Actor models\actor_weights.json
#>
param(
    [int]   $Port  = 5555,
    [string]$Actor = "$PSScriptRoot\models\actor_weights.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Exe = "$PSScriptRoot\rust_signal_server\target\release\signal_server.exe"

Write-Host ""
Write-Host "=== XAUUSD Meta-Policy v0.4 - Live Signal Engine ===" -ForegroundColor Cyan

if (-not (Test-Path $Exe)) {
    Write-Error "signal_server.exe not found. Build first:`n  cd rust_signal_server; cargo build --release"
}
if (-not (Test-Path $Actor)) {
    Write-Error "Actor weights not found: $Actor`nRun: python train_offline.py"
}

$actorKB = [Math]::Round((Get-Item $Actor).Length / 1KB)
Write-Host ("Actor  : {0}  ({1} KB, includes obs normalisation)" -f $Actor, $actorKB)
Write-Host ("Bind   : 127.0.0.1:{0}" -f $Port)
Write-Host ""

# Kill any existing instance
$old = Get-Process signal_server -ErrorAction SilentlyContinue
if ($old) {
    Write-Host "Stopping previous signal_server (PID $($old.Id))..." -ForegroundColor DarkYellow
    $old | Stop-Process -Force
    Start-Sleep -Milliseconds 500
}

# Launch signal_server as a real detached process (not a PS job).
# Logs go to $LogFile; this script tails them until Ctrl+C.
$LogFile  = "$PSScriptRoot\logs\signal_server_live.log"
$null = New-Item -ItemType Directory -Force -Path "$PSScriptRoot\logs"
"" | Out-File $LogFile -Encoding utf8  # clear previous log

$proc = Start-Process `
    -FilePath   $Exe `
    -ArgumentList "--bind 127.0.0.1:$Port --actor `"$Actor`"" `
    -RedirectStandardError  $LogFile `
    -RedirectStandardOutput "$PSScriptRoot\logs\signal_server_stdout.log" `
    -NoNewWindow `
    -PassThru

if (-not $proc -or $proc.HasExited) {
    Write-Error "Failed to start signal_server.exe"
}

Write-Host ("Signal server PID {0} started" -f $proc.Id) -ForegroundColor Green
Write-Host ("Log: {0}" -f $LogFile)

# Wait for LISTENING line in log (up to 5 s)
$ready = $false
for ($i = 0; $i -lt 25; $i++) {
    Start-Sleep -Milliseconds 200
    if (Test-Path $LogFile) {
        $content = Get-Content $LogFile -Raw -ErrorAction SilentlyContinue
        if ($content -match "listening") { $ready = $true; break }
    }
}

if ($ready) {
    Write-Host ("Port {0} LISTENING  --  ready for MT5" -f $Port) -ForegroundColor Green
} else {
    Write-Host "Waiting for server to initialise..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "--- MT5 Setup ---" -ForegroundColor Cyan
Write-Host "Option A  AutoTradeEA.mq5  (auto-executes, logs position state to position_log.csv)"
Write-Host "  1. In MetaEditor: Experts > AutoTradeEA > Compile (F7)"
Write-Host "  2. Drag onto XAUUSD M1 chart"
Write-Host "  3. Inputs: SERVER_HOST=127.0.0.1  SERVER_PORT=$Port"
Write-Host ""
Write-Host "Option B  XAUUSD_Meta.mq5  (indicator only: signal overlay + no trades)"
Write-Host "  1. Insert as custom indicator on XAUUSD M1"
Write-Host "  2. Inputs: SERVER_HOST=127.0.0.1  SERVER_PORT=$Port"
Write-Host ""
Write-Host "--- Controls ---" -ForegroundColor Cyan
Write-Host ("Stop server : Stop-Process -Id {0}" -f $proc.Id)
Write-Host ("View log    : Get-Content {0} -Wait" -f $LogFile)
Write-Host ""
Write-Host "Tailing log (Ctrl+C to exit this script -- server keeps running):" -ForegroundColor DarkGray

# Tail log until Ctrl+C or process exits
try {
    $pos = 0
    while (-not $proc.HasExited) {
        Start-Sleep -Milliseconds 400
        if (Test-Path $LogFile) {
            $lines = Get-Content $LogFile -ErrorAction SilentlyContinue
            if ($lines.Count -gt $pos) {
                $lines[$pos..($lines.Count - 1)] | Write-Host -ForegroundColor DarkGray
                $pos = $lines.Count
            }
        }
    }
    Write-Host "Server process exited (code $($proc.ExitCode))" -ForegroundColor Yellow
} catch [System.Management.Automation.PipelineStoppedException] {
    Write-Host ("`nDetached. Server PID {0} still running." -f $proc.Id) -ForegroundColor Yellow
}
