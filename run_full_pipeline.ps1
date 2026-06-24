#!/usr/bin/env pwsh
# Full training pipeline: parallel replay -> BC retrain -> server restart.
# Run after download_dukascopy.py completes.
#
# Usage:
#   .\run_full_pipeline.ps1                         # uses bars_2020_2026.csv
#   .\run_full_pipeline.ps1 -Bars bars_2020_2026.csv -Subsample 500000

param(
    [string]$Bars      = "bars_2020_2026.csv",
    [string]$Signals   = "signals_full.csv",
    [string]$Obs       = "signals_obs_full.bin",
    [string]$Actor     = "models\actor_weights.json",
    [int]   $Subsample = 500000   # rows sampled for training; 0 = use all
)

$ErrorActionPreference = "Stop"
$start = Get-Date

function Log($msg) { Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $msg" -ForegroundColor Cyan }

# ── Step 1: Parallel replay ────────────────────────────────────────────────────
Log "STEP 1/3  Parallel replay: $Bars -> $Signals + $Obs"

if (-not (Test-Path $Bars)) {
    Write-Host "ERROR: bars file not found: $Bars" -ForegroundColor Red
    Write-Host "  Run: python download_dukascopy.py --from 2020.01.01 --to 2026.06.23 --out $Bars"
    exit 1
}

$n_bars = (Get-Content $Bars | Measure-Object -Line).Lines - 1
Log "  Bars: $("{0:N0}" -f $n_bars) rows"

$server = ".\rust_signal_server\target\release\signal_server.exe"
& $server replay `
    --bars     $Bars `
    --out      $Signals `
    --obs-out  $Obs `
    --actor    $Actor `
    --parallel

if ($LASTEXITCODE -ne 0) { Write-Host "Replay failed (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }

$n_sigs = (Get-Content $Signals | Measure-Object -Line).Lines - 1
Log "  Signals: $("{0:N0}" -f $n_sigs) rows -> $Signals"
Log "  Obs bin: $([math]::Round((Get-Item $Obs).Length / 1MB, 1)) MB -> $Obs"


# ── Step 2: Retrain BC actor ───────────────────────────────────────────────────
Log "STEP 2/3  BC retrain: $("{0:N0}" -f $n_sigs) signals"

$subsample_arg = if ($Subsample -gt 0) { "--subsample", "$Subsample" } else { @() }

& ".\.venv-train\Scripts\python.exe" train_offline.py `
    --obs       $Obs `
    --signals   $Signals `
    --bars      $Bars `
    --bc-steps  10000 `
    --steps     0 `
    --device    cpu `
    @subsample_arg

if ($LASTEXITCODE -ne 0) { Write-Host "Training failed (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }


# ── Step 3: Restart live server ───────────────────────────────────────────────
Log "STEP 3/3  Restarting signal server with new actor weights..."

Stop-Process -Name signal_server -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

.\start_live.ps1

$elapsed = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
Log "Pipeline complete in $elapsed min"
Log "  Monitor: python live_dashboard.py"
Log "  Evaluate: python evaluate_performance.py $Bars $Signals"
