#Requires -Version 5.1
param(
    [string]$BarsFile  = "$env:APPDATA\MetaQuotes\Terminal\Common\Files\XAUUSD_M1_bars.csv",
    [string]$OutFile   = "$env:APPDATA\MetaQuotes\Terminal\Common\Files\signals.csv",
    [string]$ActorPath = "$PSScriptRoot\models\actor_weights.json"
)
# Full backtest pipeline: Rust replay -> signal summary report
# Usage: .\backtest.ps1 [-BarsFile <path>] [-OutFile <path>] [-ActorPath <path>]

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ServerExe = "$PSScriptRoot\rust_signal_server\target\release\signal_server.exe"

Write-Host ""
Write-Host "=== XAUUSD Backtest Pipeline ===" -ForegroundColor Cyan

if (-not (Test-Path $ServerExe)) {
    Write-Error "signal_server.exe not found. Run: cargo build --release -j1"
}
if (-not (Test-Path $BarsFile)) {
    Write-Error "Bars CSV not found: $BarsFile`nRun ExportBars.mq5 in MT5 first."
}

$barRows = ((Get-Content $BarsFile | Measure-Object -Line).Lines) - 1
Write-Host ("Bars file : {0}  ({1} bars)" -f $BarsFile, $barRows) -ForegroundColor Gray

# -- Run Rust replay ----------------------------------------------------------
Write-Host ""
Write-Host "Running replay..." -ForegroundColor Yellow

$replayArgs = @("replay", "--bars", $BarsFile, "--out", $OutFile)
if ($ActorPath -and (Test-Path $ActorPath)) {
    $replayArgs += "--actor", $ActorPath
    Write-Host ("Actor     : {0}" -f $ActorPath) -ForegroundColor Gray
} else {
    Write-Host "Actor     : not found - rule-only mode" -ForegroundColor DarkYellow
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $ServerExe @replayArgs
$sw.Stop()

if ($LASTEXITCODE -ne 0) {
    Write-Error "signal_server replay failed (exit $LASTEXITCODE)"
}
Write-Host ("Replay done in {0:F1}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green

# -- Parse output and report --------------------------------------------------
Write-Host ""
Write-Host "Analysing signals..." -ForegroundColor Yellow

$sigs     = Import-Csv $OutFile
$total    = $sigs.Count
if ($total -eq 0) { Write-Error "signals.csv is empty" }

# final_dir = actor-blended (falls back to rule when actor saturates)
$buyRows   = @($sigs | Where-Object { [double]$_.final_dir -gt  0.25 })
$sellRows  = @($sigs | Where-Object { [double]$_.final_dir -lt -0.25 })
$holdRows  = @($sigs | Where-Object { [double]$_.final_dir -ge -0.25 -and [double]$_.final_dir -le 0.25 })
$sigRows   = @($sigs | Where-Object { [double]$_.signal_strength -gt 0 })
$exitRows  = @($sigs | Where-Object { $_.should_exit -eq "1" })
$tdaGated  = @($sigs | Where-Object { [double]$_.tda_wasserstein -gt 0.35 })
$trending  = @($sigs | Where-Object { [double]$_.hurst -gt 0.50 })
# direction_bias = pure rule-based (always valid, actor-independent)
$ruleBuy   = @($sigs | Where-Object { [double]$_.direction_bias -gt  0.25 })
$ruleSell  = @($sigs | Where-Object { [double]$_.direction_bias -lt -0.25 })
$ruleHold  = @($sigs | Where-Object { [double]$_.direction_bias -ge -0.25 -and [double]$_.direction_bias -le 0.25 })
# actor saturation check
$saturated = @($sigs | Where-Object { [Math]::Abs([double]$_.actor_dir) -gt 0.98 })

function ColAvg($rows, $col) {
    if ($rows.Count -eq 0) { return 0.0 }
    $vals = $rows | ForEach-Object { [double]$_.$col }
    ($vals | Measure-Object -Average).Average
}

$avgStrSig   = ColAvg $sigRows  "signal_strength"
$avgHurst    = ColAvg $sigs     "hurst"
$avgTda      = ColAvg $sigs     "tda_wasserstein"
$avgActBuy   = ColAvg $buyRows  "actor_dir"
$avgActSell  = ColAvg $sellRows "actor_dir"

function Pct($n) { if ($total -gt 0) { "{0:F1}" -f (100.0 * $n / $total) } else { "0.0" } }

Write-Host ""
Write-Host "--- Rule-based signal (direction_bias) ---"
Write-Host ("  BUY   {0,6}  ({1}%)" -f $ruleBuy.Count,  (Pct $ruleBuy.Count))  -ForegroundColor Green
Write-Host ("  SELL  {0,6}  ({1}%)" -f $ruleSell.Count, (Pct $ruleSell.Count)) -ForegroundColor Red
Write-Host ("  HOLD  {0,6}  ({1}%)" -f $ruleHold.Count, (Pct $ruleHold.Count)) -ForegroundColor Gray

Write-Host ""
Write-Host "--- Blended signal (final_dir, actor-blended) ---"
Write-Host ("  BUY   {0,6}  ({1}%)" -f $buyRows.Count,  (Pct $buyRows.Count))  -ForegroundColor Green
Write-Host ("  SELL  {0,6}  ({1}%)" -f $sellRows.Count, (Pct $sellRows.Count)) -ForegroundColor Red
Write-Host ("  HOLD  {0,6}  ({1}%)" -f $holdRows.Count, (Pct $holdRows.Count)) -ForegroundColor Gray
Write-Host ("  EXIT  {0,6}  ({1}%)" -f $exitRows.Count, (Pct $exitRows.Count)) -ForegroundColor DarkYellow
if ($saturated.Count -gt 0) {
    $satPct = Pct $saturated.Count
    Write-Host ("  [WARN] Actor saturated on {0} bars ({1}%) - rule fallback active" -f $saturated.Count, $satPct) -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "--- Signal quality ---"
Write-Host ("  Avg strength when signal     : {0:F3}" -f $avgStrSig)
Write-Host ("  Actor dir on BUY bars        : {0:F3}  (>0 = rule/actor agree)" -f $avgActBuy)
Write-Host ("  Actor dir on SELL bars       : {0:F3}  (<0 = rule/actor agree)" -f $avgActSell)

Write-Host ""
Write-Host "--- Regime and topology ---"
Write-Host ("  Avg Hurst exponent           : {0:F3}  ({1} bars trending, {2}%)" -f $avgHurst, $trending.Count, (Pct $trending.Count))
Write-Host ("  Avg TDA Wasserstein          : {0:F3}" -f $avgTda)
Write-Host ("  TDA gate fired (>0.35)       : {0} bars  ({1}%)" -f $tdaGated.Count, (Pct $tdaGated.Count))

Write-Host ""
Write-Host "--- Coverage ---"
Write-Host ("  First signal bar : {0}" -f $sigs[0].datetime)
Write-Host ("  Last  signal bar : {0}" -f $sigs[$sigs.Count - 1].datetime)
Write-Host ("  Input bars total : {0}   Signal rows: {1}" -f $barRows, $total)

Write-Host ""
Write-Host ("Signals CSV : {0}" -f $OutFile) -ForegroundColor Green
Write-Host "Next        : attach XAUUSD_Meta.mq5 in Strategy Tester"
Write-Host "              SIGNALS_CSV input = signals.csv  (filename only, MT5 reads from Common\Files\)"
Write-Host ""
