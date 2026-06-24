# Build and Setup Guide — v0.4

## What changed from v0.3

| Area | v0.3 | v0.4 |
|------|------|------|
| MT5 → server protocol | Raw TCP SocketConnect (blocked by some brokers) | HTTP POST via WebRequest (auto-fallback) |
| Actor training data | Synthetic pre-train buffer | 99,761 real XAUUSD M1 bars (Jan–Jun 2026) |
| Actor saturation | 100% (raw scatter scale ~2500–5000 USD/oz) | 0% (z-score normalisation embedded in JSON) |
| Training method | BC + DSAC (DSAC diverged on synthetic) | BC warmup only (DSAC diverges offline — use IQL/CQL later) |
| actor_weights.json | Weights only (~380 KB) | Weights + obs_mean[118] + obs_std[118] (~3.9 MB) |
| MT5 EAs | XAUUSD_Meta.mq5 (display only) | + AutoTradeEA.mq5 (live auto-executor) |
| Server protocol | Raw JSON only | Raw JSON + HTTP POST on same port, auto-detected |
| Monitoring | MT5 terminal log | live_dashboard.py at localhost:8080 |
| Launch | Manual exe command | start_live.ps1 (launches, waits for LISTENING, tails log) |

---

## Prerequisites

- **Rust** 1.78+ (`rustup install stable`)
- **Python** 3.11+ with `.venv-train` virtualenv activated
- **MT5** build 2485+ (required for socket functions; confirmed working on build 5830)
- No external libraries in MT5 or Rust — pure standard library

---

## Step 1 — Build Rust signal server

```powershell
cd rust_signal_server
cargo build --release
# First build: ~2 min (downloads ndarray, serde, etc.)
# Binary: target\release\signal_server.exe
```

Verify:
```powershell
.\target\release\signal_server.exe --help
# Should print usage without error
```

---

## Step 2 — Export bars from MT5

In MT5:
1. MetaEditor → compile `mql5\ExportBars.mq5` (F7 → 0 errors)
2. Navigator → Scripts → drag `ExportBars` onto XAUUSD M1 chart
3. Set `FROM_DATE` / `TO_DATE` inputs (e.g. 2026.01.01 → 2026.06.24)
4. Script runs → writes `Common\Files\XAUUSD_M1_bars.csv`

```
Common\Files = %APPDATA%\MetaQuotes\Terminal\Common\Files\
```

---

## Step 3 — Run Rust replay (signals + obs binary)

```powershell
$common = "$env:APPDATA\MetaQuotes\Terminal\Common\Files"

# Serial (default — recommended for small datasets < 500K bars)
.\rust_signal_server\target\release\signal_server.exe replay `
    --bars    "$common\XAUUSD_M1_bars.csv" `
    --out     "$common\signals.csv" `
    --obs-out "$common\signals_obs.bin" `
    --actor   "models\actor_weights.json"

# Parallel (--parallel flag — ~8x faster, ideal for 2.5M+ bar datasets)
.\rust_signal_server\target\release\signal_server.exe replay `
    --bars    "bars_2020_2026.csv" `
    --out     "signals_full.csv" `
    --obs-out "signals_obs_full.bin" `
    --actor   "models\actor_weights.json" `
    --parallel
```

Expected output (serial):
```
INFO  signal_server > Replay done -- 100001 bars in, 99762 rows out -> signals.csv
INFO  signal_server > Obs binary written: 99762 rows x 118 cols -> signals_obs.bin
```

Expected output (parallel, 8 cores, 2.5M bars):
```
INFO  signal_server > Parsed 2500000 bars from bars_2020_2026.csv
INFO  signal_server > Parallel replay: 2500000 bars, 611 chunks of 4096, 8 threads
INFO  signal_server > Parallel replay done -- 2500000 bars in, 2499761 rows out -> signals_full.csv
```

`signals_obs.bin` format: `[n_rows:u32LE][n_cols=118:u32LE][n_rows×118×f32LE]`
Load in Python: `np.frombuffer(f.read(n*118*4), '<f4').reshape(n, 118)`

---

## Step 4 — Offline actor retrain (BC warmup)

```powershell
cd d:\xauusd_system
python train_offline.py
```

What it does:
1. Loads `signals_obs.bin` (118D obs matrix)
2. Loads `signals.csv` (rule-engine decisions as BC targets)
3. Computes obs z-score stats (mean/std per feature, std clamped ≥ 1.0)
4. Reinitialises actor from scratch (Kaiming init — avoids dead gradient)
5. BC warmup 10K steps, lr=1e-3, soft targets ±0.70 (not ±1.0 — prevents re-saturation)
6. Saves BC checkpoint immediately
7. Optional DSAC 20K steps (currently diverges — BC checkpoint is retained)
8. Exports `models/actor_weights.json` with weights + obs_mean + obs_std

Expected BC output:
```
Step 0: loss=2.4912 saturation=0.0%
Step 1000: loss=0.0142 saturation=0.0%
Step 10000: loss=0.00001 saturation=0.0%
BC warmup complete — exporting actor_weights.json
```

> **Note:** Raw scatter obs[0:104] are ~2500–5000 USD/oz (raw XAUUSD price scale).
> Without normalisation, Kaiming-init layer output ≈ 2943 → tanh saturates immediately.
> The z-score stats are embedded in `actor_weights.json` as `obs_mean`/`obs_std`
> and applied automatically by `actor.rs` at inference time — no change to server startup.

---

## Step 5 — Start signal server

```powershell
.\start_live.ps1
```

Expected output:
```
=== XAUUSD Meta-Policy v0.4 - Live Signal Engine ===
Actor  : models\actor_weights.json  (3973 KB, includes obs normalisation)
Bind   : 127.0.0.1:5555

Signal server PID 11168 started
Port 5555 LISTENING  --  ready for MT5
```

The server accepts **both** protocols on port 5555:
- Raw newline-delimited JSON (persistent TCP — for direct socket clients)
- HTTP POST (WebRequest from MT5 — used when broker blocks SocketConnect)

Auto-detected per connection by inspecting the first line (`POST ` prefix = HTTP).

Per-bar log line:
```
[INFO] HTTP signal: final_dir=-1.00 actor=-0.6976 strength=0.60 regime=0.00
```

---

## Step 6 — MT5 setup

### 6a — Allow WebRequest (one-time)

> **Why:** Some brokers return `ERR_FUNCTION_NOT_ALLOWED` (err 4014) on `SocketConnect`
> to localhost. Both EAs auto-detect this and switch to `WebRequest`, which uses MT5's
> own HTTP stack and is not subject to the same restriction.

`Tools → Options → Expert Advisors`
- Check **Allow WebRequest for listed URL**
- Click **+** → type `http://127.0.0.1:5555` → Enter
- Click **OK**

### 6b — AutoTradeEA (live auto-executor)

1. Copy `mql5\AutoTradeEA.mq5` →
   `%APPDATA%\MetaQuotes\Terminal\<ID>\MQL5\Experts\`
2. MetaEditor → open → **F7** → must show **0 errors, 0 warnings**
3. Drag onto **XAUUSD M1 chart** (must be M1, not H1)
4. Check **Allow live trading** in the popup → OK
5. Experts tab should show:
   ```
   AutoTradeEA: broker blocks SocketConnect (err 4014) → HTTP mode enabled
   AutoTradeEA [LIVE]: TCP failed — will retry
   ```
   Then at the next M1 bar:
   ```
   AutoTradeEA BAR 2026.06.24 09:09  pos=0 hold=0.00 sig=-1.00 → SELL
   ```

Outputs:
- `Common\Files\position_log.csv` — per-bar: datetime, pos_dir, hold_frac, unrealized_norm, action_taken, entry_price, sl_price, tp_price
- Trades visible in MT5 Account History tab

### 6c — XAUUSD_Meta (signal overlay indicator)

1. Copy `mql5\XAUUSD_Meta.mq5` →
   `%APPDATA%\MetaQuotes\Terminal\<ID>\MQL5\Indicators\`
2. MetaEditor → **F7** → 0 errors
3. Insert as indicator on the same XAUUSD M1 chart
4. Experts tab:
   ```
   SocketCreate failed (err=XXXX) → HTTP mode
   HTTP mode: ensure http://127.0.0.1:5555 is whitelisted
   ```
5. Chart overlay updates each M1 bar:
   ```
   SELL
   Str: 60%  Conf: 70%
   Lot: 0.01  SL: 2.3  TP: 4.1
   H=0.523  TDA=0.012  Bear
   Event: LOW  Actor: -0.70
   ```

Outputs:
- `Common\Files\override_log.csv` — trades that diverged from `final_dir`

---

## Step 7 — Launch monitoring dashboard

```powershell
python live_dashboard.py
# Opens http://localhost:8080 automatically
```

Dashboard panels:
- **Status bar** — server PID, last signal time, staleness
- **Current Signal** — final_dir, actor, strength, regime
- **Open Position** — dir, entry, SL/TP, hold progress bar
- **Session Stats** — BUY/SELL/HOLD counts, actor mean, saturation %
- **Signal Timeline** — Chart.js: stepped final_dir + smooth actor trace
- **Bar Log** — scrollable table of last 30 signals
- **Position Log** — live rows from position_log.csv
- **Override Log** — XAUUSD_Meta override events

Auto-refreshes every 30 seconds.

---

## Backtest performance evaluation

```powershell
# Requires both bars CSV and signals CSV
python evaluate_performance.py `
    "$env:APPDATA\MetaQuotes\Terminal\Common\Files\XAUUSD_M1_bars.csv" `
    "$env:APPDATA\MetaQuotes\Terminal\Common\Files\signals.csv"

# Use final_dir (blended actor+rule) instead of pure rule signal
python evaluate_performance.py --final-dir
```

Output: console P&L table + `logs/performance_metrics.json`

Metrics computed: win rate, profit factor, Sharpe, max drawdown, avg P&L/trade,
gross profit/loss, bull/bear regime breakdown, BUY/SELL direction split.

---

## Expand training data to post-COVID regime (2020–2026)

```powershell
# 1. Download ~2.5M M1 bars from Dukascopy (free, no API key required)
#    First: validate price divisor is correct for your download region:
python download_dukascopy.py --validate-price

#    Full download (~45–90 min):
python download_dukascopy.py --from 2020.01.01 --to 2026.06.23 --out bars_2020_2026.csv

# 2. Run parallel replay on full dataset (~2–5 min on 8 cores)
.\rust_signal_server\target\release\signal_server.exe replay `
    --bars  bars_2020_2026.csv `
    --out   signals_full.csv `
    --obs-out signals_obs_full.bin `
    --actor models\actor_weights.json `
    --parallel

# 3. Retrain with regime-diverse data
python train_offline.py  # update paths to signals_full.csv / signals_obs_full.bin first

# 4. Restart server
.\start_live.ps1
```

---

## Backtest mode (Strategy Tester)

Use pre-computed signals instead of a live server:

```powershell
# Run replay (Step 3) to generate signals.csv
# Copy signals.csv to Common\Files if not already there

# In MT5 Strategy Tester:
# Expert: AutoTradeEA.mq5 (or XAUUSD_Meta.mq5)
# Input SIGNALS_CSV = signals.csv
# Tester uses precomputed final_dir/sl_pips/tp_pips without any server connection
```

> Replay runs with `pos_dir=0 / unrealized=0 / hold_fraction=0` (flat-position assumption).
> Rule-based signal and actor entry signals are fully accurate.
> Position-management features (exit timing, hold duration) require live mode.

---

## File paths reference

| File | Path |
|------|------|
| Actor weights | `d:\xauusd_system\models\actor_weights.json` |
| Server binary | `rust_signal_server\target\release\signal_server.exe` |
| Server log | `d:\xauusd_system\logs\signal_server_live.log` |
| AutoTradeEA source | `mql5\AutoTradeEA.mq5` |
| XAUUSD_Meta source | `mql5\XAUUSD_Meta.mq5` |
| Position log | `%APPDATA%\MetaQuotes\Terminal\Common\Files\position_log.csv` |
| Override log | `%APPDATA%\MetaQuotes\Terminal\Common\Files\override_log.csv` |
| Bars CSV | `%APPDATA%\MetaQuotes\Terminal\Common\Files\XAUUSD_M1_bars.csv` |
| Obs binary | `%APPDATA%\MetaQuotes\Terminal\Common\Files\signals_obs.bin` |

---

## Dependency notes

- `ndarray 0.15` — zero rand dependency, builds cleanly on Windows MSVC
- `serde_json` — JSON serialisation for wire protocol
- `env_logger` — structured `[TIMESTAMP INFO ...]` log output
- `anyhow` — error propagation
- **Candle** — dropped (rand 0.8/0.9 conflict in candle-core CPU backend)
- **ZMQ / ZMQ4MQL5** — dropped (char/uchar conflict in MT5 build 3000+; not needed)
- Python: `torch`, `numpy`, `tqdm` in `.venv-train`; `jinja2` for dashboard
- MT5: no external libraries — only built-in `SocketCreate` / `WebRequest` functions

---

## Known issues and workarounds

| Issue | Cause | Workaround |
|-------|-------|-----------|
| `SocketConnect err=4014` | Broker blocks raw TCP to localhost | Auto-detected → HTTP mode via WebRequest |
| `SocketCreate failed` in indicators | Some brokers restrict sockets in indicators | Same HTTP fallback (g_use_http=true) |
| DSAC diverges offline | Fresh critics + offline RL = Q overestimation | Deploy BC checkpoint; use IQL/CQL for future |
| `StringFormat` truncates to 4096 chars | MQL5 StringFormat buffer limit | Build JSON by string concatenation, not StringFormat |
| position_log.csv is 0 bytes after reload | FILE_WRITE truncates; FileFlush only on first bar | Wait 1 bar; data appears after first OnTick flush |
