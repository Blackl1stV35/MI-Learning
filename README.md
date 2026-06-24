# XAUUSD Meta-Policy System — v0.4

Rule-based 5-vote signal engine fused with a BC-trained DSAC actor for XAUUSD M1.
Runs fully local: no cloud, no ZMQ, no frozen encoder at runtime.

---

## Architecture

```
MT5 OHLCV bars (240 M1 bars per request)
        │
        │  HTTP POST  (WebRequest — broker-safe fallback when raw TCP blocked)
        ▼
Rust TcpListener 127.0.0.1:5555  ←── also accepts raw newline-JSON (TCP direct)
        │
        ├── Morlet scattering  (104D, CPU)
        ├── Hurst DFA  +  TDA Wasserstein  +  ATR / RSI / BB / VWAP
        ├── Rule-based meta-policy  (5-vote, gates: event_risk / TDA / bull-low)
        └── DSAC actor  (BC-trained, 118D obs, obs z-score normalisation built-in)
                │
                ▼
        final_dir  ·  sl_pips  ·  tp_pips  ·  lot_suggestion  ·  should_exit
                │
        ┌───────┴────────┐
        ▼                ▼
AutoTradeEA.mq5      XAUUSD_Meta.mq5
(live execution)     (indicator overlay, override log)
        │
        ▼
position_log.csv  →  live_dashboard.py (localhost:8080)

Offline retrain path:
ExportBars.mq5 → XAUUSD_M1_bars.csv
    → signal_server replay --obs-out signals_obs.bin
    → train_offline.py  (BC warmup 10K + optional DSAC)
    → actor_weights.json  (with obs_mean / obs_std embedded)
```

---

## Component status

| Component | Status | Notes |
|-----------|--------|-------|
| Rust signal server | ✅ Live | HTTP POST + raw JSON on same port 5555 |
| DSAC actor | ✅ Deployed | BC-trained on 99,761 real M1 bars (Jan–Jun 2026) |
| Obs normalisation | ✅ Active | obs_mean/obs_std embedded in actor_weights.json |
| AutoTradeEA.mq5 | ✅ Live | HTTP mode; SELL at 4081.45 confirmed 2026-06-24 |
| XAUUSD_Meta.mq5 | ✅ Live | HTTP fallback + overlay panel + override_log |
| start_live.ps1 | ✅ Ready | Launches server, waits for LISTENING, tails log |
| live_dashboard.py | ✅ Ready | localhost:8080 — signal chart, position, stats |
| Rust replay mode | ✅ Ready | `replay --bars ... --out ... --obs-out ...` |
| train_offline.py | ✅ Ready | BC warmup → BC checkpoint (DSAC diverges offline) |
| MT5 ExportBars script | ✅ Ready | CopyRates → CSV for replay / training |
| Scala data pipeline | Phase 2 | |
| Multi-user aggregation | Phase 3 | |

---

## Observation vector (118D)

```
[0:104]   Morlet scattering coefficients — raw price scale (~2500–5000 USD/oz)
          z-score normalised at inference via obs_mean/obs_std in actor_weights.json
[104]     atr_ratio       tanh((close−prev) / ATR)
[105]     vwap_dev        tanh((close−session_vwap) / ATR)
[106]     rsi_63          RSI(63) normalised [0,1]
[107]     bb_pctb         Bollinger %B [0,1]
[108]     vol_zscore      tanh(rolling z-score volume, w=120)
[109]     hurst           DFA Hurst exponent [0,1]
[110]     tda_wasserstein H0 Wasserstein distance [0,1]
[111]     session_phase   0.0=Asian / 0.5=London / 1.0=NY
[112]     event_risk      0.0=low / 0.5=medium / 1.0=high
[113]     regime          0.0=Bear / 1.0=Bull (SMA+volume proxy)
[114]     pos_dir         current position −1 / 0 / +1
[115]     unrealized_norm tanh(unrealized_pnl / ATR)
[116]     hold_fraction   bars_held / max_hold [0,1]
[117]     conf_proxy      signal convergence score [0,1]
```

---

## DSAC actor

```
Linear(118→512) → ReLU → Linear(512→256) → ReLU → Linear(256→2) → Tanh
Output[0]: actor_dir   (−1=sell, 0=hold, +1=buy)
Output[1]: actor_exit  (< −0.1 → suggest exit)
```

**Training history:**
- Pre-train (synthetic, v0.3): saturated at tanh=−1 on real obs → 100% dead gradient
- Root cause: scatter obs[0:104] = raw price ~2500–5000 USD/oz; Kaiming init → first layer ≈ 2943 → tanh clips
- Fix: z-score normalisation (obs_mean/obs_std computed on 99,761 real bars, embedded in JSON)
- BC warmup (10K steps, lr=1e-3, soft targets ±0.7): loss 2.5 → 0.00001, saturation 0% → 0%
- DSAC fine-tune: diverges offline (fresh critics + no conservative constraint → Q overestimation)
- **Deployed: BC checkpoint** — actor_dir tracks rule-engine at ±0.70 with 0% saturation

---

## Meta-policy signal logic

```
Gate 1: event_risk ≥ 1.0        → HOLD (high-impact news window)
Gate 2: tda_wasserstein > 0.35   → HOLD (regime transition detected)
Gate 3: Bull regime + vol < −0.3 → suppress SELL

Vote threshold: ≥3 votes if Hurst > 0.50 (trending), else ≥4 (mean-reverting)

SELL votes: rsi<0.35  bb<0.25  atr_ratio<−0.1  vwap_dev<−0.15  Bear
BUY  votes: rsi>0.65  bb>0.75  atr_ratio>0.1   vwap_dev>0.15   Bull

final_dir: actor_dir blended with rule_dir when |actor_dir| > threshold
```

---

## MT5 broker socket restriction (err 4014)

Some brokers block `SocketConnect` to localhost (`ERR_FUNCTION_NOT_ALLOWED`).
Both EAs auto-detect this on the first connection attempt and switch to
`WebRequest` (MT5's internal HTTP stack, which is not blocked).

**One-time setup required:**
`Tools → Options → Expert Advisors → Allow WebRequest for listed URL`
Add: `http://127.0.0.1:5555`

The server handles both protocols on the same port — no config change needed.

---

## Quick start — live trading

```powershell
# 1. Build Rust server
cd rust_signal_server
cargo build --release

# 2. Start signal server + tail log
.\start_live.ps1

# 3. In MT5:
#    Tools → Options → Expert Advisors → Allow WebRequest → add http://127.0.0.1:5555
#    MetaEditor F7 → compile AutoTradeEA.mq5 (0 errors)
#    Drag AutoTradeEA onto XAUUSD M1 chart (not H1)
#    MetaEditor F7 → compile XAUUSD_Meta.mq5
#    Insert XAUUSD_Meta indicator on same XAUUSD M1 chart

# 4. Monitor
python live_dashboard.py     # opens http://localhost:8080
```

---

## Quick start — offline retrain

```powershell
# 1. Export bars from MT5 (compile + run ExportBars.mq5 on XAUUSD M1)
#    Output: Common\Files\XAUUSD_M1_bars.csv

# 2. Run replay to get signals + 118D obs binary
.\rust_signal_server\target\release\signal_server.exe replay `
    --bars  "$env:APPDATA\MetaQuotes\Terminal\Common\Files\XAUUSD_M1_bars.csv" `
    --out   "$env:APPDATA\MetaQuotes\Terminal\Common\Files\signals.csv" `
    --obs-out "$env:APPDATA\MetaQuotes\Terminal\Common\Files\signals_obs.bin" `
    --actor models\actor_weights.json

# 3. Retrain actor via BC warmup
python train_offline.py
# Output: models/actor_weights.json (includes obs_mean/obs_std)

# 4. Restart server to pick up new weights
.\start_live.ps1
```

---

## Key files

| File | Purpose |
|------|---------|
| `rust_signal_server/src/main.rs` | TCP server, HTTP handler, replay mode |
| `rust_signal_server/src/actor.rs` | DSAC actor inference + obs normalisation |
| `models/actor_weights.json` | BC-trained weights + obs_mean/obs_std (~3.9 MB) |
| `train_offline.py` | BC warmup + optional DSAC fine-tuning |
| `mql5/AutoTradeEA.mq5` | Live auto-executor (HTTP fallback, position_log) |
| `mql5/XAUUSD_Meta.mq5` | Signal overlay indicator (HTTP fallback, override_log) |
| `mql5/ExportBars.mq5` | MT5 script: CopyRates → CSV |
| `start_live.ps1` | Server launcher + log tail |
| `live_dashboard.py` | localhost:8080 monitoring dashboard |

## Log / output files (MT5 Common\Files)

| File | Written by | Content |
|------|-----------|---------|
| `position_log.csv` | AutoTradeEA | per-bar: pos_dir, hold_frac, unrealized, action |
| `override_log.csv` | XAUUSD_Meta | trades that diverged from final_dir signal |
| `signals.csv` | Rust replay | per-bar signal (rule + actor) for backtesting |
| `signals_obs.bin` | Rust replay `--obs-out` | 118D obs binary for retraining |
| `XAUUSD_M1_bars.csv` | ExportBars.mq5 | raw OHLCV for replay / training |
