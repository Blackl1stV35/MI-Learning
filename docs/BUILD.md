# Build and Setup Guide — v0.3

## Prerequisites

- Rust 1.78+  (`rustup install stable`)
- Python 3.11+ with `.venv-train` activated
- MT5 installed at `C:\Program Files\MetaTrader 5`
- ZMQ4MQL5 library (see Step 5)

---

## Step 1 — Generate synthetic buffer

```bash
cd synthetic_pretrain
python generate.py --npz ..\data\training_ready_v3b.npz --out ..\data\pretrain_buffer.npz
```

## Step 2 — Pre-train DSAC

```bash
python train_dsac.py --buffer ..\data\pretrain_buffer.npz --out ..\models\dsac_pretrained.pt
# ~7 min on GPU. Healthy: q_loss < 0.001, a_loss stable ~-0.024
```

## Step 3 — Export actor weights to JSON

Candle has a known rand version conflict on Windows — actor inference uses
pure Rust ndarray instead (no ML framework dependency, no rand conflicts).

```bash
python export_actor.py ^
    --checkpoint ..\models\dsac_pretrained.pt ^
    --json ..\models\actor_weights.json
# Output: models/actor_weights.json (~380 KB)
```

## Step 4 — Build Rust signal server

`Cargo.toml` uses `ndarray = "0.15"` only — no Candle, no rand conflicts.

```bash
cd rust_signal_server
del Cargo.lock
cargo build --release -j1
# Binary: target\release\signal_server.exe
```

## Step 5 — Install ZMQ4MQL5 for MT5

Download release from: https://github.com/dingmaotu/mql-zmq/releases

Copy files:
```
ZMQ.mqh   →  C:\Program Files\MetaTrader 5\MQL5\Include\ZMQ\ZMQ.mqh
zmq.dll   →  C:\Program Files\MetaTrader 5\MQL5\Libraries\zmq.dll
zmq.dll   →  C:\Program Files\MetaTrader 5\zmq.dll   (runtime copy)
```

Restart MT5 after copying.

## Step 6 — Start calendar parser (background)

```bash
cd calendar
python parser.py ^
    --out "C:\Program Files\MetaTrader 5\session_context.json" ^
    --schedule
```

## Step 7 — Start signal server

```bash
.\rust_signal_server\target\release\signal_server.exe ^
    --bind tcp://127.0.0.1:5555 ^
    --actor models\actor_weights.json
```

Verify startup output:
```
INFO signal_server > Signal server listening on tcp://127.0.0.1:5555
INFO signal_server > Actor loaded: models/actor_weights.json
```

## Step 8 — Install MT5 indicator

1. Copy `mql5\XAUUSD_Meta.mq5` →
   `C:\Program Files\MetaTrader 5\MQL5\Indicators\XAUUSD_Meta.mq5`
2. Open MetaEditor (F4 in MT5)
3. Open the file and compile (F7) — must show 0 errors
4. Attach to XAUUSD M1 chart from Navigator → Indicators

## File paths used by the indicator

| File | Path |
|------|------|
| Override log | `C:\Program Files\MetaTrader 5\override_log.csv` |
| Session context | `C:\Program Files\MetaTrader 5\session_context.json` |

Both created automatically on first use.

## Override log columns

```
timestamp, symbol, rule_dir, final_dir, actor_dir, signal_strength,
user_dir, user_lot, hurst, tda_w, event_risk, regime, event_type
```

`final_dir` = blended rule+actor decision.
`rule_dir`  = rule-based signal only (for comparison).
`actor_dir` = raw actor output before blending.

## Dependency notes

- `ndarray 0.15` has no rand dependency — builds cleanly on Windows
- Candle dropped due to rand 0.8/0.9 version conflict in candle-core CPU backend
- ZMQ4MQL5 requires `zmq.dll` in both `MQL5\Libraries\` AND the MT5 root dir
- `FILE_ANSI` flag used in MQL5 for Windows compatibility (not FILE_UNICODE)