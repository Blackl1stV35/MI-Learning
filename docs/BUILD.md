# Build and Setup Guide — v0.3

## What changed from v0.2

| Component | v0.2 | v0.3 |
|-----------|------|------|
| Rust actor inference | Candle (rand conflict, broken) | Pure ndarray (no conflicts) |
| MT5 communication | ZMQ4MQL5 library (char/uchar conflict) | MT5 native SocketCreate/Connect |
| Actor weights format | PyTorch .pt pickle | JSON (exported by export_actor.py) |
| ZMQ library dependency | Required | Removed entirely |

---

## Prerequisites

- Rust 1.78+  (`rustup install stable`)
- Python 3.11+ with `.venv-train` activated
- MT5 build 2485+ (required for native socket functions)

No external libraries required for MT5 or Rust.

---

## Step 1 — Generate synthetic buffer

```bash
cd synthetic_pretrain
python generate.py --npz ..\data\training_ready_v3b.npz --out ..\data\pretrain_buffer.npz
# Runtime: ~5 min (Hurst sampled on 500k bars, not full 5.68M)
```

## Step 2 — Pre-train DSAC

```bash
python train_dsac.py ^
    --buffer ..\data\pretrain_buffer.npz ^
    --out ..\models\dsac_pretrained.pt
# Runtime: ~7 min on GPU
# Healthy output: q_loss < 0.001 throughout, a_loss stable ~-0.024 in Phase 2
# Phase 1 (BC, steps 1-10000): actor learns buffer actions via supervised loss
# Phase 2 (RL, steps 10001-20000): actor fine-tunes against stable critic
```

## Step 3 — Export actor weights to JSON

```bash
python export_actor.py ^
    --checkpoint ..\models\dsac_pretrained.pt ^
    --json ..\models\actor_weights.json
# Output: models/actor_weights.json (~380 KB)
# Must print: ✓ Checkpoint valid
```

## Step 4 — Build Rust signal server

`Cargo.toml` uses `ndarray = "0.15"` only. No Candle, no rand conflicts.

```bash
cd rust_signal_server
del Cargo.lock
cargo build --release -j1
# Binary: target\release\signal_server.exe
# First build: ~2 min (downloads ndarray crate)
```

## Step 5 — Start calendar parser (background process)

```bash
cd calendar
python parser.py ^
    --out "C:\Users\%USERNAME%\AppData\Roaming\MetaQuotes\Terminal\Common\Files\session_context.json" ^
    --schedule
# Runs before London open (07:45 UTC) and NY open (12:45 UTC)
# Writes event_risk JSON consumed by MT5 indicator
```

## Step 6 — Start signal server

```bash
.\rust_signal_server\target\release\signal_server.exe ^
    --bind tcp://127.0.0.1:5555 ^
    --actor models\actor_weights.json

# Rule-based only (no actor, for testing):
.\rust_signal_server\target\release\signal_server.exe --bind tcp://127.0.0.1:5555
```

Expected startup output:
```
INFO signal_server > Signal server listening on tcp://127.0.0.1:5555
INFO signal_server > Actor loaded: models/actor_weights.json
```

**Important:** The signal server uses ZeroMQ REP socket internally.
MT5 native sockets send raw TCP — the server must be updated to use
a plain TCP listener instead of ZMQ REP. See note below.

## Step 7 — Install MT5 indicator

1. Copy `mql5\XAUUSD_Meta.mq5` →
   `C:\Users\%USERNAME%\AppData\Roaming\MetaQuotes\Terminal\<ID>\MQL5\Indicators\`
2. Open MetaEditor (F4 in MT5), open the file, compile (F7)
3. Must show **0 errors, 0 warnings**
4. Attach to XAUUSD M1 chart from Navigator → Indicators

**No ZMQ library required.** The indicator uses MT5 built-in socket functions
(`SocketCreate`, `SocketConnect`, `SocketSend`, `SocketRead`).
Requires MT5 build 2485+. Check: Help → About → build number.

## File paths

| File | Location |
|------|----------|
| Override log | `Common\Files\override_log.csv` (FILE_COMMON flag) |
| Session context | `Common\Files\session_context.json` (FILE_COMMON flag) |
| Actor weights | `models\actor_weights.json` (signal server reads at startup) |

`Common\Files` resolves to:
`C:\Users\%USERNAME%\AppData\Roaming\MetaQuotes\Terminal\Common\Files\`

## Dependency notes

- `numpy==1.26.4` pinned in `.venv-train` — do not upgrade
- `ndarray 0.15` has zero rand dependency — builds cleanly on Windows MSVC
- Candle dropped permanently due to rand 0.8/0.9 version conflict in candle-core CPU backend
- ZMQ4MQL5 dropped permanently due to char/uchar incompatibility with MT5 build 3000+
- `advapi32` still auto-linked via `build.rs` for ZMQ crate's TCP pair functions

## Note: Rust server TCP vs ZMQ framing

The current `main.rs` uses `zmq::REP` socket which adds ZMQ framing headers.
MT5 native sockets send raw TCP without ZMQ framing — the server will not
parse requests correctly until `main.rs` is updated to use `std::net::TcpListener`.
This is the next pending change. Until then, test the signal server
using the Python test client below:

```python
# test_client.py — verify server responds correctly
import socket, json

req = json.dumps({
    "bars": [[1900.0,1901.0,1899.0,1900.5,100.0,0.2,0.5,0.0]] * 240,
    "pos_dir": 0.0, "unrealized": 0.0, "hold_fraction": 0.0
}) + "\n"

s = socket.socket()
s.connect(("127.0.0.1", 5555))
s.send(req.encode())
resp = s.recv(65536)
print(json.loads(resp.decode()))
s.close()
```