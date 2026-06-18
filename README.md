# XAUUSD Meta-Policy System — v0.3

Rule-based signal engine + DSAC meta-policy for XAUUSD M1.
No frozen encoder. No external ML framework at runtime. No ZMQ library in MT5.

---

## Architecture

```
Investing.com calendar  →  Calendar parser (Python)  →  session_context.json
                                                               ↓
MT5 OHLCV bars  →  MT5 native socket  →  Rust signal server  →  118D obs vector
                                              ↓                      ↓
                                        Scattering (CPU)       ndarray actor
                                        Hurst DFA              (DSAC weights)
                                        TDA Wasserstein              ↓
                                        Rule-based signal     final_dir + should_exit
                                              ↓
                                    MT5 Indicator display
                                    (no execution — user decides)
                                              ↓ (overrides only)
                                    override_log.csv
                                              ↓ (≥500 overrides)
                                    DSAC training pipeline
                                              ↓
                                    Updated actor weights → server restart
```

---

## Component status

| Component | Language | Status | Notes |
|-----------|----------|--------|-------|
| Calendar parser | Python | ✓ Running | Investing.com ec_events_sitemap |
| Rust signal server | Rust | ✓ Built | ndarray actor, ZMQ REP socket |
| DSAC pre-training | Python | ✓ Done | BC+RL, stable convergence |
| Actor weights | JSON | ✓ Exported | actor_weights.json ~380KB |
| MT5 indicator | MQL5 | ✓ Compiled | Native sockets, no ZMQ library |
| TCP server migration | Rust | ⏳ Pending | ZMQ REP → std::net::TcpListener |
| Rust scattering kernel | Rust | Phase 2 | CPU version active in PoC |
| Scala data pipeline | Scala | Phase 2 | |
| Multi-user aggregation | FastAPI | Phase 3 | |
| C++ encoder | C++ | Gated | Requires Run 15 sell_P ≥ 0.302 |

---

## Observation vector (118D)

```
[0:104]   Scattering coefficients — 8ch × 12 Morlet filters + 8 lowpass
[104]     atr_ratio       tanh((close-prev)/ATR)
[105]     vwap_dev        tanh((close-session_vwap)/ATR)
[106]     rsi_63          RSI(63) normalised [0,1]
[107]     bb_pctb         Bollinger %B [0,1]
[108]     vol_zscore      tanh(rolling z-score volume, w=120)
[109]     hurst           DFA Hurst exponent [0,1]
[110]     tda_wasserstein H0 Wasserstein distance [0,1]
[111]     session_phase   0.0=Asian / 0.5=London / 1.0=NY
[112]     event_risk      0.0=low / 0.5=medium / 1.0=high
[113]     regime          0.0=Bear / 1.0=Bull (SMA+volume proxy)
[114]     pos_dir         current position: -1/0/+1
[115]     unrealized_norm tanh(unrealized_pnl / ATR)
[116]     hold_fraction   bars_held / max_hold [0,1]
[117]     conf_proxy      signal convergence score [0,1]
```

---

## DSAC actor architecture

```
Linear(118→512) → ReLU → Linear(512→256) → ReLU → Linear(256→2) → Tanh
Output[0]: direction_bias  (-1=sell, 0=hold, +1=buy)
Output[1]: exit_logit      (< -0.1 = suggest exit)
```

Pre-trained via:
- Phase 1 (10k steps): behavioural cloning on synthetic buffer
- Phase 2 (10k steps): RL fine-tuning against stable double-Q critic
- Convergence: q_loss ~0.00013, a_loss ~-0.024 (stable, not diverging)

---

## Meta-policy signal logic

```
Gate 1: event_risk ≥ 1.0  → HOLD (high-impact event window)
Gate 2: tda_wasserstein > 0.35 → HOLD (regime transition in progress)
Gate 3: Bull + LOW vol → suppress SELL (Bull-LOW filter from paper trade)

Signal convergence votes (need ≥3 if H>0.5, else ≥4):
  SELL votes: rsi<0.35, bb<0.25, atr_ratio<-0.1, vwap_dev<-0.15, Bear
  BUY  votes: rsi>0.65, bb>0.75, atr_ratio>0.1,  vwap_dev>0.15,  Bull

Actor blend: final_dir = rule_dir×rule_weight + actor_dir×(1-rule_weight)
  rule_weight = clamp(strength×2, 0, 1)
```

---

## Override log columns

```
timestamp, symbol, rule_dir, final_dir, actor_dir, strength,
user_dir, user_lot, hurst, tda_w, event_risk, regime, event_type
```

Used for DSAC training once ≥500 overrides accumulate.

---

## Quick start

See `docs/BUILD.md` for full setup instructions.

```bash
# 1. Build
cd rust_signal_server && cargo build --release -j1

# 2. Start calendar parser (background)
python calendar\parser.py --out "...\Common\Files\session_context.json" --schedule

# 3. Start signal server
.\rust_signal_server\target\release\signal_server.exe ^
    --bind tcp://127.0.0.1:5555 ^
    --actor models\actor_weights.json

# 4. Compile and attach XAUUSD_Meta.mq5 in MetaEditor
```

---