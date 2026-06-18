# XAUUSD Meta-Policy System

Rule-based signal engine + DSAC meta-policy for XAUUSD M1.
No frozen encoder. No GPU required at runtime.

## Components

| Component | Language | Status |
|-----------|----------|--------|
| Rust signal server | Rust | PoC ready |
| Calendar parser | Python | PoC ready |
| Synthetic pre-training | Python | PoC ready |
| DSAC training | Python | PoC ready |
| MT5 indicator | MQL5 | PoC ready |
| Rust scattering kernel | Rust | Phase 2 |
| Scala data pipeline | Scala | Phase 2 |
| ZMQ cloud migration | Rust | Phase 2 |
| Multi-user aggregation | Python/FastAPI | Phase 3 |
| C++ encoder | C++ | Gated on Run 15 |

## Observation vector (118D)

- Scattering coefficients [0:104] — 8ch × 12 filters + 8 lowpass
- ATR-ratio, VWAP-dev, RSI-63, BB%B, vol-zscore [104:109]
- Hurst DFA, TDA Wasserstein [109:111]
- Session phase, event_risk, regime proxy [111:114]
- pos_dir, unrealized_pnl_norm, hold_fraction, conf_proxy [114:118]

## Quick start

See docs/BUILD.md
