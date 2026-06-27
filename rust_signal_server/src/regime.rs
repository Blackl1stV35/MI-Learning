//! Bear/Bull regime proxy — three timescales within the 240-bar window.
//!
//! Returns [0, 1]: higher = stronger bull conditions.
//! `bull = regime > 0.5` in meta_policy_signal() requires at least the
//! medium AND long signals to agree (0.35+0.35=0.70 > 0.5).
//!
//! Weights and their roles:
//!   s_short (SMA20 > SMA60,     w=0.15) — M1 micro-trend
//!   s_med   (SMA60 > SMA240,    w=0.35) — 1hr vs 4hr trend
//!   s_long  (240-bar return > 0, w=0.35) — 4hr price momentum
//!   s_vol   (20-bar VW return > 0, w=0.15) — volume confirmation
//!
//! Minimum pairs that reach > 0.5:
//!   med + long            = 0.70  ← primary (4hr structure both agree)
//!   short + med + long    = 0.85  ← full agreement sans volume
//!   short + med + vol     = 0.65  ← micro + medium + volume
//!   long + med + vol      = 0.85  ← medium/long + volume
//!   all four              = 1.00
//!
//! Pairs that do NOT reach 0.5 (old proxy would have called bull):
//!   short + vol alone     = 0.30  ← M1 micro-uptick, no medium/long
//!   long alone            = 0.35  ← 4hr up but no 1hr confirmation
//!   med + vol             = 0.50  ← exactly 0.50, not > 0.50
//!   short + med           = 0.50  ← exactly 0.50, not > 0.50

pub fn bear_bull_proxy(close: &[f32], volume: &[f32]) -> f32 {
    let n = close.len();
    if n < 240 { return 0.5; }

    // Scale 1 — short: SMA20 vs SMA60  (20–60 bar micro-trend)
    let sma20:  f32 = close[n-20..].iter().sum::<f32>()  / 20.0;
    let sma60:  f32 = close[n-60..].iter().sum::<f32>()  / 60.0;
    let s_short = if sma20 > sma60 { 1.0f32 } else { 0.0 };

    // Scale 2 — medium: SMA60 vs SMA240  (1hr vs 4hr trend)
    let sma240: f32 = close[n-240..].iter().sum::<f32>() / 240.0;
    let s_med   = if sma60 > sma240 { 1.0f32 } else { 0.0 };

    // Scale 3 — long: 240-bar price return  (4hr momentum)
    let ret240  = close[n-1] / close[n-240].max(1e-8) - 1.0;
    let s_long  = if ret240 > 0.0 { 1.0f32 } else { 0.0 };

    // Volume-weighted return over last 20 bars  (volume confirmation)
    let vol_w: f32 = volume[n-20..].iter().sum::<f32>().max(1e-8);
    let vw_ret: f32 = close[n-20..].windows(2)
        .zip(&volume[n-19..])
        .map(|(w, v)| (w[1] - w[0]) * v)
        .sum::<f32>() / vol_w;
    let s_vol = if vw_ret > 0.0 { 1.0f32 } else { 0.0 };

    0.15 * s_short + 0.35 * s_med + 0.35 * s_long + 0.15 * s_vol
}
