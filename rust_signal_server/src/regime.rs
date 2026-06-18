//! Bear/Bull proxy from rolling price trend + volume.
//! Returns 0.0 (Bear) to 1.0 (Bull).
//! Approximates the GMM2 regime label used in NPZ without external data.

pub fn bear_bull_proxy(close: &[f32], volume: &[f32]) -> f32 {
    let n = close.len();
    if n < 60 { return 0.5; }

    // Short vs long SMA crossover (20 vs 60)
    let sma20: f32 = close[n-20..].iter().sum::<f32>() / 20.0;
    let sma60: f32 = close[n-60..].iter().sum::<f32>() / 60.0;
    let trend = if sma20 > sma60 { 1.0f32 } else { 0.0f32 };

    // Volume-weighted return direction (last 20 bars)
    let vol_w: f32 = volume[n-20..].iter().sum::<f32>().max(1e-8);
    let vw_ret: f32 = close[n-20..].windows(2)
        .zip(&volume[n-19..])
        .map(|(w, v)| (w[1]-w[0]) * v)
        .sum::<f32>() / vol_w;

    let vw_signal = if vw_ret > 0.0 { 1.0f32 } else { 0.0f32 };

    // Average of two signals
    (trend + vw_signal) / 2.0
}
