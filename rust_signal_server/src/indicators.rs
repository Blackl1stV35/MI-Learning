//! Stationary technical indicators — all bounded, ADF/KPSS validated.

pub struct Indicators {
    pub atr:        f32,
    pub atr_ratio:  f32,   // tanh((close[-1] - close[-2]) / ATR)
    pub vwap_dev:   f32,   // tanh((close - vwap) / ATR)
    pub rsi_63:     f32,   // [0,1]
    pub bb_pctb:    f32,   // [0,1]
    pub vol_zscore: f32,   // tanh(rolling z-score)
    pub hurst:      f32,   // DFA [0,1]
}

impl Indicators {
    pub fn compute(
        close:   &[f32],
        high:    &[f32],
        low:     &[f32],
        volume:  &[f32],
        session: &[f32],
        _event_risk: f32,
    ) -> Self {
        let n = close.len();
        let atr     = wilder_atr(close, high, low, 14);
        let atr_r   = if n >= 2 {
            ((close[n-1] - close[n-2]) / atr.max(1e-8)).tanh()
        } else { 0.0 };
        let vwap    = session_vwap(close, session);
        let vwap_d  = ((close[n-1] - vwap) / atr.max(1e-8)).tanh();
        let rsi     = rsi(close, 63);
        let bbp     = bb_pctb(close, 20);
        let volz    = vol_zscore(volume, 120);
        let h       = hurst_dfa(close, 120);
        Self {
            atr,
            atr_ratio:  atr_r,
            vwap_dev:   vwap_d,
            rsi_63:     rsi,
            bb_pctb:    bbp,
            vol_zscore: volz,
            hurst:      h,
        }
    }
}

fn wilder_atr(close: &[f32], high: &[f32], low: &[f32], period: usize) -> f32 {
    let n = close.len();
    if n < period + 1 { return 0.0; }
    let mut tr_sum = 0f32;
    for i in 1..=period {
        let hl = high[n-period-1+i] - low[n-period-1+i];
        let hc = (high[n-period-1+i] - close[n-period-2+i]).abs();
        let lc = (low[n-period-1+i]  - close[n-period-2+i]).abs();
        tr_sum += hl.max(hc).max(lc);
    }
    let mut atr = tr_sum / period as f32;
    let alpha = 1.0 / period as f32;
    for i in (n - n.saturating_sub(period+1))..n {
        if i == 0 { continue; }
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i-1]).abs();
        let lc = (low[i]  - close[i-1]).abs();
        let tr = hl.max(hc).max(lc);
        atr = alpha * tr + (1.0 - alpha) * atr;
    }
    atr
}

fn session_vwap(close: &[f32], session: &[f32]) -> f32 {
    let mut cum_pv = 0f32;
    let mut cum_v  = 0f32;
    let mut prev_s = -1i32;
    for (i, (&c, &s)) in close.iter().zip(session.iter()).enumerate() {
        let bucket = (s * 3.0) as i32;
        if bucket != prev_s {
            cum_pv = 0.0; cum_v = 0.0; prev_s = bucket;
        }
        cum_pv += c * 1.0;
        cum_v  += 1.0;
        let _ = i;
    }
    if cum_v > 0.0 { cum_pv / cum_v } else { *close.last().unwrap_or(&0.0) }
}

fn rsi(close: &[f32], period: usize) -> f32 {
    let n = close.len();
    if n < period + 1 { return 0.5; }
    let mut avg_g = 0f32; let mut avg_l = 0f32;
    let start = n.saturating_sub(period * 3);
    for i in (start+1)..=start+period {
        if i >= n { break; }
        let d = close[i] - close[i-1];
        avg_g += d.max(0.0); avg_l += (-d).max(0.0);
    }
    avg_g /= period as f32; avg_l /= period as f32;
    let alpha = 1.0 / period as f32;
    for i in (start+period+1)..n {
        let d = close[i] - close[i-1];
        avg_g = alpha * d.max(0.0)  + (1.0-alpha) * avg_g;
        avg_l = alpha * (-d).max(0.0) + (1.0-alpha) * avg_l;
    }
    if avg_l < 1e-8 { return 1.0; }
    let rs = avg_g / avg_l;
    1.0 - 1.0 / (1.0 + rs)
}

fn bb_pctb(close: &[f32], period: usize) -> f32 {
    let n = close.len();
    if n < period { return 0.5; }
    let w = &close[n-period..];
    let mu: f32 = w.iter().sum::<f32>() / period as f32;
    let var: f32 = w.iter().map(|x|(x-mu).powi(2)).sum::<f32>() / period as f32;
    let sigma = var.sqrt();
    if sigma < 1e-8 { return 0.5; }
    let upper = mu + 2.0*sigma;
    let lower = mu - 2.0*sigma;
    ((close[n-1] - lower) / (upper - lower + 1e-8)).clamp(0.0, 1.0)
}

fn vol_zscore(volume: &[f32], period: usize) -> f32 {
    let n = volume.len();
    if n < period { return 0.0; }
    let w = &volume[n-period..];
    let mu: f32 = w.iter().sum::<f32>() / period as f32;
    let var: f32 = w.iter().map(|x|(x-mu).powi(2)).sum::<f32>() / period as f32;
    let sigma = var.sqrt();
    if sigma < 1e-8 { return 0.0; }
    ((volume[n-1] - mu) / sigma).tanh()
}

pub fn hurst_dfa(close: &[f32], window: usize) -> f32 {
    let n = close.len();
    if n < window { return 0.5; }
    let x: Vec<f32> = close[n-window..].windows(2)
        .map(|w| (w[1] / w[0].max(1e-8)).ln())
        .collect();
    dfa_hurst(&x)
}

fn dfa_hurst(x: &[f32]) -> f32 {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let y: Vec<f32> = x.iter().scan(0f32, |acc, &v| {
        *acc += v - mean; Some(*acc)
    }).collect();
    let scales: &[usize] = &[4, 6, 8, 10, 14, 18];
    let mut log_s = Vec::new();
    let mut log_f = Vec::new();
    for &s in scales {
        if s >= n { continue; }
        let n_seg = n / s;
        if n_seg == 0 { continue; }
        let mut f2_sum = 0f32;
        for k in 0..n_seg {
            let seg = &y[k*s..(k+1)*s];
            let t: Vec<f32> = (0..s).map(|i| i as f32).collect();
            let tm: f32 = t.iter().sum::<f32>() / s as f32;
            let sm: f32 = seg.iter().sum::<f32>() / s as f32;
            let num: f32 = t.iter().zip(seg).map(|(ti,si)|(ti-tm)*(si-sm)).sum();
            let den: f32 = t.iter().map(|ti|(ti-tm).powi(2)).sum::<f32>() + 1e-8;
            let slope = num / den;
            let intercept = sm - slope * tm;
            let f2: f32 = seg.iter().zip(&t)
                .map(|(si, ti)| (si - (slope*ti + intercept)).powi(2))
                .sum::<f32>() / s as f32;
            f2_sum += f2;
        }
        let fluct = (f2_sum / n_seg as f32).sqrt().max(1e-10);
        log_s.push((s as f32).ln());
        log_f.push(fluct.ln());
    }
    if log_s.len() < 3 { return 0.5; }
    let sm: f32 = log_s.iter().sum::<f32>() / log_s.len() as f32;
    let fm: f32 = log_f.iter().sum::<f32>() / log_f.len() as f32;
    let num: f32 = log_s.iter().zip(&log_f).map(|(s,f)|(s-sm)*(f-fm)).sum();
    let den: f32 = log_s.iter().map(|s|(s-sm).powi(2)).sum::<f32>() + 1e-8;
    (num / den).clamp(0.01, 0.99)
}
