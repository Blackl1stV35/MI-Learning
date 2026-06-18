//! LearnableScatteringBlock — CPU version for PoC.
//! Matches Python architecture: J=3, Q=4, filter_len=31, pool=2.
//! Output: 104D per bar (8ch × 12 filters + 8 lowpass).
//!
//! Weights are initialised from Morlet wavelets (no trained weights needed).
//! When C++ encoder is ported, these weights can be loaded from the checkpoint.

pub const J: usize = 3;
pub const Q: usize = 4;
pub const N_FILTERS: usize = J * Q;        // 12
pub const IN_CH:     usize = 8;
pub const FILTER_LEN: usize = 31;
pub const POOL:      usize = 2;
pub const OUT_CH:    usize = IN_CH * N_FILTERS + IN_CH;  // 104
pub const T_IN:      usize = 240;
pub const T_OUT:     usize = T_IN / POOL;  // 120

pub struct ScatterBlock {
    filter_bank: Vec<f32>,   // (N_FILTERS, FILTER_LEN)
    lowpass:     Vec<f32>,   // (IN_CH, 9)
}

impl ScatterBlock {
    pub fn new() -> Self {
        let filter_bank = init_morlet_bank();
        let lp_len = 9;
        let lowpass = vec![1.0 / lp_len as f32; IN_CH * lp_len];
        Self { filter_bank, lowpass }
    }

    /// Load weights from flat binary file (future: from C++ encoder checkpoint).
    pub fn load_weights(&mut self, bank: Vec<f32>, lp: Vec<f32>) {
        assert_eq!(bank.len(), N_FILTERS * FILTER_LEN);
        assert_eq!(lp.len(),   IN_CH * 9);
        self.filter_bank = bank;
        self.lowpass     = lp;
    }

    /// Forward pass.
    /// Input: 240 bars × 8 features (first 8 channels used).
    /// Output: Vec<f32> of length OUT_CH=104 (mean-pooled over T_OUT for obs vector).
    pub fn forward(&self, bars: &[[f32; 8]]) -> Vec<f32> {
        assert_eq!(bars.len(), T_IN);
        let mut x = [[0f32; IN_CH]; T_IN];
        for (i, bar) in bars.iter().enumerate() {
            for c in 0..IN_CH { x[i][c] = bar[c]; }
        }

        let _pad = FILTER_LEN - 1;  // 30 — static

        // ── Scatter path: grouped depthwise conv1d ─────────────────────────
        // For each channel c and filter f: convolve x[:,c] with filter_bank[f,:]
        let mut scatter_out = vec![0f32; IN_CH * N_FILTERS * T_OUT];
        for c in 0..IN_CH {
            for f in 0..N_FILTERS {
                let filt = &self.filter_bank[f * FILTER_LEN..(f+1) * FILTER_LEN];
                // Causal left-pad with zeros (static pad=30)
                for t in 0..T_OUT {
                    let t2 = t * POOL;  // position before pooling
                    let mut acc = 0f32;
                    for k in 0..FILTER_LEN {
                        let src = t2 as isize - (FILTER_LEN as isize - 1 - k as isize);
                        let val = if src >= 0 && src < T_IN as isize {
                            x[src as usize][c]
                        } else { 0.0 };
                        acc += val * filt[k];
                    }
                    // Modulus
                    let idx = (c * N_FILTERS + f) * T_OUT + t;
                    scatter_out[idx] = acc.abs();
                }
            }
        }

        // ── Lowpass path ──────────────────────────────────────────────────
        let lp_len = 9usize;
        let lp_pad = 4usize;
        let mut lp_out = vec![0f32; IN_CH * T_OUT];
        for c in 0..IN_CH {
            let lp_filt = &self.lowpass[c * lp_len..(c+1)*lp_len];
            for t in 0..T_OUT {
                let t2 = t * POOL;
                let mut acc = 0f32;
                for k in 0..lp_len {
                    let src = t2 as isize - lp_pad as isize + k as isize;
                    let val = if src >= 0 && src < T_IN as isize {
                        x[src as usize][c]
                    } else if src < 0 { x[0][c] } else { x[T_IN-1][c] };
                    acc += val * lp_filt[k];
                }
                lp_out[c * T_OUT + t] = acc;
            }
        }

        // ── Mean-pool over T_OUT → single 104D vector for obs ────────────
        let mut out = vec![0f32; OUT_CH];
        for ch in 0..IN_CH * N_FILTERS {
            let s: f32 = (0..T_OUT).map(|t| scatter_out[ch * T_OUT + t]).sum();
            out[ch] = s / T_OUT as f32;
        }
        for c in 0..IN_CH {
            let s: f32 = (0..T_OUT).map(|t| lp_out[c * T_OUT + t]).sum();
            out[IN_CH * N_FILTERS + c] = s / T_OUT as f32;
        }
        out
    }
}

fn init_morlet_bank() -> Vec<f32> {
    use std::f32::consts::PI;
    let mut bank = vec![0f32; N_FILTERS * FILTER_LEN];
    let half = FILTER_LEN as f32 / 2.0;
    for j in 0..J {
        for q in 0..Q {
            let idx   = j * Q + q;
            let exp   = j as f32 + q as f32 / Q as f32;
            let sigma = 0.8 * 2f32.powf(exp);
            let xi    = PI / 2f32.powf(exp + 1.0);
            let mut sum = 0f32;
            for k in 0..FILTER_LEN {
                let t     = k as f32 - half;
                let gauss = (- t*t / (2.0 * sigma * sigma)).exp();
                let wave  = gauss * (xi * t).cos();
                bank[idx * FILTER_LEN + k] = wave;
                sum += wave.abs();
            }
            // Normalise
            if sum > 1e-8 {
                for k in 0..FILTER_LEN {
                    bank[idx * FILTER_LEN + k] /= sum;
                }
            }
        }
    }
    bank
}
