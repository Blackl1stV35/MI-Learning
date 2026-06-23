//! DSAC actor — pure Rust inference, no ML framework dependency.
//!
//! Architecture: Linear(118→512) → ReLU → Linear(512→256) → ReLU → Linear(256→2) → Tanh
//! Weights loaded from PyTorch pickle via serde_json (exported by export_actor.py).

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use std::path::Path;

pub struct DsacActor {
    w0: Array2<f32>, b0: Array1<f32>,
    w2: Array2<f32>, b2: Array1<f32>,
    w4: Array2<f32>, b4: Array1<f32>,
}

impl DsacActor {
    /// Load from JSON weights file exported by export_actor.py.
    pub fn load<P: AsRef<Path>>(json_path: P) -> Result<Self> {
        let content = std::fs::read_to_string(&json_path)
            .with_context(|| format!("cannot read {:?}", json_path.as_ref()))?;
        let v: serde_json::Value = serde_json::from_str(&content)?;

        let load_mat = |key: &str, rows: usize, cols: usize| -> Result<Array2<f32>> {
            let flat: Vec<f32> = v[key].as_array()
                .with_context(|| format!("missing key {key}"))?
                .iter()
                .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                .collect();
            Array2::from_shape_vec((rows, cols), flat)
                .with_context(|| format!("shape error for {key}"))
        };
        let load_vec = |key: &str, len: usize| -> Result<Array1<f32>> {
            let flat: Vec<f32> = v[key].as_array()
                .with_context(|| format!("missing key {key}"))?
                .iter()
                .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                .collect();
            anyhow::ensure!(flat.len() == len, "bias length mismatch for {key}");
            Ok(Array1::from_vec(flat))
        };

        Ok(Self {
            w0: load_mat("w0", 512, 118)?,  b0: load_vec("b0", 512)?,
            w2: load_mat("w2", 256, 512)?,  b2: load_vec("b2", 256)?,
            w4: load_mat("w4",   2, 256)?,  b4: load_vec("b4",   2)?,
        })
    }

    pub fn forward(&self, obs: &[f32]) -> Result<(f32, f32)> {
        anyhow::ensure!(obs.len() == 118, "obs must be 118-dim, got {}", obs.len());
        let x = Array1::from_vec(obs.to_vec());

        let x = relu(self.w0.dot(&x) + &self.b0);
        let x = relu(self.w2.dot(&x) + &self.b2);
        let x = tanh_vec(self.w4.dot(&x) + &self.b4);

        Ok((x[0], x[1]))
    }
}

fn relu(x: Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v.max(0.0))
}

fn tanh_vec(x: Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v.tanh())
}

/// Blend rule-based and actor signals into final decision.
///
/// Saturation guard: if both actor outputs are at the tanh rail (|x| > 0.98)
/// the actor is out-of-distribution (synthetic pretrain vs live scatter features).
/// Fall back to pure rule signal until the actor is retrained on override data.
pub fn actor_decision(
    rule_dir:   f32,
    actor_dir:  f32,
    actor_exit: f32,
    strength:   f32,
) -> (f32, bool, f32) {
    let saturated = actor_dir.abs() > 0.98 && actor_exit.abs() > 0.98;
    if saturated {
        let final_dir = if rule_dir > 0.25 { 1.0 }
                        else if rule_dir < -0.25 { -1.0 }
                        else { 0.0 };
        return (final_dir, false, strength);
    }

    let rule_w  = (strength * 2.0).clamp(0.0, 1.0);
    let blended = rule_dir * rule_w + actor_dir * (1.0 - rule_w);
    let final_dir = if blended > 0.25 { 1.0 }
                    else if blended < -0.25 { -1.0 }
                    else { 0.0 };
    let should_exit = actor_exit < -0.1;
    let confidence  = if rule_dir != 0.0 {
        let matched = (rule_dir * actor_dir) > 0.0;
        if matched { (strength * 1.2).min(1.0) } else { strength * 0.7 }
    } else { strength };
    (final_dir, should_exit, confidence)
}