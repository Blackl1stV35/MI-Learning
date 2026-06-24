#!/usr/bin/env python3
"""
Offline DSAC fine-tuning on real XAUUSD M1 experience from Rust replay.

Pipeline:
  1. Load 118D obs vectors from Rust replay (signals_obs.bin) -- real scatter features
  2. Load signals.csv for rule direction/sl/tp per bar
  3. Load bars.csv for forward P&L simulation (TP/SL hit detection)
  4. Build (s, a, r, s', done) buffer from rule-engine experience
  5. Load dsac_pretrained.pt, fine-tune with real data
  6. Export updated actor_weights.json for signal_server deployment

The pretrained actor saturates on real data (100% saturation) because synthetic
pretrain scatter features (tile approximation) != real Morlet scattering outputs.
This script fixes that by training on the real feature distribution.

Usage:
    python train_offline.py
        [--obs PATH]          default: MT5 Common/Files/signals_obs.bin
        [--signals PATH]      default: MT5 Common/Files/signals.csv
        [--bars PATH]         default: MT5 Common/Files/XAUUSD_M1_bars.csv
        [--pretrained PATH]   default: models/dsac_pretrained.pt
        [--out-pt PATH]       default: models/dsac_finetuned.pt
        [--out-json PATH]     default: models/actor_weights.json
        [--steps N]           gradient update steps (default 20000)
        [--batch N]           batch size (default 256)
        [--device cpu|cuda]   default: cuda if available
"""
from __future__ import annotations
import argparse, csv, json, os, struct, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'synthetic_pretrain'))
from train_dsac import DSACAgent, Actor  # noqa: E402

MT5_COMMON = Path(os.environ.get("APPDATA", "")) / "MetaQuotes/Terminal/Common/Files"
OBS_DIM    = 118
MAX_HOLD   = 80
COST_USD   = 0.612   # spread + commission per trade (USD/oz)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_obs_bin(path: str) -> np.ndarray:
    """
    Binary format written by Rust --obs-out:
      [n_rows: u32 LE][n_cols: u32 LE][n_rows * n_cols * f32 LE]
    Returns float32 array (n_rows, n_cols).
    """
    with open(path, 'rb') as f:
        n_rows, n_cols = struct.unpack('<II', f.read(8))
        data = np.frombuffer(f.read(n_rows * n_cols * 4), dtype='<f4')
    obs = data.reshape(n_rows, n_cols).copy()
    assert n_cols == OBS_DIM, f"Expected {OBS_DIM} cols, got {n_cols}"
    return obs


def load_signals(path: str) -> list[dict]:
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append({
                't':  r['datetime'],
                'fd': float(r['final_dir']),
                'st': float(r['signal_strength']),
                'sl': float(r['sl_pips']),
                'tp': float(r['tp_pips']),
            })
    return rows


def load_bars(path: str):
    """Returns (list_of_(t,c,h,l), dict[t]->(index,c,h,l))."""
    bar_list = []
    bar_idx  = {}
    with open(path, newline='', encoding='utf-8') as f:
        for i, r in enumerate(csv.DictReader(f)):
            t = r['datetime']
            c, h, l = float(r['close']), float(r['high']), float(r['low'])
            bar_list.append((t, c, h, l))
            bar_idx[t] = (i, c, h, l)
    return bar_list, bar_idx


# ── Forward P&L simulation ────────────────────────────────────────────────────

def simulate_pnl(bar_list, bi: int, direction: float, sl: float, tp: float):
    """
    sl/tp are price distances from entry (sl_pips/tp_pips from signals.csv).
    Returns (pnl_usd_per_oz, done) where done=True if TP or SL hit (not timeout).
    """
    entry  = bar_list[bi][1]
    sl_px  = entry - direction * sl
    tp_px  = entry + direction * tp
    for step in range(1, MAX_HOLD + 1):
        j = bi + step
        if j >= len(bar_list):
            break
        _, c, h, l = bar_list[j]
        if direction > 0:
            if l <= sl_px:
                return -sl - COST_USD, True
            if h >= tp_px:
                return  tp - COST_USD, True
        else:
            if h >= sl_px:
                return -sl - COST_USD, True
            if l <= tp_px:
                return  tp - COST_USD, True
    j_end   = min(bi + MAX_HOLD, len(bar_list) - 1)
    timeout = direction * (bar_list[j_end][1] - entry) - COST_USD
    return timeout, False


# ── Buffer builder ────────────────────────────────────────────────────────────

def build_buffer(obs_mat, sigs, bar_list, bar_idx):
    N = len(sigs)
    assert obs_mat.shape[0] == N, f"obs rows {obs_mat.shape[0]} != signal rows {N}"

    obs_b  = np.empty((N, OBS_DIM), np.float32)
    act_b  = np.zeros((N, 2),       np.float32)
    rew_b  = np.zeros(N,             np.float32)
    nobs_b = np.empty((N, OBS_DIM), np.float32)
    done_b = np.zeros(N,             np.float32)

    trade_n = tp_n = sl_n = to_n = 0

    for i, s in enumerate(sigs):
        obs_b[i]  = obs_mat[i]
        nobs_b[i] = obs_mat[i + 1] if i + 1 < N else obs_mat[i]

        fd  = s['fd']
        st  = s['st']
        # Action: [direction, stay_or_exit_logit]
        act_b[i] = [fd, 0.2 if st > 0 else -0.2]

        if st == 0 or fd == 0:
            # HOLD bar: no position entered, zero reward
            continue

        entry = bar_idx.get(s['t'])
        if entry is None:
            continue
        bi_     = entry[0]
        sl, tp  = s['sl'], s['tp']
        if sl <= 0 or tp <= 0:
            continue

        pnl, done = simulate_pnl(bar_list, bi_, fd, sl, tp)
        rew_b[i]  = float(pnl)
        done_b[i] = float(done)
        trade_n  += 1
        if done:
            if pnl > 0: tp_n += 1
            else:        sl_n += 1
        else:
            to_n += 1

    print(f"  Buffer: {N:,} rows  |  {trade_n:,} trades  "
          f"TP={tp_n}  SL={sl_n}  Timeout={to_n}")
    trade_rew = rew_b[rew_b != 0]
    if len(trade_rew):
        print(f"  Reward stats: mean={trade_rew.mean():.4f}  "
              f"std={trade_rew.std():.4f}  "
              f"min={trade_rew.min():.4f}  max={trade_rew.max():.4f}")
    return obs_b, act_b, rew_b, nobs_b, done_b


# ── Obs normalization ─────────────────────────────────────────────────────────
# Scatter features obs[0:104] are raw XAUUSD prices (~2500-5000 USD/oz).
# Indicators obs[104:118] are already in [-1,1] or [0,1] range.
# Normalize scatter features only; leave indicators as identity transform.

def compute_obs_stats(obs_mat: np.ndarray):
    """Per-feature mean and std for z-score normalization."""
    mean = obs_mat.mean(axis=0)
    std  = obs_mat.std(axis=0)
    # Features with std < 0.1 are near-constant in the training set (pos_dir,
    # unrealized, hold_frac are all 0 in replay mode).  Set std=1.0 to leave
    # them untransformed instead of exploding division by ~0.
    std  = np.where(std < 0.1, 1.0, std).astype(np.float32)
    return mean.astype(np.float32), std


def apply_obs_norm(obs_mat: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((obs_mat - mean) / std).astype(np.float32)


# ── Actor JSON export ─────────────────────────────────────────────────────────
# Keys match Rust actor.rs: w0/b0/w2/b2/w4/b4 (weights flattened row-major).
# Architecture: net[0]=Linear(118->512), net[2]=Linear(512->256), net[4]=Linear(256->2)
# obs_mean / obs_std: z-score normalization applied before forward pass in Rust.

def export_actor_json(actor, path: str,
                      obs_mean: np.ndarray | None = None,
                      obs_std:  np.ndarray | None = None):
    sd = actor.state_dict()
    weights = {
        'w0': sd['net.0.weight'].flatten().tolist(),
        'b0': sd['net.0.bias'].tolist(),
        'w2': sd['net.2.weight'].flatten().tolist(),
        'b2': sd['net.2.bias'].tolist(),
        'w4': sd['net.4.weight'].flatten().tolist(),
        'b4': sd['net.4.bias'].tolist(),
    }
    if obs_mean is not None:
        weights['obs_mean'] = obs_mean.tolist()
        weights['obs_std']  = obs_std.tolist()
    with open(path, 'w') as f:
        json.dump(weights, f, separators=(',', ':'))
    size_kb = Path(path).stat().st_size // 1024
    print(f"  Actor JSON: {path}  ({size_kb} KB)  "
          f"(norm={'yes' if obs_mean is not None else 'no'})")


def dsac_step_clipped(agent, obs_t, act_t, rew_t, nobs_t, done_t,
                      batch_size: int, grad_clip: float = 1.0):
    """DSACAgent.update() with gradient clipping to prevent Q-divergence."""
    from train_dsac import N_QUANT, quantile_huber, soft_update
    B   = batch_size
    idx = torch.randint(0, obs_t.shape[0], (B,))
    o   = obs_t[idx].to(agent.device)
    a   = act_t[idx].to(agent.device)
    r   = rew_t[idx].to(agent.device).unsqueeze(1)
    no  = nobs_t[idx].to(agent.device)
    d   = done_t[idx].to(agent.device).unsqueeze(1)
    taus  = torch.rand(B, N_QUANT, device=agent.device)
    taus_ = torch.rand(B, N_QUANT, device=agent.device)
    with torch.no_grad():
        na   = agent.actor(no)
        q1_t = agent.q1_t(no, na, taus_)
        q2_t = agent.q2_t(no, na, taus_)
        q_t  = torch.min(q1_t, q2_t)
        tgt  = r + (1 - d) * agent.gamma * q_t
    q1 = agent.q1(o, a, taus)
    q2 = agent.q2(o, a, taus)
    ql = quantile_huber(q1, tgt, taus) + quantile_huber(q2, tgt, taus)
    agent.q_opt.zero_grad()
    ql.backward()
    torch.nn.utils.clip_grad_norm_(
        list(agent.q1.parameters()) + list(agent.q2.parameters()), grad_clip)
    agent.q_opt.step()
    al = -agent.q1(o, agent.actor(o), taus).mean()
    agent.a_opt.zero_grad()
    al.backward()
    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), grad_clip)
    agent.a_opt.step()
    soft_update(agent.q1, agent.q1_t, agent.tau)
    soft_update(agent.q2, agent.q2_t, agent.tau)
    return float(ql.item()), float(al.item())


def saturation_pct(actor, obs_t, device, n=1024):
    with torch.no_grad():
        out = actor(obs_t[:n].to(device)).cpu().numpy()
    return (np.abs(out) > 0.98).all(axis=1).mean() * 100


def bc_warmup(agent, obs_t, act_t, steps: int, batch: int):
    """
    Behavioral cloning warmup: supervised regression on rule-engine actions.

    The pretrained actor saturates at tanh = -1 on real scatter inputs
    (distribution shift from synthetic pretraining).  At saturation,
    tanh'(x) = 1 - tanh(x)^2 ~= 0, so DSAC actor gradient is ~0 and
    the standard RL update cannot escape the dead-neuron local minimum.

    Fix: regress actor outputs toward soft targets (±0.7 not ±1.0) using
    a high learning rate, explicitly pulling weights away from saturation
    before handing off to DSAC.  Soft targets leave gradient signal at all
    levels and prevent re-saturation during BC.
    """
    device  = agent.device
    bc_opt  = torch.optim.Adam(agent.actor.parameters(), lr=1e-3)
    N       = obs_t.shape[0]

    # Soft targets: scale rule direction to ±0.7 so tanh never re-saturates
    dir_t  = (act_t[:, 0] * 0.7).to(device)   # BUY=+0.7, SELL=-0.7, HOLD=0.0
    exit_t = act_t[:, 1].to(device)            # already ±0.2

    print(f"  BC warmup: {steps:,} steps  lr=1e-3  targets=[±0.7, ±0.2]")
    for step in range(1, steps + 1):
        idx   = torch.randint(0, N, (batch,))
        obs_b = obs_t[idx].to(device)
        pred  = agent.actor(obs_b)           # (B, 2)
        loss  = (F.mse_loss(pred[:, 0], dir_t[idx]) +
                 F.mse_loss(pred[:, 1], exit_t[idx]))
        bc_opt.zero_grad()
        loss.backward()
        bc_opt.step()
        if step % 1000 == 0:
            sat = saturation_pct(agent.actor, obs_t, device)
            print(f"    BC {step:5d}/{steps}  loss={loss.item():.5f}  sat={sat:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--obs',        default=str(MT5_COMMON / 'signals_obs.bin'))
    p.add_argument('--signals',    default=str(MT5_COMMON / 'signals.csv'))
    p.add_argument('--bars',       default=str(MT5_COMMON / 'XAUUSD_M1_bars.csv'))
    p.add_argument('--pretrained', default=str(ROOT / 'models' / 'dsac_pretrained.pt'))
    p.add_argument('--out-pt',     default=str(ROOT / 'models' / 'dsac_finetuned.pt'),
                   dest='out_pt')
    p.add_argument('--out-json',   default=str(ROOT / 'models' / 'actor_weights.json'),
                   dest='out_json')
    p.add_argument('--bc-steps', type=int, default=10_000, dest='bc_steps',
                   help='behavioral cloning steps before DSAC')
    p.add_argument('--reinit-actor', action='store_true', default=True,
                   dest='reinit_actor',
                   help='reinit actor weights before BC (required when pretrained '
                        'actor saturates: tanh=-1 => gradient=0, BC cannot train)')
    p.add_argument('--steps',    type=int, default=20_000)
    p.add_argument('--batch',    type=int, default=256)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--subsample', type=int, default=0,
                   help='randomly subsample N rows for training buffer '
                        '(0=use all; recommended 500000 for 2.5M row datasets '
                        'to fit in ~8 GB RAM). obs_mean/std computed on FULL dataset.')
    args = p.parse_args()

    print("\n=== Offline DSAC Fine-Tuning ===")
    print(f"Device     : {args.device}")
    print(f"Steps      : {args.steps:,}  batch={args.batch}")

    print(f"\n[1/5] Loading obs binary  : {args.obs}")
    obs_mat = load_obs_bin(args.obs)
    print(f"  {obs_mat.shape[0]:,} rows x {obs_mat.shape[1]} cols  "
          f"({obs_mat.nbytes//1024//1024} MB)")

    print(f"\n[2/5] Loading signals.csv : {args.signals}")
    sigs = load_signals(args.signals)
    print(f"  {len(sigs):,} rows")

    print(f"\n[3/5] Loading bars.csv    : {args.bars}")
    bar_list, bar_idx = load_bars(args.bars)
    print(f"  {len(bar_list):,} bars")

    print(f"\n[4/5] Building experience buffer...")
    # Compute obs stats on full dataset (important for correct normalisation
    # across all regimes even when subsampling the training buffer).
    obs_mean, obs_std = compute_obs_stats(obs_mat)
    obs_norm = apply_obs_norm(obs_mat, obs_mean, obs_std)
    print(f"  Obs norm: scatter mean={obs_mean[:104].mean():.1f}->{0:.1f}  "
          f"std range [{obs_std.min():.3f}, {obs_std.max():.3f}]")

    # Subsample for memory efficiency on large datasets (2.5M rows = ~7 GB tensors).
    if args.subsample > 0 and obs_norm.shape[0] > args.subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(obs_norm.shape[0] - 1, size=args.subsample, replace=False)
        idx.sort()
        obs_sub  = obs_norm[idx]
        sigs_sub = [sigs[i] for i in idx]
        print(f"  Subsampled {args.subsample:,} / {obs_norm.shape[0]:,} rows for buffer")
    else:
        obs_sub  = obs_norm
        sigs_sub = sigs

    obs_b, act_b, rew_b, nobs_b, done_b = build_buffer(
        obs_sub, sigs_sub, bar_list, bar_idx)

    obs_t  = torch.FloatTensor(obs_b)
    act_t  = torch.FloatTensor(act_b)
    rew_t  = torch.FloatTensor(rew_b)
    nobs_t = torch.FloatTensor(nobs_b)
    done_t = torch.FloatTensor(done_b)

    print(f"\n[5/5] Fine-tuning DSAC...")
    print(f"  Loading checkpoint : {args.pretrained}")
    ckpt  = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    agent = DSACAgent(device=args.device)
    agent.actor.load_state_dict(ckpt['actor'])
    # The checkpoint q1/q2 use a simple net.* MLP that differs from the current
    # IQN QuantileCritic (trunk/quant_embed/head).  Load actor only; critics
    # start from random and bootstrap during the fine-tuning steps below.
    print(f"  Actor loaded  (critics start fresh -- checkpoint critic arch mismatch)")

    sat_before = saturation_pct(agent.actor, obs_t, agent.device)
    print(f"  Saturation BEFORE  : {sat_before:.1f}%")

    if args.reinit_actor:
        # The pretrained actor saturates at tanh=-1 on real XAUUSD scatter.
        # tanh'(-inf) = 0 exactly in float32, so BC gradient is identically zero
        # through the final Tanh layer.  Reinitializing with Kaiming-normal
        # (PyTorch default for Linear) gives random outputs in (-1, 1) where
        # gradients are non-zero and BC can converge.
        agent.actor = Actor().to(agent.device)
        agent.a_opt = torch.optim.Adam(agent.actor.parameters(), lr=3e-4)
        sat_reinit = saturation_pct(agent.actor, obs_t, agent.device)
        print(f"  Actor reinitialized  (sat={sat_reinit:.1f}%  -- should be ~0%)")

    if args.bc_steps > 0:
        print(f"\n  --- Phase 1: Behavioral Cloning warmup ({args.bc_steps:,} steps) ---")
        bc_warmup(agent, obs_t, act_t, args.bc_steps, args.batch)
        sat_bc = saturation_pct(agent.actor, obs_t, agent.device)
        print(f"  Saturation after BC: {sat_bc:.1f}%")
        # Save BC snapshot immediately — DSAC phase may diverge with fresh critics
        agent.save(args.out_pt)
        export_actor_json(agent.actor, args.out_json, obs_mean, obs_std)
        print(f"  BC actor saved (usable for deployment even if DSAC step skipped)")

    if args.steps > 0:
        # Normalize rewards to unit variance to prevent Q-divergence.
        # Raw rewards have std~20 USD/oz; with random critics and gamma=0.99,
        # Q-targets blow up within a few thousand steps.
        rew_std = rew_t[rew_t != 0].std().item()
        if rew_std > 1.0:
            rew_t_scaled = (rew_t / rew_std).clamp(-5.0, 5.0)
            print(f"\n  Reward normalized: std {rew_std:.2f} -> 1.0  (clipped ±5)")
        else:
            rew_t_scaled = rew_t

        print(f"\n  --- Phase 2: DSAC fine-tuning ({args.steps:,} steps, grad_clip=1.0) ---")
        t0 = time.time()
        for step in range(1, args.steps + 1):
            ql, al = dsac_step_clipped(agent, obs_t, act_t, rew_t_scaled,
                                       nobs_t, done_t, args.batch)
            if step % 2000 == 0:
                sat = saturation_pct(agent.actor, obs_t, agent.device)
                print(f"  Step {step:6d}/{args.steps}  "
                      f"q_loss={ql:.4f}  a_loss={al:.4f}  "
                      f"sat={sat:.1f}%  ({time.time()-t0:.0f}s)")
                if abs(ql) > 1e6 or abs(al) > 1e6:
                    print("  [WARN] DSAC diverged — keeping BC checkpoint")
                    break

        sat_after = saturation_pct(agent.actor, obs_t, agent.device, n=min(4096, len(obs_b)))
        with torch.no_grad():
            sample_out = agent.actor(obs_t[:2048].to(agent.device)).cpu().numpy()
        print(f"\n  Saturation AFTER DSAC: {sat_after:.1f}%")
        print(f"  Dir  range : [{sample_out[:,0].min():.4f}, {sample_out[:,0].max():.4f}]")
        print(f"  Exit range : [{sample_out[:,1].min():.4f}, {sample_out[:,1].max():.4f}]")

        if sat_after < sat_bc:
            agent.save(args.out_pt)
            export_actor_json(agent.actor, args.out_json, obs_mean, obs_std)
            print(f"  DSAC improved saturation ({sat_bc:.1f}% -> {sat_after:.1f}%) — saved")
        else:
            print(f"  BC checkpoint retained (DSAC did not improve saturation)")

    print(f"\nNext steps:")
    print(f"  1. Run backtest.ps1 to verify saturation is gone")
    print(f"  2. Watch actor_dir in backtest report (should no longer be -1.000)")
    print(f"  3. If saturation < 5%, deploy {args.out_json} to live signal_server")


if __name__ == '__main__':
    main()
