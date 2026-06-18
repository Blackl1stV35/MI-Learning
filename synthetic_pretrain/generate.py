#!/usr/bin/env python3
"""
Synthetic pre-training experience generator.

Reads training_ready_v3b.npz, generates synthetic override experience
using three user archetypes, and outputs a pre-populated DSAC replay buffer.

Usage:
    python generate.py --npz data/training_ready_v3b.npz --out data/pretrain_buffer.npz

Archetypes:
    A (60%) — signal follower: follows signal, outcome = actual forward PnL
    B (25%) — cautious overrider: holds on event_risk>0.5 or Hurst<0.5
    C (15%) — contrarian: acts against signal, zero agreement weight if profitable
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

FORWARD_BARS = 40
ATR_PERIOD   = 14
COST_USD     = 0.612
MAX_HOLD     = 80
BASE_LOT     = 0.01
N_SYNTHETIC  = 50_000   # target experience tuples

OBS_DIM      = 118       # must match Rust server output


def wilder_atr(close, high, low, period=14):
    n = len(close)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i]-low[i],
                    abs(high[i]-close[i-1]),
                    abs(low[i]-close[i-1]))
    atr = np.zeros(n)
    atr[period] = tr[1:period+1].mean()
    a = 1.0 / period
    for i in range(period+1, n):
        atr[i] = a*tr[i] + (1-a)*atr[i-1]
    return atr


def hurst_dfa_vec(log_ret, window=120):
    """Vectorised rolling Hurst — approximate, fast."""
    n = len(log_ret)
    out = np.full(n, 0.5)
    scales = np.array([4, 6, 8, 10, 14, 18])
    for i in range(window, n):
        x = log_ret[i-window:i]
        y = np.cumsum(x - x.mean())
        flucts = []
        for s in scales:
            if s >= window: continue
            n_seg = window // s
            f2 = 0.0
            for k in range(n_seg):
                seg = y[k*s:(k+1)*s]
                t   = np.arange(s, dtype=float)
                p   = np.polyfit(t, seg, 1)
                f2 += np.mean((seg - np.polyval(p, t))**2)
            flucts.append(np.sqrt(f2 / n_seg + 1e-10))
        if len(flucts) >= 3:
            ls = np.log(scales[:len(flucts)].astype(float))
            lf = np.log(np.array(flucts))
            out[i] = np.clip(np.polyfit(ls, lf, 1)[0], 0.01, 0.99)
    return out


def rule_signal(atr_ratio, vwap_dev, rsi, bb_pctb, vol_z,
                hurst, tda_w, regime, event_risk):
    """Mirror of Rust meta_policy_signal."""
    if event_risk >= 1.0: return 0
    if tda_w > 0.35:      return 0
    bull = regime > 0.5
    if bull and vol_z < -0.3: return 0

    sell_v = sum([rsi < 0.35, bb_pctb < 0.25, atr_ratio < -0.1,
                  vwap_dev < -0.15, not bull])
    buy_v  = sum([rsi > 0.65, bb_pctb > 0.75, atr_ratio > 0.1,
                  vwap_dev > 0.15, bull])
    thresh = 3 if hurst > 0.5 else 4
    if sell_v >= thresh and sell_v > buy_v: return -1
    if buy_v  >= thresh and buy_v  > sell_v: return 1
    return 0


def forward_pnl(close, t, direction, atr, n_bars=40):
    """Simulate TP/SL outcome over next n_bars bars."""
    entry  = close[t]
    tp     = atr * 1.5
    sl     = atr * 0.75
    for i in range(1, min(n_bars, len(close)-t)):
        move = (close[t+i] - entry) * direction
        if move >= tp:
            return  move - COST_USD, True   # TP hit
        if move <= -sl:
            return -sl   - COST_USD, False  # SL hit
    final = (close[min(t+n_bars, len(close)-1)] - entry) * direction
    return final - COST_USD, final > 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', default='data/training_ready_v3b.npz')
    p.add_argument('--out', default='data/pretrain_buffer.npz')
    p.add_argument('--n',   type=int, default=N_SYNTHETIC)
    args = p.parse_args()

    logger.info(f"Loading {args.npz}")
    d     = np.load(args.npz, allow_pickle=False)
    feats = d['features'].astype(np.float32)   # (N, 12)
    close = d['close'].astype(np.float64)
    high  = d.get('high', close).astype(np.float64)
    low   = d.get('low',  close).astype(np.float64)
    gmm2  = d.get('gmm2', np.full(len(close), 0.5, np.float32))
    vol_e = d.get('vol_enc', np.full(len(close), 0.5, np.float32))
    N     = len(close)

    logger.info("Computing ATR and Hurst...")
    atr      = wilder_atr(close, high, low, ATR_PERIOD).astype(np.float32)
    log_ret  = np.concatenate([[0.0], np.log(close[1:]/close[:-1])])
    hurst    = hurst_dfa_vec(log_ret.astype(np.float32), 120).astype(np.float32)

    # Indices with valid signal window
    valid = np.where(
        (np.arange(N) > 250) &
        (np.arange(N) < N - FORWARD_BARS - 1) &
        (atr > 0.1)
    )[0]

    rng = np.random.default_rng(42)
    chosen = rng.choice(valid, size=min(args.n * 3, len(valid)), replace=False)

    obs_buf   = np.zeros((args.n, OBS_DIM), np.float32)
    act_buf   = np.zeros((args.n, 2),       np.float32)
    rew_buf   = np.zeros(args.n,             np.float32)
    nobs_buf  = np.zeros((args.n, OBS_DIM), np.float32)
    done_buf  = np.zeros(args.n,             np.float32)

    stored = 0
    archetypes = ['A','A','A','A','A','A', 'B','B','B', 'C']

    for t in chosen:
        if stored >= args.n: break
        if t < 240: continue

        f = feats[t]
        # Build simplified 118D obs (scattering approximated by raw features × 8)
        # In production the Rust server computes real scatter; this is a bootstrap
        scatter_approx = np.tile(f[:8], 13)[:104].astype(np.float32)
        atr_ratio = float(np.tanh((close[t]-close[t-1]) / max(atr[t],1e-8)))
        vwap_dev  = float(np.tanh((close[t]-close[t-60:t].mean()) / max(atr[t],1e-8)))
        rsi_raw   = float(f[0]) if f.shape[0] > 0 else 0.5
        bb_pctb   = float(np.clip(f[1] if f.shape[0] > 1 else 0.5, 0, 1))
        vol_z     = float(np.tanh(f[4] if f.shape[0] > 4 else 0))
        h         = float(hurst[t])
        tda_w     = 0.1   # placeholder; Rust server computes real TDA
        regime    = float(gmm2[t])
        sess      = float(f[10] if f.shape[0] > 10 else 0.5)
        ev_risk   = 0.0   # no calendar data in NPZ

        obs_vec = np.concatenate([
            scatter_approx,
            [atr_ratio, vwap_dev, rsi_raw, bb_pctb, vol_z,
             h, tda_w, sess, ev_risk, regime,
             0.0, 0.0, 0.0, 0.5]   # pos_dir=0, unreal=0, hold=0, conf=0.5
        ]).astype(np.float32)

        direction = rule_signal(atr_ratio, vwap_dev, rsi_raw, bb_pctb,
                                vol_z, h, tda_w, regime, ev_risk)
        if direction == 0: continue

        strength  = 0.6   # average signal strength approximation
        tp_pips   = atr[t] * 1.5 * 10
        sl_pips   = atr[t] * 0.75 * 10
        lot       = round(max(0.01 * strength * 0.15, 0.01), 2)

        arch = archetypes[stored % len(archetypes)]

        if arch == 'A':
            pnl, _ = forward_pnl(close, t, direction, atr[t])
            agreement = 1.0
            action_dir = direction

        elif arch == 'B':
            if ev_risk > 0.5 or h < 0.5:
                pnl = 0.0
                agreement = 0.5
                action_dir = 0
            else:
                pnl, _ = forward_pnl(close, t, direction, atr[t])
                agreement = 1.0
                action_dir = direction

        else:   # C — contrarian
            contra = -direction
            pnl, won = forward_pnl(close, t, contra, atr[t])
            agreement = 0.0 if won else -0.3
            action_dir = contra

        reward = float(pnl) * agreement
        done   = 1.0 if abs(pnl) > 0 else 0.0

        obs_buf[stored]  = obs_vec
        act_buf[stored]  = [float(action_dir), 0.2 if action_dir != 0 else -0.2]
        rew_buf[stored]  = reward
        nobs_buf[stored] = obs_vec   # simplified: next obs = current (offline)
        done_buf[stored] = done
        stored += 1

    obs_buf   = obs_buf[:stored]
    act_buf   = act_buf[:stored]
    rew_buf   = rew_buf[:stored]
    nobs_buf  = nobs_buf[:stored]
    done_buf  = done_buf[:stored]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out,
        obs=obs_buf, actions=act_buf, rewards=rew_buf,
        next_obs=nobs_buf, dones=done_buf)

    logger.info(f"Saved {stored:,} synthetic experiences → {args.out}")
    logger.info(f"  sell actions: {(act_buf[:,0]<0).sum():,}  "
                f"buy: {(act_buf[:,0]>0).sum():,}  "
                f"hold: {(act_buf[:,0]==0).sum():,}")
    logger.info(f"  reward: mean={rew_buf.mean():.4f} std={rew_buf.std():.4f}")


if __name__ == '__main__':
    main()
