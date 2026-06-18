#!/usr/bin/env python3
"""
DSAC pre-training on synthetic experience buffer.

Loads pretrain_buffer.npz, runs offline DSAC gradient updates,
saves dsac_pretrained.pt ready for deployment in MT5.

Usage:
    python train_dsac.py --buffer data/pretrain_buffer.npz --out models/dsac_pretrained.pt
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

OBS_DIM  = 118
ACT_DIM  = 2
N_QUANT  = 32      # IQN quantiles for distributional critics
HIDDEN   = [512, 256]


# ── DSAC components ───────────────────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        layers, d = [], OBS_DIM
        for h in HIDDEN:
            layers += [nn.Linear(d, h), nn.ReLU()]; d = h
        layers += [nn.Linear(d, ACT_DIM), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, obs): return self.net(obs)


class QuantileCritic(nn.Module):
    """Implicit Quantile Network critic."""
    def __init__(self):
        super().__init__()
        layers, d = [], OBS_DIM + ACT_DIM
        for h in HIDDEN:
            layers += [nn.Linear(d, h), nn.ReLU()]; d = h
        self.trunk = nn.Sequential(*layers)
        self.quant_embed = nn.Linear(64, d)
        self.head        = nn.Linear(d, 1)

    def forward(self, obs, act, taus):
        """taus: (B, N_QUANT) uniform samples in (0,1)."""
        B  = obs.shape[0]
        sa = torch.cat([obs, act], dim=-1)
        h  = self.trunk(sa)                              # (B, d)
        # Quantile embedding
        i  = torch.arange(1, 65, device=obs.device).float()
        cos = torch.cos(taus.unsqueeze(-1) * i * 3.14159)  # (B, N, 64)
        phi = F.relu(self.quant_embed(cos))              # (B, N, d)
        h   = h.unsqueeze(1) * phi                       # (B, N, d)
        return self.head(h).squeeze(-1)                  # (B, N)


class DSACAgent:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.actor  = Actor().to(self.device)
        self.q1     = QuantileCritic().to(self.device)
        self.q2     = QuantileCritic().to(self.device)
        self.q1_t   = QuantileCritic().to(self.device)
        self.q2_t   = QuantileCritic().to(self.device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        lr = 3e-4
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.gamma = 0.99
        self.tau   = 0.005

    def update(self, obs, act, rew, nobs, done, batch_size=256):
        B = batch_size
        idx   = torch.randint(0, obs.shape[0], (B,))
        o     = obs[idx].to(self.device)
        a     = act[idx].to(self.device)
        r     = rew[idx].to(self.device).unsqueeze(1)
        no    = nobs[idx].to(self.device)
        d     = done[idx].to(self.device).unsqueeze(1)

        taus  = torch.rand(B, N_QUANT, device=self.device)
        taus_ = torch.rand(B, N_QUANT, device=self.device)

        with torch.no_grad():
            na   = self.actor(no)
            q1_t = self.q1_t(no, na, taus_)   # (B, N)
            q2_t = self.q2_t(no, na, taus_)
            q_t  = torch.min(q1_t, q2_t)
            tgt  = r + (1-d) * self.gamma * q_t  # (B, N)

        # Quantile Huber loss
        q1   = self.q1(o, a, taus)
        q2   = self.q2(o, a, taus)
        ql1  = quantile_huber(q1, tgt, taus)
        ql2  = quantile_huber(q2, tgt, taus)
        ql   = ql1 + ql2
        self.q_opt.zero_grad(); ql.backward(); self.q_opt.step()

        # Actor loss
        al   = -self.q1(o, self.actor(o), taus).mean()
        self.a_opt.zero_grad(); al.backward(); self.a_opt.step()

        # Soft target update
        soft_update(self.q1, self.q1_t, self.tau)
        soft_update(self.q2, self.q2_t, self.tau)

        return float(ql.item()), float(al.item())

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'q1':    self.q1.state_dict(),
            'q2':    self.q2.state_dict(),
            'obs_dim': OBS_DIM, 'act_dim': ACT_DIM,
            'n_quant': N_QUANT, 'hidden': HIDDEN,
        }, path)
        logger.info(f"DSAC checkpoint saved: {path}")


def quantile_huber(pred, target, taus, kappa=1.0):
    """Quantile Huber loss for IQN."""
    diff = target.unsqueeze(2) - pred.unsqueeze(1)   # (B, N, N)
    abs_diff = diff.abs()
    huber = torch.where(abs_diff < kappa,
                        0.5 * diff.pow(2),
                        kappa * (abs_diff - 0.5*kappa))
    rho = (taus.unsqueeze(2) - (diff < 0).float()).abs()
    return (rho * huber).mean()


def soft_update(src, tgt, tau):
    for sp, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.copy_(tau * sp.data + (1-tau) * tp.data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--buffer',  default='data/pretrain_buffer.npz')
    p.add_argument('--out',     default='models/dsac_pretrained.pt')
    p.add_argument('--steps',   type=int, default=20_000)
    p.add_argument('--batch',   type=int, default=256)
    p.add_argument('--device',  default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    logger.info(f"Loading buffer: {args.buffer}")
    buf   = np.load(args.buffer)
    obs   = torch.FloatTensor(buf['obs'])
    act   = torch.FloatTensor(buf['actions'])
    rew   = torch.FloatTensor(buf['rewards'])
    nobs  = torch.FloatTensor(buf['next_obs'])
    done  = torch.FloatTensor(buf['dones'])
    logger.info(f"  {len(obs):,} experiences | device={args.device}")

    agent = DSACAgent(device=args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for step in range(1, args.steps+1):
        ql, al = agent.update(obs, act, rew, nobs, done, args.batch)
        if step % 1000 == 0:
            elapsed = time.time() - t0
            logger.info(f"Step {step:6d}/{args.steps} | "
                        f"q_loss={ql:.4f} a_loss={al:.4f} | "
                        f"{elapsed:.0f}s elapsed")

    agent.save(args.out)
    logger.info(f"Pre-training complete. Deploy: models/dsac_pretrained.pt")


if __name__ == '__main__':
    main()
