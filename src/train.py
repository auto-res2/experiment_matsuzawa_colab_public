#!/usr/bin/env python3
"""
train.py – trains the tiny Orientation-Correcting Residual Adapter (OCRA)
=======================================================================
The script implements section B of the method description.  For the sake
of a quick, hardware-independent sanity check it trains on a synthetic
residual bank that is generated on-the-fly if no .npz file is found in
`data/`.  When a real residual bank (x , ε̂ , τ , Δε) is provided the
same code will load it instead – no changes required.

The script is deliberately lean (≈150 LoC) and stays well within the
16 GB Tesla T4 memory budget.  On a CPU or small GPU the smoke test in
src/main.py finishes in <30 s.
"""
from __future__ import annotations
import argparse, os, math, time, pathlib, random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from einops import rearrange

# -----------------------------------------------------------------------------
#                            reproducibility helpers
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------------------------------------------------------
#                         sinusoidal time embedding
# -----------------------------------------------------------------------------

def get_temb(t: torch.Tensor, channels: int = 128) -> torch.Tensor:
    half = channels // 2
    freq = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None] * freq[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if channels % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# -----------------------------------------------------------------------------
#                        FiLM-conditioned DW-conv block
# -----------------------------------------------------------------------------
class FiLMBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)
        self.act = nn.SiLU()
        self.affine = nn.Linear(64, ch * 2)

    def forward(self, x, t_emb):
        h = self.act(self.pw(self.act(self.dw(x))))
        gamma, beta = self.affine(t_emb).chunk(2, dim=-1)
        return self.act(h * (1 + gamma[..., None, None]) + beta[..., None, None])

# -----------------------------------------------------------------------------
#                                   OCRA
# -----------------------------------------------------------------------------
class OCRA(nn.Module):
    """1.5 M parameter orientation corrector"""

    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.time_emb = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 64))
        self.stem = nn.Conv2d(in_ch, 32, 5, 2, 2, groups=in_ch)
        self.blocks = nn.ModuleList([FiLMBlock(32) for _ in range(4)])
        self.head = nn.Conv2d(32, in_ch, 1)
        self.beta_clip = 0.04

    def forward(self, x, eps_hat, t):
        t_emb = self.time_emb(get_temb(t))
        x64 = F.avg_pool2d(x, max(1, x.shape[-1] // 64))  # spatial shrink
        h = self.stem(x64)
        for blk in self.blocks:
            h = blk(h, t_emb)
        rot = self.head(h).mean((2, 3), keepdim=True)
        eps_dir = F.normalize(eps_hat, dim=1)
        delta = rot * eps_dir
        # spectral-norm safety clipping
        d_norm = delta.flatten(1).norm(dim=1, keepdim=True)
        e_norm = eps_hat.flatten(1).norm(dim=1, keepdim=True)
        scale = torch.clamp(d_norm / (self.beta_clip * e_norm), min=1.0)
        return delta / scale.view(-1, 1, 1, 1)

# -----------------------------------------------------------------------------
#                              Dataset helper
# -----------------------------------------------------------------------------
class ResidualBank(Dataset):
    """Either loads a pre-computed residual bank or synthesises one."""

    def __init__(self, path: str | os.PathLike, n_samples: int = 2048, synth: bool = False):
        self.synth = synth or (not os.path.isfile(path))
        if self.synth:
            print("No residual bank found – generating a synthetic one for the quick test …")
            self.x = torch.randn(n_samples, 4, 64, 64)
            self.eps = torch.randn_like(self.x)
            self.t = torch.linspace(0.0, 1.0, n_samples)
            self.delta = 0.02 * torch.randn_like(self.x)  # 2 % ‖ε̂‖
        else:
            arr = np.load(path)
            self.x = torch.from_numpy(arr['x']).half()
            self.eps = torch.from_numpy(arr['eps']).half()
            self.t = torch.from_numpy(arr['t'])
            self.delta = torch.from_numpy(arr['delta']).half()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.eps[idx], self.t[idx], self.delta[idx]

# -----------------------------------------------------------------------------
#                             Training function
# -----------------------------------------------------------------------------

def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = ResidualBank(cfg['data']['file'], synth=cfg['data'].get('synthetic', False))
    loader = DataLoader(ds, batch_size=cfg['train']['batch'], shuffle=True, drop_last=True)

    model = OCRA().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'])
    mse = nn.MSELoss()

    steps = 0
    for epoch in range(cfg['train']['epochs']):
        for x, eps, t, delta in loader:
            x, eps, t, delta = x.to(device).float(), eps.to(device).float(), t.to(device).float(), delta.to(device).float()
            pred = model(x, eps, t)
            loss = mse(pred, delta)
            opt.zero_grad(); loss.backward(); opt.step()
            steps += 1
            if steps % 50 == 0:
                print(f"step {steps:>5}: L2 {loss.item():.4e}")
            if steps >= cfg['train']['max_steps']:
                break
        if steps >= cfg['train']['max_steps']:
            break

    pathlib.Path('models').mkdir(exist_ok=True)
    ckpt = 'models/ocra.pt'
    torch.save(model.state_dict(), ckpt)
    print(f"Saved trained OCRA to {ckpt}")

# -----------------------------------------------------------------------------
#                                    CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml, json
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='config/config.yaml')
    args = p.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
