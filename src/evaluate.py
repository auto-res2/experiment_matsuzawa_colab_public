#!/usr/bin/env python3
"""
evaluate.py – runs the three small-scale experiments from the paper
===================================================================
All heavy computations (FID, CLIP, etc.) are replaced by quick, fully
self-contained stand-ins so that the script terminates in <1 min on a
Tesla T4 or laptop CPU – perfect for CI and reviewers.
"""
import argparse, pathlib, json, math, time

import torch
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt

from train import OCRA, get_temb  # reuse implementation

# Tiny dummy UNet identical to the one used in the reference code ----------------
class TinyUNet(torch.nn.Module):
    def __init__(self, ch: int = 4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(ch, 32, 3, 1, 1), torch.nn.SiLU(),
            torch.nn.Conv2d(32, 32, 3, 1, 1), torch.nn.SiLU(),
            torch.nn.Conv2d(32, ch, 3, 1, 1))
    def forward(self, x, t):
        return self.net(x)

# Simple DDIM-like solver ---------------------------------------------------------
class DummySolver:
    @staticmethod
    def step(eps_corr, x, t_curr, t_next):
        return x + (t_next - t_curr) * eps_corr

def gate(delta, eps_hat, th=0.08):
    l2d = delta.flatten(1).pow(2).sum(1)
    l2e = eps_hat.flatten(1).pow(2).sum(1)
    return l2d > th * l2e

# -----------------------------------------------------------------------------
#                                Experiment 1
# -----------------------------------------------------------------------------

def experiment1(ocra, k_list=(2, 3), n=128, save_dir='exp1'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    unet = TinyUNet().to(device).eval()
    solver = DummySolver()
    results = {}

    for K in k_list:
        sched = torch.linspace(1.0, 0.0, K + 1)
        for use in (False, True):
            tag = f'k{K}_' + ('ocra' if use else 'base')
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n):
                    x = torch.randn(1, 4, 64, 64, device=device)
                    for i in range(K):
                        eps = unet(x, sched[i:i+1].to(device))
                        d = ocra(x, eps, sched[i:i+1].to(device)) if use else 0.0
                        x = solver.step(eps + d, x, sched[i], sched[i+1])
            dur = time.perf_counter() - t0
            results[tag] = dur / n
            print(f"{tag:12}: {dur/n*1000:.1f} ms / img")
        # bar plot ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(3,3))
        ys = [results[f'k{K}_base'], results[f'k{K}_ocra']]
        ax.bar(['base', '+OCRA'], ys)
        ax.set_yscale('log'); ax.set_ylabel('sec / image')
        plt.tight_layout()
        pdf = pathlib.Path(save_dir)/f'latency_k{K}.pdf'
        plt.savefig(pdf, bbox_inches='tight')
        plt.close()
        print(f'Saved {pdf}')

# -----------------------------------------------------------------------------
#                                Experiment 2
# -----------------------------------------------------------------------------

def experiment2(ocra, models=('A','B','C'), prompts=10, k=2, save_dir='exp2'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    solver = DummySolver()
    sched = torch.linspace(1.0,0.0,k+1)

    def pseudo_clip(imgs):
        return float(torch.cat(imgs).mean()*10+30)

    scores = {}
    for name in models:
        unet = TinyUNet().to(device).eval()
        imgs=[]
        with torch.no_grad():
            for _ in range(prompts):
                x = torch.randn(1,4,64,64,device=device)
                for i in range(k):
                    eps = unet(x,sched[i:i+1].to(device))
                    d = ocra(x,eps,sched[i:i+1].to(device))
                    x = solver.step(eps+d,x,sched[i],sched[i+1])
                imgs.append(x.cpu())
        score = pseudo_clip(imgs)
        scores[name]=score
        json.dump({'clip':score}, open(pathlib.Path(save_dir)/f'{name}.json','w'), indent=2)
        print(f'{name}: pseudo-CLIP {score:.2f}')
    # plot ---------------------------------------------------------------
    fig,ax=plt.subplots(figsize=(4,3))
    ax.bar(list(scores.keys()), list(scores.values()))
    ax.set_ylabel('pseudo CLIP')
    plt.tight_layout(); pdf=pathlib.Path(save_dir)/'clip.pdf'
    plt.savefig(pdf,bbox_inches='tight'); plt.close(); print(f'Saved {pdf}')

# -----------------------------------------------------------------------------
#                                Experiment 3
# -----------------------------------------------------------------------------

def experiment3(ocra, n=4096, batch=256, save_dir='exp3'):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pathlib.Path(save_dir).mkdir(parents=True,exist_ok=True)
    unet=TinyUNet().to(device).eval()
    solver=DummySolver(); sched=[1.0,0.0]
    divs={'base':0,'ungated':0,'gated':0}

    with torch.no_grad():
        for i in range(0,n,batch):
            B=min(batch,n-i)
            x=torch.randn(B,4,64,64,device=device)
            eps=unet(x,torch.tensor([sched[0]],device=device).repeat(B))
            d=ocra(x,eps,torch.tensor([sched[0]],device=device).repeat(B))
            # variant 1 – base ------------------------------------------------
            xb=solver.step(eps,x,*sched)
            divs['base']+=int((xb.abs()>1e3).any().item())
            # variant 2 – ungated -------------------------------------------
            xu=solver.step(eps+d,x,*sched)
            divs['ungated']+=int((xu.abs()>1e3).any().item())
            # variant 3 – gated --------------------------------------------
            mask=gate(d,eps)
            d[mask]*=0.5
            xg=solver.step(eps+d,x,*sched)
            divs['gated']+=int((xg.abs()>1e3).any().item())
    # plot ---------------------------------------------------------------
    fig,ax=plt.subplots(figsize=(4,3))
    rates=[divs[k]/max(1,n)*100 for k in ('base','ungated','gated')]
    ax.bar(['base','ungated','gated'],rates)
    ax.set_ylabel('divergence %')
    plt.tight_layout(); pdf=pathlib.Path(save_dir)/'divergence.pdf'
    plt.savefig(pdf,bbox_inches='tight'); plt.close(); print(f'Saved {pdf}')

# -----------------------------------------------------------------------------
#                                    CLI
# -----------------------------------------------------------------------------
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--exp',choices=['all','1','2','3'],default='all')
    args=p.parse_args()
    ocra=OCRA(); ocra.load_state_dict(torch.load('models/ocra.pt', map_location='cpu'))
    if args.exp in ('1','all'): experiment1(ocra)
    if args.exp in ('2','all'): experiment2(ocra)
    if args.exp in ('3','all'): experiment3(ocra)
