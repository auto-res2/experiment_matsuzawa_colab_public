#!/usr/bin/env python3
"""
preprocess.py – creates a *synthetic* residual bank so that training can run
without any external data.  For a real study simply replace the synthetic
part with proper DDIM inversion.
"""
import argparse, pathlib, numpy as np, torch

def main(out='data/residual_bank.npz', n=2048):
    pathlib.Path('data').mkdir(exist_ok=True)
    x = torch.randn(n,4,64,64)
    eps = torch.randn_like(x)
    t = torch.linspace(0,1,n)
    delta = 0.02*torch.randn_like(x)
    np.savez(out, x=x.numpy().astype('float16'), eps=eps.numpy().astype('float16'),
             t=t.numpy().astype('float32'), delta=delta.numpy().astype('float16'))
    print(f'Synthetic residual bank saved to {out}')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='data/residual_bank.npz')
    p.add_argument('--n',type=int,default=2048)
    a=p.parse_args(); main(a.out,a.n)
