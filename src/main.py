#!/usr/bin/env python3
"""
main.py – orchestrates the full pipeline
=======================================
   1. Pre-processing (creates residual bank if none exists)
   2. Training      (≈1 k synthetic steps)
   3. Evaluation    (three toy experiments, PDF figures!)
All plots are placed in `.research/iteration1/images` so that they are
picked up by the paper build system.
"""
import argparse, pathlib, shutil, os

from preprocess import main as preprocess_main
from train import train as train_main
from evaluate import experiment1, experiment2, experiment3, OCRA

IMG_DIR = pathlib.Path('.research/iteration1/images')


def ensure_dirs():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for d in ('data','models'):
        pathlib.Path(d).mkdir(exist_ok=True)


def copy_pdf(src_dir):
    for pdf in pathlib.Path(src_dir).glob('*.pdf'):
        shutil.copy(pdf, IMG_DIR/pdf.name)


def pipeline(cfg):
    ensure_dirs()
    # ------------------------------------------------------------------
    # 1) Pre-process synthetic residuals
    # ------------------------------------------------------------------
    preprocess_main(out='data/residual_bank.npz', n=cfg['data']['n'])
    # ------------------------------------------------------------------
    # 2) Train OCRA
    # ------------------------------------------------------------------
    train_main(cfg)
    # ------------------------------------------------------------------
    # 3) Evaluate – run the miniature experiments
    # ------------------------------------------------------------------
    ocra = OCRA(); ocra.load_state_dict(os.path.join('models','ocra.pt'))  # will be re-loaded by evaluate
    experiment1(ocra, n=64, k_list=[2], save_dir='exp1')
    experiment2(ocra, models=('M1','M2'), prompts=5, k=2, save_dir='exp2')
    experiment3(ocra, n=1024, batch=256, save_dir='exp3')
    # Collect PDFs -------------------------------------------------------
    for d in ('exp1','exp2','exp3'):
        copy_pdf(d)
    print(f'All figures copied to {IMG_DIR}')


if __name__=='__main__':
    import yaml, json
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',default='config/config.yaml')
    a=p.parse_args()
    cfg=yaml.safe_load(open(a.cfg))
    pipeline(cfg)
