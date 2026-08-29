#!/usr/bin/env python
"""Draw and save reconstruction samples (+ mean/std) for qualitative
inspection. Model-agnostic: reuses the same load_model/reconstruct dispatch
as evaluate.py so a baseline's samples are produced under identical
data/conditioning assumptions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from evaluate import build_dataset, load_model, reconstruct
from src.ct.operators import CTProjector, ParallelBeamGeometry
from src.metrics.uncertainty import mean_and_std
from src.utils.io import ensure_dir
from src.utils.seed import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None, help="Required for every model except 'fbp'.")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--sampler", type=str, default="ddpm", choices=("ddpm", "dpm_solver"))
    ap.add_argument("--dpm_solver_steps", type=int, default=20)
    ap.add_argument("--eta", type=float, default=0.1, help="DPS step-size hyperparameter; ignored unless --model dps.")
    ap.add_argument("--out_dir", type=str, default="samples")
    ap.add_argument("--max_items", type=int, default=50)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.data.split = args.split

    seed_everything(int(cfg.experiment.seed), deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W = cfg.data.image_size
    geom = ParallelBeamGeometry(image_size=int(H), angles=int(cfg.ct.angles), det_count=int(cfg.ct.det_count))
    projector = CTProjector(geom, device=device)

    model_name, model_obj = load_model(args.model, cfg, args.ckpt, device, int(H))

    ds = build_dataset(cfg)
    out_dir = ensure_dir(Path(args.out_dir))

    for idx in tqdm(range(min(len(ds), int(args.max_items))), desc="sampling"):
        item = ds[idx]
        x_gt = item["x"].unsqueeze(0).to(device)
        y = item["y"].unsqueeze(0).to(device)
        x_min = item["x_min"].unsqueeze(0).to(device)
        x_max = item["x_max"].unsqueeze(0).to(device)
        c = projector.AT(y)

        samples = reconstruct(
            model_name, model_obj, c, y, projector, int(args.K), shape=(1, 1, int(H), int(W)),
            sampler=args.sampler, dpm_solver_steps=args.dpm_solver_steps, eta=args.eta,
            x_min=x_min, x_max=x_max,
        )
        mean, std = mean_and_std(samples)

        np.savez_compressed(
            out_dir / f"item_{idx:05d}.npz",
            x_gt=x_gt.detach().cpu().numpy(),
            y=y.detach().cpu().numpy(),
            samples=samples.detach().cpu().numpy(),
            mean=mean.detach().cpu().numpy(),
            std=std.detach().cpu().numpy(),
        )


if __name__ == "__main__":
    main()
