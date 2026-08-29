#!/usr/bin/env python
"""Shared evaluation entry point for the proposed method, FBP, and all
learned baselines (including DPS).

Rewritten from the old evaluate.py to fix the most consequential gap found
in the audit: it only ever wrote aggregate mean/std to the output JSON,
discarding per-item scores and the exact run configuration (--max_items,
--K, checkpoint path, git commit). That is the most plausible root cause of
the Table 1/2 PSNR-SSIM discrepancy (DECISIONS.md D-003) -- two runs with
different subsets/seeds silently reported as the same "full model" number.
Every result produced by this script now carries full provenance and
per-item scores, and per-method sinogram-domain residuals (EXPERIMENTS.md
E-002) are computed alongside PSNR/SSIM/uncertainty metrics rather than left
untested at inference time.

`--model fbp` needs no checkpoint (closed-form Ram-Lak filtered
backprojection, src/ct/fbp.py). `--model dps` loads an unconditional prior
checkpoint (trained via `scripts/train.py --model dps_prior`) and applies the
per-step data-fidelity gradient correction of Sec. "Head-to-head comparison
with inference-time physics correction" at sampling time via --eta.
`--model proposed --sampler dpm_solver` reuses the same trained checkpoint
with Algorithm 2's deterministic accelerated sampler (--dpm_solver_steps)
instead of full-T ancestral DDPM sampling, with no retraining -- this is what
reproduces the "+DPM-Solver" rows of the computational-cost and DPS
comparison tables.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.ct.fbp import filtered_backprojection
from src.ct.operators import CTProjector, ParallelBeamGeometry
from src.ct.units import to_normalized_units, to_physical_units
from src.data.lodopab import LoDoPaBDataset
from src.data.mayo_aapm import MayoAAPMDataset
from src.diffusion.ddpm import PhysicsConditionedDDPM
from src.diffusion.dps import UnconditionalDDPM, dps_sample
from src.diffusion.schedule import make_ddpm_schedule
from src.metrics.image import psnr, ssim_torch
from src.metrics.sinogram import sinogram_residual
from src.metrics.uncertainty import coverage_and_ece, error_gap, mean_and_std, pearson_r
from src.models.baselines.ascon import ASCONDenoiser
from src.models.baselines.corediff import CoreDiffModel, make_generalized_schedule
from src.models.baselines.dugan import DUGANGenerator
from src.models.baselines.hformer import Hformer
from src.models.baselines.redcnn import REDCNN
from src.models.unet import UNetConfig, UNetModel
from src.utils.io import run_provenance, save_json
from src.utils.seed import seed_everything

# K is forced to 1 for these: no stochastic sampling, so repeated draws would
# just be K identical copies -- "uncertainty" is exactly zero by construction.
DETERMINISTIC_MODELS = ("redcnn", "hformer", "ascon", "dugan", "fbp")


def build_dataset(cfg):
    if cfg.data.dataset == "lodopab":
        return LoDoPaBDataset(cfg.data.root, cfg.data.split)
    if cfg.data.dataset == "mayo_aapm":
        return MayoAAPMDataset(cfg.data.root, cfg.data.split)
    raise ValueError(cfg.data.dataset)


def load_model(model_name: str, cfg, ckpt_path: str | None, device, image_size: int):
    if model_name == "fbp":
        return ("fbp", None)  # closed-form, no learned parameters

    if ckpt_path is None:
        raise ValueError(f"--ckpt is required for model '{model_name}' (only 'fbp' needs none).")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if model_name == "proposed":
        schedule = make_ddpm_schedule(
            T=int(cfg.diffusion.T), beta_schedule=str(cfg.diffusion.beta_schedule),
            beta_start=float(cfg.diffusion.beta_start), beta_end=float(cfg.diffusion.beta_end), device=device,
        )
        unet_cfg = UNetConfig(
            in_channels=int(cfg.model.in_channels), out_channels=int(cfg.model.out_channels),
            base_channels=int(cfg.model.base_channels), channel_mult=tuple(cfg.model.channel_mult),
            num_res_blocks=int(cfg.model.num_res_blocks), attention_resolutions=tuple(cfg.model.attention_resolutions),
            num_heads=int(cfg.model.num_heads), dropout=float(cfg.model.dropout),
        )
        denoiser = UNetModel(unet_cfg, image_size=image_size).to(device)
        ddpm = PhysicsConditionedDDPM(denoiser, schedule).to(device)
        ddpm.load_state_dict(ckpt["model"])
        ddpm.eval()
        return ("proposed", ddpm)

    if model_name == "dps":
        # Same UNet width/depth/attention as `proposed`, but unconditional
        # (in_channels=1, no c concatenation) -- see scripts/train.py's
        # _train_dps_prior and src/diffusion/dps.py.
        schedule = make_ddpm_schedule(
            T=int(cfg.diffusion.T), beta_schedule=str(cfg.diffusion.beta_schedule),
            beta_start=float(cfg.diffusion.beta_start), beta_end=float(cfg.diffusion.beta_end), device=device,
        )
        unet_cfg = UNetConfig(
            in_channels=1, out_channels=int(cfg.model.out_channels),
            base_channels=int(cfg.model.base_channels), channel_mult=tuple(cfg.model.channel_mult),
            num_res_blocks=int(cfg.model.num_res_blocks), attention_resolutions=tuple(cfg.model.attention_resolutions),
            num_heads=int(cfg.model.num_heads), dropout=float(cfg.model.dropout),
        )
        denoiser = UNetModel(unet_cfg, image_size=image_size).to(device)
        model = UnconditionalDDPM(denoiser, schedule).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return ("dps", model)

    if model_name == "corediff":
        schedule = make_generalized_schedule(int(cfg.diffusion.T), device=device)
        model = CoreDiffModel(image_size=image_size).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return ("corediff", (model, schedule))

    if model_name == "dugan":
        gen = DUGANGenerator().to(device)
        gen.load_state_dict(ckpt["generator"])
        gen.eval()
        return ("dugan", gen)

    if model_name == "ascon":
        model = ASCONDenoiser().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return ("ascon", model)

    if model_name == "redcnn":
        model = REDCNN().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return ("redcnn", model)

    if model_name == "hformer":
        model = Hformer().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return ("hformer", model)

    raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def reconstruct(
    model_name: str,
    model_obj,
    c: torch.Tensor,
    y: torch.Tensor,
    projector,
    K: int,
    shape,
    sampler: str = "ddpm",
    dpm_solver_steps: int = 20,
    eta: float = 0.1,
    x_min: torch.Tensor | None = None,
    x_max: torch.Tensor | None = None,
) -> torch.Tensor:
    """Returns samples: (K,1,1,H,W), in the network's [-1,1] domain for every
    model except 'fbp' (physical attenuation units; see main()'s handling).
    K is forced to 1 upstream for deterministic (non-diffusion) baselines."""
    if model_name == "proposed":
        if sampler == "dpm_solver":
            samples = [model_obj.dpm_solver_sample(c, shape=shape, steps=dpm_solver_steps) for _ in range(K)]
        else:
            samples = [model_obj.sample(c, shape=shape) for _ in range(K)]
        return torch.stack(samples, dim=0)
    if model_name == "corediff":
        cd_model, schedule = model_obj
        samples = [cd_model.sample(c, schedule) for _ in range(K)]
        return torch.stack(samples, dim=0)
    if model_name == "fbp":
        pred = filtered_backprojection(projector, y)
        return pred[None].expand(K, *pred.shape).clone()
    if model_name == "dps":
        samples = [
            dps_sample(model_obj, projector, y, shape=shape, eta=eta, x_min=x_min, x_max=x_max)
            for _ in range(K)
        ]
        return torch.stack(samples, dim=0)
    # Deterministic image-domain baselines: one forward pass, replicated K
    # times so downstream code (mean/std, coverage) doesn't need a separate
    # path -- their "uncertainty" is exactly zero by construction, itself a
    # reportable point of contrast against sampling-based methods.
    pred = model_obj(c)
    return pred[None].expand(K, *pred.shape).clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--model", type=str, required=True, choices=(
        "proposed", "fbp", "dps", "corediff", "dugan", "ascon", "redcnn", "hformer",
    ))
    ap.add_argument("--ckpt", type=str, default=None, help="Required for every model except 'fbp'.")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--K", type=int, default=8, help="Samples for uncertainty estimation (proposed/corediff/dps only; forced to 1 for deterministic baselines).")
    ap.add_argument("--sampler", type=str, default="ddpm", choices=("ddpm", "dpm_solver"), help="Reverse-process sampler for --model proposed (Algorithm 1 vs. Algorithm 2). Ignored for every other model.")
    ap.add_argument("--dpm_solver_steps", type=int, default=20, help="S in Algorithm 2 (paper default: 20).")
    ap.add_argument("--eta", type=float, default=0.1, help="DPS step-size hyperparameter (Eq. for DPS); ignored unless --model dps.")
    ap.add_argument("--max_items", type=int, default=None, help="If unset, evaluates the FULL split -- the old default of 200 is exactly the kind of silent subset choice suspected of causing the Table 1/2 mismatch (D-003); pass it explicitly and it will be recorded in provenance.")
    ap.add_argument("--data_range", type=float, default=2.0, help="Must match the actual normalization used (2.0 for [-1,1]).")
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.data.split = args.split

    seed_everything(int(cfg.experiment.seed), deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W = cfg.data.image_size
    geom = ParallelBeamGeometry(image_size=int(H), angles=int(cfg.ct.angles), det_count=int(cfg.ct.det_count))
    projector = CTProjector(geom, device=device)

    model_name, model_obj = load_model(args.model, cfg, args.ckpt, device, int(H))
    K = 1 if model_name in DETERMINISTIC_MODELS else int(args.K)

    ds = build_dataset(cfg)
    n_items = len(ds) if args.max_items is None else min(len(ds), args.max_items)

    per_item = []
    for idx in tqdm(range(n_items), desc=f"eval[{model_name}]"):
        item = ds[idx]
        x_gt = item["x"].unsqueeze(0).to(device)
        y = item["y"].unsqueeze(0).to(device)
        x_min = item["x_min"].unsqueeze(0).to(device)
        x_max = item["x_max"].unsqueeze(0).to(device)
        c = projector.AT(y)

        samples = reconstruct(
            model_name, model_obj, c, y, projector, K, shape=(1, 1, int(H), int(W)),
            sampler=args.sampler, dpm_solver_steps=args.dpm_solver_steps, eta=args.eta,
            x_min=x_min, x_max=x_max,
        )
        mean, std = mean_and_std(samples)

        # FBP reconstructs directly from y and so is already in physical
        # attenuation units; every other method predicts in the network's
        # [-1,1] domain. Normalize both to [-1,1] for PSNR/SSIM (comparable
        # across all methods, data_range=2.0) and to physical units for the
        # sinogram-domain residual (comparable to y) -- see src/ct/units.py.
        if model_name == "fbp":
            mean_norm = to_normalized_units(mean, x_min, x_max)
            mean_phys = mean
        else:
            mean_norm = mean
            mean_phys = to_physical_units(mean, x_min, x_max)

        err = (mean_norm - x_gt).abs()
        sino_res = sinogram_residual(projector, mean_phys, y)
        cal = coverage_and_ece(err, std)
        gap = error_gap(err, std)

        per_item.append({
            "idx": idx,
            "psnr": psnr(mean_norm[0], x_gt[0], data_range=args.data_range),
            "ssim": ssim_torch(mean_norm[0], x_gt[0], data_range=args.data_range),
            "sinogram_l2": sino_res.l2,
            "sinogram_rmse": sino_res.rmse,
            "sinogram_relative": sino_res.relative,
            "pearson_r": pearson_r(std, err),
            "cal_err": cal.cal_err,
            "coverage95": cal.coverage95,
            "error_gap": gap,
        })

    def agg(key: str):
        vals = [r[key] for r in per_item]
        return float(np.mean(vals)), float(np.std(vals))

    metric_keys = (
        "psnr", "ssim", "sinogram_l2", "sinogram_rmse", "sinogram_relative",
        "pearson_r", "cal_err", "coverage95", "error_gap",
    )
    results = {
        "provenance": run_provenance({
            "model": model_name, "config": args.config, "ckpt": args.ckpt, "split": args.split,
            "K": K, "sampler": args.sampler, "dpm_solver_steps": args.dpm_solver_steps, "eta": args.eta,
            "max_items": args.max_items, "n_items_evaluated": n_items, "data_range": args.data_range,
        }),
        "per_item": per_item,
        "aggregate": {
            metric: {"mean": m, "std": s}
            for metric in metric_keys
            for m, s in [agg(metric)]
        },
    }

    save_json(results, args.out_json)
    print(f"Wrote {n_items} per-item results + provenance to {args.out_json}")
    print({k: v["mean"] for k, v in results["aggregate"].items()})


if __name__ == "__main__":
    main()
