#!/usr/bin/env python
"""Print trainable parameter counts for every model in the paper's tables, by
instantiating each one directly (manuscript: "computed directly by
instantiating the model rather than estimated from architecture
descriptions"). No checkpoint, data, or CT operator (torch-radon) is needed
-- this only builds the nn.Module graphs.

Reference values from the manuscript (Sec. "Implementation details",
Table: computational cost and throughput):
  PCDiff (full, base_channels=128):        ~118.93 M
  PCDiff-Small (base_channels=48):          ~17.15 M
  CoreDiff:                                 ~16.89 M
  RED-CNN / DU-GAN / Hformer / ASCON:  1.85 / 31.04 / 2.29 / 1.91 M
  DPS prior: same backbone as PCDiff (in_channels=1 instead of 2), so its
  count should be within a few thousand parameters of the "proposed" row.
"""
from __future__ import annotations

import argparse

import torch.nn as nn

from src.models.baselines.ascon import ASCONDenoiser
from src.models.baselines.corediff import CoreDiffModel
from src.models.baselines.dugan import DUGANGenerator
from src.models.baselines.hformer import Hformer
from src.models.baselines.redcnn import REDCNN
from src.models.unet import UNetConfig, UNetModel


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_size", type=int, default=256, help="Only affects models whose parameter count depends on resolution (none currently do, but kept for parity with train.py's UNetModel(..., image_size=...) call).")
    ap.add_argument("--base_channels", type=int, default=128, help="PCDiff base width; use 48 to report PCDiff-Small's count.")
    args = ap.parse_args()

    proposed_cfg = UNetConfig(
        in_channels=2, out_channels=1, base_channels=args.base_channels,
        channel_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2, attention_resolutions=(16, 8), num_heads=4, dropout=0.0,
    )
    proposed = UNetModel(proposed_cfg, image_size=args.image_size)

    dps_prior_cfg = UNetConfig(
        in_channels=1, out_channels=1, base_channels=args.base_channels,
        channel_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2, attention_resolutions=(16, 8), num_heads=4, dropout=0.0,
    )
    dps_prior = UNetModel(dps_prior_cfg, image_size=args.image_size)

    rows = [
        (f"proposed (PCDiff, base_channels={args.base_channels})", count_params(proposed)),
        (f"dps_prior (unconditional, base_channels={args.base_channels})", count_params(dps_prior)),
        ("corediff", count_params(CoreDiffModel(image_size=args.image_size))),
        ("redcnn", count_params(REDCNN())),
        ("dugan (generator only)", count_params(DUGANGenerator())),
        ("hformer", count_params(Hformer())),
        ("ascon", count_params(ASCONDenoiser())),
    ]

    name_w = max(len(n) for n, _ in rows)
    for name, n in rows:
        print(f"{name.ljust(name_w)}  {n:>12,d} params  ({n / 1e6:.2f} M)")


if __name__ == "__main__":
    main()
