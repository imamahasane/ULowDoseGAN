#!/usr/bin/env python
"""Runs the four loss-ablation variants (Table 2: full / w/o physics / w/o
perceptual / w/o denoising) through ONE shared harness with explicit config
overrides, instead of separately-run, ad hoc configs.

This exists specifically to close off the bug class that most plausibly
caused both the Table 1/2 mismatch (D-003) and the perceptual-loss ambiguity
(D-004): four "variants" trained via four manually-edited config files with
no enforced record of which config produced which reported row. Here, the
base config is loaded once and each variant is an explicit, logged override
of it -- the resolved config for every run is saved next to its checkpoint
(scripts/train.py already does this via config_resolved.yaml), so which
variant used which settings is no longer reconstructed after the fact.
"wo_denoising" scales L_diff by loss.denoising_weight=0.0, a config knob
scripts/train.py's _train_proposed actually reads (not a no-op override).

NOTE: this trains 4 separate models (each ablation is a genuinely different
training run) -- there's no way to avoid that compute cost. What this script
removes is the *ambiguity* about what was run, not the training time itself.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

VARIANTS = {
    "full": {},
    "wo_physics": {"loss": {"physics_weight": 0.0}},
    "wo_perceptual": {"loss": {"perceptual": {"enabled": False}}},
    # scripts/train.py now reads loss.denoising_weight (default 1.0) and
    # scales L_diff by it, so this is a genuine code path (the eps-prediction
    # loss term is dropped from the optimized objective), not a no-op override.
    "wo_denoising": {"loss": {"denoising_weight": 0.0}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--variants", type=str, nargs="+", default=list(VARIANTS.keys()))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = OmegaConf.load(args.base_config)

    for name in args.variants:
        if name not in VARIANTS:
            print(f"SKIP {name}: not one of the defined variants ({', '.join(VARIANTS)}).")
            continue

        variant_cfg = OmegaConf.merge(base_cfg, VARIANTS[name])
        variant_cfg.experiment.name = f"{base_cfg.experiment.name}_ablation_{name}"
        variant_path = out_dir / f"config_{name}.yaml"
        OmegaConf.save(variant_cfg, str(variant_path))

        print(f"=== Training ablation variant: {name} ===")
        subprocess.run(
            [sys.executable, "scripts/train.py", "--config", str(variant_path), "--model", "proposed"],
            check=True,
        )


if __name__ == "__main__":
    main()
