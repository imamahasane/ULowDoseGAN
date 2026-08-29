#!/usr/bin/env python
"""Paired significance testing across two evaluate.py result files
(EXPERIMENTS.md E-003/E-004). Requires both files to have been produced
against the SAME test-set ordering (same --config, --split, --max_items) so
per-item scores are genuinely paired -- this is checked, not assumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.metrics.stats import bootstrap_ci, bootstrap_ci_diff, wilcoxon_signed_rank
from src.utils.io import save_json


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposed", type=str, required=True, help="evaluate.py output JSON for the proposed method")
    ap.add_argument("--baseline", type=str, required=True, help="evaluate.py output JSON for the comparison baseline (e.g. CoreDiff)")
    ap.add_argument("--metric", type=str, default="psnr", choices=["psnr", "ssim", "sinogram_l2", "sinogram_rmse", "sinogram_relative"])
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    a = _load(args.proposed)
    b = _load(args.baseline)

    n_a, n_b = len(a["per_item"]), len(b["per_item"])
    if n_a != n_b:
        raise ValueError(
            f"Item-count mismatch ({n_a} vs {n_b}) -- these two result files were not evaluated over "
            "the same test-set subset and cannot be paired. Re-run evaluate.py with matching --max_items/--split."
        )
    if a["provenance"]["config"] != b["provenance"]["config"] or a["provenance"]["split"] != b["provenance"]["split"]:
        raise ValueError(
            "Config/split mismatch between the two result files -- refusing to pair scores that may not "
            f"correspond to the same items. proposed={a['provenance']['config']}/{a['provenance']['split']}, "
            f"baseline={b['provenance']['config']}/{b['provenance']['split']}"
        )

    scores_a = [r[args.metric] for r in sorted(a["per_item"], key=lambda r: r["idx"])]
    scores_b = [r[args.metric] for r in sorted(b["per_item"], key=lambda r: r["idx"])]

    wilcoxon = wilcoxon_signed_rank(scores_a, scores_b)
    ci_a = bootstrap_ci(scores_a, n_boot=args.n_boot)
    ci_b = bootstrap_ci(scores_b, n_boot=args.n_boot)
    ci_diff = bootstrap_ci_diff(scores_a, scores_b, n_boot=args.n_boot)

    results = {
        "metric": args.metric,
        "n_paired_items": n_a,
        "proposed_model": a["provenance"]["model"],
        "baseline_model": b["provenance"]["model"],
        "wilcoxon_signed_rank": vars(wilcoxon),
        "bootstrap_ci_proposed": vars(ci_a),
        "bootstrap_ci_baseline": vars(ci_b),
        "bootstrap_ci_paired_diff": vars(ci_diff),
    }
    save_json(results, args.out_json)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
