from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class MayoAAPMDataset(Dataset):
    
    def __init__(self, root: str | Path, split: str):
        super().__init__()
        self.root = Path(root) / split
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"No .npz files found in {self.root}. Run scripts/prepare_mayo_aapm.py first."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = np.load(self.files[idx])
        x = torch.from_numpy(d["x"]).float()  # (1,H,W), normalized to [-1,1]
        y = torch.from_numpy(d["y"]).float()  # (angles,det), physical attenuation units
        # Per-slice physical-attenuation range the [-1,1] image was normalized
        # from, saved by scripts/prepare_mayo_aapm.py; required to invert that
        # normalization for the physics-consistency loss (src/ct/units.py),
        # since y was forward-projected from the pre-normalization image.
        x_min = torch.tensor(float(d["x_min"]))
        x_max = torch.tensor(float(d["x_max"]))
        return {"x": x, "y": y, "x_min": x_min, "x_max": x_max}
