from __future__ import annotations

import glob
import warnings
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class LoDoPaBDataset(Dataset):
   
    def __init__(self, root: str | Path, split: str):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.files = sorted(glob.glob(str(self.root / f"{split}*.h5")))
        if not self.files:
            raise FileNotFoundError(
                f"No HDF5 files found for split='{split}' in {self.root}.\n"
                "Place LoDoPaB h5 files under data/lodopab/, e.g., train.h5 / validation.h5 / test.h5."
            )

        self.index = []  # list of (file_path, local_idx)
        for fp in self.files:
            with h5py.File(fp, "r") as f:
                images_key, sino_key = self._infer_keys(f)
                n = f[images_key].shape[0]
            for i in range(n):
                self.index.append((fp, i))

    def _infer_keys(self, f: h5py.File) -> Tuple[str, str]:
        candidates = [
            ("images", "sinograms"),
            ("image", "sinogram"),
            ("ground_truth", "observation"),
            ("x", "y"),
        ]
        for ik, sk in candidates:
            if ik in f and sk in f:
                return ik, sk
        for g in f.keys():
            if isinstance(f[g], h5py.Group):
                grp = f[g]
                for ik, sk in candidates:
                    if ik in grp and sk in grp:
                        return f"{g}/{ik}", f"{g}/{sk}"
        raise KeyError(f"Could not infer dataset keys in file. Keys: {list(f.keys())}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fp, j = self.index[idx]
        with h5py.File(fp, "r") as f:
            images_key, sino_key = self._infer_keys(f)
            x = f[images_key][j]  # (H,W), assumed in [0,1]
            y = f[sino_key][j]  # (angles,det)

        x_arr = np.asarray(x)
        if x_arr.min() < -0.05 or x_arr.max() > 1.05:
            warnings.warn(
                f"LoDoPaB image at index {idx} has range [{x_arr.min():.3f}, {x_arr.max():.3f}], "
                "outside the expected [0,1] -- the [-1,1] rescale below assumes [0,1] input and "
                "may be wrong for this file. Verify against the actual dataset release notes.",
                stacklevel=2,
            )

        x_t = torch.from_numpy(x_arr).float().unsqueeze(0)  # (1,H,W)
        x_t = x_t * 2.0 - 1.0
        y_t = torch.from_numpy(np.asarray(y)).float()  # (angles,det)
        # LoDoPaB-CT's own convention: ground-truth images and the released
        # sinograms both live on the fixed [0,1] attenuation scale (Leuschner
        # et al. 2021) -- so x_min/x_max for the physics-loss rescale
        # (src/ct/units.py) are dataset-wide constants, not per-slice.
        x_min = torch.tensor(0.0)
        x_max = torch.tensor(1.0)
        return {"x": x_t, "y": y_t, "x_min": x_min, "x_max": x_max}
