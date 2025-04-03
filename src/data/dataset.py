import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LoDoPaBDataset(Dataset):
    def __init__(self, obs_dir, gt_dir, num_files=None):
        self.obs_files = sorted(os.listdir(obs_dir))
        if num_files:
            self.obs_files = self.obs_files[:num_files]
        self.obs_dir = obs_dir
        self.gt_dir = gt_dir

        print("Loading dataset...")
        self.obs_data = []
        self.gt_data = []
        for file_name in self.obs_files:
            obs_file = os.path.join(obs_dir, file_name)
            gt_file = os.path.join(gt_dir, file_name.replace("observation", "ground_truth"))
            with h5py.File(obs_file, 'r') as f_obs, h5py.File(gt_file, 'r') as f_gt:
                obs = f_obs['data'][:].astype(np.float16)
                gt = f_gt['data'][:].astype(np.float16)
                self.obs_data.append(obs)
                self.gt_data.append(gt)
        
        self.obs_data = np.concatenate(self.obs_data, axis=0)
        self.gt_data = np.concatenate(self.gt_data, axis=0)
        
        self.obs_data = (self.obs_data - np.min(self.obs_data)) / (np.max(self.obs_data) - np.min(self.obs_data)) * 2 - 1
        self.gt_data = (self.gt_data - np.min(self.gt_data)) / (np.max(self.gt_data) - np.min(self.gt_data)) * 2 - 1
        print(f"Dataset loaded: {self.obs_data.shape[0]} samples")

    def __len__(self):
        return len(self.obs_data)

    def __getitem__(self, idx):
        obs = torch.FloatTensor(self.obs_data[idx]).unsqueeze(0)  # [1, 1000, 513]
        gt = torch.FloatTensor(self.gt_data[idx]).unsqueeze(0)    # [1, 362, 362]
        return obs, gt