import sys
sys.path.append('.')

import os
import random
from loguru import logger

import torch
import torch.utils.data as data
import numpy as np
import open3d as o3d


class CoveredDataset(data.Dataset):
    """
    CoveredDataset dataset in "PCN: Point Completion Network". It contains 28974 training
    samples while each complete samples corresponds to 8 viewpoint partial scans, 800
    validation samples and 1200 testing samples.
    """
    
    def __init__(self, dataroot, split, category):
        assert split in ['train', 'valid', 'test'], "split error value!"

        self.dataroot = dataroot
        self.split = split
        self.category = category

        self.data_paths = self._load_data()
        logger.info("Load {} {} samples".format(len(self.data_paths), self.split))
    
    def __getitem__(self, index):
        complete_path = self.data_paths[index]

        complete_pc = self.random_sample(self.read_point_cloud(complete_path), 16384)
        partial_pc = self.random_sample(self.perturbation(complete_pc), 2048)

        return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc)

    def __len__(self):
        return len(self.data_paths)

    def _load_data(self):
        with open(os.path.join(self.dataroot, '{}.list').format(self.split), 'r') as f:
            lines = f.read().splitlines()
        data_paths = [os.path.join(self.dataroot, self.category, line) for line in lines]
        return data_paths
    
    def read_point_cloud(self, path):
        pc = o3d.io.read_point_cloud(path)
        return np.array(pc.points, np.float32)
    
    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
        return pc[idx[:n]]
    
    def perturbation(self, pc, sigma=0.01):
        idx = np.random.permutation(pc.shape[0])[:int(sigma*pc.shape[0])]

        mask = np.ones(pc.shape[0], dtype=bool)
        mask[idx] = False

        partial_pc = pc[mask]

        return partial_pc

