import sys
sys.path.append('.')

import os
import random
from loguru import logger

import torch
import torch.utils.data as data
import numpy as np
import open3d as o3d


def get_subfolder_names(root_directory):
    subfolders = []
    for entry in os.scandir(root_directory):
        if entry.is_dir():
            subfolders.append(entry.name)
    return subfolders

class SimulationDataset(data.Dataset):
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

        if category == 'all':
            self.category_names = get_subfolder_names(dataroot)

        self.num_categories = len(self.category_names)
        self.category_idx = {self.category_names[i]:i for i in range(len(self.category_names))}

        self.data_pairs_paths = self._load_data()
        logger.info("Load {} {} samples".format(len(self.data_pairs_paths)*self.num_categories, self.split))
    
    def __getitem__(self, index):
        try:
            complete_path, category_dix = self.data_pairs_paths[index]

            pcd, pcd_points = self.read_point_cloud(complete_path)

            complete_pc = self.random_sample(pcd_points, 3000)

            partial_pcd = self.perturbation(pcd)
            partial_pc = self.random_sample(np.array(partial_pcd.points, np.float32), 1000)
            encoded = torch.zeros(self.num_categories, dtype=torch.long)
            encoded[category_dix] = 1

            return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc), encoded
        
        except Exception as e:
            logger.error(self.data_pairs_paths[index])
            return self.__getitem__(index+1)

    def __len__(self):
        return len(self.data_pairs_paths)

    def _load_data(self):
        with open(os.path.join(self.dataroot, '{}.list').format(self.split), 'r') as f:
            lines = f.read().splitlines()
        for category_name in self.category_names:
            data_pairs_paths = [(os.path.join(self.dataroot, category_name, line), self.category_idx[category_name]) for line in lines]
        return data_pairs_paths
    
    def read_point_cloud(self, path):
        pcd = o3d.io.read_point_cloud(path)
        return pcd, np.array(pcd.points, np.float32)
    
    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
        return pc[idx[:n]]
    
    def perturbation(self, pc, ratio=0.01):
        if random.random() < 0.5:
            max_points = len(pc.points)
            k = random.randint(int(max_points * ratio/4), int(max_points * ratio))

            return self.random_radius_mask(pc, random.randint(0, max_points-1), k)
        else:
            
            return self.random_rect_mask(pc, random.uniform(0.4, 0.6), random.uniform(0.4, 0.6))


    def random_radius_mask(self, pcd, point_index, k):
        points = np.asarray(pcd.points)

        # Build KD tree for KNN search
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        # Find the indices of the K-nearest neighbors for the specified point index
        k_neighbors_indices = pcd_tree.search_knn_vector_3d(points[point_index], k)[1]

        # Create a mask to filter out points based on KNN indices
        mask = np.isin(np.arange(len(points)), k_neighbors_indices)

        # Remove the masked points from the point cloud
        pcd = pcd.select_by_index(np.where(~mask)[0])
        return pcd
        
    def random_rect_mask(self, pcd, scale_factor_x, scale_factor_y):
        points = np.asarray(pcd.points)

        # Generate random rectangle coordinates
        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])

        x_range = x_max - x_min
        y_range = y_max - y_min

        x1 = random.uniform(x_min, x_max - x_range * scale_factor_x)
        y1 = random.uniform(y_min, y_max - y_range * scale_factor_y)
        x2 = random.uniform(x1 + x_range * scale_factor_x, x_max)
        y2 = random.uniform(y1 + y_range * scale_factor_y, y_max)
        

        # Create a mask for points inside the rectangle
        mask = (points[:, 0] >= x1) & (points[:, 0] <= x2) & (points[:, 1] >= y1) & (points[:, 1] <= y2)

        # Apply the mask to the point cloud
        masked_points = points[mask]

        # Create a new point cloud with masked points
        masked_cloud = o3d.geometry.PointCloud()
        masked_cloud.points = o3d.utility.Vector3dVector(masked_points) 

        return masked_cloud
