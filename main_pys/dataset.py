import os
import glob
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from functools import lru_cache
from main_pys.model_inputs import create_data_object, normalize_graph_data

# --- OOM CRASH FIX: Reduced maxsize from 32 to 2 ---
@lru_cache(maxsize=2)
def load_trajectory(path):
    with np.load(path) as data:
        return data['discrete_positions'].copy(), data['expert_velocities'].copy()

@lru_cache(maxsize=2)
def load_bd(path, key):
    with np.load(path) as data:
        return data[key].copy()

class FlowMAPFDataset(Dataset):
    def __init__(self, data_dir, map_dir, bd_dir, k=4, m=5):
        self.npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
        self.map_dir = map_dir
        self.bd_dir = bd_dir
        self.k = k
        self.m = m

        print("Preloading Maps...")
        self.maps = {}
        for map_path in glob.glob(os.path.join(map_dir, "*.map")):
            map_name = os.path.basename(map_path).replace(".map", "")
            self.maps[map_name] = self._read_map(map_path)
            
        print("Building flattened timestep index...")
        self.index = []
        
        random.shuffle(self.npz_files) 
        
        for f in self.npz_files:
            try:
                with np.load(f) as data:
                    T = data['discrete_positions'].shape[1]
                    for t in range(T):
                        self.index.append((f, t))
            except Exception as e:
                pass
                
        print(f"Total training samples: {len(self.index)}")

    def _read_map(self, map_file):
        with open(map_file, 'r') as f:
            f.readline()
            height = int(f.readline().split()[1])
            width = int(f.readline().split()[1])
            f.readline()
            map_data = np.zeros((height, width), dtype=int)
            for r in range(height):
                line = f.readline().strip()
                for c in range(width):
                    if line[c] in ['@', 'T', 'O']:
                        map_data[r, c] = 1
        return np.pad(map_data, self.k, 'constant', constant_values=1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        npz_path, t_step = self.index[idx]
        
        discrete_positions, expert_velocities = load_trajectory(npz_path)
        
        filename = os.path.basename(npz_path)
        map_name = filename.split("-random-")[0]
        grid_map = self.maps[map_name]
        
        cur_locs = discrete_positions[:, t_step, :].astype(float)
        noise = np.random.normal(0, 0.15, cur_locs.shape)
        cur_locs_jittered = cur_locs + noise
        cur_locs_discrete = (np.round(cur_locs_jittered) + self.k).astype(int)
        
        max_r = grid_map.shape[0] - self.k - 1
        max_c = grid_map.shape[1] - self.k - 1
        cur_locs_discrete[:, 0] = np.clip(cur_locs_discrete[:, 0], self.k, max_r)
        cur_locs_discrete[:, 1] = np.clip(cur_locs_discrete[:, 1], self.k, max_c)

        target_velocity = expert_velocities[:, t_step, :]

        speeds = np.linalg.norm(target_velocity, axis=1)
        is_parked = speeds < 0.01
        weights = np.ones(target_velocity.shape[0], dtype=np.float32)
        parked_ratio = np.mean(is_parked)
        
        weights[is_parked] -= (parked_ratio + 0.001)
        weights = np.clip(weights, 0.0, None)
        
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            weights = weights * (len(weights) / sum_weights)
        else:
            weights = np.ones_like(weights)
        
        scen_name = filename.replace('.npz', '').rsplit('_', 1)[0]
        bd_key = f"{map_name}-random-{scen_name.split('-random-')[-1]}"
        bd_file_path = os.path.join(self.bd_dir, "large_scale", f"{scen_name}_bds.npz")
        
        bd_grid = load_bd(bd_file_path, bd_key)
        bd = bd_grid[:cur_locs.shape[0]].astype(np.float32)
            
        bd = np.pad(bd, ((0, 0), (self.k, self.k), (self.k, self.k)), 'constant', constant_values=10000)
        dummy_goals = np.zeros_like(cur_locs_discrete)

        graph_data = create_data_object(cur_locs_discrete, bd, grid_map, self.k, self.m, dummy_goals)
        graph_data = normalize_graph_data(graph_data, self.k)

        graph_data.y = torch.tensor(target_velocity, dtype=torch.float32)
        graph_data.node_weights = torch.tensor(weights, dtype=torch.float32)

        return graph_data