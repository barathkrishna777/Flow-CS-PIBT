import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from main_pys.model_inputs import create_data_object, normalize_graph_data

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
            
        print("Building flattened timestep index for Large-Scale training...")
        self.index = []
        for f in self.npz_files:
            try:
                with np.load(f) as data:
                    T = data['discrete_positions'].shape[1]
                    for t in range(T):
                        self.index.append((f, t))
            except Exception as e:
                print(f"Skipping corrupted file: {f}")
                
        print(f"Total training samples (timesteps): {len(self.index)}")

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
        data = np.load(npz_path)
        
        cur_locs = data['discrete_positions'][:, t_step, :].astype(float)
        noise = np.random.normal(0, 0.15, cur_locs.shape)
        cur_locs_jittered = cur_locs + noise
        cur_locs_discrete = (np.round(cur_locs_jittered) + self.k).astype(int)
        
        target_velocity = data['expert_velocities'][:, t_step, :]

        # --- GOAL WEIGHTING LOGIC ---
        speeds = np.linalg.norm(target_velocity, axis=1)
        is_parked = speeds < 0.01
        weights = np.ones(target_velocity.shape[0], dtype=np.float32)
        parked_ratio = np.mean(is_parked)
        
        weights[is_parked] -= (parked_ratio + 0.001)
        weights = np.clip(weights, 0.0, None) # Safety to prevent negative weights
        
        # Normalize weights
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            weights = weights * (len(weights) / sum_weights)
        else:
            weights = np.ones_like(weights)
        
        filename = os.path.basename(npz_path)
        map_name = filename.split("-random-")[0]
        scen_name = filename.split('_')[0]
        bd_key = f"{map_name}-random-{scen_name.split('-random-')[-1]}"

        grid_map = self.maps[map_name]
        
        # Lazy load the 1000-agent BD grid
        bd_file_path = os.path.join(self.bd_dir, "large_scale", f"{scen_name}_bds.npz")
        with np.load(bd_file_path) as bd_data:
            bd = bd_data[bd_key][:cur_locs.shape[0]].astype(np.float32)
            
        bd = np.pad(bd, ((0, 0), (self.k, self.k), (self.k, self.k)), 'constant', constant_values=10000)
        dummy_goals = np.zeros_like(cur_locs_discrete)

        graph_data = create_data_object(cur_locs_discrete, bd, grid_map, self.k, self.m, dummy_goals)
        graph_data = normalize_graph_data(graph_data, self.k)

        return graph_data, torch.tensor(target_velocity, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32)