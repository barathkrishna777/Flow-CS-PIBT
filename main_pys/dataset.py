import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from main_pys.model_inputs import create_data_object, normalize_graph_data

class FlowMAPFDataset(Dataset):
    def __init__(self, data_dir, map_dir, bd_dir, k=4, m=5):
        """
        data_dir: Path to flow_training_data_multi
        map_dir: Path to mapf-map
        bd_dir: Path to bd_npzs
        """
        self.npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
        self.map_dir = map_dir
        self.k = k
        self.m = m

        print("Preloading Maps and BDs...")
        self.maps = {}
        for map_path in glob.glob(os.path.join(map_dir, "*.map")):
            map_name = os.path.basename(map_path).replace(".map", "")
            self.maps[map_name] = self._read_map(map_path)
            
        self.bds = {}
        for bd_path in glob.glob(os.path.join(bd_dir, "*.npz")):
            bd_data = np.load(bd_path)
            for key in bd_data.files:
                self.bds[key] = bd_data[key]

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
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_path = self.npz_files[idx]
        data = np.load(npz_path)
        discrete_pos = data['discrete_positions']
        expert_vel = data['expert_velocities']

        filename = os.path.basename(npz_path)
        map_name = filename.split("-random-")[0]
        scen_num = filename.split("-random-")[1].split("_")[0]
        bd_key = f"{map_name}-random-{scen_num}"

        T = discrete_pos.shape[1]
        t_step = np.random.randint(0, T)

        cur_locs = discrete_pos[:, t_step, :].astype(int) + self.k
        target_velocity = expert_vel[:, t_step, :]

        grid_map = self.maps[map_name]
        bd = self.bds[bd_key][:cur_locs.shape[0]]
        bd = np.pad(bd, ((0, 0), (self.k, self.k), (self.k, self.k)), 'constant', constant_values=12345678)

        dummy_goals = np.zeros_like(cur_locs)

        graph_data = create_data_object(cur_locs, bd, grid_map, self.k, self.m, dummy_goals)
        graph_data = normalize_graph_data(graph_data, self.k)

        return graph_data, torch.tensor(target_velocity, dtype=torch.float32)