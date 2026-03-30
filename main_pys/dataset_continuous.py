import glob
import os
import random
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from main_pys.model_inputs import (
    create_continuous_data_object,
    load_grid_map_from_file,
    normalize_continuous_graph_data,
    velocity_to_direction_labels,
)


@lru_cache(maxsize=8)
def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k].copy() for k in data.files}


class ContinuousFlowDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        map_dir: str,
        k: int = 4,
        m: int = 5,
        num_directions: int = 8,
        wait_threshold: float = 0.1,
        max_speed: float = 1.0,
    ) -> None:
        self.data_dir = data_dir
        self.map_dir = map_dir
        self.k = k
        self.m = m
        self.num_directions = num_directions
        self.wait_threshold = wait_threshold
        self.max_speed = max_speed

        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            raise RuntimeError(f"No continuous dataset files found in {data_dir}")

        self.maps = self._load_maps()
        self.index = self._build_index()

    def _load_maps(self) -> Dict[str, np.ndarray]:
        maps = {}
        for map_path in glob.glob(os.path.join(self.map_dir, "*.map")):
            map_name = os.path.basename(map_path).replace(".map", "")
            maps[map_name] = load_grid_map_from_file(map_path)
        return maps

    def _build_index(self) -> List[Tuple[str, int]]:
        index = []
        for path in self.files:
            data = _load_npz(path)
            steps = int(data["positions"].shape[0] - 1)
            for t in range(steps):
                index.append((path, t))
        random.shuffle(index)
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        path, t = self.index[idx]
        data = _load_npz(path)
        map_name = str(data["map_name"].item() if data["map_name"].ndim == 0 else data["map_name"][0])
        grid = self.maps[map_name]

        positions = data["positions"][t].astype(np.float32)
        goals = data["goals"].astype(np.float32)
        velocities = data["velocities"][t].astype(np.float32)
        action_labels = data["action_labels"][t].astype(np.int64) if "action_labels" in data else velocity_to_direction_labels(
            velocities,
            num_directions=self.num_directions,
            wait_threshold=self.wait_threshold,
        )

        moving = np.linalg.norm(velocities, axis=1) >= self.wait_threshold
        moving_ratio = moving.mean() if len(moving) else 0.0
        weights = np.ones(len(velocities), dtype=np.float32)
        if len(weights):
            moving_weight = 1.0 / max(moving_ratio, 1e-3)
            waiting_weight = 1.0 / max(1.0 - moving_ratio, 1e-3)
            weights[moving] = moving_weight
            weights[~moving] = waiting_weight
            weights *= len(weights) / max(weights.sum(), 1e-6)

        graph = create_continuous_data_object(
            positions,
            goals,
            grid,
            self.k,
            self.m,
            labels=velocities,
            action_labels=action_labels,
            max_speed=self.max_speed,
        )
        graph.node_weights = torch.from_numpy(weights)
        graph.map_name = map_name
        graph = normalize_continuous_graph_data(graph, self.k, max_speed=self.max_speed)
        return graph

    def get_agent_counts(self) -> np.ndarray:
        counts = np.zeros(len(self.index), dtype=np.int32)
        for i, (path, _) in enumerate(self.index):
            data = _load_npz(path)
            counts[i] = int(data["positions"].shape[1])
        return counts


def build_continuous_weighted_sampler(dataset: ContinuousFlowDataset) -> WeightedRandomSampler:
    counts = dataset.get_agent_counts()
    bucket_edges = [0, 32, 64, 128, 256, np.inf]
    bucket_ids = np.digitize(counts, bucket_edges[1:])
    unique, bucket_counts = np.unique(bucket_ids, return_counts=True)
    count_map = {u: c for u, c in zip(unique, bucket_counts)}
    total = len(counts)
    num_buckets = len(bucket_edges) - 1
    weights = np.array([total / (num_buckets * count_map.get(b, 1)) for b in bucket_ids], dtype=np.float64)
    return WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=len(dataset), replacement=True)
