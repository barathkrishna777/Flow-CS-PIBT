import glob
import os
import random
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, WeightedRandomSampler

from main_pys.continuous_scenarios import scenario_id_from_path
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
        expert_sources: Optional[Sequence[str]] = None,
        scenario_ids: Optional[Sequence[int]] = None,
        scenario_start: Optional[int] = None,
        scenario_end: Optional[int] = None,
    ) -> None:
        self.data_dir = data_dir
        self.map_dir = map_dir
        self.k = k
        self.m = m
        self.num_directions = num_directions
        self.wait_threshold = wait_threshold
        self.max_speed = max_speed
        self.allowed_expert_sources = set(expert_sources) if expert_sources else None
        self.allowed_scenario_ids = set(int(v) for v in scenario_ids) if scenario_ids else None
        self.scenario_start = scenario_start
        self.scenario_end = scenario_end

        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            raise RuntimeError(f"No continuous dataset files found in {data_dir}")

        self.maps = self._load_maps()
        self.rollout_infos = self._build_rollout_infos()
        self.index = self._build_index()

    def _load_maps(self) -> Dict[str, np.ndarray]:
        maps = {}
        for map_path in glob.glob(os.path.join(self.map_dir, "*.map")):
            map_name = os.path.basename(map_path).replace(".map", "")
            maps[map_name] = load_grid_map_from_file(map_path)
        return maps

    def _build_rollout_infos(self) -> List[Dict[str, object]]:
        infos: List[Dict[str, object]] = []
        for path in self.files:
            data = _load_npz(path)
            if "expert_source_used" in data:
                source_arr = data["expert_source_used"]
            else:
                source_arr = data["expert_source"]
            expert_source = str(source_arr.item() if source_arr.ndim == 0 else source_arr[0])
            scenario_name = str(data["scenario_name"].item() if data["scenario_name"].ndim == 0 else data["scenario_name"][0])
            scenario_id = int(data["scenario_id"].item()) if "scenario_id" in data else scenario_id_from_path(path)
            if self.allowed_expert_sources is not None and expert_source not in self.allowed_expert_sources:
                continue
            if self.allowed_scenario_ids is not None and scenario_id not in self.allowed_scenario_ids:
                continue
            if self.scenario_start is not None and scenario_id < self.scenario_start:
                continue
            if self.scenario_end is not None and scenario_id > self.scenario_end:
                continue
            infos.append(
                {
                    "path": path,
                    "scenario_name": scenario_name,
                    "scenario_id": scenario_id,
                    "expert_source": expert_source,
                    "agent_count": int(data["positions"].shape[1]),
                    "rollout_length": int(data["positions"].shape[0] - 1),
                    "fraction_moving": float(data["fraction_moving"].item()) if "fraction_moving" in data else 0.0,
                    "mean_nearest_neighbor_distance": (
                        float(data["mean_nearest_neighbor_distance"].item())
                        if "mean_nearest_neighbor_distance" in data
                        else 0.0
                    ),
                }
            )
        if not infos:
            raise RuntimeError("No continuous dataset files matched the provided filters")
        return infos

    def _build_index(self) -> List[Tuple[int, int]]:
        index = []
        for rollout_idx, info in enumerate(self.rollout_infos):
            data = _load_npz(str(info["path"]))
            steps = int(data["positions"].shape[0] - 1)
            for t in range(steps):
                index.append((rollout_idx, t))
        random.shuffle(index)
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        rollout_idx, t = self.index[idx]
        info = self.rollout_infos[rollout_idx]
        data = _load_npz(str(info["path"]))
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
        graph.scenario_name = str(info["scenario_name"])
        graph.scenario_id = int(info["scenario_id"])
        graph.expert_source = str(info["expert_source"])
        graph.agent_count = int(info["agent_count"])
        graph.rollout_id = int(rollout_idx)
        graph.positions = torch.from_numpy(positions)
        graph.goals = torch.from_numpy(goals)
        graph = normalize_continuous_graph_data(graph, self.k, max_speed=self.max_speed)
        return graph

    def get_sample_metadata(self, indices: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
        use_indices = list(indices) if indices is not None else list(range(len(self.index)))
        counts = np.zeros(len(use_indices), dtype=np.int32)
        expert_source = np.empty(len(use_indices), dtype=object)
        difficulty = np.zeros(len(use_indices), dtype=np.float32)
        rollout_ids = np.zeros(len(use_indices), dtype=np.int32)
        scenario_ids = np.zeros(len(use_indices), dtype=np.int32)
        for out_idx, sample_idx in enumerate(use_indices):
            rollout_idx, _ = self.index[sample_idx]
            info = self.rollout_infos[rollout_idx]
            counts[out_idx] = int(info["agent_count"])
            expert_source[out_idx] = str(info["expert_source"])
            difficulty[out_idx] = float(info["fraction_moving"]) + 1.0 / max(float(info["mean_nearest_neighbor_distance"]) + 1e-6, 1e-6)
            rollout_ids[out_idx] = int(rollout_idx)
            scenario_ids[out_idx] = int(info["scenario_id"])
        return {
            "agent_count": counts,
            "expert_source": expert_source,
            "difficulty": difficulty,
            "rollout_id": rollout_ids,
            "scenario_id": scenario_ids,
        }

    def split_indices_by_rollout(self, val_split: float, seed: int) -> Tuple[List[int], List[int]]:
        if val_split <= 0.0:
            return list(range(len(self.index))), []
        rollout_ids = list(range(len(self.rollout_infos)))
        rng = random.Random(seed)
        rng.shuffle(rollout_ids)
        val_rollouts = max(1, int(round(len(rollout_ids) * val_split)))
        val_rollout_ids = set(rollout_ids[:val_rollouts])
        train_indices: List[int] = []
        val_indices: List[int] = []
        for sample_idx, (rollout_idx, _) in enumerate(self.index):
            if rollout_idx in val_rollout_ids:
                val_indices.append(sample_idx)
            else:
                train_indices.append(sample_idx)
        return train_indices, val_indices


def build_continuous_weighted_sampler(
    dataset: ContinuousFlowDataset,
    subset: Optional[Subset] = None,
    balance_agent_counts: bool = True,
    balance_expert_sources: bool = True,
    oversample_difficult: bool = False,
) -> WeightedRandomSampler:
    indices = subset.indices if subset is not None else None
    metadata = dataset.get_sample_metadata(indices)
    counts = metadata["agent_count"]
    sources = metadata["expert_source"]
    difficulty = metadata["difficulty"]

    total = len(counts)
    weights = np.ones(total, dtype=np.float64)

    if balance_agent_counts and total > 0:
        bucket_edges = [0, 32, 64, 128, 256, np.inf]
        bucket_ids = np.digitize(counts, bucket_edges[1:])
        unique, bucket_counts = np.unique(bucket_ids, return_counts=True)
        count_map = {u: c for u, c in zip(unique, bucket_counts)}
        n_buckets = max(len(unique), 1)
        weights *= np.array([total / (n_buckets * count_map.get(b, 1)) for b in bucket_ids], dtype=np.float64)

    if balance_expert_sources and total > 0:
        unique_sources, source_counts = np.unique(sources, return_counts=True)
        source_map = {src: c for src, c in zip(unique_sources.tolist(), source_counts.tolist())}
        n_sources = max(len(unique_sources), 1)
        weights *= np.array([total / (n_sources * source_map.get(src, 1)) for src in sources], dtype=np.float64)

    if oversample_difficult and total > 0:
        normalized = difficulty / max(float(np.mean(difficulty)), 1e-6)
        weights *= np.clip(normalized, 0.5, 2.0)

    return WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=total, replacement=True)
