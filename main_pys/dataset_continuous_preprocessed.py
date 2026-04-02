import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
from torch_geometric.data import Data
import random

from main_pys.model_inputs import load_grid_map_from_file


def _categorical_value(vocab: Sequence[str], ids: np.ndarray, idx: int) -> str:
    return str(vocab[int(ids[idx])])


def _compact_tensor(tensor: Optional[torch.Tensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().cpu().to(dtype).contiguous()


def _build_sample_from_compact(
    sample: Dict[str, Optional[torch.Tensor]],
    metadata: Dict[str, object],
) -> Data:
    data = Data(
        x=sample["x"].float(),
        edge_index=sample["edge_index"].to(torch.int64),
        edge_attr=sample["edge_attr"].float(),
        aux_features=sample["aux_features"].float(),
        y=sample["y"].float(),
        action_label=sample["action_label"].long(),
    )
    if sample.get("node_weights") is not None:
        data.node_weights = sample["node_weights"].float()
    if sample.get("positions") is not None:
        data.positions = sample["positions"].float()
    if sample.get("goals") is not None:
        data.goals = sample["goals"].float()

    data.map_name = str(metadata["map_name"])
    data.scenario_name = str(metadata["scenario_name"])
    data.scenario_id = int(metadata["scenario_id"])
    data.expert_source = str(metadata["expert_source"])
    data.agent_count = int(metadata["agent_count"])
    data.rollout_id = int(metadata["rollout_id"])
    return data


class PreprocessedContinuousShardDataset(Dataset):
    def __init__(
        self,
        preprocessed_dir: str,
        map_dir: str,
        expert_sources: Optional[Sequence[str]] = None,
        scenario_ids: Optional[Sequence[int]] = None,
        scenario_start: Optional[int] = None,
        scenario_end: Optional[int] = None,
        shard_cache_size: int = 2,
    ) -> None:
        self.preprocessed_dir = preprocessed_dir
        self.map_dir = map_dir
        self.allowed_expert_sources = set(expert_sources) if expert_sources else None
        self.allowed_scenario_ids = set(int(v) for v in scenario_ids) if scenario_ids else None
        self.scenario_start = scenario_start
        self.scenario_end = scenario_end
        self.shard_cache_size = max(1, int(shard_cache_size))
        self._shard_cache: "OrderedDict[int, Dict[str, object]]" = OrderedDict()

        manifest_path = os.path.join(preprocessed_dir, "manifest.pt")
        if not os.path.exists(manifest_path):
            raise RuntimeError(
                f"Missing continuous preprocessed manifest: {manifest_path}. "
                "Run `python preprocess_continuous_shards.py` first."
            )
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
        if int(manifest.get("version", 0)) != 1:
            raise RuntimeError(f"Unsupported continuous preprocessed manifest version: {manifest.get('version')}")

        self.manifest = manifest
        self.maps = self._load_maps()
        self.indices = self._build_indices()
        if not self.indices:
            raise RuntimeError("No preprocessed continuous samples matched the provided filters")

    def _load_maps(self) -> Dict[str, np.ndarray]:
        maps = {}
        for map_path in os.listdir(self.map_dir):
            if not map_path.endswith(".map"):
                continue
            map_name = map_path.replace(".map", "")
            maps[map_name] = load_grid_map_from_file(os.path.join(self.map_dir, map_path))
        return maps

    def _build_indices(self) -> List[int]:
        scenario_ids = np.asarray(self.manifest["scenario_id"], dtype=np.int32)
        source_ids = np.asarray(self.manifest["expert_source_id"], dtype=np.int16)
        expert_vocab = list(self.manifest["expert_source_vocab"])

        indices: List[int] = []
        for idx in range(int(self.manifest["total_samples"])):
            scenario_id = int(scenario_ids[idx])
            if self.allowed_scenario_ids is not None and scenario_id not in self.allowed_scenario_ids:
                continue
            if self.scenario_start is not None and scenario_id < self.scenario_start:
                continue
            if self.scenario_end is not None and scenario_id > self.scenario_end:
                continue
            expert_source = expert_vocab[int(source_ids[idx])]
            if self.allowed_expert_sources is not None and expert_source not in self.allowed_expert_sources:
                continue
            indices.append(idx)
        return indices

    def _get_shard(self, shard_idx: int) -> Dict[str, object]:
        cached = self._shard_cache.get(shard_idx)
        if cached is not None:
            self._shard_cache.move_to_end(shard_idx)
            return cached

        shard_name = self.manifest["shard_files"][shard_idx]
        shard_path = os.path.join(self.preprocessed_dir, shard_name)
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        self._shard_cache[shard_idx] = shard
        self._shard_cache.move_to_end(shard_idx)
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return shard

    def _sample_metadata(self, global_idx: int) -> Dict[str, object]:
        manifest = self.manifest
        return {
            "map_name": _categorical_value(manifest["map_vocab"], manifest["map_id"], global_idx),
            "scenario_name": _categorical_value(manifest["scenario_name_vocab"], manifest["scenario_name_id"], global_idx),
            "scenario_id": int(manifest["scenario_id"][global_idx]),
            "expert_source": _categorical_value(
                manifest["expert_source_vocab"], manifest["expert_source_id"], global_idx
            ),
            "agent_count": int(manifest["agent_count"][global_idx]),
            "rollout_id": int(manifest["rollout_id"][global_idx]),
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        global_idx = self.indices[idx]
        shard_idx = int(self.manifest["sample_to_shard"][global_idx])
        offset = int(self.manifest["sample_to_offset"][global_idx])
        shard = self._get_shard(shard_idx)
        sample = shard["samples"][offset]
        return _build_sample_from_compact(sample, self._sample_metadata(global_idx))

    def get_sample_metadata(self, indices: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
        use_indices = list(indices) if indices is not None else list(range(len(self.indices)))
        global_indices = [self.indices[idx] for idx in use_indices]
        manifest = self.manifest
        return {
            "agent_count": np.asarray(manifest["agent_count"], dtype=np.int32)[global_indices],
            "expert_source": np.asarray(
                [
                    _categorical_value(manifest["expert_source_vocab"], manifest["expert_source_id"], i)
                    for i in global_indices
                ],
                dtype=object,
            ),
            "difficulty": np.asarray(manifest["difficulty"], dtype=np.float32)[global_indices],
            "rollout_id": np.asarray(manifest["rollout_id"], dtype=np.int32)[global_indices],
            "scenario_id": np.asarray(manifest["scenario_id"], dtype=np.int32)[global_indices],
        }

    def split_indices_by_rollout(self, val_split: float, seed: int):
        if val_split <= 0.0:
            return list(range(len(self.indices))), []
        rollout_ids = self.get_sample_metadata()["rollout_id"]
        unique_rollouts = list(np.unique(rollout_ids).tolist())
        rng = random.Random(seed)
        rng.shuffle(unique_rollouts)
        val_rollouts = max(1, int(round(len(unique_rollouts) * val_split)))
        val_rollout_ids = set(unique_rollouts[:val_rollouts])
        train_indices: List[int] = []
        val_indices: List[int] = []
        for sample_idx, rollout_id in enumerate(rollout_ids.tolist()):
            if rollout_id in val_rollout_ids:
                val_indices.append(sample_idx)
            else:
                train_indices.append(sample_idx)
        return train_indices, val_indices


def build_preprocessed_continuous_weighted_sampler(
    dataset: PreprocessedContinuousShardDataset,
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


def compact_continuous_sample(data: Data) -> Dict[str, Optional[torch.Tensor]]:
    return {
        "x": _compact_tensor(data.x, torch.float16),
        "edge_index": _compact_tensor(data.edge_index, torch.int32),
        "edge_attr": _compact_tensor(data.edge_attr, torch.float16),
        "aux_features": _compact_tensor(getattr(data, "aux_features", None), torch.float16),
        "y": _compact_tensor(getattr(data, "y", None), torch.float16),
        "action_label": _compact_tensor(getattr(data, "action_label", None), torch.uint8),
        "node_weights": _compact_tensor(getattr(data, "node_weights", None), torch.float16),
        "positions": _compact_tensor(getattr(data, "positions", None), torch.float16),
        "goals": _compact_tensor(getattr(data, "goals", None), torch.float16),
    }
