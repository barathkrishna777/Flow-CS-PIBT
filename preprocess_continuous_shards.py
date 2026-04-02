"""
Precompute the continuous PyG dataset into compact shard files.

This avoids rebuilding graphs during training while keeping disk usage
much lower than one-file-per-sample preprocessing.

Each shard stores a list of compacted graph samples:
  - float tensors are stored as float16
  - edge indices are stored as int32
  - action labels are stored as uint8

Usage:
    python preprocess_continuous_shards.py \
        --data-dir data/continuous_eecbs/raw \
        --map-dir data/mapf-map \
        --out data/continuous_eecbs/preprocessed_shards
"""
import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from main_pys.dataset_continuous import ContinuousFlowDataset
from main_pys.dataset_continuous_preprocessed import compact_continuous_sample


def _get_or_add_id(vocab: List[str], vocab_map: Dict[str, int], value: str) -> int:
    existing = vocab_map.get(value)
    if existing is not None:
        return existing
    idx = len(vocab)
    vocab.append(value)
    vocab_map[value] = idx
    return idx


def _save_shard(output_dir: str, shard_idx: int, samples: List[dict]) -> str:
    shard_name = f"shard_{shard_idx:05d}.pt"
    shard_path = os.path.join(output_dir, shard_name)
    torch.save({"samples": samples}, shard_path, pickle_protocol=4)
    return shard_name


def main() -> None:
    if hasattr(torch.multiprocessing, "set_sharing_strategy"):
        torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Precompute compact continuous PyG shards")
    parser.add_argument("--data-dir", required=True, help="Directory containing raw continuous .npz rollouts")
    parser.add_argument("--map-dir", required=True, help="Directory containing .map files")
    parser.add_argument("--out", required=True, help="Output directory for shard files and manifest")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--num-directions", type=int, default=8)
    parser.add_argument("--wait-threshold", type=float, default=0.1)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--expert-sources", nargs="*", default=None)
    parser.add_argument("--scenario-ids", nargs="*", type=int, default=None)
    parser.add_argument("--scenario-start", type=int, default=None)
    parser.add_argument("--scenario-end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8, help="Dataset loader workers")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of samples fetched per loader batch")
    parser.add_argument("--shard-size", type=int, default=2048, help="Samples per shard file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output dir if it already contains shards")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    existing = [name for name in os.listdir(args.out) if name.startswith("shard_") or name == "manifest.pt"]
    if existing and not args.overwrite:
        raise RuntimeError(
            f"Output directory {args.out} already contains shard data. "
            "Pass --overwrite or choose a new directory."
        )
    for name in existing:
        os.remove(os.path.join(args.out, name))

    dataset = ContinuousFlowDataset(
        data_dir=args.data_dir,
        map_dir=args.map_dir,
        k=args.k,
        m=args.m,
        num_directions=args.num_directions,
        wait_threshold=args.wait_threshold,
        max_speed=args.max_speed,
        expert_sources=args.expert_sources,
        scenario_ids=args.scenario_ids,
        scenario_start=args.scenario_start,
        scenario_end=args.scenario_end,
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "collate_fn": list,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)

    shard_files: List[str] = []
    samples_in_shard: List[dict] = []
    sample_to_shard: List[int] = []
    sample_to_offset: List[int] = []
    scenario_ids: List[int] = []
    agent_counts: List[int] = []
    difficulty: List[float] = []
    rollout_ids: List[int] = []
    map_ids: List[int] = []
    source_ids: List[int] = []
    scenario_name_ids: List[int] = []

    map_vocab: List[str] = []
    map_vocab_map: Dict[str, int] = {}
    source_vocab: List[str] = []
    source_vocab_map: Dict[str, int] = {}
    scenario_name_vocab: List[str] = []
    scenario_name_vocab_map: Dict[str, int] = {}

    shard_idx = 0
    sample_idx = 0
    progress = tqdm(total=len(dataset), desc="Preprocessing continuous shards")
    for batch in loader:
        for data in batch:
            compact_sample = compact_continuous_sample(data)
            sample_to_shard.append(shard_idx)
            sample_to_offset.append(len(samples_in_shard))
            samples_in_shard.append(compact_sample)

            scenario_ids.append(int(data.scenario_id))
            agent_counts.append(int(data.agent_count))
            rollout_ids.append(int(data.rollout_id))
            map_ids.append(_get_or_add_id(map_vocab, map_vocab_map, str(data.map_name)))
            source_ids.append(_get_or_add_id(source_vocab, source_vocab_map, str(data.expert_source)))
            scenario_name_ids.append(
                _get_or_add_id(scenario_name_vocab, scenario_name_vocab_map, str(data.scenario_name))
            )

            positions = data.positions.detach().cpu().numpy()
            moving_ratio = float((np.linalg.norm(data.y.detach().cpu().numpy(), axis=1) >= args.wait_threshold).mean())
            if len(positions) <= 1:
                difficulty_score = moving_ratio
            else:
                deltas = positions[:, None, :] - positions[None, :, :]
                dists = np.linalg.norm(deltas, axis=2)
                np.fill_diagonal(dists, np.inf)
                mean_nnd = float(np.mean(np.min(dists, axis=1)))
                difficulty_score = moving_ratio + 1.0 / max(mean_nnd + 1e-6, 1e-6)
            difficulty.append(difficulty_score)

            sample_idx += 1
            progress.update(1)
            if len(samples_in_shard) >= args.shard_size:
                shard_files.append(_save_shard(args.out, shard_idx, samples_in_shard))
                samples_in_shard = []
                shard_idx += 1

    if samples_in_shard:
        shard_files.append(_save_shard(args.out, shard_idx, samples_in_shard))

    progress.close()

    manifest = {
        "version": 1,
        "total_samples": sample_idx,
        "num_shards": len(shard_files),
        "shard_size": args.shard_size,
        "shard_files": shard_files,
        "sample_to_shard": np.asarray(sample_to_shard, dtype=np.int32),
        "sample_to_offset": np.asarray(sample_to_offset, dtype=np.int32),
        "scenario_id": np.asarray(scenario_ids, dtype=np.int32),
        "agent_count": np.asarray(agent_counts, dtype=np.int16),
        "difficulty": np.asarray(difficulty, dtype=np.float32),
        "rollout_id": np.asarray(rollout_ids, dtype=np.int32),
        "map_vocab": map_vocab,
        "map_id": np.asarray(map_ids, dtype=np.int16 if len(map_vocab) < 2 ** 15 else np.int32),
        "expert_source_vocab": source_vocab,
        "expert_source_id": np.asarray(
            source_ids, dtype=np.int16 if len(source_vocab) < 2 ** 15 else np.int32
        ),
        "scenario_name_vocab": scenario_name_vocab,
        "scenario_name_id": np.asarray(
            scenario_name_ids, dtype=np.int16 if len(scenario_name_vocab) < 2 ** 15 else np.int32
        ),
    }
    manifest_path = os.path.join(args.out, "manifest.pt")
    torch.save(manifest, manifest_path, pickle_protocol=4)

    total_bytes = 0
    for shard_name in shard_files:
        total_bytes += os.path.getsize(os.path.join(args.out, shard_name))
    total_bytes += os.path.getsize(manifest_path)
    print(
        "Done. "
        f"samples={sample_idx:,} shards={len(shard_files):,} "
        f"size_gb={total_bytes / (1024 ** 3):.2f} "
        f"output={args.out}"
    )


if __name__ == "__main__":
    main()
