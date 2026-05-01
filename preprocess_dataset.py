"""
Preprocess the full dataset into ready-to-load PyG .pt files.

Runs the expensive CPU pipeline (Savitzky-Golay, BD lookups, graph construction,
normalization) once and saves each sample as a torch .pt file. Training then
does a simple torch.load() per sample — no computation, pure I/O.

Usage:
    python preprocess_dataset.py [--workers 32] [--out /media/.../preprocessed_data]
    python preprocess_dataset.py --migrate-from data/preprocessed --out /media/.../preprocessed_data
    python preprocess_dataset.py --data-dir data/flow_training_data_custom_coordination \
        --map-dir data/custom_coordination/maps --bd-dir data/bd_npzs/custom_coordination \
        --out data/preprocessed_custom_coordination --output-prefix custom
"""
import os
import sys
import glob
import shutil
import argparse
import numpy as np
import torch
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

from main_pys.model_inputs import (
    create_data_object,
    discrete_action_labels_from_positions,
    normalize_graph_data,
)


# ── Map loading (same logic as dataset.py) ──────────────────────────────────
def read_map(map_file, k):
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
    return np.pad(map_data, k, 'constant', constant_values=1)


# ── Process one (file, timestep) pair ───────────────────────────────────────
def process_sample(args, maps, k, m, out_dir, bd_dir, output_prefix):
    """Build and save one PyG Data object. Returns output path or None on error."""
    npz_path, t_step, sample_idx = args
    out_path = os.path.join(out_dir, f"{output_prefix}_{sample_idx:08d}.pt")
    if os.path.exists(out_path) and os.path.getsize(out_path) >= 100:
        return out_path  # already processed and not corrupt, skip
    try:
        with np.load(npz_path) as data:
            discrete_positions = data['discrete_positions']
            expert_velocities = data['expert_velocities']

        filename = os.path.basename(npz_path)
        map_name = filename.split("-random-")[0]
        grid_map = maps[map_name]

        cur_locs = discrete_positions[:, t_step, :].astype(float)
        # Deterministic jitter with seed for reproducibility
        rng = np.random.RandomState(sample_idx)
        noise = rng.normal(0, 0.15, cur_locs.shape)
        cur_locs_jittered = cur_locs + noise
        cur_locs_discrete = (np.round(cur_locs_jittered) + k).astype(int)

        max_r = grid_map.shape[0] - k - 1
        max_c = grid_map.shape[1] - k - 1
        cur_locs_discrete[:, 0] = np.clip(cur_locs_discrete[:, 0], k, max_r)
        cur_locs_discrete[:, 1] = np.clip(cur_locs_discrete[:, 1], k, max_c)

        target_velocity = expert_velocities[:, t_step, :]
        action_labels = discrete_action_labels_from_positions(discrete_positions, t_step)

        # Goal weighting
        speeds = np.linalg.norm(target_velocity, axis=1)
        is_parked = speeds < 0.01
        parked_ratio = np.mean(is_parked)
        weights = np.ones(target_velocity.shape[0], dtype=np.float32)
        moving_weight = 1.0 / (1.0 - parked_ratio + 1e-3)
        parked_weight = 1.0 / (parked_ratio + 1e-3)
        weights[~is_parked] = moving_weight
        weights[is_parked] = parked_weight
        weights = weights * (len(weights) / weights.sum())

        # BD loading
        scen_name = filename.replace('.npz', '').rsplit('_', 1)[0]
        bd_file_path = os.path.join(bd_dir, f"{scen_name}_bds.npz")
        bd_key = f"{map_name}-random-{scen_name.split('-random-')[-1]}"
        with np.load(bd_file_path) as bd_data:
            bd_grid = bd_data[bd_key][:cur_locs.shape[0]].astype(np.float32)
        bd_grid = np.pad(bd_grid, ((0, 0), (k, k), (k, k)), 'constant', constant_values=10000)

        dummy_goals = np.zeros_like(cur_locs_discrete)

        graph_data = create_data_object(cur_locs_discrete, bd_grid, grid_map, k, m, dummy_goals)
        graph_data = normalize_graph_data(graph_data, k)
        graph_data.y = torch.tensor(target_velocity, dtype=torch.float32)
        graph_data.action_y = torch.tensor(action_labels, dtype=torch.long)
        graph_data.node_weights = torch.tensor(weights, dtype=torch.float32)

        torch.save(graph_data, out_path)
        return out_path
    except Exception as e:
        return None


def worker_init(maps_dict, k_val, m_val, out_dir_val, bd_dir_val, output_prefix_val):
    """Store shared data in each worker process."""
    global _maps, _k, _m, _out_dir, _bd_dir, _output_prefix
    _maps = maps_dict
    _k = k_val
    _m = m_val
    _out_dir = out_dir_val
    _bd_dir = bd_dir_val
    _output_prefix = output_prefix_val


def worker_fn(args):
    """Wrapper that uses global worker state."""
    return process_sample(args, _maps, _k, _m, _out_dir, _bd_dir, _output_prefix)


def split_dirs(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def migrate_existing(old_dir, new_dir):
    """Validate and move existing .pt files from old_dir to new_dir."""
    old_files = sorted(glob.glob(os.path.join(old_dir, "sample_*.pt")))
    if not old_files:
        print(f"  No existing files found in {old_dir}")
        return 0

    print(f"  Found {len(old_files):,} files in {old_dir}")
    print(f"  Validating and moving to {new_dir}...")
    os.makedirs(new_dir, exist_ok=True)

    moved = 0
    corrupt = 0
    for f in tqdm(old_files, desc="Migrating"):
        fname = os.path.basename(f)
        try:
            if os.path.getsize(f) < 100:
                os.remove(f)
                corrupt += 1
                continue
        except OSError:
            corrupt += 1
            continue

        dst = os.path.join(new_dir, fname)
        if not os.path.exists(dst):
            shutil.move(f, dst)
            moved += 1
        else:
            # Already in new dir, just delete old copy
            os.remove(f)
            moved += 1

    print(f"  Migrated {moved:,} valid files, removed {corrupt:,} corrupt files")

    # Clean up old directory if empty
    remaining = glob.glob(os.path.join(old_dir, "*.pt"))
    if not remaining:
        print(f"  Old directory {old_dir} is now empty")

    return moved


def main():
    parser = argparse.ArgumentParser(description="Preprocess MAPF dataset into PyG .pt files")
    parser.add_argument("--data-dir", default="data/flow_training_data_multi",
                        help="Directory with trajectory .npz files")
    parser.add_argument("--map-dir", default="data/mapf-map",
                        help="Directory with .map files. Comma-separated dirs are supported.")
    parser.add_argument("--bd-dir", default=os.path.join("data", "bd_npzs", "large_scale"),
                        help="Directory with *_bds.npz files")
    parser.add_argument("--out", default="data/preprocessed",
                        help="Output directory for .pt files")
    parser.add_argument("--output-prefix", default="sample",
                        help="Filename prefix for .pt files (default: sample)")
    parser.add_argument("--migrate-from", default=None,
                        help="Old preprocessed dir to validate and move files from before processing")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (default: all CPU cores)")
    parser.add_argument("--k", type=int, default=4, help="Local region size")
    parser.add_argument("--m", type=int, default=5, help="Nearest neighbors")
    parser.add_argument("--exclude-maps", nargs="*", default=None,
                        help="Map names to exclude from preprocessing (e.g. den312d empty-48-48)")
    parser.add_argument("--include-maps", nargs="*", default=None,
                        help="If set, only preprocess these map names. Useful for curriculum/staged datasets.")
    args = parser.parse_args()
    exclude_set = set(args.exclude_maps) if args.exclude_maps else set()
    include_set = set(args.include_maps) if args.include_maps else None
    if include_set and exclude_set:
        overlap = include_set & exclude_set
        if overlap:
            raise ValueError(f"Maps cannot be both included and excluded: {sorted(overlap)}")

    k, m = args.k, args.m
    num_workers = args.workers if args.workers > 0 else cpu_count()

    # 0) Migrate existing data from old location if specified
    if args.migrate_from and os.path.isdir(args.migrate_from):
        print(f"Migrating existing data from {args.migrate_from} -> {args.out}")
        migrate_existing(args.migrate_from, args.out)
    elif args.migrate_from:
        print(f"  Migration source {args.migrate_from} not found, skipping")

    # 1) Load all maps
    print("Loading maps...")
    maps = {}
    for map_dir in split_dirs(args.map_dir):
        for map_path in glob.glob(os.path.join(map_dir, "*.map")):
            map_name = os.path.basename(map_path).replace(".map", "")
            if map_name in maps:
                print(f"  Duplicate map name {map_name}; keeping first copy")
                continue
            maps[map_name] = read_map(map_path, k)
    print(f"  Loaded {len(maps)} maps")

    # 2) Build flat index of (file, timestep, global_idx)
    print("Building sample index...")
    npz_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if include_set:
        before = len(npz_files)
        npz_files = [f for f in npz_files
                     if os.path.basename(f).split("-random-")[0] in include_set]
        print(f"  Included {len(npz_files)} of {before} files from maps: {include_set}")
    if exclude_set:
        before = len(npz_files)
        npz_files = [f for f in npz_files
                     if os.path.basename(f).split("-random-")[0] not in exclude_set]
        print(f"  Excluded {before - len(npz_files)} files from maps: {exclude_set}")
    index = []
    for f in tqdm(npz_files, desc="Scanning"):
        try:
            with np.load(f) as data:
                T = data['discrete_positions'].shape[1]
                for t in range(T):
                    index.append((f, t, len(index)))
        except Exception:
            pass
    print(f"  Total samples: {len(index):,}")

    # 3) Create output dir
    os.makedirs(args.out, exist_ok=True)

    # 4) Process in parallel
    print(f"Processing with {num_workers} workers -> {args.out}/")
    with Pool(
        num_workers,
        initializer=worker_init,
        initargs=(maps, k, m, args.out, args.bd_dir, args.output_prefix),
    ) as pool:
        results = list(tqdm(
            pool.imap_unordered(worker_fn, index, chunksize=64),
            total=len(index),
            desc="Preprocessing"
        ))

    succeeded = sum(1 for r in results if r is not None)
    failed = sum(1 for r in results if r is None)
    print(f"\nDone! {succeeded:,} samples OK, {failed:,} failed.")
    print(f"Output: {args.out}/")


if __name__ == "__main__":
    main()
