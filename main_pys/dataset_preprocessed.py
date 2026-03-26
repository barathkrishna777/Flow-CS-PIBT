"""
Lightning-fast dataset that loads pre-built PyG .pt files.
No CPU preprocessing at all — just torch.load().

On init, removes obviously corrupt files (<100 bytes).
At runtime, catches any remaining corrupt files and returns
a random valid sample instead of crashing the DataLoader.
"""
import os
import glob
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm


class PreprocessedFlowMAPFDataset(Dataset):
    def __init__(self, preprocessed_dir="data/preprocessed", validate=True):
        # Support multiple directories (comma-separated string or list)
        if isinstance(preprocessed_dir, str):
            dirs = [d.strip() for d in preprocessed_dir.split(",") if d.strip()]
        elif isinstance(preprocessed_dir, (list, tuple)):
            dirs = list(preprocessed_dir)
        else:
            dirs = [preprocessed_dir]

        self.preprocessed_dir = dirs[0]  # primary dir (for cache files)
        all_files = []
        for d in dirs:
            all_files.extend(glob.glob(os.path.join(d, "*.pt")))
        all_files.sort()

        if len(all_files) == 0:
            raise RuntimeError(
                f"No .pt files found in {dirs}. "
                f"Run `python preprocess_dataset.py` first."
            )
        if len(dirs) > 1:
            for d in dirs:
                n = len(glob.glob(os.path.join(d, "*.pt")))
                print(f"  {d}: {n:,} files")
            print(f"  Combined: {len(all_files):,} files")

        if validate:
            print(f"Validating {len(all_files):,} preprocessed files...")
            self.files = []
            corrupt = 0
            for f in tqdm(all_files, desc="Validating", mininterval=5):
                try:
                    if os.path.getsize(f) < 100:
                        os.remove(f)
                        corrupt += 1
                        continue
                except OSError:
                    corrupt += 1
                    continue
                self.files.append(f)

            if corrupt > 0:
                print(f"  Removed {corrupt:,} corrupt files (<100 bytes). "
                      f"Re-run `python preprocess_dataset.py` to regenerate them.")
        else:
            self.files = all_files

        print(f"Preprocessed dataset: {len(self.files):,} valid samples from {preprocessed_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Try loading the requested file; if corrupt, try random alternatives
        for attempt in range(5):
            try:
                target = idx if attempt == 0 else random.randint(0, len(self.files) - 1)
                return torch.load(self.files[target], weights_only=False)
            except Exception:
                continue
        # All 5 attempts failed — shouldn't happen, but return a minimal dummy
        # rather than crashing the entire training run
        raise RuntimeError(f"Failed to load any sample after 5 attempts (started at idx {idx})")

    def get_agent_counts(self):
        """Return agent count per sample for weighted sampling.

        Uses a cached JSON file (agent_counts.json) in the preprocessed dir.
        If not found, scans all .pt files to build it (slow first time, fast after).
        """
        cache_path = os.path.join(self.preprocessed_dir, "agent_counts.json")

        if os.path.exists(cache_path):
            print("Loading cached agent counts...")
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            # Map cached filenames to current file list
            fname_to_count = {os.path.basename(k): v for k, v in cached.items()}
            counts = []
            missing = 0
            for f in self.files:
                c = fname_to_count.get(os.path.basename(f))
                if c is not None:
                    counts.append(c)
                else:
                    missing += 1
                    counts.append(50)  # default fallback
            if missing > 0:
                print(f"  {missing} files not in cache — rebuilding...")
            else:
                print(f"  Loaded {len(counts):,} agent counts from cache.")
                return np.array(counts, dtype=np.int32)

        # Scan all files (slow but only done once)
        print(f"Building agent count cache for {len(self.files):,} files...")
        counts = {}
        for f in tqdm(self.files, desc="Scanning agent counts", mininterval=5):
            try:
                data = torch.load(f, weights_only=False)
                counts[os.path.basename(f)] = data.x.shape[0]
            except Exception:
                counts[os.path.basename(f)] = 50  # fallback

        with open(cache_path, 'w') as fp:
            json.dump(counts, fp)
        print(f"  Cached to {cache_path}")

        return np.array([counts[os.path.basename(f)] for f in self.files], dtype=np.int32)


def build_weighted_sampler(dataset):
    """Build a WeightedRandomSampler that upsamples high-agent-count scenarios.

    Buckets samples by agent count and weights each bucket by inverse frequency,
    so rare high-density scenarios are sampled as often as common low-density ones.
    """
    agent_counts = dataset.get_agent_counts()

    # Bucket edges: [0,35), [35,75), [75,150), [150,300), [300,inf)
    bucket_edges = [0, 35, 75, 150, 300, np.inf]
    bucket_labels = ["20", "50", "100", "200", "400+"]
    bucket_ids = np.digitize(agent_counts, bucket_edges[1:])  # 0-indexed bucket

    # Count per bucket
    unique, counts = np.unique(bucket_ids, return_counts=True)
    bucket_counts = {u: c for u, c in zip(unique, counts)}

    print("Agent count distribution for weighted sampling:")
    for i, label in enumerate(bucket_labels):
        c = bucket_counts.get(i, 0)
        print(f"  {label:>5s} agents: {c:>8,d} samples")

    # Inverse-frequency weight per bucket
    total = len(agent_counts)
    n_buckets = len(bucket_labels)
    bucket_weights = {}
    for i in range(n_buckets):
        c = bucket_counts.get(i, 1)
        bucket_weights[i] = total / (n_buckets * c)

    # Per-sample weight
    sample_weights = np.array([bucket_weights[b] for b in bucket_ids], dtype=np.float64)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(dataset),
        replacement=True
    )
    return sampler
