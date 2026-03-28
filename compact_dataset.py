"""
In-place dataset compaction: shrink preprocessed .pt files by ~6x.

Optimizations applied:
  1. Downsample: keep every Nth file (default N=3), delete the rest
  2. Dtype compression: float32 → float16 for node features, edge_attr,
     velocities (y), node_weights, bd_pred; int64 → int32 for edge_index
  3. Add discrete action labels derived from velocity targets

The script processes files in-place (overwrite + delete) to minimize
disk usage during migration. Progress is checkpointed so it can resume
after interruption.

Usage:
    python compact_dataset.py --dir data/preprocessed
    python compact_dataset.py --dir data/preprocessed --keep-every 3 --dry-run
    python compact_dataset.py --dir data/preprocessed --resume
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch
import numpy as np
from tqdm import tqdm


# Action vectors: 0=wait(0,0), 1=right(0,1), 2=down(1,0), 3=up(-1,0), 4=left(0,-1)
ACTION_VECS = torch.tensor([[0, 0], [0, 1], [1, 0], [-1, 0], [0, -1]], dtype=torch.float32)
WAIT_MAGNITUDE_THRESH = 0.3  # velocities below this norm → wait


def derive_discrete_actions(velocities: torch.Tensor) -> torch.Tensor:
    """Map continuous velocity targets to discrete action labels (0-4).

    Args:
        velocities: (N, 2) float tensor of expert velocities.

    Returns:
        (N,) int8 tensor of action labels.
    """
    # Dot product with cardinal direction vectors
    scores = velocities.float() @ ACTION_VECS.T  # (N, 5)
    actions = scores.argmax(dim=1)  # (N,)

    # Near-zero velocity → wait (action 0)
    norms = velocities.float().norm(dim=1)
    actions[norms < WAIT_MAGNITUDE_THRESH] = 0

    return actions.to(torch.int8)


def compact_one_file(filepath: str) -> tuple[int, int]:
    """Load a .pt file, compress dtypes, add action labels, save back.

    Returns:
        (old_size, new_size) in bytes.
    """
    old_size = os.path.getsize(filepath)

    data = torch.load(filepath, weights_only=False)

    # --- Dtype compression ---
    # Node features: (N, 3, 9, 9) float32 → float16
    if data.x.dtype == torch.float32:
        data.x = data.x.half()

    # Edge index: (2, E) int64 → int32
    if data.edge_index.dtype == torch.int64:
        data.edge_index = data.edge_index.to(torch.int32)

    # Edge attributes: (E, 2) float32 → float16
    if data.edge_attr.dtype == torch.float32:
        data.edge_attr = data.edge_attr.half()

    # Target velocity: (N, 2) float32 → float16
    if data.y.dtype == torch.float32:
        data.y = data.y.half()

    # Node weights: (N,) float32 → float16
    if hasattr(data, 'node_weights') and data.node_weights is not None:
        if data.node_weights.dtype == torch.float32:
            data.node_weights = data.node_weights.half()

    # BD predictions: (N, 5) float32 → float16
    if hasattr(data, 'bd_pred') and data.bd_pred is not None:
        if data.bd_pred.dtype == torch.float32:
            data.bd_pred = data.bd_pred.half()

    # --- Add discrete action labels ---
    if not hasattr(data, 'action_label') or data.action_label is None:
        data.action_label = derive_discrete_actions(data.y)

    # --- Mark as compact format ---
    data.compact_version = 1

    # Save back (overwrite)
    torch.save(data, filepath, pickle_protocol=4)
    new_size = os.path.getsize(filepath)

    return old_size, new_size


def main():
    parser = argparse.ArgumentParser(description="In-place dataset compaction")
    parser.add_argument("--dir", required=True,
                        help="Directory containing sample_*.pt files")
    parser.add_argument("--keep-every", type=int, default=3,
                        help="Keep every Nth file, delete the rest (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without modifying files")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-compacted files (detected by dtype)")
    parser.add_argument("--no-downsample", action="store_true",
                        help="Only compress dtypes, don't delete any files")
    parser.add_argument("--verify-count", type=int, default=5,
                        help="Number of random files to verify after compaction")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"ERROR: directory not found: {args.dir}", file=sys.stderr)
        sys.exit(1)

    # Discover files
    pattern = os.path.join(args.dir, "sample_*.pt")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        print(f"No sample_*.pt files found in {args.dir}")
        sys.exit(1)

    total_files = len(all_files)
    print(f"Found {total_files:,} files in {args.dir}")

    # Checkpoint file for resumable processing
    checkpoint_path = os.path.join(args.dir, ".compact_checkpoint.json")
    processed_set = set()
    deleted_set = set()

    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        processed_set = set(ckpt.get("processed", []))
        deleted_set = set(ckpt.get("deleted", []))
        print(f"Resuming: {len(processed_set):,} already compacted, "
              f"{len(deleted_set):,} already deleted")

    # Decide which files to keep vs delete
    if args.no_downsample:
        keep_files = all_files
        delete_files = []
    else:
        keep_files = all_files[::args.keep_every]
        delete_files = [f for f in all_files if f not in set(keep_files)]

    print(f"Keep: {len(keep_files):,} files | Delete: {len(delete_files):,} files")
    print(f"Keep ratio: 1/{args.keep_every} = {len(keep_files)/total_files:.1%}")

    # Estimate sizes
    sample_files = all_files[:min(20, len(all_files))]
    avg_old_size = sum(os.path.getsize(f) for f in sample_files) / len(sample_files)
    est_old_total = avg_old_size * total_files / 1e9
    est_new_total = avg_old_size * 0.5 * len(keep_files) / 1e9  # ~50% from dtype
    print(f"Estimated current total: {est_old_total:.1f} GB")
    print(f"Estimated after compaction: {est_new_total:.1f} GB")
    print(f"Estimated savings: {est_old_total - est_new_total:.1f} GB")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
        return

    # Phase 1: Delete files we don't need (frees space for compaction)
    if delete_files:
        # Filter out already-deleted files
        to_delete = [f for f in delete_files
                     if os.path.basename(f) not in deleted_set and os.path.exists(f)]
        if to_delete:
            print(f"\nPhase 1: Deleting {len(to_delete):,} downsampled files...")
            freed = 0
            for f in tqdm(to_delete, desc="Deleting"):
                try:
                    freed += os.path.getsize(f)
                    os.remove(f)
                    deleted_set.add(os.path.basename(f))
                except OSError:
                    pass
            print(f"  Freed {freed / 1e9:.2f} GB")

            # Save checkpoint
            with open(checkpoint_path, 'w') as f:
                json.dump({"processed": list(processed_set),
                           "deleted": list(deleted_set)}, f)
        else:
            print("\nPhase 1: All downsample deletions already done.")

    # Phase 2: Compress kept files in-place
    to_compact = [f for f in keep_files
                  if os.path.basename(f) not in processed_set and os.path.exists(f)]

    if not to_compact:
        print("\nPhase 2: All files already compacted.")
    else:
        print(f"\nPhase 2: Compacting {len(to_compact):,} files...")
        total_old = 0
        total_new = 0
        errors = 0
        save_interval = 1000  # checkpoint every N files

        for i, filepath in enumerate(tqdm(to_compact, desc="Compacting")):
            try:
                old_sz, new_sz = compact_one_file(filepath)
                total_old += old_sz
                total_new += new_sz
                processed_set.add(os.path.basename(filepath))
            except Exception as e:
                errors += 1
                if errors <= 5:
                    tqdm.write(f"  Error: {filepath}: {e}")

            # Periodic checkpoint
            if (i + 1) % save_interval == 0:
                with open(checkpoint_path, 'w') as f:
                    json.dump({"processed": list(processed_set),
                               "deleted": list(deleted_set)}, f)

        # Final checkpoint
        with open(checkpoint_path, 'w') as f:
            json.dump({"processed": list(processed_set),
                       "deleted": list(deleted_set)}, f)

        if total_old > 0:
            ratio = total_new / total_old
            print(f"  Compressed: {total_old / 1e9:.2f} GB → {total_new / 1e9:.2f} GB "
                  f"({ratio:.2%} of original)")
        if errors > 0:
            print(f"  Errors: {errors}")

    # Phase 3: Verify random samples
    remaining_files = sorted(glob.glob(os.path.join(args.dir, "sample_*.pt")))
    print(f"\nFinal file count: {len(remaining_files):,}")

    if args.verify_count > 0 and remaining_files:
        import random
        verify_files = random.sample(remaining_files,
                                     min(args.verify_count, len(remaining_files)))
        print(f"Verifying {len(verify_files)} random files...")
        for vf in verify_files:
            try:
                data = torch.load(vf, weights_only=False)
                assert data.x.dtype == torch.float16, f"x dtype: {data.x.dtype}"
                assert data.edge_index.dtype == torch.int32, f"edge_index dtype: {data.edge_index.dtype}"
                assert data.y.dtype == torch.float16, f"y dtype: {data.y.dtype}"
                assert hasattr(data, 'action_label'), "missing action_label"
                assert data.action_label.dtype == torch.int8, f"action_label dtype: {data.action_label.dtype}"
                n_agents = data.x.shape[0]
                assert data.x.shape == (n_agents, 3, 9, 9), f"x shape: {data.x.shape}"
                assert data.y.shape == (n_agents, 2), f"y shape: {data.y.shape}"
                assert data.action_label.shape == (n_agents,), f"action_label shape: {data.action_label.shape}"
                tqdm.write(f"  ✓ {os.path.basename(vf)}: {n_agents} agents, "
                           f"{os.path.getsize(vf)/1024:.0f} KB")
            except Exception as e:
                tqdm.write(f"  ✗ {os.path.basename(vf)}: {e}")

    # Clean up checkpoint if fully done
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # Invalidate old agent_counts cache (needs rebuilding with new file list)
    cache_path = os.path.join(args.dir, "agent_counts.json")
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print("Removed stale agent_counts.json cache (will rebuild on next training run)")

    # Summary
    final_size = sum(os.path.getsize(f) for f in remaining_files) / 1e9
    print(f"\nDone! {len(remaining_files):,} files, {final_size:.2f} GB total")


if __name__ == "__main__":
    main()
