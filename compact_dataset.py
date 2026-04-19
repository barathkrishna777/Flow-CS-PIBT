"""
In-place dataset compaction: shrink preprocessed .pt files by ~6x.

Optimizations applied:
  1. Downsample: keep every Nth file (default N=3), delete the rest
  2. Dtype compression: float32 → float16 for node features, edge_attr,
     velocities (y), node_weights, bd_pred; int64 → int32 for edge_index
  3. Add discrete action labels derived from velocity targets

The script processes files in-place (overwrite + delete) to minimize
disk usage during migration. Progress is checkpointed so it can resume
after interruption. Compaction is parallelized across CPU workers.

Usage:
    python compact_dataset.py --dir data/preprocessed
    python compact_dataset.py --dir data/preprocessed --keep-every 3 --dry-run
    python compact_dataset.py --dir data/preprocessed --workers 64 --resume
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import torch
import numpy as np
from tqdm import tqdm

from main_pys.grid_actions import get_action_vectors

WAIT_MAGNITUDE_THRESH = 0.3  # velocities below this norm → wait


def derive_discrete_actions(velocities: torch.Tensor, action_mode: str = "grid4") -> torch.Tensor:
    """Map continuous velocity targets to discrete action labels.

    Args:
        velocities: (N, 2) float tensor of expert velocities.

    Returns:
        (N,) int8 tensor of action labels.
    """
    # Dot product with cardinal direction vectors
    action_vecs = torch.as_tensor(get_action_vectors(action_mode, normalize=True), dtype=torch.float32)
    scores = velocities.float() @ action_vecs.T
    actions = scores.argmax(dim=1)  # (N,)

    # Near-zero velocity → wait (action 0)
    norms = velocities.float().norm(dim=1)
    actions[norms < WAIT_MAGNITUDE_THRESH] = 0

    return actions.to(torch.int8)


def compact_one_file(filepath: str, action_mode: str = "grid4") -> tuple[str, int, int, str | None]:
    """Load a .pt file, compress dtypes, add action labels, save back.

    Returns:
        (basename, old_size, new_size, error_or_None)
    """
    basename = os.path.basename(filepath)
    try:
        old_size = os.path.getsize(filepath)

        data = torch.load(filepath, weights_only=False)

        # --- Dtype compression ---
        if data.x.dtype == torch.float32:
            data.x = data.x.half()
        if data.edge_index.dtype == torch.int64:
            data.edge_index = data.edge_index.to(torch.int32)
        if data.edge_attr.dtype == torch.float32:
            data.edge_attr = data.edge_attr.half()
        if data.y.dtype == torch.float32:
            data.y = data.y.half()
        if hasattr(data, 'node_weights') and data.node_weights is not None:
            if data.node_weights.dtype == torch.float32:
                data.node_weights = data.node_weights.half()
        if hasattr(data, 'bd_pred') and data.bd_pred is not None:
            if data.bd_pred.dtype == torch.float32:
                data.bd_pred = data.bd_pred.half()

        # --- Add discrete action labels ---
        if (
            not hasattr(data, 'action_label')
            or data.action_label is None
            or getattr(data, 'action_mode', None) != action_mode
        ):
            data.action_label = derive_discrete_actions(data.y, action_mode=action_mode)
            data.action_mode = action_mode

        # --- Mark as compact format ---
        data.compact_version = 1

        # Save back (overwrite)
        torch.save(data, filepath, pickle_protocol=4)
        new_size = os.path.getsize(filepath)

        return basename, old_size, new_size, None
    except Exception as e:
        return basename, 0, 0, str(e)


def delete_one_file(filepath: str) -> int:
    """Delete a file and return its size (0 on error)."""
    try:
        sz = os.path.getsize(filepath)
        os.remove(filepath)
        return sz
    except OSError:
        return 0


def main():
    parser = argparse.ArgumentParser(description="In-place dataset compaction")
    parser.add_argument("--dir", required=True,
                        help="Directory containing sample_*.pt files")
    parser.add_argument("--keep-every", type=int, default=3,
                        help="Keep every Nth file, delete the rest (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without modifying files")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-compacted files (detected by checkpoint)")
    parser.add_argument("--no-downsample", action="store_true",
                        help="Only compress dtypes, don't delete any files")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (default: all CPU cores)")
    parser.add_argument("--verify-count", type=int, default=5,
                        help="Number of random files to verify after compaction")
    parser.add_argument("--action-mode", choices=["grid4", "grid8"], default="grid4",
                        help="Discrete action mode for newly derived action labels")
    args = parser.parse_args()

    num_workers = args.workers if args.workers > 0 else cpu_count()

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
    print(f"Workers: {num_workers}")

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

    # ── Phase 1: Delete files we don't need (parallel) ──────────────────
    if delete_files:
        to_delete = [f for f in delete_files
                     if os.path.basename(f) not in deleted_set and os.path.exists(f)]
        if to_delete:
            print(f"\nPhase 1: Deleting {len(to_delete):,} downsampled files "
                  f"({num_workers} workers)...", flush=True)
            freed = 0
            t0 = time.time()
            log_interval = max(1, len(to_delete) // 20)  # ~20 log lines

            with Pool(num_workers) as pool:
                for i, sz in enumerate(pool.imap_unordered(
                        delete_one_file, to_delete, chunksize=256)):
                    freed += sz
                    if (i + 1) % log_interval == 0 or (i + 1) == len(to_delete):
                        elapsed = time.time() - t0
                        rate = (i + 1) / elapsed
                        eta = (len(to_delete) - i - 1) / rate if rate > 0 else 0
                        print(f"  [{i+1:,}/{len(to_delete):,}] "
                              f"{freed/1e9:.2f} GB freed | "
                              f"{rate:.0f} files/s | "
                              f"ETA {eta:.0f}s", flush=True)

            deleted_set.update(os.path.basename(f) for f in to_delete)
            elapsed = time.time() - t0
            print(f"  Done: freed {freed / 1e9:.2f} GB in {elapsed:.1f}s "
                  f"({len(to_delete)/elapsed:.0f} files/s)", flush=True)

            with open(checkpoint_path, 'w') as f:
                json.dump({"processed": list(processed_set),
                           "deleted": list(deleted_set)}, f)
        else:
            print("\nPhase 1: All downsample deletions already done.")

    # ── Phase 2: Compress kept files in-place (parallel) ────────────────
    to_compact = [f for f in keep_files
                  if os.path.basename(f) not in processed_set and os.path.exists(f)]

    if not to_compact:
        print("\nPhase 2: All files already compacted.")
    else:
        print(f"\nPhase 2: Compacting {len(to_compact):,} files "
              f"({num_workers} workers)...", flush=True)
        total_old = 0
        total_new = 0
        errors = 0
        save_interval = 5000  # checkpoint every N files
        log_interval = max(1, len(to_compact) // 50)  # ~50 log lines
        t0 = time.time()

        with Pool(num_workers) as pool:
            compact_fn = partial(compact_one_file, action_mode=args.action_mode)
            results_iter = pool.imap_unordered(compact_fn, to_compact, chunksize=64)
            for i, (basename, old_sz, new_sz, err) in enumerate(results_iter):
                if err is not None:
                    errors += 1
                    if errors <= 5:
                        print(f"  Error: {basename}: {err}", flush=True)
                else:
                    total_old += old_sz
                    total_new += new_sz
                    processed_set.add(basename)

                # Progress log
                if (i + 1) % log_interval == 0 or (i + 1) == len(to_compact):
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (len(to_compact) - i - 1) / rate if rate > 0 else 0
                    pct = (i + 1) / len(to_compact) * 100
                    ratio_str = (f"{total_new/total_old:.1%}"
                                 if total_old > 0 else "N/A")
                    print(f"  [{i+1:,}/{len(to_compact):,} {pct:.0f}%] "
                          f"{total_old/1e9:.2f} GB → {total_new/1e9:.2f} GB "
                          f"({ratio_str}) | "
                          f"{rate:.0f} files/s | "
                          f"ETA {eta:.0f}s | "
                          f"errors: {errors}", flush=True)

                # Periodic checkpoint
                if (i + 1) % save_interval == 0:
                    with open(checkpoint_path, 'w') as f:
                        json.dump({"processed": list(processed_set),
                                   "deleted": list(deleted_set)}, f)

        # Final checkpoint
        with open(checkpoint_path, 'w') as f:
            json.dump({"processed": list(processed_set),
                       "deleted": list(deleted_set)}, f)

        elapsed = time.time() - t0
        if total_old > 0:
            ratio = total_new / total_old
            print(f"\n  Compressed: {total_old / 1e9:.2f} GB → {total_new / 1e9:.2f} GB "
                  f"({ratio:.2%} of original) in {elapsed:.1f}s", flush=True)
        if errors > 0:
            print(f"  Errors: {errors}")

    # ── Phase 3: Verify random samples ──────────────────────────────────
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
                print(f"  OK {os.path.basename(vf)}: {n_agents} agents, "
                      f"{os.path.getsize(vf)/1024:.0f} KB")
            except Exception as e:
                print(f"  FAIL {os.path.basename(vf)}: {e}")

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
