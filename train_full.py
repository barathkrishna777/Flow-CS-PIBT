#!/usr/bin/env python3
"""
One-command training script for Flow-CS-PIBT.

Usage:
    python train_full.py \
        --base-data /path/to/data.zip \
        --trajectories /path/to/massive_flow_dataset_large_scale.zip

That's it. The script will:
  1. Extract both zips into data/
  2. Print dataset statistics
  3. Launch full training (10 epochs, all fixes applied)
"""
import argparse
import os
import sys
import glob
import shutil
import zipfile
import subprocess
import numpy as np
from tqdm import tqdm


def extract_data(base_zip, traj_zip, project_dir):
    data_dir = os.path.join(project_dir, "data")

    # Clean old data
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # 1) Base support data: maps + BDs + other base assets
    print("Extracting base data...")
    with zipfile.ZipFile(base_zip, 'r') as z:
        z.extractall(project_dir)

    # 2) Replace trajectories with freshly generated ones
    flow_data_dir = os.path.join(data_dir, "flow_training_data_multi")
    if os.path.exists(flow_data_dir):
        shutil.rmtree(flow_data_dir)

    print("Extracting trajectories...")
    with zipfile.ZipFile(traj_zip, 'r') as z:
        z.extractall(project_dir)

    # 3) Sanity check
    npz_files = glob.glob(os.path.join(data_dir, "flow_training_data_multi", "*.npz"))
    map_files = glob.glob(os.path.join(data_dir, "mapf-map", "*.map"))
    bd_files = glob.glob(os.path.join(data_dir, "bd_npzs", "**", "*.npz"), recursive=True)

    print(f"\n--- DATA INVENTORY ---")
    print(f"Trajectories : {len(npz_files)}")
    print(f"Maps         : {len(map_files)}")
    print(f"BD files     : {len(bd_files)}")

    if len(npz_files) == 0:
        print("ERROR: No trajectory files found! Check your zip structure.")
        sys.exit(1)

    # Count total training samples
    total_timesteps = 0
    print("Counting training samples...")
    for f in tqdm(npz_files, desc="Scanning"):
        try:
            with np.load(f) as data:
                total_timesteps += data['discrete_positions'].shape[1]
        except Exception:
            pass

    print(f"Total training samples: {total_timesteps:,}")
    print(f"----------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Train Flow-CS-PIBT model")
    parser.add_argument("--base-data", required=True,
                        help="Path to data.zip (maps, BDs, scen files)")
    parser.add_argument("--trajectories", required=True,
                        help="Path to massive_flow_dataset_large_scale.zip")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction if data/ already exists")
    args = parser.parse_args()

    # Resolve project directory (where this script lives = repo root)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    print(f"Project dir: {project_dir}")

    # Validate inputs
    for path, name in [(args.base_data, "base-data"), (args.trajectories, "trajectories")]:
        if not os.path.exists(path):
            print(f"ERROR: --{name} path does not exist: {path}")
            sys.exit(1)

    # Extract data
    if args.skip_extract and os.path.exists(os.path.join(project_dir, "data", "flow_training_data_multi")):
        print("Skipping extraction (--skip-extract and data/ exists)")
    else:
        extract_data(args.base_data, args.trajectories, project_dir)

    # Launch training
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "main_pys.train_flow"],
        cwd=project_dir
    )

    if result.returncode != 0:
        print(f"\nTraining failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("Checkpoints saved as: large_scale_flow_epoch_*.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
