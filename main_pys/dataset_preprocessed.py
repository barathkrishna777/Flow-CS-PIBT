"""
Lightning-fast dataset that loads pre-built PyG .pt files.
No CPU preprocessing at all — just torch.load().

On init, removes obviously corrupt files (<100 bytes).
At runtime, catches any remaining corrupt files and returns
a random valid sample instead of crashing the DataLoader.
"""
import os
import glob
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PreprocessedFlowMAPFDataset(Dataset):
    def __init__(self, preprocessed_dir="data/preprocessed", validate=True):
        all_files = sorted(glob.glob(os.path.join(preprocessed_dir, "*.pt")))
        if len(all_files) == 0:
            raise RuntimeError(
                f"No .pt files found in {preprocessed_dir}. "
                f"Run `python preprocess_dataset.py` first."
            )

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
