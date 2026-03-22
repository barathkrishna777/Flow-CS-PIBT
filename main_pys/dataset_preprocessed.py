"""
Lightning-fast dataset that loads pre-built PyG .pt files.
No CPU preprocessing at all — just torch.load().

On init, validates all files and removes corrupt ones so the
DataLoader never hits an EOFError mid-training.
"""
import os
import glob
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
                # Quick check: files under 100 bytes are definitely corrupt
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
                print(f"  Removed {corrupt:,} corrupt files. "
                      f"Re-run `python preprocess_dataset.py` to regenerate them.")
        else:
            self.files = all_files

        print(f"Preprocessed dataset: {len(self.files):,} valid samples from {preprocessed_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=False)
