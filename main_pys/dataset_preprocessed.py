"""
Lightning-fast dataset that loads pre-built PyG .pt files.
No CPU preprocessing at all — just torch.load().
"""
import os
import glob
import torch
from torch.utils.data import Dataset


class PreprocessedFlowMAPFDataset(Dataset):
    def __init__(self, preprocessed_dir="data/preprocessed"):
        self.files = sorted(glob.glob(os.path.join(preprocessed_dir, "*.pt")))
        if len(self.files) == 0:
            raise RuntimeError(
                f"No .pt files found in {preprocessed_dir}. "
                f"Run `python preprocess_dataset.py` first."
            )
        print(f"Preprocessed dataset: {len(self.files):,} samples from {preprocessed_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=False)
