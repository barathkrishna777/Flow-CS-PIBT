"""Train a local Rishi-like discrete-action classifier.

This is the Phase 3 sanity check from FLOW_VS_SSIL_ABLATION_PLAN.md. It uses
preprocessed PyG samples with exact next-action labels in `action_y`.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from main_pys.dataset_preprocessed import PreprocessedFlowMAPFDataset
from main_pys.rishi_like_model import RishiLikeClassifier


def action_labels_from_batch(batch, device):
    if not hasattr(batch, "action_y") or batch.action_y is None:
        raise RuntimeError(
            "Rishi-like classifier training requires exact action_y labels. "
            "Re-run scripts/preprocess_dataset.py after the exact-action-label change."
        )
    return batch.action_y.view(-1).long().to(device)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_nodes = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if training else "val"):
            batch = batch.to(device)
            labels = action_labels_from_batch(batch, device)
            _, logits = model(batch)
            loss = F.cross_entropy(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_nodes += labels.numel()

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = total_correct / max(total_nodes, 1)
    return avg_loss, accuracy


def save_checkpoint(path, model, optimizer, epoch, val_loss, args):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "model_config": {
                "k": args.k,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "in_channels": args.in_channels,
                "aux_feature_dim": 5,
                "action_dim": 5,
            },
        },
        path,
    )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    dataset = PreprocessedFlowMAPFDataset(args.preprocessed_dir)
    val_size = int(len(dataset) * args.val_split) if args.val_split > 0 else 0
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )

    model = RishiLikeClassifier(
        k=args.k,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        in_channels=args.in_channels,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, f"{args.run_name}_best.pt")
    final_path = os.path.join(args.out_dir, f"{args.run_name}_final.pt")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        if val_loader is not None:
            val_loss, val_acc = run_epoch(model, val_loader, device)
        else:
            val_loss, val_acc = train_loss, train_acc

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, model, optimizer, epoch, val_loss, args)
            print(f"  saved best: {best_path}")

    save_checkpoint(final_path, model, optimizer, args.epochs, best_val, args)
    print(f"Done. Best checkpoint: {best_path}")
    print(f"Final checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a Rishi-like local classifier")
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--run-name", default="rishi_like_classifier")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
