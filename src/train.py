"""Training entrypoint for the improved from-scratch MSRA10K SOD model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from data_loader import DataConfig, build_dataloaders
from sod_model import ImprovedSODNet, count_parameters
from utils import BCEIoULoss, compute_metrics, find_best_threshold, load_checkpoint, save_checkpoint, save_history, set_seed


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    desc: str = "Evaluating",
) -> dict[str, float]:
    """Evaluate loss and thresholded metrics on a split."""
    model.eval()
    total_loss = 0.0
    metric_totals = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for images, masks in tqdm(dataloader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        preds = model(images)
        loss = criterion(preds, masks)
        metrics = compute_metrics(preds, masks, threshold=threshold)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        for key, value in metrics.items():
            metric_totals[key] += value * batch_size

    n = len(dataloader.dataset)
    return {
        "loss": total_loss / n,
        "iou": metric_totals["iou"] / n,
        "precision": metric_totals["precision"] / n,
        "recall": metric_totals["recall"] / n,
        "f1": metric_totals["f1"] / n,
    }


def run_training(args: argparse.Namespace) -> list[dict[str, float]]:
    """Train the improved model and save best/last checkpoints."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = DataConfig(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    dataloaders = build_dataloaders(config)

    model = ImprovedSODNet().to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = BCEIoULoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    history_path = Path(args.output_dir) / "training_history.json"
    best_checkpoint_path = checkpoint_dir / args.best_checkpoint_name
    last_checkpoint_path = checkpoint_dir / args.last_checkpoint_name

    best_val_iou = -1.0
    best_val_loss = float("inf")
    epochs_without_loss_improvement = 0
    start_epoch = 1
    history: list[dict[str, float]] = []

    if args.resume and last_checkpoint_path.exists():
        checkpoint = load_checkpoint(last_checkpoint_path, model, optimizer=optimizer, device=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_iou = float(checkpoint.get("best_val_iou", -1.0))
        history = list(checkpoint.get("history", []))
        if history:
            best_val_loss = min(row.get("val_loss", float("inf")) for row in history)
        print(f"Resumed from {last_checkpoint_path} at epoch {start_epoch}.")
    elif args.resume:
        print(f"Resume requested, but no checkpoint found at {last_checkpoint_path}. Starting fresh.")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_metrics = evaluate_loader(model, dataloaders["val"], criterion, device, threshold=0.5, desc="Validation")
        scheduler.step(val_metrics["iou"])
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "lr": current_lr,
        }
        history.append(row)

        print(
            "train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            "val_iou={val_iou:.4f} | val_f1={val_f1:.4f} | lr={lr:.6f}".format(**row)
        )

        save_checkpoint(last_checkpoint_path, model, optimizer, epoch, best_val_iou, history)
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            save_checkpoint(best_checkpoint_path, model, optimizer, epoch, best_val_iou, history)
            print(f"Saved best checkpoint: {best_checkpoint_path}")

        save_history(history, history_path)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_loss_improvement = 0
        else:
            epochs_without_loss_improvement += 1

        if epochs_without_loss_improvement >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}: validation loss did not improve for "
                f"{args.early_stop_patience} consecutive epochs."
            )
            break

    load_checkpoint(best_checkpoint_path, model, device=device)
    best_threshold, threshold_results = find_best_threshold(
        model,
        dataloaders["val"],
        device=device,
        criterion=criterion,
    )
    print(f"\nBest validation threshold by F1: {best_threshold:.2f}")
    for row in threshold_results:
        print(
            "threshold={threshold:.2f} | IoU={iou:.4f} | F1={f1:.4f} | "
            "Precision={precision:.4f} | Recall={recall:.4f}".format(**row)
        )

    save_history(threshold_results, Path(args.output_dir) / "threshold_search.json")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train improved from-scratch SOD model on MSRA10K.")
    parser.add_argument("--data-root", required=True, help="Path to MSRA10K folder containing images/ and masks/.")
    parser.add_argument("--checkpoint-dir", default="/content/drive/MyDrive/sod_data/checkpoints")
    parser.add_argument("--output-dir", default="/content/drive/MyDrive/sod_data/outputs")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--best-checkpoint-name", default="one_day_sod_best_model.pth")
    parser.add_argument("--last-checkpoint-name", default="one_day_sod_last_model.pth")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint if it exists.")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
