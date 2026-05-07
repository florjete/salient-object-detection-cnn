"""Losses, metrics, threshold search, checkpointing, and visualization."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TF


DEMO_THRESHOLD = 0.40


def set_seed(seed: int = 42) -> None:
    """Make notebook/script runs more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class BCEIoULoss(nn.Module):
    """BCE + 0.5 * (1 - soft IoU), matching the working notebook."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(preds, targets)
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
        soft_iou = ((intersection + 1e-7) / (union + 1e-7)).mean()
        return bce_loss + 0.5 * (1 - soft_iou)


# Backward-compatible name used by the first project version.
BCEDiceIoULoss = BCEIoULoss


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Compute IoU, precision, recall, and F1 after thresholding predictions."""
    preds_bin = (preds >= threshold).float()
    targets_bin = (targets >= 0.5).float()

    tp = (preds_bin * targets_bin).sum(dim=(1, 2, 3))
    fp = (preds_bin * (1 - targets_bin)).sum(dim=(1, 2, 3))
    fn = ((1 - preds_bin) * targets_bin).sum(dim=(1, 2, 3))
    intersection = tp
    union = preds_bin.sum(dim=(1, 2, 3)) + targets_bin.sum(dim=(1, 2, 3)) - intersection

    iou = ((intersection + 1e-7) / (union + 1e-7)).mean()
    precision = ((tp + 1e-7) / (tp + fp + 1e-7)).mean()
    recall = ((tp + 1e-7) / (tp + fn + 1e-7)).mean()
    f1 = (2 * precision * recall) / (precision + recall + 1e-7)

    return {
        "iou": float(iou.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "f1": float(f1.item()),
    }


@torch.no_grad()
def find_best_threshold(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str,
    criterion: nn.Module | None = None,
    thresholds: list[float] | None = None,
) -> tuple[float, list[dict[str, float]]]:
    """Select the threshold with the best validation F1."""
    if thresholds is None:
        thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    model.eval()
    results: list[dict[str, float]] = []
    best_threshold = DEMO_THRESHOLD
    best_f1 = -1.0

    for threshold in thresholds:
        totals = {"loss": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        for images, masks in dataloader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            preds = model(images)

            batch_size = images.size(0)
            if criterion is not None:
                totals["loss"] += float(criterion(preds, masks).item()) * batch_size
            metrics = compute_metrics(preds, masks, threshold=threshold)
            for key, value in metrics.items():
                totals[key] += value * batch_size

        n = len(dataloader.dataset)
        row = {"threshold": threshold, **{key: value / n for key, value in totals.items()}}
        results.append(row)
        if row["f1"] > best_f1:
            best_f1 = row["f1"]
            best_threshold = threshold

    return best_threshold, results


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_iou: float,
    history: list[dict[str, float]],
    threshold: float | None = None,
) -> None:
    """Save training state without touching any existing file unless this path is chosen by the caller."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": model.__class__.__name__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_iou": best_val_iou,
            "threshold": threshold,
            "image_size": 128,
            "history": history,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """Load model and optionally optimizer state."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def save_history(history: list[dict[str, float]], path: str | Path) -> None:
    """Save metrics history as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def plot_training_history(history: list[dict[str, float]], save_path: str | Path | None = None) -> None:
    """Plot training loss and validation scores."""
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train Loss")
    plt.plot(epochs, [row["val_loss"] for row in history], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["val_iou"] for row in history], marker="o", label="Val IoU")
    plt.plot(epochs, [row["val_f1"] for row in history], marker="o", label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Scores")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def visualize_batch(images: torch.Tensor, masks: torch.Tensor, max_items: int = 4) -> None:
    """Show image/mask pairs for sanity checks."""
    count = min(max_items, images.size(0))
    plt.figure(figsize=(3 * count, 6))

    for idx in range(count):
        plt.subplot(2, count, idx + 1)
        plt.imshow(images[idx].cpu().permute(1, 2, 0))
        plt.axis("off")
        plt.title("Image")

        plt.subplot(2, count, count + idx + 1)
        plt.imshow(masks[idx, 0].cpu(), cmap="gray")
        plt.axis("off")
        plt.title("Mask")

    plt.tight_layout()
    plt.show()


def overlay_mask(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    """Overlay a predicted binary/probability mask on a 0..1 image tensor."""
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    mask_np = np.clip(mask.detach().cpu().squeeze().numpy(), 0, 1)
    red = np.zeros_like(image_np)
    red[..., 0] = 1.0
    overlay = (1 - alpha * mask_np[..., None]) * image_np + (alpha * mask_np[..., None]) * red
    return np.clip(overlay, 0, 1)


@torch.no_grad()
def visualize_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str,
    threshold: float = DEMO_THRESHOLD,
    max_items: int = 4,
) -> None:
    """Display input, ground truth, thresholded prediction, and overlay."""
    model.eval()
    images, masks = next(iter(dataloader))
    preds = model(images.to(device)).cpu()
    count = min(max_items, images.size(0))

    plt.figure(figsize=(4 * count, 12))
    for idx in range(count):
        pred_binary = (preds[idx, 0] >= threshold).float()

        plt.subplot(4, count, idx + 1)
        plt.imshow(images[idx].permute(1, 2, 0))
        plt.axis("off")
        plt.title("Input")

        plt.subplot(4, count, count + idx + 1)
        plt.imshow(masks[idx, 0], cmap="gray")
        plt.axis("off")
        plt.title("Ground Truth")

        plt.subplot(4, count, 2 * count + idx + 1)
        plt.imshow(pred_binary, cmap="gray")
        plt.axis("off")
        plt.title("Prediction")

        plt.subplot(4, count, 3 * count + idx + 1)
        plt.imshow(overlay_mask(images[idx], pred_binary))
        plt.axis("off")
        plt.title("Overlay")

    plt.tight_layout()
    plt.show()


def predict_image(
    model: nn.Module,
    image_path: str | Path,
    device: torch.device | str,
    image_size: int = 128,
    threshold: float = DEMO_THRESHOLD,
) -> tuple[Image.Image, np.ndarray, np.ndarray, float]:
    """Run single-image inference and return probability mask, binary mask, and timing."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    resized = TF.resize(image, (image_size, image_size), interpolation=TF.InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(resized).unsqueeze(0).to(device)

    model.eval()
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        pred = model(tensor)
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    pred_mask = pred[0, 0].detach().cpu().numpy()
    binary = (pred_mask >= threshold).astype(np.float32)
    return resized, pred_mask, binary, elapsed


def save_prediction_overlay(
    model: nn.Module,
    image_path: str | Path,
    output_path: str | Path,
    device: torch.device | str,
    image_size: int = 128,
    threshold: float = DEMO_THRESHOLD,
) -> None:
    """Save an overlay visualization for a single image."""
    image, _, binary, _ = predict_image(model, image_path, device, image_size=image_size, threshold=threshold)
    image_np = np.asarray(image).astype(np.float32) / 255.0
    red = np.zeros_like(image_np)
    red[..., 0] = 1.0
    overlay = (1 - 0.45 * binary[..., None]) * image_np + (0.45 * binary[..., None]) * red

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8)).save(output_path)
