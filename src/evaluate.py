"""Evaluation and demo inference for the official improved SOD model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data_loader import DataConfig, build_dataloaders
from sod_model import ImprovedSODNet
from train import evaluate_loader
from utils import (
    DEMO_THRESHOLD,
    BCEIoULoss,
    find_best_threshold,
    load_checkpoint,
    predict_image,
    save_prediction_overlay,
    set_seed,
    visualize_predictions,
)


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, float]:
    """Evaluate the checkpoint on the test split."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    criterion = BCEIoULoss()

    if args.search_threshold:
        threshold, threshold_results = find_best_threshold(model, dataloaders["val"], device, criterion=criterion)
        print("Validation threshold search:")
        for row in threshold_results:
            print(
                "threshold={threshold:.2f} | IoU={iou:.4f} | F1={f1:.4f} | "
                "Precision={precision:.4f} | Recall={recall:.4f}".format(**row)
            )
        print(f"Selected threshold from validation F1: {threshold:.2f}")
    else:
        threshold = args.threshold

    metrics = evaluate_loader(
        model,
        dataloaders["test"],
        criterion,
        device,
        threshold=threshold,
        desc="Test",
    )
    print("Test metrics:")
    print(f"Threshold: {threshold:.2f}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    if args.visualize:
        visualize_predictions(
            model,
            dataloaders["test"],
            device=device,
            threshold=threshold,
            max_items=args.max_visuals,
        )

    if "threshold" not in checkpoint or checkpoint.get("threshold") is None:
        print("Note: checkpoint does not store a threshold; using CLI/validation-selected threshold.")

    return metrics


def run_single_image_inference(args: argparse.Namespace) -> None:
    """Run inference on one image and save an overlay result."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedSODNet().to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    # Threshold 0.40 is the official demo threshold because it gave the best
    # validation F1 in the improved notebook experiment.
    image, _, _, elapsed = predict_image(
        model,
        args.image,
        device=device,
        image_size=args.image_size,
        threshold=args.threshold,
    )
    save_prediction_overlay(
        model=model,
        image_path=args.image,
        output_path=args.output,
        device=device,
        image_size=args.image_size,
        threshold=args.threshold,
    )
    print(f"Processed image size: {image.size}")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Inference time per image: {elapsed * 1000:.2f} ms")
    print(f"Saved overlay to: {Path(args.output)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or demo the improved SOD model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("test", help="Evaluate checkpoint on the test split.")
    eval_parser.add_argument("--data-root", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--batch-size", type=int, default=16)
    eval_parser.add_argument("--image-size", type=int, default=128)
    eval_parser.add_argument("--num-workers", type=int, default=0)
    eval_parser.add_argument("--val-ratio", type=float, default=0.15)
    eval_parser.add_argument("--test-ratio", type=float, default=0.15)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--threshold", type=float, default=DEMO_THRESHOLD)
    eval_parser.add_argument("--search-threshold", action="store_true")
    eval_parser.add_argument("--visualize", action="store_true")
    eval_parser.add_argument("--max-visuals", type=int, default=4)

    infer_parser = subparsers.add_parser("infer", help="Run inference on a single image.")
    infer_parser.add_argument("--image", required=True)
    infer_parser.add_argument("--checkpoint", required=True)
    infer_parser.add_argument("--output", required=True)
    infer_parser.add_argument("--image-size", type=int, default=128)
    infer_parser.add_argument("--threshold", type=float, default=DEMO_THRESHOLD)

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.command == "test":
        evaluate_checkpoint(parsed_args)
    elif parsed_args.command == "infer":
        run_single_image_inference(parsed_args)
