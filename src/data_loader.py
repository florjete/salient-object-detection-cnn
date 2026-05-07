"""MSRA10K dataset loading that matches the official demo notebook."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class DataConfig:
    data_root: str
    image_size: int = 128
    batch_size: int = 16
    num_workers: int = 0
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


class MSRA10KDataset(Dataset):
    """Load paired RGB images and binary saliency masks.

    Preprocessing intentionally matches the working Colab notebook:
    images are resized to 128x128 and converted to 0..1 tensors; masks are
    resized with nearest-neighbor interpolation and thresholded to 0/1.
    """

    def __init__(self, pairs: list[tuple[Path, Path]], image_size: int = 128, augment: bool = False) -> None:
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.NEAREST)

        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Image-only color augmentation keeps mask geometry unchanged.
        if self.augment and random.random() < 0.3:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        if self.augment and random.random() < 0.3:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        if self.augment and random.random() < 0.2:
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))

        image_tensor = TF.to_tensor(image)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        return image_tensor, mask_tensor


def find_image_mask_pairs(data_root: str | Path) -> list[tuple[Path, Path]]:
    """Return sorted image/mask pairs matched by filename stem."""
    data_root = Path(data_root)
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {masks_dir}")

    image_paths = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.jpeg")])
    mask_paths = sorted(masks_dir.glob("*.png"))
    image_by_stem = {path.stem: path for path in image_paths}
    mask_by_stem = {path.stem: path for path in mask_paths}

    missing_masks = sorted(set(image_by_stem) - set(mask_by_stem))
    missing_images = sorted(set(mask_by_stem) - set(image_by_stem))
    if not image_paths:
        raise RuntimeError(f"No .jpg/.jpeg images found in {images_dir}")
    if not mask_paths:
        raise RuntimeError(f"No .png masks found in {masks_dir}")
    if missing_masks or missing_images:
        raise RuntimeError(
            "Image/mask filename mismatch. "
            f"Missing masks examples: {missing_masks[:5]}; "
            f"missing images examples: {missing_images[:5]}"
        )

    return [(image_by_stem[stem], mask_by_stem[stem]) for stem in sorted(image_by_stem)]


def create_splits(config: DataConfig) -> dict[str, list[tuple[Path, Path]]]:
    """Create 70/15/15 train/validation/test splits."""
    pairs = find_image_mask_pairs(config.data_root)
    train_pairs, temp_pairs = train_test_split(
        pairs,
        test_size=config.val_ratio + config.test_ratio,
        random_state=config.seed,
        shuffle=True,
    )
    relative_test_ratio = config.test_ratio / (config.val_ratio + config.test_ratio)
    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        test_size=relative_test_ratio,
        random_state=config.seed,
        shuffle=True,
    )
    return {"train": train_pairs, "val": val_pairs, "test": test_pairs}


def build_dataloaders(config: DataConfig) -> dict[str, DataLoader]:
    """Build train, validation, and test DataLoaders."""
    splits = create_splits(config)
    loaders: dict[str, DataLoader] = {}

    for split_name, pairs in splits.items():
        dataset = MSRA10KDataset(
            pairs=pairs,
            image_size=config.image_size,
            augment=(split_name == "train"),
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split_name == "train"),
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders
