"""Organize the raw MSRA10K folder into image and mask folders.

Expected raw structure:
    raw_dir/
        101.jpg
        101.png
        ...

Output structure:
    output_dir/
        images/
            101.jpg
        masks/
            101.png

The script copies files by default, so the raw dataset remains untouched.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
MASK_EXTENSIONS = {".png"}


def organize_msra10k(raw_dir: str | Path, output_dir: str | Path, move: bool = False) -> None:
    """Split mixed JPG/PNG files into images/ and masks/ directories."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    operation = shutil.move if move else shutil.copy2
    copied_images = 0
    copied_masks = 0

    for path in raw_dir.iterdir():
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            operation(str(path), str(images_dir / path.name))
            copied_images += 1
        elif suffix in MASK_EXTENSIONS:
            operation(str(path), str(masks_dir / path.name))
            copied_masks += 1

    image_ids = {p.stem for p in images_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS}
    mask_ids = {p.stem for p in masks_dir.glob("*") if p.suffix.lower() in MASK_EXTENSIONS}
    missing_masks = sorted(image_ids - mask_ids)
    missing_images = sorted(mask_ids - image_ids)

    print(f"Images organized: {copied_images}")
    print(f"Masks organized: {copied_masks}")
    print(f"Matched pairs: {len(image_ids & mask_ids)}")

    if missing_masks:
        print(f"Warning: {len(missing_masks)} images do not have masks. Example: {missing_masks[:5]}")
    if missing_images:
        print(f"Warning: {len(missing_images)} masks do not have images. Example: {missing_images[:5]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize MSRA10K JPG/PNG files.")
    parser.add_argument("--raw-dir", required=True, help="Folder containing mixed .jpg images and .png masks.")
    parser.add_argument("--output-dir", required=True, help="Destination folder that will contain images/ and masks/.")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    organize_msra10k(args.raw_dir, args.output_dir, move=args.move)
