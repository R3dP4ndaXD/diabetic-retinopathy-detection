"""
Visual inspection script for the updated augmentation pipeline.

Shows:
  - Grid of independently augmented versions (RandAugment, RandomAffine, flips, erasing)
  - MixUp / CutMix demo (two images blended)
  - Wavelet channel visualisation (LL, LH, HL, HH sub-bands)

Usage
-----
# Basic — single image, augmentation grid only
python scripts/inspect_augmentations.py --image path/to/sample.jpg

# Two images to also show MixUp / CutMix
python scripts/inspect_augmentations.py --image path/to/img1.jpg --image2 path/to/img2.jpg

# Custom image size and number of augmented views
python scripts/inspect_augmentations.py --image path/to/sample.jpg --size 224 --runs 12

# Show wavelet channels
python scripts/inspect_augmentations.py --image path/to/sample.jpg --show-wavelets
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from timm.data import Mixup
from torchvision.transforms import v2 as T

from src.data_module import DRDataModule
from src.model import WaveletChannelTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_visual_transform(image_size: int) -> T.Compose:
    """
    Full train augmentation pipeline WITHOUT the final Normalize step
    so the output stays in [0, 1] for matplotlib.
    """
    return T.Compose([
        T.Resize((image_size, image_size), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomAffine(
            degrees=(0, 360),
            scale=(0.9, 1.1),
            shear=(-11, 11),
            fill=(128, 128, 128),
        ),
        T.RandAugment(num_ops=2, magnitude=9, fill=(128, 128, 128)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomErasing(p=0.25, scale=(0.02, 0.1), value=0.5),
        # No Normalize — keep in [0,1] for display
    ])


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """[C,H,W] float32 → [H,W,C] uint8-compatible float clipped to [0,1]."""
    return t.permute(1, 2, 0).clamp(0, 1).numpy()


def _load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def _to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    """PIL → [1,3,H,W] float32 in [0,1], resized to size×size."""
    transform = T.Compose([
        T.Resize((size, size), antialias=True),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])
    return transform(img).unsqueeze(0)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def plot_augmentation_grid(
    img: Image.Image,
    image_size: int,
    runs: int,
    axes_iter,
) -> None:
    """Plot original + N independently augmented views."""
    transform = _build_visual_transform(image_size)

    # Original
    ax = next(axes_iter)
    ax.imshow(img)
    ax.set_title("Original", fontweight="bold", fontsize=9)
    ax.axis("off")

    for i in range(runs):
        aug = transform(img)
        ax = next(axes_iter)
        ax.imshow(_tensor_to_numpy(aug))
        ax.set_title(f"Aug {i+1}", fontsize=8)
        ax.axis("off")


def plot_mixup_cutmix(
    img1: Image.Image,
    img2: Image.Image,
    image_size: int,
    axes_iter,
) -> None:
    """Show MixUp and CutMix blends of two images at a few lambda values."""
    t1 = _to_tensor(img1, image_size)
    t2 = _to_tensor(img2, image_size)
    batch = torch.cat([t1, t2], dim=0)         # [2, 3, H, W]
    labels = torch.tensor([0, 1], dtype=torch.long)

    configs = [
        ("MixUp λ=0.5",  Mixup(mixup_alpha=1.0, cutmix_alpha=0.0, prob=1.0,
                                switch_prob=0.0, mode="elem", num_classes=5)),
        ("CutMix λ≈0.5", Mixup(mixup_alpha=0.0, cutmix_alpha=1.0, prob=1.0,
                                switch_prob=1.0, mode="elem", num_classes=5)),
        ("Mix/Cut switch", Mixup(mixup_alpha=0.4, cutmix_alpha=1.0, prob=1.0,
                                 switch_prob=0.5, mode="batch", num_classes=5)),
    ]

    # Show originals in this section
    for orig, title in [(img1, "Image 1"), (img2, "Image 2")]:
        ax = next(axes_iter)
        ax.imshow(orig.resize((image_size, image_size)))
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for title, fn in configs:
        mixed, _ = fn(batch.clone(), labels.clone())
        ax = next(axes_iter)
        ax.imshow(_tensor_to_numpy(mixed[0]))
        ax.set_title(title, fontsize=8)
        ax.axis("off")


def plot_wavelets(
    img: Image.Image,
    image_size: int,
    axes_iter,
    wavelet_name: str = "sym2",
    levels: int = 2,
) -> None:
    """Show wavelet sub-band channels (LL + detail bands) side by side."""
    tensor = _to_tensor(img, image_size)  # [1,3,H,W]

    transform = WaveletChannelTransform(
        wave=wavelet_name,
        mode="symmetric",
        J=levels,
        include_lowpass_channel=True,
    )
    with torch.no_grad():
        coeffs = transform(tensor)   # [1, 1+3*levels, H, W]

    band_names = ["LL"]
    for lvl in range(1, levels + 1):
        for sub in ["LH", "HL", "HH"]:
            band_names.append(f"{sub} L{lvl}")

    # RGB image for reference
    ax = next(axes_iter)
    ax.imshow(_tensor_to_numpy(tensor[0]))
    ax.set_title("RGB input", fontsize=8)
    ax.axis("off")

    for ch_idx, name in enumerate(band_names):
        ch = coeffs[0, ch_idx].numpy()   # [H, W]
        ax = next(axes_iter)
        ax.imshow(ch, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"Wavelet: {name}", fontsize=8)
        ax.axis("off")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect augmentation pipeline")
    parser.add_argument("--image",  required=True,       help="Path to primary sample image")
    parser.add_argument("--image2", default=None,        help="Second image for MixUp/CutMix demo")
    parser.add_argument("--size",   type=int, default=224, help="Display/resize resolution")
    parser.add_argument("--runs",   type=int, default=12,  help="Number of augmented views in grid")
    parser.add_argument("--show-wavelets", action="store_true",
                        help="Add wavelet sub-band visualisation panel")
    parser.add_argument("--wavelet",  default="sym2",     help="Wavelet family (default: sym2)")
    parser.add_argument("--wav-levels", type=int, default=2, help="DWT decomposition levels")
    parser.add_argument("--output",   default=None,
                        help="Output path (default: artifacts/augmentation_grid.png)")
    args = parser.parse_args()

    output_path = args.output or os.path.join(PROJECT_ROOT, "artifacts", "augmentation_grid.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img1 = _load_image(args.image)
    img2 = _load_image(args.image2) if args.image2 else None
    show_mixup = img2 is not None
    show_wavelets = args.show_wavelets

    # ── Decide layout ──────────────────────────────────────────────────────
    COLS = 5
    n_aug_cells   = 1 + args.runs                             # original + runs
    n_mixup_cells = 5 if show_mixup else 0                    # 2 originals + 3 blends
    n_wav_cells   = (1 + 1 + 3 * args.wav_levels) if show_wavelets else 0  # RGB + bands

    section_titles = []
    section_widths = []
    if n_aug_cells:
        section_titles.append("Augmentation pipeline  (RandAugment + RandomAffine + RandomErasing)")
        section_widths.append(n_aug_cells)
    if n_mixup_cells:
        section_titles.append("MixUp / CutMix")
        section_widths.append(n_mixup_cells)
    if n_wav_cells:
        section_titles.append(f"Wavelet channels  ({args.wavelet}, J={args.wav_levels})")
        section_widths.append(n_wav_cells)

    total_cells = n_aug_cells + n_mixup_cells + n_wav_cells
    rows = (total_cells + COLS - 1) // COLS

    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 3.2, rows * 3.2))
    axes = axes.flatten()
    axes_iter = iter(axes)

    # ── Sections ──────────────────────────────────────────────────────────
    print(f"Augmentation grid  : {args.runs} views at {args.size}×{args.size}")
    plot_augmentation_grid(img1, args.size, args.runs, axes_iter)

    if show_mixup:
        print("MixUp / CutMix demo: enabled")
        plot_mixup_cutmix(img1, img2, args.size, axes_iter)

    if show_wavelets:
        print(f"Wavelet channels   : {args.wavelet}, J={args.wav_levels}")
        plot_wavelets(img1, args.size, axes_iter, args.wavelet, args.wav_levels)

    # Hide unused axes
    for ax in axes_iter:
        ax.axis("off")

    # ── Section header annotations ─────────────────────────────────────────
    # Draw a thin coloured line above each section using figure text
    section_colours = ["#2196F3", "#4CAF50", "#FF9800"]
    cell = 0
    for sec_idx, (title, width) in enumerate(zip(section_titles, section_widths)):
        row_of_first = (cell // COLS)
        col_of_first = cell % COLS
        # Approximate x position in figure normalised coords
        x = (col_of_first + width / 2) / COLS
        y = 1.0 - row_of_first / rows
        fig.text(x, y, title,
                 ha="center", va="bottom",
                 fontsize=9, fontweight="bold",
                 color=section_colours[sec_idx % len(section_colours)])
        cell += width

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
