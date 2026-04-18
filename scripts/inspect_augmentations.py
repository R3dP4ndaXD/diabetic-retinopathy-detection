"""
Visual inspection script for the augmentation and frequency-transform pipelines.

Shows:
  - Grid of independently augmented versions (RandAugment, RandomAffine, flips, erasing)
  - MixUp / CutMix demo (two images blended)
  - Wavelet channel visualisation (LL, LH, HL, HH sub-bands)
  - Block-DCT channel visualisation (zigzag-ordered coefficient maps)
  - Fourier k-space channel visualisation (real, imaginary, magnitude — MRIQA-style)
  - Fourier high-pass filter visualisation (FFT → mask low-freq → IFFT → pixel-space edges)

Usage
-----
# Basic — single image, augmentation grid only
python scripts/inspect_augmentations.py --image path/to/sample.jpg

# Two images to also show MixUp / CutMix
python scripts/inspect_augmentations.py --image img1.jpg --image2 img2.jpg

# Show wavelet channels
python scripts/inspect_augmentations.py --image sample.jpg --show-wavelets

# Show DCT channels
python scripts/inspect_augmentations.py --image sample.jpg --show-dct

# Show Fourier k-space channels (MRIQA-style)
python scripts/inspect_augmentations.py --image sample.jpg --show-fourier

# Show Fourier high-pass filter (pixel-space, per-channel)
python scripts/inspect_augmentations.py --image sample.jpg --show-fourier-hpf

# Show all frequency transforms
python scripts/inspect_augmentations.py --image sample.jpg \\
    --show-wavelets --show-dct --show-fourier --show-fourier-hpf

# Full demo
python scripts/inspect_augmentations.py --image img1.jpg --image2 img2.jpg \\
    --show-wavelets --show-dct --show-fourier --show-fourier-hpf \\
    --wavelet sym2 --wav-levels 2 --dct-block-size 8 --dct-num-coeffs 6 \\
    --fourier-hpf-radius 0.1 --size 224
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

from src.model import (
    DCTChannelTransform,
    FourierChannelTransform,
    FourierHighPassTransform,
    WaveletChannelTransform,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_visual_transform(image_size: int) -> T.Compose:
    """Full train pipeline WITHOUT Normalize so output stays in [0,1]."""
    return T.Compose([
        T.Resize((image_size, image_size), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomAffine(degrees=(0, 360), scale=(0.9, 1.1), shear=(-11, 11),
                       fill=(128, 128, 128)),
        #T.RandAugment(num_ops=2, magnitude=9, fill=(128, 128, 128)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomErasing(p=0.25, scale=(0.02, 0.1), value=0.5),
    ])


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """[C,H,W] float32 → [H,W,C] clipped to [0,1]."""
    return t.permute(1, 2, 0).clamp(0, 1).numpy()


def _load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def _to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    """PIL → [1,3,H,W] float32 in [0,1], resized to size×size."""
    t = T.Compose([
        T.Resize((size, size), antialias=True),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])
    return t(img).unsqueeze(0)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def plot_augmentation_grid(img: Image.Image, image_size: int, runs: int,
                           axes_iter) -> None:
    """Original + N independently augmented views."""
    transform = _build_visual_transform(image_size)

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


def plot_mixup_cutmix(img1: Image.Image, img2: Image.Image, image_size: int,
                      axes_iter) -> None:
    """MixUp and CutMix blends of two images."""
    t1 = _to_tensor(img1, image_size)
    t2 = _to_tensor(img2, image_size)
    batch  = torch.cat([t1, t2], dim=0)
    labels = torch.tensor([0, 1], dtype=torch.long)

    configs = [
        ("MixUp λ=0.5",   Mixup(mixup_alpha=1.0, cutmix_alpha=0.0, prob=1.0,
                                  switch_prob=0.0, mode="elem", num_classes=5)),
        ("CutMix λ≈0.5",  Mixup(mixup_alpha=0.0, cutmix_alpha=1.0, prob=1.0,
                                  switch_prob=1.0, mode="elem", num_classes=5)),
        ("Mix/Cut switch", Mixup(mixup_alpha=0.4, cutmix_alpha=1.0, prob=1.0,
                                  switch_prob=0.5, mode="batch", num_classes=5)),
    ]

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


def plot_wavelets(img: Image.Image, image_size: int, axes_iter,
                  wavelet_name: str = "sym2", levels: int = 2) -> None:
    """Wavelet sub-band channels (LL + detail bands)."""
    tensor = _to_tensor(img, image_size)  # [1,3,H,W]
    transform = WaveletChannelTransform(wave=wavelet_name, mode="symmetric",
                                        J=levels, include_lowpass_channel=True)
    with torch.no_grad():
        coeffs = transform(tensor)  # [1, 1+3*levels, H, W]

    band_names = ["LL"]
    for lvl in range(1, levels + 1):
        for sub in ["LH", "HL", "HH"]:
            band_names.append(f"{sub} L{lvl}")

    ax = next(axes_iter)
    ax.imshow(_tensor_to_numpy(tensor[0]))
    ax.set_title("RGB input", fontsize=8)
    ax.axis("off")

    for ch_idx, name in enumerate(band_names):
        ch = coeffs[0, ch_idx].numpy()
        ax = next(axes_iter)
        ax.imshow(ch, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"Wavelet: {name}", fontsize=8)
        ax.axis("off")


def plot_dct(img: Image.Image, image_size: int, axes_iter,
             block_size: int = 8, num_coeffs: int = 6) -> None:
    """Block-DCT coefficient maps in zigzag order."""
    tensor = _to_tensor(img, image_size)  # [1,3,H,W]
    transform = DCTChannelTransform(block_size=block_size, num_coeffs=num_coeffs)
    with torch.no_grad():
        coeffs = transform(tensor)  # [1, num_coeffs, H/block_size, W/block_size]

    # Build zigzag position labels (row, col) for each coefficient
    N = block_size
    zigzag_pos = []
    for s in range(2 * N - 1):
        if s % 2 == 0:
            r, c = min(s, N - 1), s - min(s, N - 1)
            while r >= 0 and c < N:
                zigzag_pos.append((r, c))
                r -= 1; c += 1
        else:
            c, r = min(s, N - 1), s - min(s, N - 1)
            while c >= 0 and r < N:
                zigzag_pos.append((r, c))
                r += 1; c -= 1

    ax = next(axes_iter)
    ax.imshow(_tensor_to_numpy(tensor[0]))
    ax.set_title("RGB input", fontsize=8)
    ax.axis("off")

    for ch_idx in range(num_coeffs):
        ch = coeffs[0, ch_idx].numpy()
        r, c = zigzag_pos[ch_idx] if ch_idx < len(zigzag_pos) else (ch_idx, 0)
        ax = next(axes_iter)
        ax.imshow(ch, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"DCT ({r},{c})", fontsize=8)
        ax.axis("off")


def plot_fourier(img: Image.Image, image_size: int, axes_iter,
                 shift: bool = False) -> None:
    """
    Fourier k-space channel visualisation (MRIQA-style).

    Shows the three log-compressed k-space channels:
      Re channel  : sign(Re{F}) * log1p(|Re{F}|)
      Im channel  : sign(Im{F}) * log1p(|Im{F}|)
      Mag channel : log1p(|F|)
    """
    tensor = _to_tensor(img, image_size)  # [1,3,H,W]
    transform = FourierChannelTransform(shift=shift)
    with torch.no_grad():
        coeffs = transform(tensor)  # [1, 3, H, W]

    ax = next(axes_iter)
    ax.imshow(_tensor_to_numpy(tensor[0]))
    ax.set_title("RGB input", fontsize=8)
    ax.axis("off")

    channel_names = ["Re: sign·log1p|Re{F}|", "Im: sign·log1p|Im{F}|", "Mag: log1p|F|"]
    for ch_idx, name in enumerate(channel_names):
        ch = coeffs[0, ch_idx].numpy()
        ax = next(axes_iter)
        ax.imshow(ch, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"FFT {name}", fontsize=7)
        ax.axis("off")


def plot_fourier_hpf(img: Image.Image, image_size: int, axes_iter,
                     lpf_radius: float = 0.1) -> None:
    """
    Fourier high-pass filter visualisation (pixel-space, per-channel RGB).

    Applies FFT → zero-out low-freq circle of radius `lpf_radius * min(H,W)`
    → IFFT → real part.  The result highlights edges, vessels, and fine
    structures while suppressing bulk illumination gradients.
    """
    tensor = _to_tensor(img, image_size)  # [1,3,H,W]
    transform = FourierHighPassTransform(lpf_radius=lpf_radius)
    with torch.no_grad():
        filtered = transform(tensor)  # [1, 3, H, W]

    ax = next(axes_iter)
    ax.imshow(_tensor_to_numpy(tensor[0]))
    ax.set_title("RGB input", fontsize=8)
    ax.axis("off")

    for ch_idx, name in enumerate(["R-HPF", "G-HPF", "B-HPF"]):
        ch = filtered[0, ch_idx].numpy()
        ax = next(axes_iter)
        ax.imshow(ch, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"{name} (r={lpf_radius})", fontsize=7)
        ax.axis("off")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect augmentation pipeline")
    parser.add_argument("--image",   required=True,          help="Primary sample image")
    parser.add_argument("--image2",  default=None,           help="Second image for MixUp/CutMix")
    parser.add_argument("--size",    type=int, default=224,  help="Display resolution")
    parser.add_argument("--runs",    type=int, default=12,   help="Number of augmented views")
    parser.add_argument("--show-wavelets",      action="store_true", help="Show wavelet sub-bands")
    parser.add_argument("--wavelet",            default="sym2",      help="Wavelet family (default: sym2)")
    parser.add_argument("--wav-levels",         type=int, default=2, help="DWT decomposition levels")
    parser.add_argument("--show-dct",           action="store_true", help="Show block-DCT coefficient maps")
    parser.add_argument("--dct-block-size",     type=int, default=8, help="DCT block side length (default: 8)")
    parser.add_argument("--dct-num-coeffs",     type=int, default=6, help="Zigzag coefficients to show (default: 6)")
    parser.add_argument("--show-fourier",       action="store_true", help="Show Fourier k-space channels (MRIQA-style)")
    parser.add_argument("--fourier-shift",      action="store_true", help="Apply fftshift (DC at centre) for Fourier display")
    parser.add_argument("--show-fourier-hpf",   action="store_true", help="Show Fourier high-pass filtered image")
    parser.add_argument("--fourier-hpf-radius", type=float, default=0.1, help="Low-freq cutoff radius as fraction of min(H,W) (default: 0.1)")
    parser.add_argument("--output",   default=None, help="Output path (default: artifacts/augmentation_grid.png)")
    args = parser.parse_args()

    output_path = args.output or os.path.join(PROJECT_ROOT, "artifacts", "augmentation_grid.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img1 = _load_image(args.image)
    img2 = _load_image(args.image2) if args.image2 else None
    show_mixup       = img2 is not None
    show_wavelets    = args.show_wavelets
    show_dct         = args.show_dct
    show_fourier     = args.show_fourier
    show_fourier_hpf = args.show_fourier_hpf

    # ── Cell counts per section ────────────────────────────────────────────
    COLS = 5
    n_aug_cells         = 1 + args.runs
    n_mixup_cells       = 5 if show_mixup else 0
    n_wav_cells         = (1 + 1 + 3 * args.wav_levels) if show_wavelets else 0
    n_dct_cells         = (1 + args.dct_num_coeffs)     if show_dct      else 0
    n_fourier_cells     = 4 if show_fourier else 0      # RGB + 3 k-space channels
    n_fourier_hpf_cells = 4 if show_fourier_hpf else 0

    section_titles = []
    section_widths = []
    section_titles.append("Augmentation pipeline  (RandAugment + RandomAffine + RandomErasing)")
    section_widths.append(n_aug_cells)
    if n_mixup_cells:
        section_titles.append("MixUp / CutMix")
        section_widths.append(n_mixup_cells)
    if n_wav_cells:
        section_titles.append(f"Wavelet  ({args.wavelet}, J={args.wav_levels})")
        section_widths.append(n_wav_cells)
    if n_dct_cells:
        section_titles.append(f"Block DCT  ({args.dct_block_size}×{args.dct_block_size} blocks, {args.dct_num_coeffs} coeffs)")
        section_widths.append(n_dct_cells)
    if n_fourier_cells:
        shift_label = "shifted" if args.fourier_shift else "raw"
        section_titles.append(f"Fourier k-space  (Re, Im, Mag — {shift_label})")
        section_widths.append(n_fourier_cells)
    if n_fourier_hpf_cells:
        section_titles.append(f"Fourier HPF  (r={args.fourier_hpf_radius}, per-channel)")
        section_widths.append(n_fourier_hpf_cells)

    total_cells = n_aug_cells + n_mixup_cells + n_wav_cells + n_dct_cells + n_fourier_cells + n_fourier_hpf_cells
    rows = (total_cells + COLS - 1) // COLS

    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 3.2, rows * 3.2))
    axes = np.array(axes).flatten()
    axes_iter = iter(axes)

    # ── Render sections ────────────────────────────────────────────────────
    print(f"Augmentation grid : {args.runs} views at {args.size}×{args.size}")
    plot_augmentation_grid(img1, args.size, args.runs, axes_iter)

    if show_mixup:
        print("MixUp / CutMix    : enabled")
        plot_mixup_cutmix(img1, img2, args.size, axes_iter)

    if show_wavelets:
        print(f"Wavelet channels  : {args.wavelet}, J={args.wav_levels}")
        plot_wavelets(img1, args.size, axes_iter, args.wavelet, args.wav_levels)

    if show_dct:
        print(f"DCT channels      : block {args.dct_block_size}×{args.dct_block_size}, {args.dct_num_coeffs} coefficients")
        plot_dct(img1, args.size, axes_iter, args.dct_block_size, args.dct_num_coeffs)

    if show_fourier:
        shift_label = "shifted" if args.fourier_shift else "raw"
        print(f"Fourier k-space   : Re/Im/Mag channels ({shift_label})")
        plot_fourier(img1, args.size, axes_iter, shift=args.fourier_shift)

    if show_fourier_hpf:
        print(f"Fourier HPF       : radius={args.fourier_hpf_radius}, per-channel RGB")
        plot_fourier_hpf(img1, args.size, axes_iter, lpf_radius=args.fourier_hpf_radius)

    for ax in axes_iter:
        ax.axis("off")

    # ── Section header annotations ─────────────────────────────────────────
    section_colours = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    cell = 0
    for sec_idx, (title, width) in enumerate(zip(section_titles, section_widths)):
        row_of_first = cell // COLS
        col_of_first = cell % COLS
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
