"""
Quick visual test for the preprocessing pipeline.

Processes sample images and saves a side-by-side comparison grid
(original → cropped → denoised → histogram-equalized → final resized)
plus prints Laplacian variance for each image.

Usage:
    python scripts/test_preprocessing.py
    # Then open artifacts/preprocessing_test.png
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

from src.utils import compute_laplacian_variance, preprocess_image


SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data", "sample")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "preprocessing_test.png")
TARGET_SIZE = (512, 512)
THRESHOLD = 10


def preprocessing_stages(image_path, threshold=THRESHOLD, target_size=TARGET_SIZE):
    """Run each preprocessing step individually and return intermediate results."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load: {image_path}")

    # Stage 1: Background crop
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No foreground: {image_path}")
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = image[y : y + h, x : x + w]

    # Stage 2: Gaussian blur
    blurred = cv2.GaussianBlur(cropped, (3, 3), 0)

    # Stage 3: Histogram equalization on Y channel
    yuv = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # Stage 3: CLAHE on Y channel
    # yuv = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    
    # Stage 4: Resize
    resized = cv2.resize(equalized, target_size, interpolation=cv2.INTER_AREA)

    # Convert all BGR → RGB for matplotlib
    def to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return {
        "Original": to_rgb(image),
        "Cropped": to_rgb(cropped),
        "Denoised": to_rgb(blurred),
        "Hist-Eq": to_rgb(equalized),
        "Final": to_rgb(resized),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample_files = sorted([
        os.path.join(SAMPLE_DIR, f)
        for f in os.listdir(SAMPLE_DIR)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    ])

    if not sample_files:
        print(f"No sample images found in {SAMPLE_DIR}")
        return

    # Use up to 4 images for a manageable grid
    # sample_files = sample_files[:4]
    stage_names = ["Original", "Cropped", "Denoised", "Hist-Eq", "Final"]

    n_images = len(sample_files)
    n_stages = len(stage_names)

    fig, axes = plt.subplots(
        n_images, n_stages, figsize=(4 * n_stages, 4 * n_images)
    )
    if n_images == 1:
        axes = axes[np.newaxis, :]

    for row, image_path in enumerate(sample_files):
        filename = os.path.basename(image_path)

        # Laplacian variance
        lap_var = compute_laplacian_variance(image_path)
        print(f"{filename}: Laplacian variance = {lap_var:.2f}")

        # Get stages
        stages = preprocessing_stages(image_path)

        for col, stage_name in enumerate(stage_names):
            ax = axes[row, col]
            ax.imshow(stages[stage_name])
            ax.axis("off")
            if row == 0:
                ax.set_title(stage_name, fontsize=14, fontweight="bold")
            if col == 0:
                ax.set_ylabel(filename, fontsize=10, rotation=0, labelpad=80, va="center")

    # Also verify that preprocess_image() produces a valid PIL image
    print("\n--- End-to-end preprocess_image() check ---")
    for image_path in sample_files:
        result = preprocess_image(image_path, target_size=TARGET_SIZE)
        assert isinstance(result, Image.Image), f"Expected PIL Image, got {type(result)}"
        assert result.size == TARGET_SIZE, f"Expected {TARGET_SIZE}, got {result.size}"
        print(f"  {os.path.basename(image_path)}: OK  (size={result.size}, mode={result.mode})")

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison grid to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
