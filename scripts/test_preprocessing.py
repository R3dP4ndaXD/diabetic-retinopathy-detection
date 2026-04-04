"""
Quick visual test for the preprocessing pipeline.

Processes sample images and saves a side-by-side comparison grid
(Original → Squared & Resized → Ben Graham → CLAHE → Final)
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


def pad_to_square(image, pad_value=0):
    """Pads an image with a solid color to make it square, preserving aspect ratio."""
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left
    
    return cv2.copyMakeBorder(image, top, bottom, left, right, 
                              cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value])

def ben_graham_preprocessing_step(image, sigmaX=10):
    """Applies Ben Graham's blending natively to all 3 color channels."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX)
    return cv2.addWeighted(image, 4, blurred, -4, 128)

def preprocessing_stages(image_path, threshold=THRESHOLD, target_size=TARGET_SIZE):
    """Run each preprocessing step individually and return intermediate results."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load: {image_path}")

    # 1. Background crop & Pristine Mask extraction
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_gray = cv2.medianBlur(gray, 7)
    _, binary = cv2.threshold(blurred_gray, threshold, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No foreground found in image: {image_path}")
        
    main_contour = max(contours, key=cv2.contourArea)
    full_mask = np.zeros_like(gray)
    cv2.drawContours(full_mask, [main_contour], -1, 255, thickness=cv2.FILLED)
    
    x, y, w, h = cv2.boundingRect(main_contour)
    cropped = image[y : y + h, x : x + w].copy()
    mask = full_mask[y : y + h, x : x + w]

    # 2. Pre-fill the background with 128 (Gray) BEFORE resizing
    cropped[mask == 0] = [128, 128, 128]

    # 3. Pad to square using 128 (Gray)
    squared_img = pad_to_square(cropped, pad_value=128)
    squared_mask = pad_to_square(mask, pad_value=0)

    # 4. Resize safely to TARGET_SIZE (e.g., 512x512)
    resized_img = cv2.resize(squared_img, target_size, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(squared_mask, target_size, interpolation=cv2.INTER_NEAREST)

    # 5. Ben Graham Enhancement
    enhanced = ben_graham_preprocessing_step(resized_img, sigmaX=10)
    
    # 6. Clean up any slight blur bleed by strictly enforcing the gray background again
    enhanced[resized_mask == 0] = [128, 128, 128]

    # 7. CLAHE on LAB's L channel
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    cl = clahe.apply(l)
        
    merged = cv2.merge((cl, a, b))
    lab_enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # 8. Mild Denoising
    denoised = cv2.medianBlur(lab_enhanced, 3)

    # Convert all BGR → RGB for matplotlib
    def to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return {
        "Original": to_rgb(image),
        "Squared & Resized": to_rgb(resized_img),
        "Ben Graham": to_rgb(enhanced),
        "CLAHE": to_rgb(lab_enhanced),
        "Final": to_rgb(denoised),
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

    stage_names = ["Original", "Squared & Resized", "Ben Graham", "CLAHE", "Final"]

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

    # Verify that preprocess_image() produces a valid PIL image
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