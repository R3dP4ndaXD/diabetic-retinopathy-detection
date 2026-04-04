"""
Visual inspection script for a Grayscale-First Preprocessing Pipeline.

Usage:
    python scripts/inspect_grayscale_pipeline.py
    # Then open artifacts/grayscale_pipeline_test.png
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data", "sample")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "grayscale_pipeline_test.png")
TARGET_SIZE = (512, 512)
THRESHOLD = 10


def pad_to_square_gray(image, pad_value=0):
    """Pads a 1-channel grayscale image to make it square."""
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left
    
    return cv2.copyMakeBorder(image, top, bottom, left, right, 
                              cv2.BORDER_CONSTANT, value=pad_value)


def preprocessing_stages_gray(image_path, threshold=THRESHOLD, target_size=TARGET_SIZE):
    """Run the pipeline assuming Grayscale conversion happens at step 1."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load: {image_path}")

    # 1. IMMEDIATE GRAYSCALE CONVERSION
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Pristine Mask extraction (using the gray image)
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
    
    # 3. Crop 
    cropped = gray[y : y + h, x : x + w].copy()
    mask = full_mask[y : y + h, x : x + w]

    # Pre-fill the background with 128 (50% Gray) 
    cropped[mask == 0] = 128

    # 4. Pad to square using 128
    squared_img = pad_to_square_gray(cropped, pad_value=128)
    squared_mask = pad_to_square_gray(mask, pad_value=0)

    # 5. Resize safely
    resized_img = cv2.resize(squared_img, target_size, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(squared_mask, target_size, interpolation=cv2.INTER_NEAREST)

    # 6. Ben Graham Enhancement (Directly on 1D grayscale array)
    blurred_for_bg = cv2.GaussianBlur(resized_img, (0, 0), 10)
    enhanced = cv2.addWeighted(resized_img, 4, blurred_for_bg, -4, 128)
    
    # Clean up mask edges
    enhanced[resized_mask == 0] = 128

    # 7. CLAHE (Directly on 1D grayscale array — no LAB conversion needed!)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    cl = clahe.apply(enhanced)

    # 8. Mild Denoising
    denoised = cv2.medianBlur(cl, 3)

    # Helper to convert 1D grayscale back to 3D RGB solely for matplotlib to display it properly
    def to_rgb_plot(img_gray):
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    return {
        "Original (RGB)": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        "Original (Gray)": to_rgb_plot(gray),
        "Squared & Resized": to_rgb_plot(resized_img),
        "Ben Graham": to_rgb_plot(enhanced),
        "CLAHE (Final)": to_rgb_plot(denoised),
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

    stage_names = ["Original (RGB)", "Original (Gray)", "Squared & Resized", "Ben Graham", "CLAHE (Final)"]

    n_images = len(sample_files)
    n_stages = len(stage_names)

    fig, axes = plt.subplots(n_images, n_stages, figsize=(4 * n_stages, 4 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    for row, image_path in enumerate(sample_files):
        filename = os.path.basename(image_path)
        stages = preprocessing_stages_gray(image_path)

        for col, stage_name in enumerate(stage_names):
            ax = axes[row, col]
            ax.imshow(stages[stage_name])
            ax.axis("off")
            if row == 0:
                ax.set_title(stage_name, fontsize=14, fontweight="bold")
            if col == 0:
                ax.set_ylabel(filename, fontsize=10, rotation=0, labelpad=80, va="center")

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved Grayscale comparison grid to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()