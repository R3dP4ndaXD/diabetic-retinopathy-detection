"""
Visual inspection script for DRDataModule augmentations.

Usage:
    python scripts/inspect_augmentations.py --image path/to/sample.jpg
"""

import argparse
import os
import sys

# Ensure the project root is in the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import v2 as T

# Adjust this import based on exactly where you saved your DRDataModule
from src.data_module import DRDataModule 


def get_visual_transform(datamodule: DRDataModule):
    """
    Extracts the training pipeline from the DataModule but strips out 
    the Normalize step so the images remain in the visual [0, 1] range.
    """
    visual_transforms = []
    
    # Iterate through the composed transforms
    for transform in datamodule.train_transform.transforms:
        # Skip normalization so we can plot the raw colors
        if isinstance(transform, T.Normalize):
            continue
        visual_transforms.append(transform)
        
    return T.Compose(visual_transforms)


def main():
    parser = argparse.ArgumentParser(description="Inspect Data Augmentations")
    parser.add_argument("--image", type=str, required=True, help="Path to a sample preprocessed image.")
    parser.add_argument("--runs", type=int, default=15, help="Number of augmented versions to generate.")
    parser.add_argument("--size", type=int, default=512, help="Image size.")
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_ROOT, "artifacts")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "augmentation_grid.png")

    # 1. Load the image
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Could not find image at {args.image}")
    
    # PIL is the native format expected before ToImage() and ToDtype()
    original_img = Image.open(args.image).convert("RGB")

    # 2. Instantiate the DataModule (we only need it for its transform pipeline)
    # We pass dummy paths for the CSVs since we aren't calling setup()
    dm = DRDataModule(
        train_csv_path="dummy.csv",
        val_csv_path="dummy.csv",
        image_size=args.size,
        normalization_mode="dataset_by_size" # Or whatever your default is
    )

    # 3. Get the cleanly viewable transform pipeline
    visual_transform = get_visual_transform(dm)

    # 4. Set up the Matplotlib Grid (1 Original + N Augmentations)
    total_images = args.runs + 1
    cols = 4
    rows = (total_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    # Plot Original
    axes[0].imshow(original_img)
    axes[0].set_title("Original", fontweight="bold")
    axes[0].axis("off")

    print(f"Applying augmentations {args.runs} times...")

    # Plot Augmentations
    for i in range(1, total_images):
        # Apply the transform
        # The pipeline outputs a [C, H, W] float32 tensor in the [0.0, 1.0] range
        aug_tensor = visual_transform(original_img)
        
        # Convert from [C, H, W] to [H, W, C] for Matplotlib
        aug_img = aug_tensor.permute(1, 2, 0).numpy()
        
        axes[i].imshow(aug_img)
        axes[i].set_title(f"Augmentation {i}")
        axes[i].axis("off")

    # Turn off any unused axes
    for i in range(total_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Successfully saved augmentation grid to: {output_path}")

if __name__ == "__main__":
    main()