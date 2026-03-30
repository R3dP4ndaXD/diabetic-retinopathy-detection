import os
from datetime import datetime

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from zoneinfo import ZoneInfo


def compute_laplacian_variance(image_path, crop_threshold=10):
    """Compute the Laplacian variance of an image (edge/sharpness measure).

    The black background is cropped first so the retina-to-background edge
    doesn't dominate the variance.

    Args:
        image_path (str): Path to the input image file.
        crop_threshold (int): Binary threshold for background removal.

    Returns:
        float: Variance of the Laplacian. Low = blurry, very high = corrupted.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Crop out the black background before computing the Laplacian
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, crop_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        gray = gray[y : y + h, x : x + w]

    return cv2.Laplacian(gray, cv2.CV_64F).var()


def pad_to_square(image, pad_value=0):
    """Pads an image with a solid color to make it square, preserving aspect ratio."""
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    # Calculate padding amounts
    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left
    
    # Pad with black (0 by default)
    return cv2.copyMakeBorder(image, top, bottom, left, right, 
                              cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value])


def ben_graham_preprocessing(image, sigmaX=10):
    """Applies Ben Graham's blending natively to all 3 color channels."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX)
    return cv2.addWeighted(image, 4, blurred, -4, 128)


def preprocess_image(
    image_path,
    threshold=10,
    target_size=(512, 512),
    use_clahe=True,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
):
    """Preprocess a retinal fundus image with the following pipeline:

    1. Background crop via binary thresholding + contour bounding box
    2. Pad to square to preserve aspect ratio
    3. Resize to target_size
    4. Ben Graham enhancement (local average color subtraction)
    5. Luminance contrast enhancement on LAB's L channel (CLAHE or global hist-eq)
    6. Median blur (3x3) for mild denoising

    Args:
        image_path (str): Path to the input image file.
        threshold (int): Binary threshold for background segmentation.
        target_size (tuple): Target (width, height) for the output image.
        use_clahe (bool): If True, use CLAHE. Otherwise, use global equalization.
        clahe_clip_limit (float): CLAHE clip limit.
        clahe_tile_grid_size (tuple): CLAHE tile grid size.

    Returns:
        PIL.Image.Image: Preprocessed image ready to be saved.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # 1. Background crop
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No foreground found in image: {image_path}")
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = image[y : y + h, x : x + w]

    # 2. Pad to a perfect square to MAINTAIN ASPECT RATIO
    squared = pad_to_square(cropped, pad_value=0)

    # 3. Resize safely (No distortion since the image is already square)
    resized = cv2.resize(squared, target_size, interpolation=cv2.INTER_AREA)

    # 4. Ben Graham Enhancement
    enhanced = ben_graham_preprocessing(resized)

    # 5. Luminance enhancement on LAB color space
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size,
        )
        cl = clahe.apply(l)
    else:
        # Fallback to global equalization if CLAHE is disabled
        cl = cv2.equalizeHist(l)
        
    merged = cv2.merge((cl, a, b))
    lab_enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # 6. Mild Denoising
    denoised = cv2.medianBlur(lab_enhanced, 3)

    # Convert BGR → RGB → PIL
    rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def track_files(folder_path, extensions=(".jpg", ".jpeg", ".png")):
    """
    Track all the files in a folder and its subfolders.

    Args:
        folder_path (str): The path of the folder to track files in.
        extensions (tuple, optional): Tuple of file extensions to track. Default is ('.jpg', '.jpeg', '.png').

    Returns:
        list: A list containing the paths of all files in the folder and its subfolders.
    """
    # Validate folder_path
    if not os.path.isdir(folder_path):
        raise ValueError("Invalid folder path provided.")

    # Convert extensions to lowercase for case-insensitive comparison
    extensions = tuple(ext.lower() for ext in extensions)

    # Initialize file_list
    file_list = []

    # Walk through the folder and its subfolders
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            _, extension = os.path.splitext(file_path)
            # Check if the file extension is in the list of extensions
            if extension.lower() in extensions:
                file_list.append(file_path)

    return file_list


def crop_circle_roi(image_path):
    """
    Crop the circular Region of Interest (ROI) from a fundus image.

    Args:
    - image_path (str): Path to the fundus image.

    Returns:
    - cropped_roi (numpy.ndarray): The cropped circular Region of Interest.
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to binarize the image
    _, thresholded_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Assuming the largest contour corresponds to the ROI
    contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Crop the circular ROI using the bounding rectangle
    cropped_roi = image[y : y + h, x : x + w]

    return cropped_roi


def plot_image_grid(image_paths, roi_crop=False):
    """
    Create a grid plot with a maximum of 16 images.

    Args:
    - image_paths (list): A list of image paths to be plotted.

    Returns:
    - None
    """
    num_images = min(len(image_paths), 16)
    num_rows = (num_images - 1) // 4 + 1
    fig, axes = plt.subplots(num_rows, 4, figsize=(12, 3 * num_rows))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            if roi_crop:
                img = crop_and_pad_image(image_paths[i])
            else:
                img = mpimg.imread(image_paths[i])
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def generate_run_id(zone: ZoneInfo = ZoneInfo("Europe/Bucharest")) -> str:
    """Generate a unique run ID using current UTC date and time.

    Args:
        zone (ZoneInfo, optional): Timezone information. Defaults to Europe/Bucharest.

    Returns:
        str: A unique run ID in the format 'run-YYYY-MM-DD-HH-MM-SS'.
    """
    try:
        now = datetime.now(tz=zone)
        formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        return f"run-{formatted_time}"
    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error generating run ID: {e}")
        return None  # Or raise an exception if appropriate


if __name__ == "__main__":
    print(generate_run_id())
