import os
from datetime import datetime

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from zoneinfo import ZoneInfo


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
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
    sigmaX=10,
):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # 1. Background crop & Mask extraction
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No foreground found in image: {image_path}")
        
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = image[y : y + h, x : x + w]
    mask = binary[y : y + h, x : x + w]

    # 2. Pre-fill the background with 128 (Gray) BEFORE resizing to prevent the halo edge artifact
    cropped[mask == 0] = [128, 128, 128]

    # 3. Pad to square using 128 (Gray)
    squared_img = pad_to_square(cropped, pad_value=128)
    
    # We must also pad the binary mask to keep track of where the true background is
    squared_mask = pad_to_square(mask, pad_value=0)

    # 4. Resize safely to TARGET_SIZE (e.g., 512x512)
    resized_img = cv2.resize(squared_img, target_size, interpolation=cv2.INTER_AREA)
    # Use INTER_NEAREST for the mask to ensure it stays strictly binary (0 or 255) during resize
    resized_mask = cv2.resize(squared_mask, target_size, interpolation=cv2.INTER_NEAREST)

    # 5. Ben Graham Enhancement (NOW correctly happening at the standardized 512x512 resolution)
    enhanced = ben_graham_preprocessing(resized_img, sigmaX=sigmaX)
    
    # 6. Clean up any slight blur bleed by strictly enforcing the gray background again
    enhanced[resized_mask == 0] = [128, 128, 128]

    # 7. CLAHE on LAB's L channel
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    cl = clahe.apply(l)

        
    merged = cv2.merge((cl, a, b))
    lab_enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # 8. Mild Denoising
    denoised = cv2.medianBlur(lab_enhanced, 3)

    return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))

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

def build_run_tag(cfg) -> str:
    """Build a concise, human-readable run tag from the training config.

    Every meaningful field is always included so the tag stays accurate
    regardless of what the config defaults happen to be.

    Single-stream example : ``convnext_base_none_samp_ce_mx``
    Dual-stream example   : ``convnext_base__effb0_xattn_wav_samp_ce_nomx``
    """
    parts: list[str] = []

    # ── Backbone(s) ──────────────────────────────────────────────────────────
    if cfg.get("rgb_model_name"):
        wav = str(cfg.get("wav_model_name", ""))
        backbone = str(cfg.rgb_model_name)
        if wav and wav != cfg.rgb_model_name:
            backbone += f"__{wav}"
        parts.append(backbone)
        fusion_short = {"cross_attention": "xattn"}
        fusion = str(cfg.get("fusion_type", "mlp"))
        parts.append(fusion_short.get(fusion, fusion))
    else:
        parts.append(str(cfg.model_name))

    # ── Image size (only when explicitly set, otherwise model picks it) ──────
    if cfg.get("image_size"):
        parts.append(f"im{cfg.image_size}")

    # ── Frequency transform / stream ─────────────────────────────────────────
    freq_short = {"wavelet": "wav", "dct": "dct", "fourier": "fft", "fourier_hpf": "hpf", "none": "none"}
    freq = str(cfg.get("freq_stream") or cfg.get("freq_transform") or "none")
    parts.append(freq_short.get(freq, freq))

    # ── Class balancing ───────────────────────────────────────────────────────
    bal_short = {"naive_oversample": "over", "sampler": "samp", "weighted_loss": "wloss", "smote": "smote"}
    bal = str(cfg.get("balancing_mode", "sampler"))
    parts.append(bal_short.get(bal, bal))

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_short = {"cross_entropy": "ce", "focal": "focal"}
    loss = str(cfg.get("loss_name", "cross_entropy"))
    parts.append(loss_short.get(loss, loss))

    # ── Technique flags ───────────────────────────────────────────────────────
    parts.append("mx" if cfg.get("use_mixup", False) else "nomx")
    if cfg.get("tta_enabled", False):
        parts.append("tta")

    # ── Manual suffix: if run_tag is explicitly set in config, append it ─────
    override = str(cfg.get("run_tag", "")).strip()
    if override:
        parts.append(override.replace("-", "_"))

    return "_".join(parts)


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
