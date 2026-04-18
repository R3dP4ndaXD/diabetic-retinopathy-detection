import os
import sys
import argparse
from os import cpu_count

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.concurrent_task_executor import concurrent_task_executor
from src.utils import preprocess_image, track_files

from typing import NamedTuple, Tuple

parser = argparse.ArgumentParser(description="Crop and Resize Images in a folder")
parser.add_argument("--src", type=str, help="source folder", required=True)
parser.add_argument("--dest", type=str, help="destination folder", required=True)
parser.add_argument(
    "--size",
    type=int,
    nargs=2,
    metavar=("WIDTH", "HEIGHT"),
    help="Output size in pixels (width height).",
    default=(512, 512),
)
parser.add_argument(
    "--threshold",
    type=int,
    default=10,
    help="Threshold for background cropping mask.",
)
parser.add_argument(
    "--workers",
    type=int,
    default=min(8, cpu_count() or 1),
    help="Number of worker threads.",
)
parser.add_argument(
    "--skip-existing",
    action="store_true",
    help="Skip files that already exist in destination.",
)
parser.add_argument(
    "--clahe-clip-limit",
    type=float,
    default=2.0,
    help="CLAHE clip limit.",
)
parser.add_argument(
    "--clahe-tile-grid",
    type=int,
    nargs=2,
    metavar=("W", "H"),
    default=(8, 8),
    help="CLAHE tile grid size.",
)
parser.add_argument(
    "--sigmaX",
    type=float,
    default=10.0,
    help="Gaussian blur sigma for Ben Graham enhancement (default: 10).",
)


class FileInfo(NamedTuple):
    src: str
    dest: str
    size: Tuple[int, int]  # (width, height) tuple.
    threshold: int
    clahe_clip_limit: float
    clahe_tile_grid: Tuple[int, int]
    sigmaX: float = 10.0

def preprocess_and_save(file_info: FileInfo):
    try:
        # Added threshold parameter to match the updated pipeline
        result = preprocess_image(
            file_info.src,
            threshold=file_info.threshold,
            target_size=file_info.size,
            clahe_clip_limit=file_info.clahe_clip_limit,
            clahe_tile_grid_size=file_info.clahe_tile_grid,
            sigmaX=file_info.sigmaX,
        )
        result.save(file_info.dest)
    except Exception as e:
        print(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    args = parser.parse_args()
    src_folder = args.src
    dst_folder = args.dest
    size = tuple(args.size)

    # check if destination folder exists[cite: 3]
    if not os.path.exists(dst_folder):
        print("Destination folder does not exist. Creating folder...")
        os.makedirs(dst_folder, exist_ok=True)

    files = []
    for src_image_path in track_files(src_folder):
        # FIX: Recreate the exact relative folder structure
        rel_path = os.path.relpath(src_image_path, src_folder)
        dest_path = os.path.join(dst_folder, rel_path)
        
        files.append(
            FileInfo(
                src=src_image_path,
                dest=dest_path,
                size=size,
                threshold=args.threshold,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_tile_grid=tuple(args.clahe_tile_grid),
                sigmaX=args.sigmaX,
            )
        )

    if args.skip_existing:
        files = [file_info for file_info in files if not os.path.exists(file_info.dest)]

    print(f"Processing {len(files)} images with {args.workers} workers...")

    concurrent_task_executor(
        preprocess_and_save,
        files,
        max_workers=args.workers,
        description="Processing images",
    )