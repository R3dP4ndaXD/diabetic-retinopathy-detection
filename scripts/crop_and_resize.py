import os
import sys
import argparse
from os import cpu_count

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.concurrent_task_executor import concurrent_task_executor
from src.utils import compute_laplacian_variance, preprocess_image, track_files

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
    "--filter-blurry",
    action="store_true",
    help="Filter out blurry/corrupted images using the Laplacian variance.",
)
parser.add_argument(
    "--laplacian-low",
    type=float,
    default=10.0,
    help="Minimum Laplacian variance — images below this are too blurry.",
)
parser.add_argument(
    "--laplacian-high",
    type=float,
    default=10000.0,
    help="Maximum Laplacian variance — images above this are likely corrupted.",
)
parser.add_argument(
    "--filtered-log",
    type=str,
    default="",
    help="Path to write filtered-out image paths (one per line).",
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
    filter_blurry: bool
    laplacian_low: float
    laplacian_high: float
    clahe_clip_limit: float
    clahe_tile_grid: Tuple[int, int]
    sigmaX: float = 10.0


# Shared list for filtered-out paths (written at the end, not per-worker)[cite: 3]
_filtered_out: list[str] = []


def preprocess_and_save(file_info: FileInfo):
    try:
        # Laplacian quality filter (on the raw source image)[cite: 3]
        if file_info.filter_blurry:
            variance = compute_laplacian_variance(file_info.src)
            if variance < file_info.laplacian_low or variance > file_info.laplacian_high:
                _filtered_out.append(file_info.src)
                return

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
                filter_blurry=args.filter_blurry,
                laplacian_low=args.laplacian_low,
                laplacian_high=args.laplacian_high,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_tile_grid=tuple(args.clahe_tile_grid),
                sigmaX=args.sigmaX,
            )
        )

    if args.skip_existing:
        files = [file_info for file_info in files if not os.path.exists(file_info.dest)]

    print(f"Processing {len(files)} images with {args.workers} workers...")
    if args.filter_blurry:
        print(f"Laplacian filter enabled: keeping variance in [{args.laplacian_low}, {args.laplacian_high}]")

    results = concurrent_task_executor(
        preprocess_and_save,
        files,
        max_workers=args.workers,
        description="Processing images",
    )
    _filtered_out = [res for res in results if res is not None]

    if args.filter_blurry and _filtered_out:
        print(f"Filtered out {len(_filtered_out)} images")
        if args.filtered_log:
            with open(args.filtered_log, "w") as f:
                f.write("\n".join(_filtered_out) + "\n")
            print(f"Wrote filtered paths to {args.filtered_log}")