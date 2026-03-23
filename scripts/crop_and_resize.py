import os
import sys
import argparse
from os import cpu_count

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.concurrent_task_executor import concurrent_task_executor
from src.utils import crop_and_pad_image, track_files

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


class FileInfo(NamedTuple):
    src: str
    dest: str
    size: Tuple[int, int]  # (width, height) tuple.


def crop_and_save_image(file_info: FileInfo):
    try:
        if os.path.exists(file_info.dest):
            return
        cropped_image = crop_and_pad_image(file_info.src, target_size=file_info.size)
        cropped_image.save(file_info.dest)
    except Exception as e:
        print(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    args = parser.parse_args()
    src_folder = args.src
    dst_folder = args.dest
    size = tuple(args.size)

    # check if destination folder exists
    if not os.path.exists(dst_folder):
        print("Destination folder does not exist. Creating folder...")
        os.makedirs(dst_folder, exist_ok=True)

    files = [
        FileInfo(
            src_image_path,
            os.path.join(dst_folder, os.path.basename(src_image_path)),
            size
        )
        for src_image_path in track_files(src_folder)
    ]

    if args.skip_existing:
        files = [file_info for file_info in files if not os.path.exists(file_info.dest)]

    print(f"Processing {len(files)} images with {args.workers} workers...")
    # cropping and resizing images and saving them to the destination folder
    concurrent_task_executor(
        crop_and_save_image,
        files,
        max_workers=args.workers,
        description="Processing images",
    )
