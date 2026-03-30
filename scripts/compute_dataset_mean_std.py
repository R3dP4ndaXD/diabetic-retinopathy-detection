import argparse
import csv
import json
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def iter_images_from_dir(data_dir: str, extensions: Sequence[str]) -> Iterable[str]:
    exts = tuple(ext.lower() for ext in extensions)
    for root, _, files in os.walk(data_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                yield os.path.join(root, name)


def iter_images_from_csv(
    csv_path: str,
    path_column: str,
    base_dir: Optional[str] = None,
) -> Iterable[str]:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if path_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{path_column}' not found in CSV. Available columns: {reader.fieldnames}"
            )

        for row in reader:
            raw_path = row[path_column].strip()
            if not raw_path:
                continue

            if os.path.isabs(raw_path):
                yield raw_path
            elif base_dir:
                yield os.path.join(base_dir, raw_path)
            else:
                yield raw_path


def compute_mean_std(
    image_paths: Iterable[str],
    resize: Optional[Tuple[int, int]] = None,
    max_images: Optional[int] = None,
    report_every: int = 1000,
) -> dict:
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    n_total = 0
    n_ok = 0
    n_skipped = 0

    for path in image_paths:
        n_total += 1
        if max_images is not None and n_ok >= max_images:
            break

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            n_skipped += 1
            continue

        if resize is not None:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)

        # Convert BGR -> RGB and scale to [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        pixels = img.reshape(-1, 3)
        channel_sum += pixels.sum(axis=0)
        channel_sum_sq += np.square(pixels).sum(axis=0)
        total_pixels += pixels.shape[0]
        n_ok += 1

        if report_every > 0 and n_ok % report_every == 0:
            print(f"Processed {n_ok} images...")

    if n_ok == 0 or total_pixels == 0:
        raise RuntimeError("No valid images were processed.")

    mean = channel_sum / total_pixels
    var = channel_sum_sq / total_pixels - np.square(mean)
    std = np.sqrt(np.maximum(var, 0.0))

    return {
        "num_images_seen": n_total,
        "num_images_processed": n_ok,
        "num_images_skipped": n_skipped,
        "total_pixels": int(total_pixels),
        "mean_rgb": mean.tolist(),
        "std_rgb": std.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-channel RGB mean/std for a dataset (values in [0, 1])."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-dir", type=str, help="Directory with images (recursive).")
    group.add_argument("--csv-path", type=str, help="CSV containing image paths.")

    parser.add_argument(
        "--path-column",
        type=str,
        default="image_path",
        help="CSV column that contains image paths (used with --csv-path).",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Optional base directory for relative CSV paths.",
    )

    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="File extensions to include when using --data-dir.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional resize before stats (useful if training always resizes).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for quick estimates.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=5000,
        help="Print progress every N processed images (0 to disable).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write results as JSON.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.data_dir:
        image_paths = iter_images_from_dir(args.data_dir, args.extensions)
    else:
        image_paths = iter_images_from_csv(args.csv_path, args.path_column, args.base_dir)

    resize = tuple(args.resize) if args.resize is not None else None

    results = compute_mean_std(
        image_paths=image_paths,
        resize=resize,
        max_images=args.max_images,
        report_every=args.report_every,
    )

    print("Dataset statistics (RGB, [0, 1]):")
    print(f"  images_seen      : {results['num_images_seen']}")
    print(f"  images_processed : {results['num_images_processed']}")
    print(f"  images_skipped   : {results['num_images_skipped']}")
    print(f"  mean_rgb         : {results['mean_rgb']}")
    print(f"  std_rgb          : {results['std_rgb']}")

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
