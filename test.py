import argparse
import csv
from os.path import basename
from pathlib import Path

import lightning as L
import torch

from src.data_module import DRDataModule
from src.model import DRModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the labeled test set."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/dr-model.ckpt",
        help="Path to the Lightning checkpoint.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/diabetic-retinopathy-dataset/test.csv",
        help="CSV with image_path and label columns for the test split.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size expected by the model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default="",
        help="Optional path to save per-image predictions CSV.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation (average predictions over multiple augmented views).",
    )
    parser.add_argument(
        "--tta-runs",
        type=int,
        default=5,
        help="Number of augmented views per image when TTA is enabled.",
    )
    return parser.parse_args()


def _resolve_test_csv_path(configured_path: str) -> str:
    path = Path(configured_path)
    if path.is_file():
        return str(path)

    candidate = Path("/data") / basename(configured_path)
    if candidate.is_file():
        return str(candidate)

    return configured_path


def main() -> None:
    args = parse_args()

    test_csv_path = Path(_resolve_test_csv_path(args.test_csv))
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not test_csv_path.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    dm = DRDataModule(
        train_csv_path=str(test_csv_path),  # required by constructor but unused for eval-only usage
        val_csv_path=str(test_csv_path),
        test_csv_path=str(test_csv_path),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    model = DRModel.load_from_checkpoint(str(checkpoint_path))

    # Enable TTA if requested
    if args.tta:
        model.tta_enabled = True
        model.tta_runs = args.tta_runs
        print(f"TTA enabled with {args.tta_runs} augmented views per image")

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
    )
    results = trainer.test(model, datamodule=dm)
    print(results)

    # Optionally save per-image predictions
    if args.predictions_csv:
        predictions_path = Path(args.predictions_csv)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)

        device = next(model.parameters()).device
        model.eval()

        all_rows: list[dict] = []
        with torch.inference_mode():
            for images, labels in dm.test_dataloader():
                images = images.to(device)

                if args.tta:
                    avg_probs = torch.zeros(images.size(0), model.num_classes, device=device)
                    for _ in range(args.tta_runs):
                        augmented = model._tta_transform(images)
                        logits = model(augmented)
                        avg_probs += torch.softmax(logits, dim=1)
                    avg_probs /= args.tta_runs
                    preds = torch.argmax(avg_probs, dim=1).cpu().tolist()
                else:
                    logits = model(images)
                    preds = torch.argmax(logits, dim=1).cpu().tolist()

                labels = labels.tolist()
                for pred, label in zip(preds, labels):
                    all_rows.append({"label": label, "prediction": pred})

        with predictions_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "prediction"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved per-image predictions to {predictions_path}")


if __name__ == "__main__":
    main()