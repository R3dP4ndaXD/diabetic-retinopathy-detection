import argparse
import csv
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
        default=64,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default="",
        help="Optional path to save per-image predictions CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    test_csv_path = Path(args.test_csv)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not test_csv_path.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    # Build a data module with the test split
    dm = DRDataModule(
        train_csv_path=str(test_csv_path),  # needed for setup but unused
        val_csv_path=str(test_csv_path),     # needed for setup but unused
        test_csv_path=str(test_csv_path),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    model = DRModel.load_from_checkpoint(str(checkpoint_path))

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