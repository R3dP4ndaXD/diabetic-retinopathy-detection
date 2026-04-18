from __future__ import annotations

import argparse
import csv
from os.path import basename
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_module import DRDataModule
from src.model import DRModel

_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


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
        "--save-probs",
        action="store_true",
        help="When --predictions-csv is used, also save per-class probability columns.",
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
    parser.add_argument(
        "--mc-dropout",
        action="store_true",
        help="Enable Monte Carlo Dropout uncertainty estimation.",
    )
    parser.add_argument(
        "--mc-dropout-runs",
        type=int,
        default=20,
        help="Number of stochastic forward passes per image when --mc-dropout is enabled.",
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


def _enable_dropout_only(model: nn.Module) -> None:
    """Keep model in eval mode except dropout layers, used for MC Dropout."""
    model.eval()
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES):
            module.train()


def _mc_dropout_forward(
    model: DRModel,
    images: torch.Tensor,
    runs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probs_runs: list[torch.Tensor] = []
    for _ in range(runs):
        logits = model(images)
        probs_runs.append(torch.softmax(logits, dim=1))

    stacked = torch.stack(probs_runs, dim=0)  # [R, B, C]
    mean_probs = stacked.mean(dim=0)
    var_probs = stacked.var(dim=0, unbiased=False)

    entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-8))).sum(dim=1)
    variance_mean = var_probs.mean(dim=1)
    confidence = mean_probs.max(dim=1).values

    return mean_probs, entropy, variance_mean, confidence


def main() -> None:
    args = parse_args()

    if args.mc_dropout and args.tta:
        raise ValueError("Use either --tta or --mc-dropout, not both at the same time.")

    test_csv_path = Path(_resolve_test_csv_path(args.test_csv))
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not test_csv_path.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    model = DRModel.load_from_checkpoint(str(checkpoint_path))

    # Rebuild normalization to match training: read timm data_config from
    # the same backbone that was used during training.
    from src.models.factory import ModelFactory

    tmp_model = ModelFactory(name=model.hparams.model_name, num_classes=model.num_classes)()
    timm_data_config = tmp_model.data_config
    del tmp_model

    dm = DRDataModule(
        train_csv_path=str(test_csv_path),  # required by constructor but unused for eval-only usage
        val_csv_path=str(test_csv_path),
        test_csv_path=str(test_csv_path),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalization_mode="timm",
        timm_data_config=timm_data_config,
    )
    dm.setup()

    if args.mc_dropout:
        print(f"MC Dropout enabled with {args.mc_dropout_runs} stochastic passes")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        _enable_dropout_only(model)

        y_true_all: list[int] = []
        y_pred_all: list[int] = []
        all_rows: list[dict] = []

        with torch.inference_mode():
            for images, labels in dm.test_dataloader():
                images = images.to(device)
                labels = labels.to(device)

                probs, entropy, variance_mean, confidence = _mc_dropout_forward(
                    model=model,
                    images=images,
                    runs=args.mc_dropout_runs,
                )
                preds = probs.argmax(dim=1)

                y_true = labels.cpu().tolist()
                y_pred = preds.cpu().tolist()
                y_true_all.extend(y_true)
                y_pred_all.extend(y_pred)

                if args.predictions_csv:
                    probs_cpu = probs.cpu().numpy()
                    for i, (label, pred) in enumerate(zip(y_true, y_pred)):
                        row = {
                            "label": label,
                            "prediction": pred,
                            "confidence": float(confidence[i].cpu()),
                            "uncertainty_entropy": float(entropy[i].cpu()),
                            "uncertainty_variance_mean": float(variance_mean[i].cpu()),
                        }
                        if args.save_probs:
                            for c in range(model.num_classes):
                                row[f"prob_{c}"] = float(probs_cpu[i, c])
                        all_rows.append(row)

        acc = accuracy_score(y_true_all, y_pred_all)
        kappa = cohen_kappa_score(y_true_all, y_pred_all, weights="quadratic")
        prec = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
        rec = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)
        f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)

        report = classification_report(
            y_true_all,
            y_pred_all,
            labels=sorted(set(y_true_all) | set(y_pred_all)),
            target_names=[f"class_{i}" for i in sorted(set(y_true_all) | set(y_pred_all))],
            digits=4,
            zero_division=0,
        )

        print(
            [
                {
                    "test_acc": acc,
                    "test_kappa": kappa,
                    "test_precision": prec,
                    "test_recall": rec,
                    "test_f1": f1,
                    "mc_dropout_runs": args.mc_dropout_runs,
                }
            ]
        )
        print("\nTest Classification Report:\n")
        print(report)

        if args.predictions_csv:
            predictions_path = Path(args.predictions_csv)
            predictions_path.parent.mkdir(parents=True, exist_ok=True)

            fieldnames = [
                "label",
                "prediction",
                "confidence",
                "uncertainty_entropy",
                "uncertainty_variance_mean",
            ]
            if args.save_probs:
                fieldnames.extend([f"prob_{c}" for c in range(model.num_classes)])

            with predictions_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"Saved per-image MC Dropout predictions to {predictions_path}")

        return

    # Non-MC path keeps the existing Lightning test loop.
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
                    probs = avg_probs / args.tta_runs
                else:
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)

                preds = torch.argmax(probs, dim=1).cpu().tolist()
                labels_list = labels.tolist()
                probs_np = probs.cpu().numpy()

                for i, (pred, label) in enumerate(zip(preds, labels_list)):
                    row = {
                        "label": label,
                        "prediction": pred,
                        "confidence": float(np.max(probs_np[i])),
                    }
                    if args.save_probs:
                        for c in range(model.num_classes):
                            row[f"prob_{c}"] = float(probs_np[i, c])
                    all_rows.append(row)

        fieldnames = ["label", "prediction", "confidence"]
        if args.save_probs:
            fieldnames.extend([f"prob_{c}" for c in range(model.num_classes)])

        with predictions_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved per-image predictions to {predictions_path}")


if __name__ == "__main__":
    main()
