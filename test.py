import argparse
import csv
from os.path import basename
from pathlib import Path

import lightning as L
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)
import torch

from src.data_module import DRDataModule
from src.ensemble import EnsemblePredictor
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
        "--ensemble-checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of checkpoints for weighted soft-voting ensemble inference.",
    )
    parser.add_argument(
        "--ensemble-weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional weights for ensemble checkpoints (same length as --ensemble-checkpoints).",
    )
    parser.add_argument(
        "--tune-ensemble-weights",
        action="store_true",
        help="Tune ensemble weights on validation CSV to maximize quadratic kappa.",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="data/diabetic-retinopathy-dataset/val.csv",
        help="Validation CSV used when --tune-ensemble-weights is enabled.",
    )
    parser.add_argument(
        "--weight-search-trials",
        type=int,
        default=300,
        help="Number of random weight vectors to evaluate during tuning.",
    )
    parser.add_argument(
        "--weight-search-seed",
        type=int,
        default=42,
        help="Random seed for ensemble weight search.",
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


def _build_eval_datamodule(csv_path: Path, args: argparse.Namespace) -> DRDataModule:
    dm = DRDataModule(
        train_csv_path=str(csv_path),  # required by constructor but unused for eval-only usage
        val_csv_path=str(csv_path),
        test_csv_path=str(csv_path),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()
    return dm


def _collect_member_probabilities(
    checkpoint_paths: list[Path],
    dataloader,
    tta: bool,
    tta_runs: int,
    device: torch.device,
):
    all_member_probs = []
    y_true_ref = None

    for checkpoint_path in checkpoint_paths:
        model = DRModel.load_from_checkpoint(str(checkpoint_path)).to(device)
        model.eval()

        probs_batches = []
        labels_batches = []

        with torch.inference_mode():
            for images, labels in dataloader:
                images = images.to(device)
                if tta:
                    avg_probs = torch.zeros(images.size(0), model.num_classes, device=device)
                    for _ in range(tta_runs):
                        augmented = model._tta_transform(images)
                        logits = model(augmented)
                        avg_probs += torch.softmax(logits, dim=1)
                    probs = avg_probs / tta_runs
                else:
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)

                probs_batches.append(probs.cpu())
                labels_batches.append(labels.cpu())

        member_probs = torch.cat(probs_batches, dim=0)
        labels_cat = torch.cat(labels_batches, dim=0)

        if y_true_ref is None:
            y_true_ref = labels_cat
        elif not torch.equal(y_true_ref, labels_cat):
            raise RuntimeError("Validation labels mismatch across ensemble members.")

        all_member_probs.append(member_probs)

    return torch.stack(all_member_probs, dim=0), y_true_ref


def _tune_weights_from_cached_probs(
    member_probs: torch.Tensor,
    y_true: torch.Tensor,
    trials: int,
    seed: int,
):
    num_models = member_probs.shape[0]
    rng = np.random.default_rng(seed)

    candidates = [np.full(num_models, 1.0 / num_models, dtype=np.float32)]
    for idx in range(num_models):
        one_hot = np.zeros(num_models, dtype=np.float32)
        one_hot[idx] = 1.0
        candidates.append(one_hot)

    for _ in range(max(0, trials)):
        candidates.append(rng.dirichlet(np.ones(num_models)).astype(np.float32))

    probs_np = member_probs.numpy()
    y_true_np = y_true.numpy()

    best_weights = candidates[0]
    best_kappa = float("-inf")

    for weights in candidates:
        ensemble_probs = np.tensordot(weights, probs_np, axes=(0, 0))
        y_pred = ensemble_probs.argmax(axis=1)
        kappa = cohen_kappa_score(y_true_np, y_pred, weights="quadratic")
        if kappa > best_kappa:
            best_kappa = float(kappa)
            best_weights = weights

    return best_weights.tolist(), best_kappa


def main() -> None:
    args = parse_args()

    test_csv_path = Path(_resolve_test_csv_path(args.test_csv))
    val_csv_path = Path(_resolve_test_csv_path(args.val_csv))

    use_ensemble = args.ensemble_checkpoints is not None and len(args.ensemble_checkpoints) > 0

    if use_ensemble:
        checkpoint_paths = [Path(p) for p in args.ensemble_checkpoints]
        missing = [str(p) for p in checkpoint_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Ensemble checkpoints not found: {missing}")
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not test_csv_path.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    if args.tune_ensemble_weights:
        if not use_ensemble:
            raise ValueError("--tune-ensemble-weights requires --ensemble-checkpoints")
        if args.ensemble_weights is not None:
            raise ValueError("Use either --ensemble-weights or --tune-ensemble-weights, not both.")
        if not val_csv_path.is_file():
            raise FileNotFoundError(f"Validation CSV not found: {val_csv_path}")

    # Build a data module with the test split
    dm = _build_eval_datamodule(test_csv_path, args)

    if use_ensemble and args.ensemble_weights is not None:
        if len(args.ensemble_weights) != len(args.ensemble_checkpoints):
            raise ValueError(
                "--ensemble-weights must have the same number of values as --ensemble-checkpoints"
            )

    model = None
    if not use_ensemble:
        model = DRModel.load_from_checkpoint(str(checkpoint_path))

    # Enable TTA if requested
    if args.tta:
        if model is not None:
            model.tta_enabled = True
            model.tta_runs = args.tta_runs
        print(f"TTA enabled with {args.tta_runs} augmented views per image")

    if use_ensemble and args.tune_ensemble_weights:
        print("Tuning ensemble weights on validation set...")
        val_dm = _build_eval_datamodule(val_csv_path, args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        member_probs, y_true_val = _collect_member_probabilities(
            checkpoint_paths=checkpoint_paths,
            dataloader=val_dm.test_dataloader(),
            tta=args.tta,
            tta_runs=args.tta_runs,
            device=device,
        )
        tuned_weights, best_val_kappa = _tune_weights_from_cached_probs(
            member_probs=member_probs,
            y_true=y_true_val,
            trials=args.weight_search_trials,
            seed=args.weight_search_seed,
        )
        args.ensemble_weights = tuned_weights
        print(f"Best tuned weights: {tuned_weights}")
        print(f"Best validation quadratic kappa: {best_val_kappa:.6f}")

    if not use_ensemble:
        trainer = L.Trainer(
            accelerator="auto",
            devices="auto",
            logger=False,
        )
        results = trainer.test(model, datamodule=dm)
        print(results)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ensemble = EnsemblePredictor(
            checkpoint_paths=[str(p) for p in checkpoint_paths],
            weights=args.ensemble_weights,
            tta_enabled=args.tta,
            tta_runs=args.tta_runs,
        ).to(device)

        y_true_all = []
        y_pred_all = []
        all_rows = []

        with torch.inference_mode():
            for images, labels in dm.test_dataloader():
                images = images.to(device)
                preds = ensemble.predict(images).cpu()
                labels_cpu = labels.cpu()

                y_true_all.extend(labels_cpu.tolist())
                y_pred_all.extend(preds.tolist())

                if args.predictions_csv:
                    for pred, label in zip(preds.tolist(), labels_cpu.tolist()):
                        all_rows.append({"label": label, "prediction": pred})

        acc = accuracy_score(y_true_all, y_pred_all)
        kappa = cohen_kappa_score(y_true_all, y_pred_all, weights="quadratic")
        prec = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
        rec = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)
        f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)

        report = classification_report(
            y_true_all,
            y_pred_all,
            labels=list(range(ensemble.num_classes)),
            target_names=[f"class_{i}" for i in range(ensemble.num_classes)],
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
                }
            ]
        )
        print("\nTest Classification Report:\n")
        print(report)
        print(f"Final Test Quadratic Kappa: {kappa:.6f}")
        print(
            "Final Test Summary: "
            f"acc={acc:.6f}, "
            f"kappa={kappa:.6f}, "
            f"precision_macro={prec:.6f}, "
            f"recall_macro={rec:.6f}, "
            f"f1_macro={f1:.6f}"
        )

    # Optionally save per-image predictions
    if args.predictions_csv:
        predictions_path = Path(args.predictions_csv)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)

        if not use_ensemble:
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