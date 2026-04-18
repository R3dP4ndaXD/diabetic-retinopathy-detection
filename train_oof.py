from __future__ import annotations

import json
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold

from src.config_checks import validate_training_recipe
from src.data_module import DRDataModule
from src.model import ContiguousGradCallback, DRModel, ModelProfilerCallback
from src.models.factory import get_normalization_stats, resolve_image_size
from src.utils import build_run_tag, generate_run_id


def _patient_id_from_image_path(image_path: str) -> str:
        """
        Extract patient id from EyePACS-like filenames:
            - <patient_id>_left.jpeg
            - <patient_id>_right.jpeg
        Falls back to full stem when the pattern is not present.
        """
        stem = Path(image_path).stem
        if stem.endswith("_left") or stem.endswith("_right"):
                return stem.rsplit("_", 1)[0]
        return stem


def _collect_probabilities(
    model: DRModel,
    dataloader,
    device: torch.device,
    tta_enabled: bool,
    tta_runs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probs_batches: list[torch.Tensor] = []
    labels_batches: list[torch.Tensor] = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)

            if tta_enabled:
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

    return torch.cat(probs_batches, dim=0), torch.cat(labels_batches, dim=0)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_oof(cfg: DictConfig) -> None:
    validate_training_recipe(cfg, context="train_oof")

    run_id = generate_run_id()
    run_id = f"{run_id}-{build_run_tag(cfg)}"

    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    source_csv = Path(str(cfg.get("oof_source_csv_path", cfg.train_csv_path)))
    if not source_csv.is_file():
        raise FileNotFoundError(f"OOF source CSV not found: {source_csv}")

    n_splits = int(cfg.get("oof_num_folds", 5))
    oof_seed = int(cfg.get("oof_seed", cfg.seed))
    output_root = Path(str(cfg.get("oof_output_dir", "artifacts/oof"))) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    test_csv = cfg.test_csv_path if cfg.get("test_csv_path") else None
    has_test = bool(test_csv) and Path(str(test_csv)).is_file()

    print(f"Run ID             : {run_id}")
    print(f"OOF source CSV     : {source_csv}")
    print(f"OOF folds          : {n_splits}")
    print(f"OOF output dir     : {output_root}")
    if has_test:
        print(f"OOF test CSV       : {test_csv}")

    df = pd.read_csv(source_csv)
    required = {"image_path", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{source_csv} must contain columns: {sorted(required)}")

    labels = df["label"].to_numpy()
    groups = df["image_path"].apply(_patient_id_from_image_path).to_numpy()
    num_classes = int(np.max(labels)) + 1

    oof_probs = np.zeros((len(df), num_classes), dtype=np.float32)
    oof_preds = np.full(len(df), -1, dtype=np.int64)
    fold_assignment = np.full(len(df), -1, dtype=np.int64)

    test_df = pd.read_csv(test_csv) if has_test else None
    test_probs_sum = np.zeros((len(test_df), num_classes), dtype=np.float32) if test_df is not None else None

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=oof_seed)

    checkpoint_paths: list[str] = []
    fold_kappas: list[float] = []
    norm       = get_normalization_stats(cfg.model_name)
    image_size = resolve_image_size(cfg.model_name, cfg.get("image_size"))

    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(df, labels, groups=groups),
        start=1,
    ):
        fold_dir = output_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        overlap = train_groups & val_groups
        if overlap:
            sample_overlap = list(sorted(overlap))[:10]
            raise RuntimeError(
                f"Fold {fold_idx}: patient-level leakage detected. "
                f"Overlapping patient IDs (sample): {sample_overlap}"
            )

        fold_train_csv = fold_dir / "train.csv"
        fold_val_csv = fold_dir / "val.csv"

        df.iloc[train_idx].to_csv(fold_train_csv, index=False)
        df.iloc[val_idx].to_csv(fold_val_csv, index=False)

        dm = DRDataModule(
            train_csv_path=str(fold_train_csv),
            val_csv_path=str(fold_val_csv),
            test_csv_path=str(test_csv) if has_test else None,
            norm_mean=norm["mean"],
            norm_std=norm["std"],
            image_size=image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            balancing_mode=cfg.get("balancing_mode", "weighted_loss"),
            use_mixup=cfg.get("use_mixup", False),
            mixup_alpha=cfg.get("mixup_alpha", 0.4),
            cutmix_alpha=cfg.get("cutmix_alpha", 1.0),
            mixup_prob=cfg.get("mixup_prob", 0.5),
            label_smoothing=cfg.get("label_smoothing", 0.1),
            num_classes=num_classes,
        )
        dm.prepare_data()
        dm.setup()

        model = DRModel(
            num_classes=dm.num_classes,
            model_name=cfg.model_name,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.get("weight_decay", 1e-4),
            use_scheduler=cfg.use_scheduler,
            scheduler_type=cfg.get("scheduler_type", "auto"),
            freeze_backbone=cfg.get("freeze_backbone", False),
            class_weights=dm.class_weights,
            label_smoothing=cfg.get("label_smoothing", 0.0),
            loss_name=cfg.get("loss_name", "cross_entropy"),
            focal_gamma=cfg.get("focal_gamma", 2.0),
            warmup_epochs=cfg.get("warmup_epochs", 0),
            scheduler_monitor=cfg.get("scheduler_monitor", "val_kappa"),
            scheduler_monitor_mode=cfg.get("scheduler_monitor_mode", "max"),
            tta_enabled=cfg.get("tta_enabled", False),
            tta_runs=cfg.get("tta_runs", 5),
            freq_transform=cfg.get("freq_transform", "none"),
            wavelet_name=cfg.get("wavelet_name", "haar"),
            wavelet_mode=cfg.get("wavelet_mode", "symmetric"),
            wavelet_levels=cfg.get("wavelet_levels", 1),
            wavelet_include_lowpass_channel=cfg.get("wavelet_include_lowpass_channel", True),
            dct_block_size=cfg.get("dct_block_size", 8),
            dct_num_coeffs=cfg.get("dct_num_coeffs", 6),
            fourier_shift=cfg.get("fourier_shift", False),
            fourier_hpf_radius=cfg.get("fourier_hpf_radius", 0.1),
            mixup_fn=dm.mixup_fn,
            drop_rate=cfg.get("drop_rate", 0.3),
            drop_path_rate=cfg.get("drop_path_rate", 0.2),
            layer_lr_decay=cfg.get("layer_lr_decay", 0.75),
            mc_dropout_enabled=False,   # MC Dropout is inference-only, not during CV training
            mc_dropout_samples=cfg.get("mc_dropout_samples", 50),
        )

        logger = TensorBoardLogger(save_dir=cfg.logs_dir, name=run_id, version=f"fold_{fold_idx}")
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg_dict, dict):
            cfg_dict.update({"run_id": run_id, "oof_fold": fold_idx})
            logger.log_hyperparams(cfg_dict)

        checkpoint_cb = ModelCheckpoint(
            monitor=cfg.get("checkpoint_monitor", "val_kappa"),
            mode=cfg.get("checkpoint_monitor_mode", "max"),
            save_top_k=1,
            dirpath=str(fold_dir / "checkpoints"),
            filename="{epoch}-{step}-{val_loss:.2f}-{val_acc:.2f}-{val_f1:.2f}-{val_kappa:.2f}",
        )
        early_stop = EarlyStopping(
            monitor=cfg.get("early_stopping_monitor", "val_loss"),
            patience=7,
            verbose=True,
            mode=cfg.get("early_stopping_mode", "min"),
        )

        callbacks = [
            checkpoint_cb,
            LearningRateMonitor(logging_interval="step"),
            early_stop,
            ContiguousGradCallback(),
            ModelProfilerCallback(image_size=image_size),
        ]

        num_gpus = cfg.get("num_gpus", 1)
        trainer = L.Trainer(
            max_epochs=cfg.max_epochs,
            accelerator="auto",
            devices=num_gpus,
            strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            if num_gpus > 1
            else "auto",
            precision=cfg.get("precision", "bf16-mixed"),
            gradient_clip_val=cfg.get("gradient_clip_val", 1.0),
            logger=logger,
            callbacks=callbacks,
        )

        print(f"\n--- Training fold {fold_idx}/{n_splits} ---")
        trainer.fit(model, dm)

        best_ckpt = checkpoint_cb.best_model_path
        checkpoint_paths.append(best_ckpt)
        if best_ckpt:
            infer_model = DRModel.load_from_checkpoint(
                best_ckpt,
                class_weights=dm.class_weights,
                mixup_fn=None,
            )
        else:
            infer_model = model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        infer_model = infer_model.to(device)

        val_probs_t, val_labels_t = _collect_probabilities(
            model=infer_model,
            dataloader=dm.val_dataloader(),
            device=device,
            tta_enabled=cfg.get("tta_enabled", False),
            tta_runs=cfg.get("tta_runs", 5),
        )

        expected_labels = df.iloc[val_idx]["label"].to_numpy()
        if not np.array_equal(val_labels_t.numpy(), expected_labels):
            raise RuntimeError(
                "Validation label order mismatch while building OOF predictions."
            )

        val_probs = val_probs_t.numpy()
        val_preds = val_probs.argmax(axis=1)

        oof_probs[val_idx] = val_probs
        oof_preds[val_idx] = val_preds
        fold_assignment[val_idx] = fold_idx

        fold_kappa = cohen_kappa_score(expected_labels, val_preds, weights="quadratic")
        fold_kappas.append(float(fold_kappa))
        print(f"Fold {fold_idx} QWK: {fold_kappa:.4f}")

        if test_df is not None and test_probs_sum is not None:
            test_probs_t, _ = _collect_probabilities(
                model=infer_model,
                dataloader=dm.test_dataloader(),
                device=device,
                tta_enabled=cfg.get("tta_enabled", False),
                tta_runs=cfg.get("tta_runs", 5),
            )
            test_probs_sum += test_probs_t.numpy()

    if np.any(fold_assignment < 0):
        raise RuntimeError("Some samples did not receive OOF predictions.")

    oof_df = df.copy()
    oof_df["fold"] = fold_assignment
    for c in range(num_classes):
        oof_df[f"prob_{c}"] = oof_probs[:, c]
    oof_df["pred_label"] = oof_preds

    oof_kappa = cohen_kappa_score(oof_df["label"].to_numpy(), oof_df["pred_label"].to_numpy(), weights="quadratic")
    print(f"\nOOF QWK across all folds: {oof_kappa:.4f}")

    oof_path = output_root / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved OOF predictions: {oof_path}")

    test_path = None
    test_kappa = None
    if test_df is not None and test_probs_sum is not None:
        mean_test_probs = test_probs_sum / n_splits
        test_pred = mean_test_probs.argmax(axis=1)

        test_out = test_df.copy()
        for c in range(num_classes):
            test_out[f"prob_{c}"] = mean_test_probs[:, c]
        test_out["pred_label"] = test_pred

        if "label" in test_out.columns:
            test_kappa = cohen_kappa_score(
                test_out["label"].to_numpy(),
                test_out["pred_label"].to_numpy(),
                weights="quadratic",
            )
            print(f"Test QWK (mean over fold models): {test_kappa:.4f}")

        test_path = output_root / "test_predictions.csv"
        test_out.to_csv(test_path, index=False)
        print(f"Saved test probabilities: {test_path}")

    metrics = {
        "run_id": run_id,
        "oof_num_folds": n_splits,
        "oof_qwk": float(oof_kappa),
        "fold_qwk": fold_kappas,
        "fold_checkpoints": checkpoint_paths,
        "oof_predictions_csv": str(oof_path),
        "test_predictions_csv": str(test_path) if test_path else None,
        "test_qwk": float(test_kappa) if test_kappa is not None else None,
    }
    metrics_path = output_root / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    train_oof()
