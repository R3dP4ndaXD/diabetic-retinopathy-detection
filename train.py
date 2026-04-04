from __future__ import annotations

import os
from os.path import join

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from src.data_module import DRDataModule
from src.model import DRModel, EMACallback
from src.models.factory import ModelFactory, get_recommended_input_size
from src.utils import generate_run_id


def _resolve_csv_path(configured_path: str) -> str:
    """
    Resolve a CSV path for both local and cluster (/data bind-mount) setups.

    Priority:
      1. configured_path if it exists
      2. /data/<basename> if it exists (cluster bind-mount fallback)
      3. configured_path unchanged (preserves original error on open)
    """
    if os.path.exists(configured_path):
        return configured_path
    candidate = join("/data", os.path.basename(configured_path))
    if os.path.exists(candidate):
        return candidate
    return configured_path


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    run_id  = generate_run_id()
    run_tag = str(cfg.get("run_tag", "")).replace("-", "_")
    if run_tag:
        run_id = f"{run_id}-{run_tag}"

    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # ── Resolve CSV paths ────────────────────────────────────────────────────
    train_csv = _resolve_csv_path(cfg.train_csv_path)
    val_csv   = _resolve_csv_path(cfg.val_csv_path)
    test_csv  = _resolve_csv_path(cfg.test_csv_path) if cfg.get("test_csv_path") else None

    for name, orig, resolved in [
        ("train", cfg.train_csv_path, train_csv),
        ("val",   cfg.val_csv_path,   val_csv),
    ]:
        if resolved != orig:
            print(f"[path-fallback] {name}_csv_path → '{resolved}'")

    print(f"Train CSV : {train_csv}")
    print(f"Val CSV   : {val_csv}")
    if test_csv:
        print(f"Test CSV  : {test_csv}")

    # ── Build model first so we can read its data_config ────────────────────
    model_name       = cfg.model_name
    normalization_mode = cfg.get("normalization_mode", "timm")
    num_classes      = 5   # DR grades 0–4; DataModule will confirm after setup

    # Warn on image-size mismatch
    recommended = get_recommended_input_size(model_name)
    if recommended is not None and int(cfg.image_size) != int(recommended):
        print(
            f"[size-mismatch] '{model_name}' expects {recommended}px "
            f"but cfg.image_size={cfg.image_size}."
        )

    # Build a temporary model just to get data_config; real model built below
    _tmp_model = ModelFactory(name=model_name, num_classes=num_classes)()
    timm_data_config = _tmp_model.data_config if normalization_mode == "timm" else None
    del _tmp_model

    # ── DataModule ───────────────────────────────────────────────────────────
    dm = DRDataModule(
        train_csv_path=train_csv,
        val_csv_path=val_csv,
        test_csv_path=test_csv,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        balancing_mode=cfg.get("balancing_mode", "weighted_loss"),
        normalization_mode=normalization_mode,
        timm_data_config=timm_data_config,
        custom_mean=cfg.get("custom_mean"),
        custom_std=cfg.get("custom_std"),
        use_mixup=cfg.get("use_mixup", True),
        mixup_alpha=cfg.get("mixup_alpha", 0.4),
        cutmix_alpha=cfg.get("cutmix_alpha", 1.0),
        mixup_prob=cfg.get("mixup_prob", 0.5),
        label_smoothing=cfg.get("label_smoothing", 0.1),
        num_classes=num_classes,
    )
    dm.prepare_data()
    dm.setup()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DRModel(
        num_classes=dm.num_classes,
        model_name=model_name,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.get("weight_decay", 1e-4),
        use_scheduler=cfg.use_scheduler,
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
        use_wavelet_channel_input=cfg.get("use_wavelet_channel_input", False),
        wavelet_name=cfg.get("wavelet_name", "haar"),
        wavelet_mode=cfg.get("wavelet_mode", "symmetric"),
        wavelet_levels=cfg.get("wavelet_levels", 1),
        wavelet_include_lowpass_channel=cfg.get("wavelet_include_lowpass_channel", True),
        mixup_fn=dm.mixup_fn,
        drop_rate=cfg.get("drop_rate", 0.3),
        drop_path_rate=cfg.get("drop_path_rate", 0.2),
        layer_lr_decay=cfg.get("layer_lr_decay", 0.75),
    )

    # ── Logger ───────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(save_dir=cfg.logs_dir, name="", version=run_id)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg_dict, dict):
        cfg_dict.update({
            "resolved_train_csv": train_csv,
            "resolved_val_csv":   val_csv,
            "resolved_test_csv":  test_csv,
            "run_id": run_id,
        })
        logger.log_hyperparams(cfg_dict)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        monitor=cfg.get("checkpoint_monitor", "val_kappa"),
        mode=cfg.get("checkpoint_monitor_mode", "max"),
        save_top_k=2,
        dirpath=join(cfg.checkpoint_dirpath, run_id),
        filename="{epoch}-{step}-{val_loss:.2f}-{val_acc:.2f}-{val_kappa:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop  = EarlyStopping(
        monitor=cfg.get("early_stopping_monitor", "val_loss"),
        patience=7,
        verbose=True,
        mode=cfg.get("early_stopping_mode", "min"),
    )

    callbacks = [checkpoint_cb, lr_monitor, early_stop]

    if cfg.get("use_ema", True):
        callbacks.append(EMACallback(decay=cfg.get("ema_decay", 0.9998)))

    # ── Trainer ───────────────────────────────────────────────────────────────
    num_gpus = cfg.get("num_gpus", 1)
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=num_gpus,
        strategy="ddp_find_unused_parameters_false" if num_gpus > 1 else "auto",
        precision=cfg.get("precision", "bf16-mixed"),
        gradient_clip_val=cfg.get("gradient_clip_val", 1.0),
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, dm)

    # ── Test on best checkpoint ────────────────────────────────────────────
    if test_csv:
        best_ckpt = checkpoint_cb.best_model_path
        if best_ckpt:
            model = DRModel.load_from_checkpoint(
                best_ckpt, class_weights=dm.class_weights, mixup_fn=None
            )
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    train()
