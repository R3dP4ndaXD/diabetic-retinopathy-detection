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
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from src.config_checks import validate_training_recipe
from src.data_module import DRDataModule
from src.model import ContiguousGradCallback, DRModel, ModelProfilerCallback
from src.models.factory import ModelFactory, get_normalization_stats, resolve_image_size
from src.utils import build_run_tag, generate_run_id


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    run_id  = generate_run_id()
    run_id  = f"{run_id}-{build_run_tag(cfg)}"
    
    print(f"Run ID    : {run_id}")

    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    validate_training_recipe(cfg, context="train")

    # ── CSV paths ────────────────────────────────────────────────────
    train_csv = cfg.train_csv_path
    val_csv   = cfg.val_csv_path
    test_csv  = cfg.test_csv_path if cfg.get("test_csv_path") else None

    print(f"Train CSV : {train_csv}")
    print(f"Val CSV   : {val_csv}")
    if test_csv:
        print(f"Test CSV  : {test_csv}")

    model_name  = cfg.model_name
    num_classes = 5   # DR grades 0–4; DataModule will confirm after setup
    image_size  = resolve_image_size(model_name, cfg.get("image_size"))
    norm        = get_normalization_stats(model_name)

    # ── DataModule ───────────────────────────────────────────────────────────
    dm = DRDataModule(
        train_csv_path=train_csv,
        val_csv_path=val_csv,
        test_csv_path=test_csv,
        norm_mean=norm["mean"],
        norm_std=norm["std"],
        image_size=image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        balancing_mode=cfg.get("balancing_mode", "weighted_loss"),
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
        mc_dropout_enabled=cfg.get("mc_dropout_enabled", False),
        mc_dropout_samples=cfg.get("mc_dropout_samples", 50),
    )

    # ── Logger ───────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(save_dir=cfg.logs_dir, name="", version=run_id)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg_dict, dict):
        cfg_dict.update({
            "run_id": run_id,
        })
        logger.log_hyperparams(cfg_dict)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        monitor=cfg.get("checkpoint_monitor", "val_kappa"),
        mode=cfg.get("checkpoint_monitor_mode", "max"),
        save_top_k=2,
        dirpath=join(cfg.checkpoint_dirpath, run_id),
        filename="{epoch}-{step}-{val_loss:.2f}-{val_acc:.2f}-{val_f1:.2f}-{val_kappa:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop  = EarlyStopping(
        monitor=cfg.get("early_stopping_monitor", "val_kappa"),
        patience=cfg.get("early_stopping_patience", 10),
        verbose=True,
        mode=cfg.get("early_stopping_mode", "max"),
    )

    callbacks = [checkpoint_cb, lr_monitor, early_stop, ContiguousGradCallback(), ModelProfilerCallback(image_size=image_size)]

    # ── Trainer ───────────────────────────────────────────────────────────────
    num_gpus = cfg.get("num_gpus", 1)
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=num_gpus,
        strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True) if num_gpus > 1 else "auto",
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
