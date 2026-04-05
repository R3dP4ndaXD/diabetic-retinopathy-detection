"""
Dual-stream training entrypoint.

Uses conf/config_dual_stream.yaml by default; all keys can be overridden
on the command line via Hydra (key=value syntax).

Example
-------
python train_dual_stream.py \
    rgb_model_name=convnext_base wav_model_name=efficientnet_b0 \
    fusion_type=mlp image_size=224 batch_size=32 max_epochs=35
"""
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

from src.data_module import DRDataModule
from src.dual_stream_model import DualStreamDRModel
from src.model import ContiguousGradCallback, EMACallback
from src.models.factory import ModelFactory
from src.utils import generate_run_id


@hydra.main(version_base=None, config_path="conf", config_name="config_dual_stream")
def train(cfg: DictConfig) -> None:
    run_id  = generate_run_id()
    run_tag = str(cfg.get("run_tag", "")).replace("-", "_")
    if run_tag:
        run_id = f"{run_id}-{run_tag}"

    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    train_csv = cfg.train_csv_path
    val_csv   = cfg.val_csv_path
    test_csv  = cfg.test_csv_path if cfg.get("test_csv_path") else None

    print(f"Train CSV : {train_csv}")
    print(f"Val CSV   : {val_csv}")
    if test_csv:
        print(f"Test CSV  : {test_csv}")

    # Use RGB backbone's timm data_config for normalization
    normalization_mode = cfg.get("normalization_mode", "timm")
    _tmp = ModelFactory(name=cfg.rgb_model_name, num_classes=5)()
    timm_data_config = _tmp.data_config if normalization_mode == "timm" else None
    del _tmp

    # ── DataModule ────────────────────────────────────────────────────────────
    num_classes = 5
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
    model = DualStreamDRModel(
        num_classes=dm.num_classes,
        rgb_model_name=cfg.rgb_model_name,
        wav_model_name=cfg.wav_model_name,
        fusion_type=cfg.get("fusion_type", "mlp"),
        fusion_hidden=cfg.get("fusion_hidden", 512),
        fusion_dropout=cfg.get("fusion_dropout", 0.3),
        fusion_num_heads=cfg.get("fusion_num_heads", 8),
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
        wavelet_name=cfg.get("wavelet_name", "sym2"),
        wavelet_mode=cfg.get("wavelet_mode", "symmetric"),
        wavelet_levels=cfg.get("wavelet_levels", 2),
        wavelet_include_lowpass_channel=cfg.get("wavelet_include_lowpass_channel", True),
        mixup_fn=dm.mixup_fn,
        drop_rate=cfg.get("drop_rate", 0.3),
        drop_path_rate=cfg.get("drop_path_rate", 0.2),
    )

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(save_dir=cfg.logs_dir, name="", version=run_id)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg_dict, dict):
        cfg_dict["run_id"] = run_id
        logger.log_hyperparams(cfg_dict)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        monitor=cfg.get("checkpoint_monitor", "val_kappa"),
        mode=cfg.get("checkpoint_monitor_mode", "max"),
        save_top_k=2,
        dirpath=join(cfg.checkpoint_dirpath, run_id),
        filename="{epoch}-{step}-{val_loss:.2f}-{val_acc:.2f}-{val_kappa:.2f}",
    )
    callbacks = [
        checkpoint_cb,
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor=cfg.get("early_stopping_monitor", "val_loss"),
            patience=7,
            verbose=True,
            mode=cfg.get("early_stopping_mode", "min"),
        ),
        ContiguousGradCallback(),
    ]
    if cfg.get("use_ema", True):
        callbacks.append(EMACallback(decay=cfg.get("ema_decay", 0.9998)))

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

    if test_csv:
        best_ckpt = checkpoint_cb.best_model_path
        if best_ckpt:
            model = DualStreamDRModel.load_from_checkpoint(
                best_ckpt, class_weights=dm.class_weights, mixup_fn=None
            )
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    train()
