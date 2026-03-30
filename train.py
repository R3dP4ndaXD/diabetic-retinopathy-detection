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
from src.model import DRModel
from src.utils import generate_run_id


def _resolve_csv_path(configured_path: str) -> str:
    """Resolve CSV path for local and cluster (/data bind) setups.

    Priority:
    1. Configured path if it exists
    2. /data/<basename(configured_path)> if it exists
    3. Return configured path unchanged (to preserve error behavior)
    """
    if os.path.exists(configured_path):
        return configured_path

    data_bound_candidate = join("/data", os.path.basename(configured_path))
    if os.path.exists(data_bound_candidate):
        return data_bound_candidate

    return configured_path


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    # generate unique run id based on current date & time
    run_id = generate_run_id()
    run_tag = cfg.get("run_tag", "")
    if run_tag:
        run_tag = str(run_tag).replace("-", "_")
        run_id = f"{run_id}-{run_tag}"

    # Seed everything for reproducibility
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    train_csv_path = _resolve_csv_path(cfg.train_csv_path)
    val_csv_path = _resolve_csv_path(cfg.val_csv_path)
    test_csv_path = _resolve_csv_path(cfg.get("test_csv_path")) if cfg.get("test_csv_path") else None

    print(f"Using train CSV: {train_csv_path}")
    print(f"Using val CSV:   {val_csv_path}")
    if test_csv_path:
        print(f"Using test CSV:  {test_csv_path}")

    # Initialize DataModule
    dm = DRDataModule(
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        test_csv_path=test_csv_path,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        balancing_mode=cfg.get("balancing_mode", "weighted_loss"),
        normalization_mode=cfg.get("normalization_mode", "dataset_by_size"),
        custom_mean=cfg.get("custom_mean"),
        custom_std=cfg.get("custom_std"),
    )
    dm.setup()

    # Init model from datamodule's attributes
    model = DRModel(
        num_classes=dm.num_classes,
        model_name=cfg.model_name,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.get("weight_decay", 1e-4),
        use_scheduler=cfg.use_scheduler,
        freeze_backbone=cfg.get("freeze_backbone", True),
        class_weights=dm.class_weights,
        label_smoothing=cfg.get("label_smoothing", 0.0),
        loss_name=cfg.get("loss_name", "cross_entropy"),
        focal_gamma=cfg.get("focal_gamma", 2.0),
        warmup_epochs=cfg.get("warmup_epochs", 0),
        scheduler_monitor=cfg.get("scheduler_monitor", "val_kappa"),
        scheduler_monitor_mode=cfg.get("scheduler_monitor_mode", "max"),
        tta_enabled=cfg.get("tta_enabled", False),
        tta_runs=cfg.get("tta_runs", 5),
    )

    # Init logger
    logger = TensorBoardLogger(save_dir=cfg.logs_dir, name="", version=run_id)

    # Track the full experiment config in TensorBoard hparams (not only model args)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg_dict, dict):
        cfg_dict["resolved_train_csv_path"] = train_csv_path
        cfg_dict["resolved_val_csv_path"] = val_csv_path
        cfg_dict["resolved_test_csv_path"] = test_csv_path
        cfg_dict["run_id"] = run_id
        logger.log_hyperparams(cfg_dict)

    # Init callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.get("checkpoint_monitor", "val_kappa"),
        mode=cfg.get("checkpoint_monitor_mode", "max"),
        save_top_k=2,
        dirpath=join(cfg.checkpoint_dirpath, run_id),
        filename="{epoch}-{step}-{val_loss:.2f}-{val_acc:.2f}-{val_kappa:.2f}",
    )

    # Init LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # early stopping
    early_stopping = EarlyStopping(
        monitor=cfg.get("early_stopping_monitor", "val_kappa"),
        patience=7,
        verbose=True,
        mode=cfg.get("early_stopping_mode", "max"),
    )

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=cfg.get("precision", "32-true"),
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
    )

    # Train the model
    trainer.fit(model, dm)

    # Evaluate on the test set if available
    if cfg.get("test_csv_path"):
        # Load best checkpoint before testing
        best_ckpt_path = checkpoint_callback.best_model_path
        if best_ckpt_path:
            model = DRModel.load_from_checkpoint(
                best_ckpt_path,
                class_weights=dm.class_weights,
            )
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    train()
