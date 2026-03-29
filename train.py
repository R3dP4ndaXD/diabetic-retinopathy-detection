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
from omegaconf import DictConfig
from src.data_module import DRDataModule
from src.model import DRModel
from src.utils import generate_run_id


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    # generate unique run id based on current date & time
    run_id = generate_run_id()
    run_tag = cfg.get("run_tag", "")
    if run_tag:
        run_id = f"{run_id}-{run_tag}"

    # Seed everything for reproducibility
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # Initialize DataModule
    dm = DRDataModule(
        train_csv_path=cfg.train_csv_path,
        val_csv_path=cfg.val_csv_path,
        test_csv_path=cfg.get("test_csv_path"),
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_oversampling=cfg.get("use_oversampling", True),
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
        warmup_epochs=cfg.get("warmup_epochs", 0),
    )

    # Init logger
    logger = TensorBoardLogger(save_dir=cfg.logs_dir, name="", version=run_id)
    # Init callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        dirpath=join(cfg.checkpoint_dirpath, run_id),
        filename="{epoch}-{step}-{val_loss:.2f}-{val_acc:.2f}-{val_kappa:.2f}",
    )

    # Init LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=7,
        verbose=True,
        mode="min",
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
            model = DRModel.load_from_checkpoint(best_ckpt_path)
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    train()
