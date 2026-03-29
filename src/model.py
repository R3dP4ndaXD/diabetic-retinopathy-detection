import lightning as L
import torch
from torch import nn
from torchmetrics.functional import accuracy, cohen_kappa, f1_score, precision, recall
from torchvision.transforms import v2 as T
from src.models.factory import ModelFactory


class DRModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model_name: str = "densenet121",
        learning_rate: float = 4e-5,
        weight_decay: float = 1e-4,
        use_scheduler: bool = True,
        freeze_backbone: bool = True,
        class_weights=None,
        label_smoothing: float = 0.0,
        warmup_epochs: int = 0,
        tta_enabled: bool = False,
        tta_runs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.warmup_epochs = warmup_epochs
        self.tta_enabled = tta_enabled
        self.tta_runs = tta_runs

        # Define the model
        self.model = ModelFactory(name=model_name, num_classes=num_classes, freeze_backbone=freeze_backbone)()

        # Define the loss function
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

        # TTA augmentations (applied on already-normalized tensors)
        self._tta_transform = T.Compose([
            T.RandomRotation(degrees=(0, 360), fill=0),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _compute_metrics(self, preds, y):
        metric_args = dict(task="multiclass", num_classes=self.num_classes)
        return {
            "acc": accuracy(preds, y, **metric_args),
            "kappa": cohen_kappa(preds, y, **metric_args, weights="quadratic"),
            "precision": precision(preds, y, **metric_args, average="macro"),
            "recall": recall(preds, y, **metric_args, average="macro"),
            "f1": f1_score(preds, y, **metric_args, average="macro"),
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        metrics = self._compute_metrics(preds, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for name, value in metrics.items():
            self.log(f"val_{name}", value, on_step=True, on_epoch=True, prog_bar=(name in ("acc", "kappa")))

    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.tta_enabled:
            # Average softmax probabilities over multiple augmented views
            avg_probs = torch.zeros(x.size(0), self.num_classes, device=x.device)
            for _ in range(self.tta_runs):
                augmented = self._tta_transform(x)
                logits = self.model(augmented)
                avg_probs += torch.softmax(logits, dim=1)
            avg_probs /= self.tta_runs
            preds = torch.argmax(avg_probs, dim=1)
            # Compute loss using the averaged probabilities (log for NLL-style loss)
            loss = self.criterion(avg_probs.log(), y)
        else:
            logits = self.model(x)
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)

        metrics = self._compute_metrics(preds, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, value in metrics.items():
            self.log(f"test_{name}", value, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        configuration = {
            "optimizer": optimizer,
            "monitor": "val_loss",
        }

        if self.use_scheduler:
            if self.warmup_epochs > 0:
                # CosineAnnealingLR is compatible with SequentialLR (unlike ReduceLROnPlateau)
                cosine_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.trainer.max_epochs - self.warmup_epochs
                )
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-2, total_iters=self.warmup_epochs
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup, cosine_lr], milestones=[self.warmup_epochs]
                )
                configuration["lr_scheduler"] = {
                    "scheduler": scheduler,
                }
            else:
                reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.3,
                    patience=2,
                    threshold=0.001,
                )
                configuration["lr_scheduler"] = reduce_lr

        return configuration
