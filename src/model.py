import lightning as L
import torch.nn.functional as F
import torch
from sklearn.metrics import classification_report, cohen_kappa_score
from torch import nn
from torchmetrics.functional import accuracy, cohen_kappa, f1_score, precision, recall
from torchvision.transforms import v2 as T
from src.models.factory import ModelFactory


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional class weights."""

    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


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
        loss_name: str = "cross_entropy",  # cross_entropy | focal
        focal_gamma: float = 2.0,
        warmup_epochs: int = 0,
        scheduler_monitor: str = "val_kappa",
        scheduler_monitor_mode: str = "max",
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
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_monitor_mode = scheduler_monitor_mode
        self.tta_enabled = tta_enabled
        self.tta_runs = tta_runs

        # Define the model
        self.model = ModelFactory(name=model_name, num_classes=num_classes, freeze_backbone=freeze_backbone)()

        # Define the loss function
        if loss_name == "focal":
            self.criterion = FocalLoss(
                gamma=focal_gamma,
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing,
            )

        # TTA augmentations (applied on already-normalized tensors)
        self._tta_transform = T.Compose([
            T.RandomRotation(degrees=(0, 360), fill=0),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])
        self._test_preds = []
        self._test_targets = []

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

        self._test_preds.append(preds.detach().cpu())
        self._test_targets.append(y.detach().cpu())

    def on_test_epoch_start(self):
        self._test_preds = []
        self._test_targets = []

    def on_test_epoch_end(self):
        if not self._test_preds or not self._test_targets:
            return

        y_pred = torch.cat(self._test_preds).numpy()
        y_true = torch.cat(self._test_targets).numpy()
        labels = list(range(self.num_classes))
        target_names = [f"class_{idx}" for idx in labels]
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
        kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        summary = f"Quadratic Kappa: {kappa:.4f}"

        print("\nTest Classification Report:\n")
        print(report)
        print(summary)

        # Log full report to TensorBoard text panel when available
        if self.logger is not None and hasattr(self.logger, "experiment"):
            experiment = self.logger.experiment
            if hasattr(experiment, "add_text"):
                experiment.add_text(
                    "test/classification_report",
                    f"<pre>{report}\n{summary}</pre>",
                    global_step=self.global_step,
                )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        configuration = {
            "optimizer": optimizer,
            "monitor": self.scheduler_monitor,
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
                    mode=self.scheduler_monitor_mode,
                    factor=0.3,
                    patience=2,
                    threshold=0.001,
                )
                configuration["lr_scheduler"] = {
                    "scheduler": reduce_lr,
                    "monitor": self.scheduler_monitor,
                }

        return configuration
