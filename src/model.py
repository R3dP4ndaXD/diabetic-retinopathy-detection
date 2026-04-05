from __future__ import annotations

import copy
import io
import random

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_wavelets import DWTForward
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from timm.data import Mixup
from timm.utils import ModelEmaV2
from torch import nn
from torchmetrics.functional import accuracy, cohen_kappa, f1_score, precision, recall
from torchvision.transforms import v2 as T

from src.models.factory import ModelFactory, TIMM_MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multiclass focal loss with optional class weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight=None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_unweighted = F.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_unweighted)
        ce_weighted = F.cross_entropy(
            logits, targets, weight=self.weight, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        return ((1 - pt) ** self.gamma * ce_weighted).mean()


# ---------------------------------------------------------------------------
# Wavelet channel transform
# ---------------------------------------------------------------------------

class WaveletChannelTransform(nn.Module):
    """
    Compute DWT coefficients and return them as extra channels.

    With J decomposition levels and include_lowpass_channel=True the output
    has  1 + 3*J  channels (LL + {LH,HL,HH} per level).
    """

    def __init__(
        self,
        wave: str = "haar",
        mode: str = "symmetric",
        J: int = 1,
        include_lowpass_channel: bool = True,
    ) -> None:
        super().__init__()
        self.J = J
        self.include_lowpass_channel = include_lowpass_channel
        self.dwt = DWTForward(J=J, wave=wave, mode=mode)
        self.register_buffer(
            "rgb_weights",
            torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RGB → grayscale
        gray = (x * self.rgb_weights).sum(dim=1, keepdim=True) if x.shape[1] == 3 \
               else x.mean(dim=1, keepdim=True)

        input_dtype = gray.dtype
        yl, yh = self.dwt(gray.float())

        h, w = x.shape[-2:]
        channels: list[torch.Tensor] = []

        if self.include_lowpass_channel:
            ll = F.interpolate(yl.to(input_dtype), size=(h, w), mode="nearest")
            channels.append(ll)

        for level_coeff in yh:
            # level_coeff: [B, 1, 3, H_l, W_l]
            details = level_coeff.to(input_dtype).squeeze(1)  # [B, 3, H_l, W_l]
            details = F.interpolate(details, size=(h, w), mode="nearest")
            channels.append(details)

        coeffs = torch.cat(channels, dim=1)
        # Global normalisation
        denom = coeffs.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return coeffs / denom

    @property
    def out_channels(self) -> int:
        return (1 if self.include_lowpass_channel else 0) + 3 * self.J


# ---------------------------------------------------------------------------
# EMA callback
# ---------------------------------------------------------------------------

class EMACallback(L.Callback):
    """
    Exponential Moving Average of model weights.

    Swaps to EMA weights before validation/test and restores live weights
    afterwards so training gradients are unaffected.
    """

    def __init__(self, decay: float = 0.9998) -> None:
        super().__init__()
        self.decay = decay
        self._ema: ModelEmaV2 | None = None
        self._backup: dict | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._ema = ModelEmaV2(pl_module, decay=self.decay)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        if self._ema is not None:
            self._ema.update(pl_module)

    # --- swap to EMA for eval ---
    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._swap_to_ema(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._restore_live(pl_module)

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._swap_to_ema(pl_module)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self._restore_live(pl_module)

    def _swap_to_ema(self, pl_module: L.LightningModule) -> None:
        if self._ema is None:
            return
        self._backup = copy.deepcopy(pl_module.state_dict())
        # timm>=1.0 removed copy_to; keep compatibility with older/newer timm.
        if hasattr(self._ema, "copy_to"):
            self._ema.copy_to(pl_module)
        else:
            pl_module.load_state_dict(self._ema.module.state_dict())

    def _restore_live(self, pl_module: L.LightningModule) -> None:
        if self._backup is None:
            return
        pl_module.load_state_dict(self._backup)
        self._backup = None


class ContiguousGradCallback(L.Callback):
    """
    Registers backward hooks that make non-contiguous gradients contiguous
    before DDP bucket reduction.

    Depthwise-conv layers (e.g. in ConvNeXt, EfficientNet) produce gradients
    with strides that differ from what DDP's bucket view expects, causing the
    warning:
      "Grad strides do not match bucket view strides."
    gradient_as_bucket_view=True alone does not suppress this because the
    gradient is never created as the bucket view — the layout mismatch is in
    cuDNN's depthwise backward kernel.  This hook copies only when needed
    (is_contiguous() is False), so contiguous params have zero overhead.
    """

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        for param in pl_module.parameters():
            if param.requires_grad:
                param.register_hook(
                    lambda g: g.contiguous() if not g.is_contiguous() else g
                )


# ---------------------------------------------------------------------------
# Shared test-result logging helper
# ---------------------------------------------------------------------------

def log_test_results(
    pl_module: L.LightningModule,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    num_classes: int,
    prefix: str = "test",
) -> None:
    """
    Log full-dataset test metrics to TensorBoard:
      - Scalar: {prefix}/kappa, {prefix}/acc, {prefix}/precision_{i},
                {prefix}/recall_{i}, {prefix}/f1_{i}  (per class)
      - Text:   {prefix}/classification_report  (Text tab)
      - Image:  {prefix}/confusion_matrix       (Images tab)

    Only runs on global rank 0 (avoids duplicate DDP writes).
    Flushes the SummaryWriter so data appears even if the process exits
    immediately after.
    """
    # Always print on every rank so the Slurm log is complete.
    labels = list(range(num_classes))
    class_names = [f"DR{i}" for i in labels]

    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    acc   = float((y_pred == y_true).mean())
    per_prec, per_rec, per_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(f"\n{'='*60}\n{prefix.upper()} Results\n{'='*60}")
    print(report)
    print(f"Quadratic Weighted Kappa: {kappa:.4f}")

    # Only rank 0 writes to TensorBoard.
    if not pl_module.trainer.is_global_zero:
        return
    if pl_module.logger is None or not hasattr(pl_module.logger, "experiment"):
        return
    exp = pl_module.logger.experiment
    if not hasattr(exp, "add_scalar"):
        return

    step = pl_module.global_step

    # ── Scalars ──────────────────────────────────────────────────────────────
    exp.add_scalar(f"{prefix}/kappa", kappa, step)
    exp.add_scalar(f"{prefix}/acc",   acc,   step)
    for i, name in enumerate(class_names):
        exp.add_scalar(f"{prefix}/precision_{name}", float(per_prec[i]), step)
        exp.add_scalar(f"{prefix}/recall_{name}",    float(per_rec[i]),  step)
        exp.add_scalar(f"{prefix}/f1_{name}",        float(per_f1[i]),   step)

    # ── Text (visible in TensorBoard → Text tab) ──────────────────────────────
    exp.add_text(
        f"{prefix}/classification_report",
        f"<pre>{report}\nQuadratic Weighted Kappa: {kappa:.4f}</pre>",
        step,
    )

    # ── Confusion matrix image (TensorBoard → Images tab) ─────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=labels, yticks=labels,
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted", ylabel="True",
        title=f"{prefix} confusion matrix  (QWK={kappa:.3f})",
    )
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    plt.close(fig)

    # TensorBoard expects [C, H, W]
    exp.add_image(f"{prefix}/confusion_matrix", img.transpose(2, 0, 1), step)

    # Flush so data is written to disk before the process exits.
    exp.flush()


# ---------------------------------------------------------------------------
# Main LightningModule
# ---------------------------------------------------------------------------

_VIT_FAMILIES = {"vit_base", "swin_base"}


class DRModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model_name: str = "efficientnet_b2",
        learning_rate: float = 4e-5,
        weight_decay: float = 1e-4,
        use_scheduler: bool = True,
        freeze_backbone: bool = False,
        class_weights=None,
        label_smoothing: float = 0.0,
        loss_name: str = "cross_entropy",
        focal_gamma: float = 2.0,
        warmup_epochs: int = 0,
        scheduler_monitor: str = "val_kappa",
        scheduler_monitor_mode: str = "max",
        tta_enabled: bool = False,
        tta_runs: int = 5,
        # Wavelet channel input (single-stream only)
        use_wavelet_channel_input: bool = False,
        wavelet_name: str = "haar",
        wavelet_mode: str = "symmetric",
        wavelet_levels: int = 1,
        wavelet_include_lowpass_channel: bool = True,
        # MixUp (injected from DataModule)
        mixup_fn: Mixup | None = None,
        # Regularisation
        drop_rate: float = 0.3,
        drop_path_rate: float = 0.2,
        layer_lr_decay: float = 0.75,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "mixup_fn"])

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.warmup_epochs = warmup_epochs
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_monitor_mode = scheduler_monitor_mode
        self.tta_enabled = tta_enabled
        self.tta_runs = tta_runs
        self.layer_lr_decay = layer_lr_decay
        self.mixup_fn = mixup_fn

        # ── Wavelet transform (optional) ─────────────────────────────────────
        self.use_wavelet_channel_input = use_wavelet_channel_input
        if use_wavelet_channel_input:
            self.wavelet_transform = WaveletChannelTransform(
                wave=wavelet_name,
                mode=wavelet_mode,
                J=wavelet_levels,
                include_lowpass_channel=wavelet_include_lowpass_channel,
            )
            wavelet_ch = self.wavelet_transform.out_channels
            input_channels = wavelet_ch
        else:
            self.wavelet_transform = None
            input_channels = 3

        # ── Backbone ─────────────────────────────────────────────────────────
        self.model = ModelFactory(
            name=model_name,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            input_channels=input_channels,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )()

        # ── Loss ─────────────────────────────────────────────────────────────
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

        # ── TTA augmentations ─────────────────────────────────────────────────
        self._tta_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.Lambda(lambda x: torch.rot90(x, k=random.randint(0, 3), dims=[-2, -1])),
        ])

        self._test_preds: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []

        # store model_name for use in configure_optimizers
        self._model_name = model_name

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_wavelet_channel_input and self.wavelet_transform is not None:
            x = self.wavelet_transform(x)
        return self.model(x)

    # ── Steps ────────────────────────────────────────────────────────────────

    def training_step(self, batch):
        x, y = batch
        if self.mixup_fn is not None:
            x, y = self.mixup_fn(x, y)
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        return loss

    def _compute_metrics(self, preds: torch.Tensor, y: torch.Tensor) -> dict:
        kwargs = dict(task="multiclass", num_classes=self.num_classes)
        return {
            "acc":       accuracy(preds, y, **kwargs),
            "kappa":     cohen_kappa(preds, y, **kwargs, weights="quadratic"),
            "precision": precision(preds, y, **kwargs, average="macro"),
            "recall":    recall(preds, y, **kwargs, average="macro"),
            "f1":        f1_score(preds, y, **kwargs, average="macro"),
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        metrics = self._compute_metrics(preds, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        for name, value in metrics.items():
            self.log(f"val_{name}", value, on_step=True, on_epoch=True,
                     prog_bar=(name in ("acc", "kappa")), sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.tta_enabled:
            avg_probs = torch.zeros(x.size(0), self.num_classes, device=x.device)
            avg_loss  = 0.0
            for _ in range(self.tta_runs):
                aug = self._tta_transform(x)
                logits = self(aug)
                avg_probs += torch.softmax(logits, dim=1)
                avg_loss  += self.criterion(logits, y)
            avg_probs /= self.tta_runs
            loss = avg_loss / self.tta_runs
            preds = torch.argmax(avg_probs, dim=1)
        else:
            logits = self(x)
            loss   = self.criterion(logits, y)
            preds  = torch.argmax(logits, dim=1)

        metrics = self._compute_metrics(preds, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        for name, value in metrics.items():
            self.log(f"test_{name}", value, on_step=False, on_epoch=True,
                     prog_bar=True, sync_dist=True)

        self._test_preds.append(preds.detach().cpu())
        self._test_targets.append(y.detach().cpu())

    def on_test_epoch_start(self) -> None:
        self._test_preds = []
        self._test_targets = []

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        y_pred = torch.cat(self._test_preds).numpy()
        y_true = torch.cat(self._test_targets).numpy()
        log_test_results(self, y_pred, y_true, self.num_classes)

    # ── Optimiser ────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        # Layer-wise LR decay for ViT-family models
        is_vit = self._model_name in _VIT_FAMILIES
        if is_vit and self.layer_lr_decay < 1.0:
            try:
                from timm.optim import param_groups_layer_decay
                param_groups = param_groups_layer_decay(
                    self.model.backbone,
                    weight_decay=self.weight_decay,
                    layer_decay=self.layer_lr_decay,
                )
                optimizer = torch.optim.AdamW(
                    param_groups, lr=self.learning_rate
                )
            except Exception:
                # Fallback if model doesn't support layer decay
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        configuration: dict = {"optimizer": optimizer, "monitor": self.scheduler_monitor}

        if self.use_scheduler:
            if self.warmup_epochs > 0:
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, self.trainer.max_epochs - self.warmup_epochs),
                )
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-2, total_iters=self.warmup_epochs
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[self.warmup_epochs],
                )
                configuration["lr_scheduler"] = {"scheduler": scheduler}
            else:
                reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=self.scheduler_monitor_mode,
                    factor=0.5,   
                    patience=5,   
                    threshold=1e-3,
                    min_lr=1e-7,
                )
                configuration["lr_scheduler"] = {
                    "scheduler": reduce_lr,
                    "monitor": self.scheduler_monitor,
                }

        return configuration
