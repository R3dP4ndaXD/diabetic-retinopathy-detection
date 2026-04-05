"""
DualStreamDRModel — two independent timm backbones fused before the classifier.

Architecture
------------
Input [B,3,H,W]
  ├── RGB stream   ──► heavy timm backbone ──► forward_head(pre_logits=True) ──► f_rgb [B, D_rgb]
  └── Wavelet stream ──► WaveletChannelTransform(J=2) ──► light timm backbone
                                                          ──► f_wav [B, D_wav]
                              ↓
                     FusionHead(f_rgb, f_wav) ──► logits [B, num_classes]

Both streams use their own timm data_config for normalization (handled externally
by DualStreamDataModule or by normalizing with the RGB backbone's stats and
letting the wavelet stream learn to adapt — the default approach here).
"""
from __future__ import annotations

import copy
import random

import lightning as L
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from sklearn.metrics import classification_report, cohen_kappa_score  # noqa: F401 (kept for compat)
from timm.data import Mixup
from torch import nn
from torchmetrics.functional import accuracy, cohen_kappa, f1_score, precision, recall
from torchvision.transforms import v2 as T

from src.fusion import build_fusion_head
from src.model import EMACallback, FocalLoss, WaveletChannelTransform, log_test_results
from src.models.factory import ModelFactory


class DualStreamDRModel(L.LightningModule):
    """
    Parameters
    ----------
    rgb_model_name   : timm short-name for the RGB (heavy) backbone
    wav_model_name   : timm short-name for the wavelet (light) backbone
    fusion_type      : "mlp" | "cross_attention"
    wavelet_name     : wavelet family, e.g. "sym2", "haar", "db2"
    wavelet_levels   : DWT decomposition levels (J). 2 recommended for DR.
    wavelet_include_lowpass_channel : include the LL subband channel
    mixup_fn         : timm Mixup instance injected from DRDataModule
    """

    def __init__(
        self,
        num_classes: int,
        rgb_model_name: str = "convnext_base",
        wav_model_name: str = "efficientnet_b0",
        fusion_type: str = "mlp",
        fusion_hidden: int = 512,
        fusion_dropout: float = 0.3,
        fusion_num_heads: int = 8,
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
        wavelet_name: str = "sym2",
        wavelet_mode: str = "symmetric",
        wavelet_levels: int = 2,
        wavelet_include_lowpass_channel: bool = True,
        mixup_fn: Mixup | None = None,
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
        self.mixup_fn = mixup_fn
        self.layer_lr_decay = layer_lr_decay

        # ── Wavelet transform ────────────────────────────────────────────────
        self.wavelet_transform = WaveletChannelTransform(
            wave=wavelet_name,
            mode=wavelet_mode,
            J=wavelet_levels,
            include_lowpass_channel=wavelet_include_lowpass_channel,
        )
        wav_input_channels = self.wavelet_transform.out_channels

        # ── RGB backbone (num_classes=0 → feature-extractor mode) ────────────
        self.rgb_backbone = ModelFactory(
            name=rgb_model_name,
            num_classes=0,           # no head; we use get_features()
            freeze_backbone=freeze_backbone,
            input_channels=3,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )()

        # ── Wavelet backbone ──────────────────────────────────────────────────
        self.wav_backbone = ModelFactory(
            name=wav_model_name,
            num_classes=0,
            freeze_backbone=freeze_backbone,
            input_channels=wav_input_channels,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )()

        d_rgb = self.rgb_backbone.get_feature_dim()
        d_wav = self.wav_backbone.get_feature_dim()

        # ── Fusion head ───────────────────────────────────────────────────────
        self.fusion_head = build_fusion_head(
            fusion_type=fusion_type,
            d_rgb=d_rgb,
            d_wav=d_wav,
            num_classes=num_classes,
            hidden=fusion_hidden,
            dropout=fusion_dropout,
            num_heads=fusion_num_heads,
        )

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

        # ── TTA ───────────────────────────────────────────────────────────────
        self._tta_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.Lambda(lambda x: torch.rot90(x, k=random.randint(0, 3), dims=[-2, -1])),
        ])

        self._test_preds: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []
        self._rgb_model_name = rgb_model_name

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run both streams on input x (RGB image).

        Returns
        -------
        f_rgb : [B, D_rgb]  pooled RGB features
        f_wav : [B, D_wav]  pooled wavelet features
        """
        # RGB stream
        rgb_feats   = self.rgb_backbone.forward_features(x)
        f_rgb       = self.rgb_backbone.forward_head(rgb_feats, pre_logits=True)

        # Wavelet stream
        wav_input   = self.wavelet_transform(x)
        wav_feats   = self.wav_backbone.forward_features(wav_input)
        f_wav       = self.wav_backbone.forward_head(wav_feats, pre_logits=True)

        return f_rgb, f_wav

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_rgb, f_wav = self._encode(x)
        return self.fusion_head(f_rgb, f_wav)

    # ── Steps ─────────────────────────────────────────────────────────────────

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

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
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
                    factor=0.3,
                    patience=2,
                    threshold=0.001,
                )
                configuration["lr_scheduler"] = {
                    "scheduler": reduce_lr,
                    "monitor": self.scheduler_monitor,
                }

        return configuration
