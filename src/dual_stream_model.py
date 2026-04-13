"""
DualStreamDRModel — two independent timm backbones fused before the classifier.

Architecture
------------
Input [B,3,H,W]
  ├── RGB stream   ──► heavy timm backbone ──► forward_head(pre_logits=True) ──► f_rgb [B, D_rgb]
  └── Freq stream  ──► FreqTransform ──► light timm backbone ──► f_freq [B, D_freq]
                        FreqTransform is one of:
                          "wavelet"     → WaveletChannelTransform(J=2)   — 1+3J channels, H/2^J × W/2^J
                          "dct"         → DCTChannelTransform             — k channels, H/B × W/B
                          "fourier"     → FourierChannelTransform         — 3 channels (Re/Im/Mag), H × W
                          "fourier_hpf" → FourierHighPassTransform        — 1 or 3 channels, H × W
                              ↓
                     FusionHead(f_rgb, f_freq) ──► logits [B, num_classes]

Both streams use the RGB backbone's timm data_config for normalization by default.
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
from src.model import (
    DCTChannelTransform,
    EMACallback,
    FocalLoss,
    FourierChannelTransform,
    FourierHighPassTransform,
    SupConLoss,
    WaveletChannelTransform,
    log_test_results,
)
from src.models.factory import ModelFactory


class DualStreamDRModel(L.LightningModule):
    """
    Parameters
    ----------
    rgb_model_name   : timm short-name for the RGB (heavy) backbone
    wav_model_name   : timm short-name for the frequency-stream (light) backbone
    fusion_type      : "mlp" | "cross_attention"
    freq_stream      : "wavelet" | "dct" | "fourier" | "fourier_hpf"
    wavelet_name     : wavelet family (freq_stream="wavelet"), e.g. "sym2", "haar"
    wavelet_levels   : DWT decomposition levels J (freq_stream="wavelet")
    fourier_shift    : fftshift DC to centre (freq_stream="fourier")
    fourier_hpf_radius   : low-freq cutoff radius as fraction of min(H,W)
    fourier_hpf_grayscale: grayscale HPF output (1 ch) vs per-channel (3 ch)
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
        # Frequency stream: "wavelet" | "dct" | "fourier" | "fourier_hpf"
        freq_stream: str = "wavelet",
        wavelet_name: str = "sym2",
        wavelet_mode: str = "symmetric",
        wavelet_levels: int = 2,
        wavelet_include_lowpass_channel: bool = True,
        dct_block_size: int = 8,
        dct_num_coeffs: int = 6,
        fourier_shift: bool = False,
        fourier_hpf_radius: float = 0.1,
        fourier_hpf_grayscale: bool = False,
        mixup_fn: Mixup | None = None,
        drop_rate: float = 0.3,
        drop_path_rate: float = 0.2,
        layer_lr_decay: float = 0.75,
        # Supervised Contrastive Learning
        use_supcon: bool = False,
        supcon_weight: float = 0.2,
        supcon_temperature: float = 0.07,
        supcon_ordinal: bool = True,
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
        self.use_supcon = use_supcon
        self.supcon_weight = supcon_weight

        # ── Frequency stream transform ────────────────────────────────────────
        _fs = freq_stream.lower()
        if _fs == "dct":
            self.freq_transform = DCTChannelTransform(
                block_size=dct_block_size,
                num_coeffs=dct_num_coeffs,
            )
        elif _fs == "fourier":
            self.freq_transform = FourierChannelTransform(shift=fourier_shift)
        elif _fs == "fourier_hpf":
            self.freq_transform = FourierHighPassTransform(
                lpf_radius=fourier_hpf_radius,
                grayscale=fourier_hpf_grayscale,
            )
        else:  # "wavelet" (default)
            self.freq_transform = WaveletChannelTransform(
                wave=wavelet_name,
                mode=wavelet_mode,
                J=wavelet_levels,
                include_lowpass_channel=wavelet_include_lowpass_channel,
            )
        wav_input_channels = self.freq_transform.out_channels

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

        # ── Supervised Contrastive projection heads ───────────────────────────
        if use_supcon:
            self.proj_head_rgb = nn.Sequential(
                nn.Linear(d_rgb, d_rgb), nn.ReLU(), nn.Linear(d_rgb, 128)
            )
            self.proj_head_wav = nn.Sequential(
                nn.Linear(d_wav, d_wav), nn.ReLU(), nn.Linear(d_wav, 128)
            )
            self.supcon_criterion = SupConLoss(
                temperature=supcon_temperature,
                ordinal_weights=supcon_ordinal,
            )
        else:
            self.proj_head_rgb = None
            self.proj_head_wav = None
            self.supcon_criterion = None

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

        # Frequency stream
        wav_input   = self.freq_transform(x)
        wav_feats   = self.wav_backbone.forward_features(wav_input)
        f_wav       = self.wav_backbone.forward_head(wav_feats, pre_logits=True)

        return f_rgb, f_wav

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_rgb, f_wav = self._encode(x)
        return self.fusion_head(f_rgb, f_wav)

    # ── Steps ─────────────────────────────────────────────────────────────────

    def training_step(self, batch):
        x, y = batch

        # ── Supervised Contrastive auxiliary loss (free: uses existing views) ─
        # f_rgb and f_wav are two complementary representations of the same
        # image — no extra augmented pass needed.
        f_rgb, f_wav = self._encode(x)
        supcon_loss = torch.tensor(0.0, device=x.device)
        if self.use_supcon and self.proj_head_rgb is not None:
            z_rgb = self.proj_head_rgb(f_rgb)  # [B, 128]
            z_wav = self.proj_head_wav(f_wav)  # [B, 128]
            z = torch.stack([z_rgb, z_wav], dim=1)  # [B, 2, 128]
            supcon_loss = self.supcon_criterion(z, y)

        # ── Classification loss ───────────────────────────────────────────────
        if self.mixup_fn is not None:
            x, y = self.mixup_fn(x, y)
            logits = self(x)
        else:
            logits = self.fusion_head(f_rgb, f_wav)  # reuse already-encoded features
        ce_loss = self.criterion(logits, y)

        loss = ce_loss + self.supcon_weight * supcon_loss
        self.log("train_ce_loss",     ce_loss,     on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_supcon_loss", supcon_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_loss",        loss,        on_step=True, on_epoch=True, prog_bar=True,
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
                configuration["lr_scheduler"] = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
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
