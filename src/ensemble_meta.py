"""
MetaLearner — trainable late-fusion over frozen base-model probability vectors.

Four fusion strategies
----------------------
temperature     : per-model temperature scaling; calibrates overconfident models
learned_weights : softmax-normalised learnable scalar weights (soft voting++)
mlp             : flatten all probs → two-layer MLP → logits
cross_attention : each model's prob vector is a token; cross-attend then classify

Training protocol
-----------------
1. Train all base models independently (DRModel or DualStreamDRModel).
2. Freeze them and collect val-set probability matrices:
       member_probs : [N_models, N_val, num_classes]
       y_true       : [N_val]
3. Train MetaLearner on those cached probs (fast — no images needed).
4. Save meta_learner.ckpt.
5. In test_ensemble.py, optionally load --meta-learner-checkpoint.

See train_meta.py for the full training script.
"""
from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import cohen_kappa_score
from torchmetrics.functional import accuracy, cohen_kappa


# ---------------------------------------------------------------------------
# Dataset wrapper for cached probabilities
# ---------------------------------------------------------------------------

class ProbDataset(TensorDataset):
    """
    Wraps pre-collected model probability tensors.

    Parameters
    ----------
    member_probs : [N_models, N_samples, num_classes]  float32
    labels       : [N_samples]  long
    """

    def __init__(self, member_probs: torch.Tensor, labels: torch.Tensor) -> None:
        # Stack per-sample: item i → [N_models, num_classes]
        # TensorDataset expects first dim = N_samples
        probs_T = member_probs.permute(1, 0, 2)  # [N_samples, N_models, num_classes]
        super().__init__(probs_T, labels)

    def __getitem__(self, idx):
        probs, label = super().__getitem__(idx)
        return probs, label   # [N_models, num_classes], scalar


# ---------------------------------------------------------------------------
# MetaLearner LightningModule
# ---------------------------------------------------------------------------

class MetaLearner(L.LightningModule):
    """
    Trainable meta-fusion over N frozen base-model probability vectors.

    Parameters
    ----------
    num_models   : number of base models in the ensemble
    num_classes  : number of DR grades (5)
    fusion_type  : "temperature" | "learned_weights" | "mlp" | "cross_attention"
    hidden       : hidden dim for MLP / cross-attention heads
    lr           : learning rate (Adam)
    """

    def __init__(
        self,
        num_models: int,
        num_classes: int,
        fusion_type: str = "mlp",
        hidden: int = 64,
        dropout: float = 0.2,
        num_heads: int = 4,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.num_models = num_models
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.lr = lr

        # ── Fusion parameters ────────────────────────────────────────────────
        if fusion_type == "temperature":
            # One temperature scalar per model (log-space for positivity)
            self.log_temps = nn.Parameter(torch.zeros(num_models))

        elif fusion_type == "learned_weights":
            # Unnormalised weights; softmax gives the actual mixing coefficients
            self.log_weights = nn.Parameter(torch.zeros(num_models))

        elif fusion_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(num_models * num_classes, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )

        elif fusion_type == "cross_attention":
            # Each model's num_classes vector is one "token"
            d = hidden
            if d % num_heads != 0:
                d = ((d // num_heads) + 1) * num_heads
            self.input_proj = nn.Linear(num_classes, d)
            self.attn = nn.MultiheadAttention(
                embed_dim=d, num_heads=num_heads,
                dropout=dropout, batch_first=True,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d),
                nn.Dropout(dropout),
                nn.Linear(d, num_classes),
            )

        else:
            raise ValueError(
                f"Unknown fusion_type '{fusion_type}'. "
                "Expected: temperature, learned_weights, mlp, cross_attention."
            )

        self.criterion = nn.CrossEntropyLoss()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        probs : [B, N_models, num_classes]  softmax probabilities

        Returns
        -------
        logits : [B, num_classes]
        """
        B = probs.size(0)

        if self.fusion_type == "temperature":
            # Convert probs → log-probs, scale by temperature, re-normalise
            temps = self.log_temps.exp().view(1, -1, 1)          # [1, N, 1]
            scaled_logits = torch.log(probs.clamp_min(1e-8)) / temps
            # Weighted average in probability space
            w = torch.softmax(torch.zeros(self.num_models, device=self.device), dim=0)
            weighted = (scaled_logits * w.view(1, -1, 1)).sum(dim=1)  # [B, C]
            return weighted

        elif self.fusion_type == "learned_weights":
            w = torch.softmax(self.log_weights, dim=0)            # [N]
            return (probs * w.view(1, -1, 1)).sum(dim=1)          # [B, C]

        elif self.fusion_type == "mlp":
            flat = probs.reshape(B, -1)                           # [B, N*C]
            return self.mlp(flat)

        elif self.fusion_type == "cross_attention":
            tokens = self.input_proj(probs)                       # [B, N, d]
            # Self-attention over model tokens, then pool by mean
            out, _ = self.attn(tokens, tokens, tokens)            # [B, N, d]
            pooled = out.mean(dim=1)                              # [B, d]
            return self.head(pooled)

    # ── Steps ─────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        probs, y = batch
        logits = self(probs)
        loss = self.criterion(logits, y)
        self.log("meta_train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        probs, y = batch
        logits = self(probs)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        kappa = cohen_kappa(preds, y, task="multiclass",
                            num_classes=self.num_classes, weights="quadratic")
        acc   = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)
        self.log("meta_val_loss",  loss,  prog_bar=True)
        self.log("meta_val_kappa", kappa, prog_bar=True)
        self.log("meta_val_acc",   acc,   prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "meta_val_kappa"},
        }

    # ── Convenience: predict on cached probs ──────────────────────────────────

    @torch.inference_mode()
    def predict_from_probs(self, member_probs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        member_probs : [N_models, N_samples, num_classes]

        Returns
        -------
        predictions : [N_samples]  class indices
        """
        self.eval()
        probs_T = member_probs.permute(1, 0, 2)  # [N, N_models, C]
        logits = self(probs_T.to(self.device))
        return torch.argmax(logits, dim=1)
