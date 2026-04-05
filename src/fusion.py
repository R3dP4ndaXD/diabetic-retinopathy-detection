"""
Fusion heads for the dual-stream (RGB + Wavelet) model.

Two strategies are provided:
  - MLPFusionHead         : concatenate feature vectors → MLP → logits
  - CrossAttentionFusionHead : RGB features query wavelet features via
                               cross-attention, then project to logits
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPFusionHead(nn.Module):
    """
    Concatenation + two-layer MLP fusion.

    Parameters
    ----------
    d_rgb       : feature dimension from the RGB backbone
    d_wav       : feature dimension from the wavelet backbone
    num_classes : number of output classes
    hidden      : hidden layer width (default 512)
    dropout     : dropout probability applied after hidden activation
    """

    def __init__(
        self,
        d_rgb: int,
        d_wav: int,
        num_classes: int,
        hidden: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_rgb + d_wav, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, f_rgb: torch.Tensor, f_wav: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        f_rgb : [B, d_rgb]
        f_wav : [B, d_wav]

        Returns
        -------
        logits : [B, num_classes]
        """
        return self.net(torch.cat([f_rgb, f_wav], dim=1))


class CrossAttentionFusionHead(nn.Module):
    """
    Cross-attention fusion: RGB features query wavelet features.

    The RGB embedding acts as the query; the wavelet embedding supplies both
    key and value.  A single attention step produces a context vector that is
    then projected to logits.

    Parameters
    ----------
    d_rgb       : feature dimension from the RGB backbone
    d_wav       : feature dimension from the wavelet backbone
    num_classes : number of output classes
    num_heads   : multi-head attention heads (must divide d evenly)
    dropout     : attention dropout
    """

    def __init__(
        self,
        d_rgb: int,
        d_wav: int,
        num_classes: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Project both streams to a common dimension d
        d = max(d_rgb, d_wav)
        # Make d divisible by num_heads
        if d % num_heads != 0:
            d = ((d // num_heads) + 1) * num_heads

        self.d = d
        self.q_proj  = nn.Linear(d_rgb, d)
        self.kv_proj = nn.Linear(d_wav, d)
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes),
        )

    def forward(self, f_rgb: torch.Tensor, f_wav: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        f_rgb : [B, d_rgb]   — RGB stream pooled features (query)
        f_wav : [B, d_wav]   — wavelet stream pooled features (key/value)

        Returns
        -------
        logits : [B, num_classes]
        """
        # Treat each sample as a sequence of length 1
        q  = self.q_proj(f_rgb).unsqueeze(1)   # [B, 1, d]
        kv = self.kv_proj(f_wav).unsqueeze(1)  # [B, 1, d]

        context, _ = self.attn(q, kv, kv)      # [B, 1, d]
        return self.head(context.squeeze(1))    # [B, num_classes]


def build_fusion_head(
    fusion_type: str,
    d_rgb: int,
    d_wav: int,
    num_classes: int,
    hidden: int = 512,
    dropout: float = 0.3,
    num_heads: int = 8,
) -> nn.Module:
    """
    Factory function.

    Parameters
    ----------
    fusion_type : "mlp" | "cross_attention"
    """
    if fusion_type == "mlp":
        return MLPFusionHead(d_rgb, d_wav, num_classes, hidden=hidden, dropout=dropout)
    elif fusion_type == "cross_attention":
        return CrossAttentionFusionHead(
            d_rgb, d_wav, num_classes, num_heads=num_heads, dropout=dropout
        )
    else:
        raise ValueError(
            f"Unknown fusion_type '{fusion_type}'. Expected 'mlp' or 'cross_attention'."
        )
