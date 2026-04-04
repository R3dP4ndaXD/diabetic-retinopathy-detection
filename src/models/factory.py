from __future__ import annotations

import timm
import torch
from timm.data import resolve_data_config
from torch import nn

# ---------------------------------------------------------------------------
# Registry: short name  →  (timm model id, recommended input size)
# ---------------------------------------------------------------------------
TIMM_MODEL_REGISTRY: dict[str, tuple[str, int]] = {
    # ── Heavy backbones (RGB stream / single-stream training) ────────────────
    "efficientnetv2_m":  ("efficientnetv2_rw_m.agc_in1k",                       416),
    "efficientnetv2_s":  ("efficientnetv2_rw_s.ra2_in1k",                        384),
    "convnext_base":     ("convnext_base.fb_in22k_ft_in1k",                       224),
    "convnext_large":    ("convnext_large.fb_in22k_ft_in1k",                      224),
    "coatnet_2":         ("coatnet_2_rw_224.sw_in12k_ft_in1k",                    224),
    "maxvit_base":       ("maxvit_base_tf_512.in21k_ft_in1k",                     512),
    "swin_base":         ("swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",   256),
    "vit_base":          ("vit_base_patch16_224.augreg2_in21k_ft_in1k",           224),
    # ── Lightweight backbones (wavelet stream / fast ablations) ─────────────
    "efficientnet_b0":   ("efficientnet_b0.ra_in1k",                              224),
    "efficientnet_b2":   ("efficientnet_b2.ra_in1k",                              260),
    "mobilenetv3_large": ("mobilenetv3_large_100.ra_in1k",                        224),
}


def get_recommended_input_size(model_name: str) -> int | None:
    entry = TIMM_MODEL_REGISTRY.get(model_name)
    return entry[1] if entry else None


def list_supported_models() -> list[str]:
    return sorted(TIMM_MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Core model wrapper
# ---------------------------------------------------------------------------

class TimmModel(nn.Module):
    """
    Thin wrapper around a timm backbone.

    Parameters
    ----------
    model_name       : short name from TIMM_MODEL_REGISTRY
    num_classes      : output classes (0 = feature-only mode, no head)
    drop_rate        : dropout before final classifier
    drop_path_rate   : stochastic depth rate (ViTs / ConvNeXt benefit most)
    input_channels   : 3 for RGB; >3 for wavelet-augmented input
    freeze_backbone  : if True freeze all params except the classifier head
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        drop_rate: float = 0.3,
        drop_path_rate: float = 0.2,
        input_channels: int = 3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if model_name not in TIMM_MODEL_REGISTRY:
            valid = ", ".join(sorted(TIMM_MODEL_REGISTRY))
            raise ValueError(
                f"Unknown model_name '{model_name}'. Valid options: {valid}"
            )

        timm_id, _ = TIMM_MODEL_REGISTRY[model_name]

        # timm handles in_chans adaptation (weight inflation) internally
        self.backbone = timm.create_model(
            timm_id,
            pretrained=True,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            in_chans=input_channels,
        )

        # Expose the data config so callers can read mean/std/input_size
        self.data_config: dict = resolve_data_config({}, model=self.backbone)

        if freeze_backbone:
            self._freeze_backbone()

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial feature map (before pooling)."""
        return self.backbone.forward_features(x)

    def forward_head(self, features: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """
        Run the pooling + classifier head on pre-computed features.

        Parameters
        ----------
        pre_logits : if True return the pooled vector *before* the linear
                     classifier — useful as a feature embedding for fusion.
        """
        return self.backbone.forward_head(features, pre_logits=pre_logits)

    def get_feature_dim(self) -> int:
        """Dimensionality of the pooled embedding (before classifier)."""
        return self.backbone.num_features

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        """Freeze everything except the classification head."""
        head_names = {"head", "classifier", "fc", "heads"}
        for name, param in self.backbone.named_parameters():
            top_module = name.split(".")[0]
            param.requires_grad = top_module in head_names

    def unfreeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class ModelFactory:
    """
    Convenience factory — mirrors the old API so callers don't change.

    Usage
    -----
        factory = ModelFactory(
            name="convnext_base",
            num_classes=5,
            drop_path_rate=0.2,
        )
        model = factory()           # returns TimmModel instance
        data_cfg = model.data_config
    """

    def __init__(
        self,
        name: str,
        num_classes: int,
        freeze_backbone: bool = False,
        input_channels: int = 3,
        drop_rate: float = 0.3,
        drop_path_rate: float = 0.2,
    ) -> None:
        self.name = name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.input_channels = input_channels
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate

    def __call__(self) -> TimmModel:
        return TimmModel(
            model_name=self.name,
            num_classes=self.num_classes,
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
            input_channels=self.input_channels,
            freeze_backbone=self.freeze_backbone,
        )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for name in ["convnext_base", "efficientnet_b0", "vit_base"]:
        m = ModelFactory(name, num_classes=5)()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = m(x)
        print(f"{name}: output={out.shape}  features={m.get_feature_dim()}  "
              f"mean={m.data_config['mean']}")
