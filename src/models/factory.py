from __future__ import annotations

import timm
import torch
from timm.data import resolve_data_config
from torch import nn

# ---------------------------------------------------------------------------
# Precomputed normalization stats per model
# Verified by running resolve_data_config on each pretrained checkpoint.
# Most use standard ImageNet stats; attention-based models (CoAtNet, MaxViT,
# ViT) were pretrained with 0.5/0.5 normalisation.
# ---------------------------------------------------------------------------
_IN = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))   # ImageNet
_HF = ((0.5,   0.5,   0.5  ), (0.5,   0.5,   0.5  ))   # 0.5-normalised

_NORM_STATS: dict[str, tuple[tuple, tuple]] = {
    "efficientnetv2_m":  _IN,
    "efficientnetv2_s":  _IN,
    "convnext_tiny":     _IN,
    "convnext_base":     _IN,
    "convnext_large":    _IN,
    "coatnet_2":         _HF,
    "maxvit_base":       _HF,
    "swin_base":         _IN,
    "vit_base":          _HF,
    "efficientnet_b0":   _IN,
    "efficientnet_b2":   _IN,
    "mobilenetv3_large": _IN,
}


def get_normalization_stats(model_name: str) -> dict:
    """Return {"mean": list[float], "std": list[float]} for *model_name*.

    Values are precomputed from each model's timm data_config and stored
    in ``_NORM_STATS`` — no model instantiation required at call time.
    """
    if model_name not in _NORM_STATS:
        raise ValueError(
            f"No precomputed normalization stats for '{model_name}'. "
            f"Add an entry to _NORM_STATS in src/models/factory.py."
        )
    mean, std = _NORM_STATS[model_name]
    return {"mean": list(mean), "std": list(std)}


# ---------------------------------------------------------------------------
# Registry: short name  →  (timm model id, recommended input size)
# ---------------------------------------------------------------------------
TIMM_MODEL_REGISTRY: dict[str, tuple[str, int]] = {
    # ── Heavy backbones (RGB stream / single-stream training) ────────────────
    "efficientnetv2_m":  ("efficientnetv2_rw_m.agc_in1k",                         416),
    "efficientnetv2_s":  ("efficientnetv2_rw_s.ra2_in1k",                         384),
    "convnext_tiny":     ("convnext_tiny.fb_in22k_ft_in1k",                       224),
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


def resolve_image_size(model_name: str, cfg_image_size: int | None) -> int:
    """Return the image size to use for training.

    If *cfg_image_size* is set (non-null), that value wins and a warning is
    printed when it differs from the model's recommendation.  Otherwise the
    model's recommended size is used.  Falls back to 224 if no recommendation
    is registered.
    """
    recommended = get_recommended_input_size(model_name)
    if cfg_image_size is not None:
        size = int(cfg_image_size)
        if recommended is not None and size != recommended:
            print(
                f"[size-mismatch] '{model_name}' recommends {recommended}px "
                f"but image_size={size} was explicitly set."
            )
        return size
    if recommended is not None:
        print(f"[image_size] using model default: {recommended}px for '{model_name}'")
        return recommended
    print(f"[image_size] no recommendation for '{model_name}', defaulting to 224px")
    return 224


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
