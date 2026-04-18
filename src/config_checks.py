from __future__ import annotations

import warnings

from omegaconf import DictConfig


def validate_training_recipe(cfg: DictConfig, context: str = "train") -> None:
    """
    Check for known conflicting or counter-productive technique combinations.

    When strict_compatibility_checks=true (default) hard incompatibilities raise
    ValueError.  Soft conflicts (degraded effectiveness, not crashes) always emit
    warnings regardless of the strict flag.
    """
    strict = bool(cfg.get("strict_compatibility_checks", True))

    def _error(msg: str) -> None:
        full = f"[{context}] Incompatible recipe: {msg}"
        if strict:
            raise ValueError(full)
        warnings.warn(full, stacklevel=3)

    def _warn(msg: str) -> None:
        warnings.warn(f"[{context}] Recipe warning: {msg}", stacklevel=3)

    use_mixup      = bool(cfg.get("use_mixup", False))
    balancing_mode = str(cfg.get("balancing_mode", "weighted_loss")).lower()
    loss_name      = str(cfg.get("loss_name", "cross_entropy")).lower()
    label_smooth   = float(cfg.get("label_smoothing", 0.0))
    scheduler_type = str(cfg.get("scheduler_type", "auto")).lower()
    warmup_epochs  = int(cfg.get("warmup_epochs", 0))
    mc_enabled     = bool(cfg.get("mc_dropout_enabled", False))
    drop_rate      = float(cfg.get("drop_rate", 0.3))

    # ── Hard incompatibilities ────────────────────────────────────────────────
    if mc_enabled and drop_rate == 0.0:
        _error(
            "mc_dropout_enabled=true requires drop_rate > 0. "
            "All MC passes are identical when there are no Dropout layers."
        )

    # ── Soft conflicts (warn, but allow) ─────────────────────────────────────
    if use_mixup and loss_name == "focal":
        _warn(
            "MixUp produces soft labels but FocalLoss is designed for hard targets. "
            "The focal modulation factor (1-pt)^gamma is less meaningful with mixed labels. "
            "Consider loss_name='cross_entropy' when use_mixup=true."
        )

    if use_mixup and label_smooth > 0.0:
        _warn(
            f"use_mixup=true AND label_smoothing={label_smooth} > 0 both soften targets. "
            "Using both is redundant and may hurt gradient signal. "
            "Set label_smoothing=0 when use_mixup=true."
        )

    if use_mixup and balancing_mode == "weighted_loss" and loss_name != "cross_entropy":
        _warn(
            "balancing_mode='weighted_loss' with use_mixup=true has explicit "
            "weighted soft-target support only for loss_name='cross_entropy'. "
            f"Current loss_name='{loss_name}' may not apply class weighting as intended."
        )


