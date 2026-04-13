from __future__ import annotations

import copy
import io
import random

import torch_dct

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
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Pulls together embeddings of the same class while pushing apart those of
    different classes within the batch.

    Parameters
    ----------
    temperature : float
        Softmax temperature τ. Lower = harder negatives. Paper default: 0.07.
    ordinal_weights : bool
        If True, scale the loss contribution of each (anchor, negative) pair
        by |y_i - y_j| + 1, so distant DR grades are pushed harder than
        adjacent ones.  Positives (same class) are always weight 1.
    """

    def __init__(self, temperature: float = 0.07, ordinal_weights: bool = False) -> None:
        super().__init__()
        self.temperature = temperature
        self.ordinal_weights = ordinal_weights

    def forward(
        self,
        features: torch.Tensor,  # [B, n_views, D]  — will be L2-normalised here
        labels: torch.Tensor,    # [B]  integer class labels
    ) -> torch.Tensor:
        B, n_views, D = features.shape
        device = features.device

        # Flatten → [N, D] and L2-normalise
        flat = F.normalize(features.reshape(B * n_views, D), dim=1)

        # Repeat labels so each view has its label
        labels_rep = labels.repeat_interleave(n_views)  # [N]

        # Cosine-similarity matrix scaled by temperature
        sim = torch.mm(flat, flat.T) / self.temperature  # [N, N]

        N = B * n_views
        self_mask = torch.eye(N, device=device, dtype=torch.bool)
        pos_mask  = (labels_rep.unsqueeze(0) == labels_rep.unsqueeze(1)) & ~self_mask

        if not pos_mask.any():
            # No positives in this batch (all different classes) — skip gracefully
            return flat.sum() * 0.0

        # Numerically stable log-sum-exp over all non-self pairs
        sim_max = sim.detach().max(dim=1, keepdim=True).values
        exp_sim  = torch.exp(sim - sim_max).masked_fill(self_mask, 0.0)
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8) + sim_max

        # log-likelihood for each (anchor, positive) pair
        log_prob = sim - log_denom  # [N, N]

        if self.ordinal_weights:
            dist = (labels_rep.unsqueeze(0) - labels_rep.unsqueeze(1)).abs().float()
            # positives have dist=0 → weight 1; negatives have dist≥1 → weight ≥2
            weight = dist + 1.0
            log_prob = log_prob * weight

        # Average log-prob over positives per anchor
        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(log_prob * pos_mask).sum(dim=1) / pos_count
        return loss.mean()


# ---------------------------------------------------------------------------
# Wavelet channel transform
# ---------------------------------------------------------------------------

class WaveletChannelTransform(nn.Module):
    """
    Compute DWT coefficients and return them as extra channels.

    With J decomposition levels and include_lowpass_channel=True the output
    has  1 + 3*J  channels (LL + {LH,HL,HH} per level).

    All sub-bands are bilinearly upsampled back to the original input
    resolution [H, W] so the backbone receives its expected spatial size.
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
        target_size = (x.shape[-2], x.shape[-1])  # original H × W
        yl, yh = self.dwt(gray.float())

        channels: list[torch.Tensor] = []

        if self.include_lowpass_channel:
            ll = yl.to(input_dtype)
            if ll.shape[-2:] != target_size:
                ll = F.interpolate(ll, size=target_size, mode="bilinear", align_corners=False)
            channels.append(ll)

        for level_coeff in yh:
            # level_coeff: [B, 1, 3, H_l, W_l]
            details = level_coeff.to(input_dtype).squeeze(1)  # [B, 3, H_l, W_l]
            if details.shape[-2:] != target_size:
                details = F.interpolate(details, size=target_size, mode="bilinear", align_corners=False)
            channels.append(details)

        coeffs = torch.cat(channels, dim=1)
        # Global normalisation
        denom = coeffs.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return coeffs / denom

    @property
    def out_channels(self) -> int:
        return (1 if self.include_lowpass_channel else 0) + 3 * self.J


# ---------------------------------------------------------------------------
# Block DCT channel transform
# ---------------------------------------------------------------------------

class DCTChannelTransform(nn.Module):
    """
    Block 2D DCT-II → k lowest-frequency coefficient maps.

    Divides the image into non-overlapping block_size×block_size blocks,
    applies 2D DCT-II (via torch-dct) to each block, then selects the k
    lowest-frequency coefficients in zigzag order.  Each coefficient becomes
    a spatial map that is bilinearly upsampled back to [H, W] so the backbone
    receives its expected input resolution.

    Parameters
    ----------
    block_size  : int   Side length of each DCT block (8 = JPEG standard).
    num_coeffs  : int   How many zigzag-ordered coefficients to keep (≤ block_size²).
                        4–6 gives roughly the same channel count as one wavelet level.
    """

    def __init__(self, block_size: int = 8, num_coeffs: int = 6) -> None:
        super().__init__()
        self.block_size = block_size
        self.num_coeffs = min(num_coeffs, block_size * block_size)
        self.register_buffer(
            "rgb_weights",
            torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1),
        )
        self.register_buffer("zigzag_idx", self._build_zigzag(block_size))

    @staticmethod
    def _build_zigzag(N: int) -> torch.Tensor:
        """Flat indices of an N×N matrix in diagonal zigzag order (low → high freq)."""
        idx: list[int] = []
        for s in range(2 * N - 1):
            if s % 2 == 0:
                r = min(s, N - 1)
                c = s - r
                while r >= 0 and c < N:
                    idx.append(r * N + c)
                    r -= 1
                    c += 1
            else:
                c = min(s, N - 1)
                r = s - c
                while c >= 0 and r < N:
                    idx.append(r * N + c)
                    r += 1
                    c -= 1
        return torch.tensor(idx, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = self.block_size
        input_dtype = x.dtype

        # RGB → grayscale  [B, 1, H, W]
        gray = (x * self.rgb_weights).sum(dim=1, keepdim=True) if C == 3 \
               else x.mean(dim=1, keepdim=True)

        # Pad to a multiple of block_size
        pad_h = (N - H % N) % N
        pad_w = (N - W % N) % N
        if pad_h or pad_w:
            gray = F.pad(gray, (0, pad_w, 0, pad_h), mode="reflect")
        Hp, Wp = gray.shape[-2:]
        bH, bW = Hp // N, Wp // N

        # Unfold into blocks: [B, 1, bH, bW, N, N]
        blocks = gray.unfold(2, N, N).unfold(3, N, N)
        # Merge batch and block dims for vectorised DCT: [B*bH*bW, N, N]
        blocks = blocks.contiguous().reshape(B * bH * bW, N, N).float()

        # 2-D DCT-II via torch-dct (separable 1-D transforms)
        dct_blocks = torch_dct.dct_2d(blocks, norm="ortho")  # [B*bH*bW, N, N]

        # Select k zigzag-ordered coefficients: [B*bH*bW, k]
        flat = dct_blocks.reshape(B * bH * bW, N * N)
        selected = flat[:, self.zigzag_idx[: self.num_coeffs]]

        # Reshape to spatial maps: [B, k, bH, bW]
        selected = selected.reshape(B, bH, bW, self.num_coeffs).permute(0, 3, 1, 2)

        # Upsample coefficient maps back to original [H, W] so the backbone
        # receives its expected input resolution.
        out = selected.to(input_dtype)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        # Per-sample normalisation (same convention as WaveletChannelTransform)
        denom = out.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return out / denom

    @property
    def out_channels(self) -> int:
        return self.num_coeffs


# ---------------------------------------------------------------------------
# Fourier channel transform (MRIQA-style)
# ---------------------------------------------------------------------------

class FourierChannelTransform(nn.Module):
    """
    2D FFT → 3 k-space channels, log-compressed.

    Directly adapted from MRIQA (ICME 2024) which applies 3D FFT to MRI
    volumes and stacks (real, imaginary, magnitude) as three channels fed
    to a separate encoder.  Here we use torch.fft.fft2 on grayscale fundus
    images and apply the same signed-log compression:

        real_ch  = sign(Re{F}) * log1p(|Re{F}|)
        imag_ch  = sign(Im{F}) * log1p(|Im{F}|)
        mag_ch   = log1p(|F|)

    The output is in **k-space** — each "pixel" is a spatial frequency
    component, not an image pixel.  Spatial resolution stays at [H, W].
    The backbone sees 3 input channels, same as an RGB image.

    Parameters
    ----------
    shift : bool
        If True, apply fftshift so the DC component sits at the centre of
        the output map (more interpretable; cosine/sine patterns centred).
        Default False (raw FFT layout — no overhead, backbone learns both).
    """

    def __init__(self, shift: bool = False) -> None:
        super().__init__()
        self.shift = shift
        self.register_buffer(
            "rgb_weights",
            torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RGB → grayscale  [B, 1, H, W]
        gray = (x * self.rgb_weights).sum(dim=1, keepdim=True) if x.shape[1] == 3 \
               else x.mean(dim=1, keepdim=True)

        # 2D FFT — always in float32 for numerical stability
        F_complex = torch.fft.fft2(gray.float())  # [B, 1, H, W] complex

        if self.shift:
            F_complex = torch.fft.fftshift(F_complex, dim=(-2, -1))

        real = F_complex.real  # [B, 1, H, W]
        imag = F_complex.imag
        mag  = F_complex.abs()

        # Signed log-compression (MRIQA convention)
        real_ch = torch.sign(real) * torch.log1p(real.abs())  # [B, 1, H, W]
        imag_ch = torch.sign(imag) * torch.log1p(imag.abs())
        mag_ch  = torch.log1p(mag)

        out = torch.cat([real_ch, imag_ch, mag_ch], dim=1)  # [B, 3, H, W]

        # Per-sample normalisation to [-1, 1]
        denom = out.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return (out / denom).to(x.dtype)

    @property
    def out_channels(self) -> int:
        return 3


# ---------------------------------------------------------------------------
# Fourier high-pass filter transform
# ---------------------------------------------------------------------------

class FourierHighPassTransform(nn.Module):
    """
    Per-channel FFT → high-pass mask → IFFT → pixel-space detail image.

    Common in retinal / medical imaging literature: suppress the low-frequency
    bulk illumination signal and keep the high-frequency content (vessels,
    microaneurysms, exudates, haemorrhages — the clinically relevant structures).

    The low-frequency region is a circular mask centred on DC.  Radius is
    expressed as a fraction of the smaller spatial dimension (e.g. 0.1 = 10%).

    Parameters
    ----------
    lpf_radius : float
        Fraction of min(H, W) to zero-out around DC.  Smaller → more low-freq
        preserved (weaker high-pass).  Typical range: 0.05–0.25.
    grayscale : bool
        If True, convert RGB to grayscale first and output 1 channel.
        If False, apply per-channel FFT → 3 output channels.
    """

    def __init__(self, lpf_radius: float = 0.1, grayscale: bool = False) -> None:
        super().__init__()
        self.lpf_radius = lpf_radius
        self.grayscale = grayscale
        self.register_buffer(
            "rgb_weights",
            torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1),
        )
        # Mask is built lazily (depends on input spatial size)
        self._mask_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _get_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W)
        if key not in self._mask_cache:
            # High-pass mask: 1 everywhere EXCEPT a circle of low frequencies
            cy, cx = H // 2, W // 2
            ys = torch.arange(H, device=device).float() - cy
            xs = torch.arange(W, device=device).float() - cx
            dist = torch.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
            r = self.lpf_radius * min(H, W)
            mask = (dist > r).float()  # 1 = keep (high-freq), 0 = remove (low-freq)
            self._mask_cache[key] = mask
        return self._mask_cache[key].to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        input_dtype = x.dtype

        if self.grayscale:
            inp = (x * self.rgb_weights).sum(dim=1, keepdim=True).float()  # [B,1,H,W]
        else:
            inp = x.float()  # [B, C, H, W]

        # 2D FFT per channel, shift so DC is at centre
        F_complex = torch.fft.fft2(inp)               # [B, Cout, H, W] complex
        F_shifted = torch.fft.fftshift(F_complex, dim=(-2, -1))

        # Apply high-pass mask
        mask = self._get_mask(H, W, x.device)         # [H, W]
        F_filtered = F_shifted * mask.unsqueeze(0).unsqueeze(0)

        # IFFT back to pixel space (unshift first)
        F_unshifted = torch.fft.ifftshift(F_filtered, dim=(-2, -1))
        out = torch.fft.ifft2(F_unshifted).real        # [B, Cout, H, W]

        # Per-sample normalisation to [-1, 1]
        denom = out.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return (out / denom).to(input_dtype)

    @property
    def out_channels(self) -> int:
        return 1 if self.grayscale else 3


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
        # Frequency channel input: "none" | "wavelet" | "dct" | "fourier" | "fourier_hpf"
        freq_transform: str = "none",
        # Wavelet options (used when freq_transform="wavelet")
        wavelet_name: str = "haar",
        wavelet_mode: str = "symmetric",
        wavelet_levels: int = 1,
        wavelet_include_lowpass_channel: bool = True,
        # DCT options (used when freq_transform="dct")
        dct_block_size: int = 8,
        dct_num_coeffs: int = 6,
        # Fourier options (used when freq_transform="fourier" or "fourier_hpf")
        fourier_shift: bool = False,
        fourier_hpf_radius: float = 0.1,
        fourier_hpf_grayscale: bool = False,
        # MixUp (injected from DataModule)
        mixup_fn: Mixup | None = None,
        # Regularisation
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
        self.layer_lr_decay = layer_lr_decay
        self.mixup_fn = mixup_fn
        self.use_supcon = use_supcon
        self.supcon_weight = supcon_weight

        # ── Frequency channel transform (optional) ────────────────────────────
        freq_transform = freq_transform.lower()
        if freq_transform == "wavelet":
            self.freq_transform_layer: nn.Module | None = WaveletChannelTransform(
                wave=wavelet_name,
                mode=wavelet_mode,
                J=wavelet_levels,
                include_lowpass_channel=wavelet_include_lowpass_channel,
            )
            input_channels = self.freq_transform_layer.out_channels
        elif freq_transform == "dct":
            self.freq_transform_layer = DCTChannelTransform(
                block_size=dct_block_size,
                num_coeffs=dct_num_coeffs,
            )
            input_channels = self.freq_transform_layer.out_channels
        elif freq_transform == "fourier":
            self.freq_transform_layer = FourierChannelTransform(shift=fourier_shift)
            input_channels = self.freq_transform_layer.out_channels
        elif freq_transform == "fourier_hpf":
            self.freq_transform_layer = FourierHighPassTransform(
                lpf_radius=fourier_hpf_radius,
                grayscale=fourier_hpf_grayscale,
            )
            input_channels = self.freq_transform_layer.out_channels
        else:
            self.freq_transform_layer = None
            input_channels = 3
        self._use_freq_transform = self.freq_transform_layer is not None

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

        # ── Supervised Contrastive projection head ────────────────────────────
        if use_supcon:
            feat_dim = self.model.get_feature_dim()
            self.proj_head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, 128),
            )
            self.supcon_criterion = SupConLoss(
                temperature=supcon_temperature,
                ordinal_weights=supcon_ordinal,
            )
        else:
            self.proj_head = None
            self.supcon_criterion = None

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
        if self._use_freq_transform:
            x = self.freq_transform_layer(x)
        return self.model(x)

    # ── Steps ────────────────────────────────────────────────────────────────

    def training_step(self, batch):
        x, y = batch

        # ── Supervised Contrastive auxiliary loss ────────────────────────────
        # Computed before MixUp because SupCon needs hard integer labels.
        supcon_loss = torch.tensor(0.0, device=x.device)
        if self.use_supcon and self.proj_head is not None:
            x2 = self._tta_transform(x)  # second augmented view
            feats1 = self.proj_head(
                self.model.forward_head(self.model.forward_features(x),  pre_logits=True)
            )
            feats2 = self.proj_head(
                self.model.forward_head(self.model.forward_features(x2), pre_logits=True)
            )
            z = torch.stack([feats1, feats2], dim=1)  # [B, 2, 128]
            supcon_loss = self.supcon_criterion(z, y)

        # ── Classification loss ───────────────────────────────────────────────
        if self.mixup_fn is not None:
            x, y = self.mixup_fn(x, y)
        logits = self(x)
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
                configuration["lr_scheduler"] = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
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
