from __future__ import annotations

import io
import random
import time
import warnings

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
from torch import nn
import torchmetrics
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAccuracy,
    MulticlassCalibrationError,
    MulticlassCohenKappa,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
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
    """

    def __init__(self, lpf_radius: float = 0.1) -> None:
        super().__init__()
        self.lpf_radius = lpf_radius
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
        inp = x.float()  # [B, C, H, W] — per-channel

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
        return 3


# ---------------------------------------------------------------------------
class ModelProfilerCallback(L.Callback):
    """Logs parameter count, GFLOPs, and inference throughput once per run."""

    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size

    def _log_model_profile(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        from fvcore.nn import FlopCountAnalysis

        total_params     = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)

        dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=pl_module.device)
        flops = FlopCountAnalysis(pl_module, dummy)
        flops.unsupported_ops_warnings(False)
        gflops = flops.total() / 1e9

        print(f"\nModel profile:")
        print(f"  Total params     : {total_params / 1e6:.2f}M")
        print(f"  Trainable params : {trainable_params / 1e6:.2f}M")
        print(f"  GFLOPs           : {gflops:.2f}")

        if not trainer.is_global_zero:
            return
        if pl_module.logger is None or not hasattr(pl_module.logger, "experiment"):
            return
        exp = pl_module.logger.experiment
        if not hasattr(exp, "add_scalar"):
            return
        exp.add_scalar("model/total_params_M",     total_params / 1e6,     0)
        exp.add_scalar("model/trainable_params_M", trainable_params / 1e6, 0)
        exp.add_scalar("model/gflops",             gflops,                 0)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._log_model_profile(trainer, pl_module)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._log_model_profile(trainer, pl_module)


# ---------------------------------------------------------------------------
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
    y_probs: np.ndarray,
    num_classes: int,
    throughput: float | None = None,
    prefix: str = "test",
) -> None:
    """
    Log full-dataset test metrics to TensorBoard:
      - Scalar: {prefix}/kappa, {prefix}/acc, {prefix}/auc, {prefix}/ece,
                {prefix}/precision, {prefix}/recall, {prefix}/f1,
                {prefix}/throughput_imgs_per_sec
      - Text:   {prefix}/classification_report  (Text tab)
      - Image:  {prefix}/confusion_matrix       (Images tab)

    Only runs on global rank 0 (avoids duplicate DDP writes).
    """
    labels = list(range(num_classes))
    class_names = [f"DR{i}" for i in labels]

    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    acc   = float((y_pred == y_true).mean())
    per_prec, per_rec, per_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    probs_t  = torch.from_numpy(y_probs)
    labels_t = torch.from_numpy(y_true).long()
    auc = float(MulticlassAUROC(num_classes=num_classes, average="macro")(probs_t, labels_t))
    ece = float(MulticlassCalibrationError(num_classes=num_classes, n_bins=15)(probs_t, labels_t))

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(f"\n{'='*60}\n{prefix.upper()} Results\n{'='*60}")
    print(report)
    print(f"Quadratic Weighted Kappa : {kappa:.4f}")
    print(f"Macro OvR AUC            : {auc:.4f}")
    print(f"Expected Calibration Err : {ece:.4f}")
    if throughput is not None:
        print(f"Throughput               : {throughput:.1f} imgs/sec")

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
    exp.add_scalar(f"{prefix}/kappa",     kappa, step)
    exp.add_scalar(f"{prefix}/acc",       acc,   step)
    exp.add_scalar(f"{prefix}/auc",       auc,   step)
    exp.add_scalar(f"{prefix}/ece",       ece,   step)
    exp.add_scalar(f"{prefix}/precision", float(per_prec.mean()), step)
    exp.add_scalar(f"{prefix}/recall",    float(per_rec.mean()),  step)
    exp.add_scalar(f"{prefix}/f1",        float(per_f1.mean()),   step)
    if throughput is not None:
        exp.add_scalar(f"{prefix}/throughput_imgs_per_sec", throughput, step)

    # ── Text (visible in TensorBoard → Text tab) ──────────────────────────────
    exp.add_text(
        f"{prefix}/classification_report",
        f"<pre>{report}\nQWK: {kappa:.4f}  AUC: {auc:.4f}  ECE: {ece:.4f}</pre>",
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

# Models known to expose timm's `group_matcher`, so `param_groups_layer_decay`
# can split params into depth-ordered groups. Widened from ViT-only to cover
# every ConvNeXt / CoAtNet / MaxViT / EfficientNet backbone in the registry.
_LAYER_DECAY_MODELS = {
    "vit_base",
    "swin_base",
    "convnext_tiny",
    "convnext_base",
    "convnext_large",
    "coatnet_2",
    "maxvit_base",
    "efficientnetv2_m",
    "efficientnetv2_s",
    "efficientnet_b0",
    "efficientnet_b2",
}


class DRModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model_name: str = "efficientnet_b2",
        learning_rate: float = 4e-5,
        weight_decay: float = 1e-4,
        use_scheduler: bool = True,
        scheduler_type: str = "auto",
        freeze_backbone: bool = False,
        class_weights=None,
        label_smoothing: float = 0.0,
        loss_name: str = "cross_entropy",
        focal_gamma: float = 2.0,
        warmup_epochs: int = 0,
        scheduler_monitor: str = "val_kappa",
        scheduler_monitor_mode: str = "max",
        tta_enabled: bool = False,
        # `tta_runs` is deprecated: TTA now enumerates D4 views deterministically
        # (8 views per image). Retained as a keyword arg only so existing
        # checkpoints that saved `tta_runs` in their hparams still load.
        tta_runs: int | None = None,  # noqa: ARG002  (kept for ckpt compat)
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
        # MixUp (injected from DataModule)
        mixup_fn: Mixup | None = None,
        # Regularisation
        drop_rate: float = 0.3,
        drop_path_rate: float = 0.2,
        layer_lr_decay: float = 0.75,
        # Monte Carlo Dropout (uncertainty quantification at test time)
        mc_dropout_enabled: bool = False,
        mc_dropout_samples: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "mixup_fn"])

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_monitor_mode = scheduler_monitor_mode
        self.tta_enabled = tta_enabled
        self.layer_lr_decay = layer_lr_decay
        self.mixup_fn = mixup_fn
        self.mc_dropout_enabled = mc_dropout_enabled
        self.mc_dropout_samples = mc_dropout_samples
        self.loss_name = str(loss_name).lower()

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
            )
            input_channels = self.freq_transform_layer.out_channels
        else:
            self.freq_transform_layer = None
            input_channels = 3
        self._use_freq_transform = self.freq_transform_layer is not None
        # Stored so forward() can assert post-transform channel count matches
        # the backbone's expected in_chans on the first batch.
        self._backbone_in_chans = input_channels
        self._channel_check_done = False

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

        # MixUp produces soft targets; store class weights for explicit
        # weighted soft-target CE during training when needed.
        if class_weights is not None:
            self.register_buffer(
                "_mixup_class_weights",
                class_weights.detach().clone().float(),
            )
        else:
            self._mixup_class_weights = None

        # ── Epoch-level metrics (stateful — accumulate full confusion matrix) ──────
        _mc = dict(num_classes=num_classes)
        self._val_metrics = torchmetrics.MetricCollection({
            # Use micro accuracy so val/test_acc matches sklearn's report accuracy.
            "acc":       MulticlassAccuracy(**_mc, average="micro"),
            "kappa":     MulticlassCohenKappa(**_mc, weights="quadratic"),
            "f1":        MulticlassF1Score(**_mc, average="macro"),
            "precision": MulticlassPrecision(**_mc, average="macro"),
            "recall":    MulticlassRecall(**_mc, average="macro"),
        }, prefix="val_")
        self._test_metrics = self._val_metrics.clone(prefix="test_")

        # ── Prob-based metrics (require softmax probs, not argmax preds) ─────
        self._val_prob_metrics = torchmetrics.MetricCollection({
            "auc": MulticlassAUROC(**_mc, average="macro"),
            "ece": MulticlassCalibrationError(**_mc, n_bins=15),
        }, prefix="val_")
        self._test_prob_metrics = self._val_prob_metrics.clone(prefix="test_")

        # ── TTA: enumerated D4 (rotations) × {id, hflip}. VFlip is *omitted*
        # because fundus images have a hard up/down axis (optic disc ↔ macula)
        # and a vertical flip produces an anatomically impossible view. Result:
        # 4 × 2 = 8 deterministic forward passes averaged in probability space.
        # `tta_runs` is accepted for ckpt-load compatibility but unused.
        self._tta_rotations = (0, 1, 2, 3)
        self._tta_flip = (False, True)
        # Legacy random-view transform retained for callers in src/ensemble.py,
        # train_oof.py, and test_ensemble.py that still drive per-view TTA
        # externally. New code should use `tta_d4_predict` instead.
        self._tta_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda t: torch.rot90(t, k=random.randint(0, 3), dims=[-2, -1])),
        ])

        self._test_preds: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []
        self._test_probs: list[torch.Tensor] = []
        self._test_start_time: float = 0.0
        self._test_n_samples: int = 0

        # store model_name for use in configure_optimizers
        self._model_name = model_name

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_freq_transform:
            x = self.freq_transform_layer(x)
        if not self._channel_check_done:
            actual = x.shape[1]
            if actual != self._backbone_in_chans:
                raise RuntimeError(
                    f"Channel mismatch after freq_transform: got {actual} "
                    f"channels but backbone was built with "
                    f"in_chans={self._backbone_in_chans}. Check that the "
                    f"frequency transform layer's `out_channels` matches the "
                    f"value passed to ModelFactory."
                )
            self._channel_check_done = True
        return self.model(x)

    def _weighted_soft_target_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy for soft targets with optional class weighting.

            For MixUp targets with class weights, compute the exact weighted
            soft-target CE:
                numerator_i   = -sum_c target_i,c * weight_c * log_prob_i,c
                denominator_i =  sum_c target_i,c * weight_c
            and return sum_i numerator_i / sum_i denominator_i.
        """
        log_probs = F.log_softmax(logits, dim=1)

        if self._mixup_class_weights is None:
            per_sample_loss = -(targets * log_probs).sum(dim=1)
            return per_sample_loss.mean()

        weights = self._mixup_class_weights.to(device=logits.device, dtype=logits.dtype)
        weighted_targets = targets * weights.unsqueeze(0)
        per_sample_numerator = -(weighted_targets * log_probs).sum(dim=1)
        per_sample_denominator = weighted_targets.sum(dim=1)
        return per_sample_numerator.sum() / per_sample_denominator.sum().clamp_min(1e-8)

    # ── Steps ────────────────────────────────────────────────────────────────

    def training_step(self, batch):
        x, y = batch

        if self.mixup_fn is not None:
            x, y = self.mixup_fn(x, y)
        logits = self(x)

        # MixUp returns soft targets [B, C]. Use explicit soft-target CE so
        # class_weights remain effective with balancing_mode=weighted_loss.
        if y.ndim == 2 and self.loss_name == "cross_entropy":
            loss = self._weighted_soft_target_cross_entropy(logits, y)
        else:
            loss = self.criterion(logits, y)

        # on_epoch only: avoid emitting both train_loss_step and train_loss_epoch
        # series to TensorBoard. The running epoch average is what we compare
        # to val_loss anyway.
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # ── Monte Carlo Dropout ───────────────────────────────────────────────────

    @torch.no_grad()
    def tta_d4_predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enumerated D4 TTA (without VFlip): 4 rotations × {identity, HFlip} = 8
        forward passes. Returns the mean softmax probability tensor
        `[B, num_classes]`. Averaging in probability space (not logit space)
        matches standard practice.
        """
        probs = torch.zeros(x.size(0), self.num_classes, device=x.device)
        n_views = 0
        for k in self._tta_rotations:
            rot = torch.rot90(x, k=k, dims=[-2, -1]) if k else x
            for do_flip in self._tta_flip:
                view = torch.flip(rot, dims=[-1]) if do_flip else rot
                probs += torch.softmax(self(view), dim=1)
                n_views += 1
        return probs / n_views

    def mc_predict(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run `mc_dropout_samples` stochastic forward passes with dropout enabled.

        Returns
        -------
        mean_probs : [B, num_classes]  mean softmax probability across all passes
        entropy    : [B]               predictive entropy H = -Σ p log p
                                       (higher → more uncertain; flag for review)
        """
        # Enable only Dropout layers; everything else stays in eval mode.
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        with torch.no_grad():
            passes = torch.stack(
                [torch.softmax(self(x), dim=1) for _ in range(self.mc_dropout_samples)],
                dim=0,
            )  # [S, B, C]

        # Restore full eval mode.
        self.eval()

        mean_probs = passes.mean(dim=0)  # [B, C]
        entropy = -(mean_probs * (mean_probs + 1e-10).log()).sum(dim=1)  # [B]
        return mean_probs, entropy

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        self._val_metrics.update(preds, y)
        self._val_prob_metrics.update(probs, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        prog_bar_keys = {"val_kappa", "val_acc", "val_f1"}
        for name, value in {**self._val_metrics.compute(), **self._val_prob_metrics.compute()}.items():
            self.log(name, value, prog_bar=(name in prog_bar_keys), sync_dist=True)
        self._val_metrics.reset()
        self._val_prob_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.mc_dropout_enabled:
            probs, entropy = self.mc_predict(x)
            loss  = self.criterion(torch.log(probs.clamp_min(1e-8)), y)
            self.log("test_mc_entropy", entropy.mean(), on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=True)
        elif self.tta_enabled:
            probs = self.tta_d4_predict(x)
            # Loss computed once on the identity view so it stays comparable
            # to non-TTA test_loss. Per-view loss averaging would mix
            # orientations into a single scalar with no physical meaning.
            loss = self.criterion(self(x), y)
        else:
            logits = self(x)
            loss   = self.criterion(logits, y)
            probs  = torch.softmax(logits, dim=1)

        preds = torch.argmax(probs, dim=1)
        self._test_metrics.update(preds, y)
        self._test_prob_metrics.update(probs, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)

        self._test_preds.append(preds.detach().cpu())
        self._test_targets.append(y.detach().cpu())
        self._test_probs.append(probs.detach().cpu())
        self._test_n_samples += x.size(0)

    def on_test_epoch_start(self) -> None:
        self._test_preds = []
        self._test_targets = []
        self._test_probs = []
        self._test_n_samples = 0
        self._test_start_time = time.perf_counter()

    def on_test_epoch_end(self) -> None:
        elapsed = time.perf_counter() - self._test_start_time
        throughput = self._test_n_samples / elapsed if elapsed > 0 else None

        for name, value in {**self._test_metrics.compute(), **self._test_prob_metrics.compute()}.items():
            self.log(name, value, prog_bar=True, sync_dist=True)
        self._test_metrics.reset()
        self._test_prob_metrics.reset()

        if not self._test_preds:
            return
        y_pred  = torch.cat(self._test_preds).numpy()
        y_true  = torch.cat(self._test_targets).numpy()
        y_probs = torch.cat(self._test_probs).numpy()

        # In DDP, gather rank-local test outputs so report metrics are global.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: list[dict] = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(
                gathered,
                {
                    "y_pred": y_pred,
                    "y_true": y_true,
                    "y_probs": y_probs,
                    "n_samples": int(self._test_n_samples),
                },
            )

            if not self.trainer.is_global_zero:
                return

            y_pred = np.concatenate([item["y_pred"] for item in gathered], axis=0)
            y_true = np.concatenate([item["y_true"] for item in gathered], axis=0)
            y_probs = np.concatenate([item["y_probs"] for item in gathered], axis=0)
            global_n_samples = sum(int(item["n_samples"]) for item in gathered)
            throughput = global_n_samples / elapsed if elapsed > 0 else None

        log_test_results(self, y_pred, y_true, y_probs, self.num_classes, throughput=throughput)

    # ── Optimiser ────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        supports_layer_decay = self._model_name in _LAYER_DECAY_MODELS
        if supports_layer_decay and self.layer_lr_decay < 1.0:
            try:
                from timm.optim import param_groups_layer_decay
                param_groups = param_groups_layer_decay(
                    self.model.backbone,
                    weight_decay=self.weight_decay,
                    layer_decay=self.layer_lr_decay,
                )
                optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate)
                self.print(
                    f"[layer-decay] {self._model_name}: "
                    f"{len(param_groups)} param groups, layer_decay={self.layer_lr_decay}"
                )
            except (AttributeError, NotImplementedError, KeyError, TypeError, ValueError) as e:
                warnings.warn(
                    f"[layer-decay] `{self._model_name}` is in _LAYER_DECAY_MODELS but "
                    f"timm.param_groups_layer_decay raised {type(e).__name__}: {e}. "
                    "Falling back to a flat AdamW. Remove the model from _LAYER_DECAY_MODELS "
                    "or fix the backbone so this fallback is not silently hit.",
                    stacklevel=2,
                )
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

        if not self.use_scheduler:
            return configuration

        scheduler_type = str(self.scheduler_type).lower()
        if scheduler_type == "auto":
            scheduler_type = "warmup_cosine" if self.warmup_epochs > 0 else "plateau"

        if scheduler_type in {"warmup_cosine", "cosine_warmup"}:
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

        elif scheduler_type in {"plateau", "reduce_on_plateau"}:
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

        else:
            raise ValueError(
                f"Unsupported scheduler_type '{self.scheduler_type}'. "
                "Expected: auto, warmup_cosine, plateau."
            )

        return configuration
