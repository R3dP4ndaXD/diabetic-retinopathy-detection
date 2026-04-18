from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning as L
import numpy as np
import pandas as pd
import random
import torch
import os
from sklearn.utils.class_weight import compute_class_weight
from timm.data import Mixup
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2 as T

from src.dataset import DRDataset


def _worker_init_fn(worker_id: int) -> None:
    """
    Seed NumPy and Python's `random` per DataLoader worker. Without this, every
    worker inherits the same NumPy seed at process fork and Albumentations /
    NumPy-based augmentations produce the same sequence across epochs inside
    a worker. `torch.initial_seed()` already differs per-worker per-epoch
    because Lightning/PyTorch seed workers with `base_seed + worker_id`.
    """
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class AlbumentationsWrapper(nn.Module):
    """
    Wraps an Albumentations transform so it can be used as a torchvision-style
    transform.  DRDataset feeds uint8 tensors [C, H, W]; Albumentations expects
    numpy [H, W, C].  This bridge handles both directions.
    """

    def __init__(self, transform: A.Compose) -> None:
        super().__init__()
        self.transform = transform

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous uint8 [C, H, W] before numpy conversion.
        # read_image() returns uint8; guard against float inputs (e.g. cached tensors).
        if img.dtype != torch.uint8:
            img = (img * 255).clamp(0, 255).to(torch.uint8)
        arr = img.permute(1, 2, 0).contiguous().numpy()  # [H, W, C] uint8
        out = self.transform(image=arr)["image"]          # float32 tensor [C, H, W]
        return out


class DRDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for diabetic retinopathy grading.

    Parameters
    ----------
    norm_mean / norm_std
        Per-channel normalisation statistics.  Obtain from
        ``src.models.factory.get_normalization_stats(model_name)``.

    balancing_mode
        "naive_oversample" – duplicate minority rows in a sidecar CSV
        "sampler"          – WeightedRandomSampler with 1/count weights
        "smote"            – load pre-generated SMOTE CSV (train_smote.csv); run
                             scripts/generate_smote_images.py first to create it
        "weighted_loss"    – inverse-frequency class weights passed to loss

    use_mixup / mixup_*
        If use_mixup=True a timm Mixup object is created and exposed as
        self.mixup_fn for the LightningModule's training_step to consume.
    """

    def __init__(
        self,
        train_csv_path: str,
        val_csv_path: str,
        test_csv_path: str | None,
        norm_mean: list[float],
        norm_std: list[float],
        image_size: int = 224,
        batch_size: int = 8,
        num_workers: int = 4,
        balancing_mode: str = "weighted_loss",
        # MixUp / CutMix
        use_mixup: bool = True,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        label_smoothing: float = 0.1,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.test_csv_path = test_csv_path
        self.balancing_mode = balancing_mode

        self.sampler = None
        self.class_weights = None

        self.dataset_mean = list(norm_mean)
        self.dataset_std  = list(norm_std)

        # ── Transforms ───────────────────────────────────────────────────────
        # Old torchvision pipeline (kept for reference):
        self.train_transform = T.Compose([
            T.Resize((image_size, image_size), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomAffine(
                degrees=(0, 360),
                scale=(0.9, 1.1),
                shear=(-11, 11),
                fill=(128, 128, 128),
            ),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.RandomErasing(p=0.25, scale=(0.02, 0.1), value=0.5),
            T.Normalize(mean=self.dataset_mean, std=self.dataset_std),
        ])
        self.val_transform = T.Compose([
            T.Resize((image_size, image_size), antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=self.dataset_mean, std=self.dataset_std),
        ])

        # Albumentations pipeline matching the EyePACS-APTOS-Messidor baseline:
        #   A.CLAHE         — stochastic contrast enhancement (p=0.7)
        #   A.RandomRotate90 — 0/90/180/270° with equal probability
        # ToTensorV2 outputs a float32 tensor [C, H, W] already in [0, 1].
        # self.train_transform = AlbumentationsWrapper(A.Compose([
        #     A.Resize(image_size, image_size),
        #     A.HorizontalFlip(p=0.5),
        #     A.VerticalFlip(p=0.5),
        #     A.RandomRotate90(p=0.5),
        #     A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.7),
        #     A.Normalize(mean=self.dataset_mean, std=self.dataset_std),
        #     ToTensorV2(),
        # ]))

        # self.val_transform = AlbumentationsWrapper(A.Compose([
        #     A.Resize(image_size, image_size),
        #     A.Normalize(mean=self.dataset_mean, std=self.dataset_std),
        #     ToTensorV2(),
        # ]))
        # ── MixUp / CutMix ───────────────────────────────────────────────────
        # Exposed as self.mixup_fn; the LightningModule applies it in
        # training_step.  We build it here so it shares label_smoothing with
        # the rest of the config, but we need num_classes to be known.
        if use_mixup:
            self.mixup_fn: Mixup | None = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                prob=mixup_prob,
                switch_prob=0.5,
                mode="batch",
                label_smoothing=label_smoothing,
                num_classes=num_classes,
            )
        else:
            self.mixup_fn = None

    # ── Oversampling helpers ─────────────────────────────────────────────────

    def _get_oversampled_csv_path(self) -> str:
        return self.train_csv_path.replace(".csv", "_oversampled.csv")

    def _ensure_oversampled_csv_exists(self) -> str:
        oversampled_csv = self._get_oversampled_csv_path()
        if os.path.exists(oversampled_csv):
            return oversampled_csv

        train_df = pd.read_csv(self.train_csv_path)
        class_counts = train_df["label"].value_counts()
        max_count = class_counts.max()
        parts = []

        for _, group in train_df.groupby("label"):
            n = len(group)
            if n < max_count:
                full_repeats = max_count // n
                remainder    = max_count % n
                repeated = pd.concat([group] * full_repeats, ignore_index=True)
                if remainder > 0:
                    repeated = pd.concat(
                        [repeated, group.sample(n=remainder, random_state=42)],
                        ignore_index=True,
                    )
                parts.append(repeated)
            else:
                parts.append(group)

        oversampled_df = pd.concat(parts, ignore_index=True)
        oversampled_df.to_csv(oversampled_csv, index=False)
        return oversampled_csv

    # ── Lightning hooks ──────────────────────────────────────────────────────

    def prepare_data(self) -> None:
        """Runs once on the main process only."""
        if self.balancing_mode == "naive_oversample":
            self._ensure_oversampled_csv_exists()

    def setup(self, stage: str | None = None) -> None:
        if self.balancing_mode == "naive_oversample":
            oversampled_csv = self._ensure_oversampled_csv_exists()
            self.train_dataset = DRDataset(oversampled_csv, transform=self.train_transform)
            train_labels = self.train_dataset.labels.numpy()

        elif self.balancing_mode == "sampler":
            self.train_dataset = DRDataset(self.train_csv_path, transform=self.train_transform)
            train_labels = self.train_dataset.labels.numpy()
            counts = np.bincount(train_labels)
            class_sample_weights = 1.0 / counts
            sample_weights = class_sample_weights[train_labels]
            self.sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_labels),
                replacement=True,
            )

        elif self.balancing_mode == "smote":
            smote_csv = self.train_csv_path.replace(".csv", "_smote.csv")
            if not os.path.exists(smote_csv):
                raise FileNotFoundError(
                    f"SMOTE CSV not found: {smote_csv}\n"
                    "Run scripts/generate_smote_images.py first to generate it."
                )
            self.train_dataset = DRDataset(smote_csv, transform=self.train_transform)
            train_labels = self.train_dataset.labels.numpy()

        elif self.balancing_mode == "weighted_loss":
            self.train_dataset = DRDataset(self.train_csv_path, transform=self.train_transform)
            train_labels = self.train_dataset.labels.numpy()
            weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(train_labels),
                y=train_labels,
            )
            self.class_weights = torch.tensor(weights, dtype=torch.float32)

        else:
            raise ValueError(
                f"Invalid balancing_mode '{self.balancing_mode}'. "
                "Expected: naive_oversample, sampler, smote, weighted_loss."
            )

        self.num_classes = len(np.unique(train_labels))
        self.val_dataset  = DRDataset(self.val_csv_path,  transform=self.val_transform)
        if self.test_csv_path:
            self.test_dataset = DRDataset(self.test_csv_path, transform=self.val_transform)
        else:
            self.test_dataset = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            drop_last=self.mixup_fn is not None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_worker_init_fn,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("No test_csv_path was provided.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_worker_init_fn,
        )
