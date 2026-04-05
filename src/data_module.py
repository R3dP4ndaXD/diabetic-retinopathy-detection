from __future__ import annotations

import lightning as L
import numpy as np
import pandas as pd
import torch
import os
from sklearn.utils.class_weight import compute_class_weight
from timm.data import Mixup
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2 as T

from src.dataset import DRDataset


class DRDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for diabetic retinopathy grading.

    Parameters
    ----------
    normalization_mode
        "timm"           – use mean/std from a timm model's data_config
                           (pass timm_data_config as well)
        "imagenet"       – standard ImageNet stats
        "dataset_by_size"– pre-computed dataset stats keyed by image_size
                           (supported sizes: 224, 260)

    balancing_mode
        "naive_oversample" – duplicate minority rows in a sidecar CSV
        "sampler"          – WeightedRandomSampler on the original CSV
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
        image_size: int = 224,
        batch_size: int = 8,
        num_workers: int = 4,
        balancing_mode: str = "weighted_loss",
        normalization_mode: str = "timm",
        timm_data_config: dict | None = None,
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

        # ── Normalization ────────────────────────────────────────────────────
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std  = [0.229, 0.224, 0.225]

        dataset_stats_by_size = {
            260: {
                "mean": [0.531122041515047,  0.5182638704826283, 0.5062381632560897],
                "std":  [0.16232073010217313, 0.1638348223458083, 0.15570514150297027],
            },
            224: {
                "mean": [0.5402626813088397,  0.5251738955848213, 0.5114480908180413],
                "std":  [0.17024819115805778, 0.17033830508027195, 0.16347060175144904],
            },
        }

        if normalization_mode == "timm":
            if timm_data_config is None:
                raise ValueError(
                    "timm_data_config must be provided when normalization_mode='timm'. "
                    "Create the model first, then pass model.data_config here."
                )
            self.dataset_mean = list(timm_data_config["mean"])
            self.dataset_std  = list(timm_data_config["std"])

        elif normalization_mode == "imagenet":
            self.dataset_mean = imagenet_mean
            self.dataset_std  = imagenet_std

        elif normalization_mode == "dataset_by_size":
            if image_size not in dataset_stats_by_size:
                available = ", ".join(str(k) for k in sorted(dataset_stats_by_size))
                raise ValueError(
                    f"No dataset stats for image_size={image_size}. "
                    f"Available: {available}. "
                    "Use normalization_mode='imagenet'."
                )
            self.dataset_mean = dataset_stats_by_size[image_size]["mean"]
            self.dataset_std  = dataset_stats_by_size[image_size]["std"]

        else:
            raise ValueError(
                f"Invalid normalization_mode '{normalization_mode}'. "
                "Expected one of: timm, imagenet, dataset_by_size."
            )

        # ── Transforms ───────────────────────────────────────────────────────
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
            T.RandAugment(num_ops=2, magnitude=9, fill=(128, 128, 128)),
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

        elif self.balancing_mode in {"sampler", "weighted_sampler"}:
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
                "Expected: naive_oversample, sampler, weighted_loss."
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
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
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
        )
