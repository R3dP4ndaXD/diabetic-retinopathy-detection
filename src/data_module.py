import lightning as L
import numpy as np
import pandas as pd
import torch
import os
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2 as T

from src.dataset import DRDataset


class DRDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_csv_path,
        val_csv_path,
        test_csv_path,
        image_size: int = 224,
        batch_size: int = 8,
        num_workers: int = 4,
        balancing_mode: str = "weighted_loss",
        normalization_mode: str = "dataset_by_size",
        custom_mean=None,
        custom_std=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.test_csv_path = test_csv_path
        self.balancing_mode = balancing_mode
        self.normalization_mode = normalization_mode
        self.custom_mean = custom_mean
        self.custom_std = custom_std

        self.sampler = None
        self.class_weights = None

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        dataset_stats_by_size = {
            260: {
                "mean": [
                    0.531122041515047,
                    0.5182638704826283,
                    0.5062381632560897
                ],
                "std": [
                    0.16232073010217313,
                    0.1638348223458083,
                    0.15570514150297027
                ]
            },
            224: {
                  "mean": [
                    0.5402626813088397,
                    0.5251738955848213,
                    0.5114480908180413
                ],
                "std": [
                    0.17024819115805778,
                    0.17033830508027195,
                    0.16347060175144904
                ]
            },
        }

        if self.normalization_mode == "imagenet":
            self.dataset_mean = imagenet_mean
            self.dataset_std = imagenet_std
        elif self.normalization_mode == "dataset_by_size":
            if image_size not in dataset_stats_by_size:
                available = ", ".join(str(k) for k in sorted(dataset_stats_by_size.keys()))
                raise ValueError(
                    f"No dataset stats configured for image_size={image_size}. Available sizes: {available}. "
                    "Use normalization_mode='imagenet' or provide custom_mean/custom_std."
                )
            self.dataset_mean = dataset_stats_by_size[image_size]["mean"]
            self.dataset_std = dataset_stats_by_size[image_size]["std"]
        elif self.normalization_mode == "custom":
            if not self.custom_mean or not self.custom_std:
                raise ValueError(
                    "custom_mean and custom_std must be provided when normalization_mode='custom'."
                )
            if len(self.custom_mean) != 3 or len(self.custom_std) != 3:
                raise ValueError("custom_mean and custom_std must each have exactly 3 values (RGB).")
            self.dataset_mean = list(self.custom_mean)
            self.dataset_std = list(self.custom_std)
        else:
            raise ValueError(
                "Invalid normalization_mode. Expected one of: imagenet, dataset_by_size, custom"
            )
        
        self.train_transform = T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.RandomAffine(
                    degrees=(0, 360),
                    scale=(0.9, 1.1),
                    shear=(-11, 11), 
                    fill=(128, 128, 128) # Fixed: Use tuple instead of list for safety
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                
                # Fixed: Erase in the [0, 1] space before normalization distorts the values
                T.RandomErasing(p=0.3, scale=(0.02, 0.1), value=0.5),
                
                T.Normalize(mean=self.dataset_mean, std=self.dataset_std),
            ]
        )

        self.val_transform = T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=self.dataset_mean, std=self.dataset_std),
            ]
        )

    def _get_oversampled_csv_path(self) -> str:
        return self.train_csv_path.replace(".csv", "_oversampled.csv")

    def _ensure_oversampled_csv_exists(self) -> str:
        oversampled_csv = self._get_oversampled_csv_path()

        if os.path.exists(oversampled_csv):
            return oversampled_csv

        train_df = pd.read_csv(self.train_csv_path)
        class_counts = train_df["label"].value_counts()
        max_count = class_counts.max()
        oversampled_parts = []

        for _, group in train_df.groupby("label"):
            n = len(group)
            if n < max_count:
                full_repeats = max_count // n
                remainder = max_count % n
                repeated = pd.concat([group] * full_repeats, ignore_index=True)
                if remainder > 0:
                    repeated = pd.concat(
                        [repeated, group.sample(n=remainder, random_state=42)],
                        ignore_index=True,
                    )
                oversampled_parts.append(repeated)
            else:
                oversampled_parts.append(group)

        oversampled_df = pd.concat(oversampled_parts, ignore_index=True)
        oversampled_df.to_csv(oversampled_csv, index=False)
        return oversampled_csv

    def prepare_data(self):
        """
        Runs exactly once on the main process. 
        """
        if self.balancing_mode == "naive_oversample":
            self._ensure_oversampled_csv_exists()

    def setup(self, stage=None):
        """Set up datasets for training, validation, and optionally test."""
        
        if self.balancing_mode == "naive_oversample":
            # setup can be called before prepare_data in custom flows.
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
                weights=sample_weights, num_samples=len(train_labels), replacement=True
            )
            
        elif self.balancing_mode == "weighted_loss":
            self.train_dataset = DRDataset(self.train_csv_path, transform=self.train_transform)
            train_labels = self.train_dataset.labels.numpy()
            weights = compute_class_weight(
                class_weight="balanced", classes=np.unique(train_labels), y=train_labels
            )
            self.class_weights = torch.tensor(weights, dtype=torch.float32)
            
        else:
            raise ValueError(
                "Invalid balancing_mode. Expected one of: naive_oversample, sampler, weighted_loss"
            )

        self.num_classes = len(np.unique(train_labels))
        self.val_dataset = DRDataset(self.val_csv_path, transform=self.val_transform)
        self.test_dataset = DRDataset(self.test_csv_path, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )