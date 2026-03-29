import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2 as T

from src.dataset import DRDataset


class DRDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_csv_path,
        val_csv_path,
        test_csv_path=None,
        image_size: int = 224,
        batch_size: int = 8,
        num_workers: int = 4,
        use_oversampling: bool = False,
        balancing_mode: str = "weighted_loss",  # "naive_oversample", "weighted_sampler", "weighted_loss"
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.test_csv_path = test_csv_path
        self.use_oversampling = use_oversampling
        self.balancing_mode = balancing_mode

        self.sampler = None

        # Training augmentations
        self.train_transform = T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.RandomRotation(degrees=(0, 360), fill=0),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomAffine(degrees=0, scale=(0.9, 1.1), fill=0),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.3, scale=(0.02, 0.1)),
            ]
        )

        self.train_transform_jupiter = T.Compose([
            T.Resize((image_size, image_size), antialias=True),
            T.RandomAffine(degrees=10, translate=(0.01, 0.01), scale=(0.99, 1.01)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.01),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomVerticalFlip(p=0.5),
            T.GaussianBlur(kernel_size=3),  # You can adjust kernel size as needed
            T.RandomResizedCrop(size=(image_size, image_size), scale=(0.99, 1.0), ratio=(0.75, 1.333)),
            T.RandomAdjustSharpness(sharpness_factor=2),
            T.RandomAutocontrast(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.2, scale=(0.05, 0.05), ratio=(0.5, 0.5)),
        ])
        # Val/test: deterministic resize + normalize only
        self.val_transform = T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def setup(self, stage=None):
        """Set up datasets for training, validation, and optionally test."""
        train_df = pd.read_csv(self.train_csv_path)
        class_counts = train_df["label"].value_counts()

        if self.balancing_mode == "naive_oversample":           # Oversample minority classes to match the majority class count
            max_count = class_counts.max()
            oversampled_parts = []
            for label, group in train_df.groupby("label"):
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
            oversampled_csv = self.train_csv_path.replace(".csv", "_oversampled.csv")
            oversampled_df.to_csv(oversampled_csv, index=False)
            self.train_dataset = DRDataset(oversampled_csv, transform=self.train_transform)
            self.class_weights = None
        elif self.balancing_mode == "weighted_sampler":
            self.train_dataset = DRDataset(self.train_csv_path, transform=self.train_transform)
            labels = self.train_dataset.labels.numpy()
            counts = np.bincount(labels)
            class_sample_weights = 1.0 / counts
            sample_weights = class_sample_weights[labels]
            self.sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=len(labels), replacement=True
            )
            self.class_weights = None
        else:
            # No oversampling — compute class weights for the loss function instead
            self.train_dataset = DRDataset(self.train_csv_path, transform=self.train_transform)
            labels = self.train_dataset.labels.numpy()
            weights = compute_class_weight(
                class_weight="balanced", classes=np.unique(labels), y=labels
            )
            self.class_weights = torch.tensor(weights, dtype=torch.float32)

        self.val_dataset = DRDataset(self.val_csv_path, transform=self.val_transform)

        if self.test_csv_path:
            self.test_dataset = DRDataset(
                self.test_csv_path, transform=self.val_transform
            )

        self.num_classes = len(class_counts)

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
