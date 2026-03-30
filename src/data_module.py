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
        balancing_mode: str = "weighted_loss",  # "naive_oversample", "sampler", "weighted_loss"
        normalization_mode: str = "dataset_by_size",  # "imagenet", "dataset_by_size", "custom"
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

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        dataset_stats_by_size = {
            260: {
                "mean": [0.5157322866781002, 0.5127125195333987, 0.5109832913220571],
                "std": [0.22462066781120452, 0.20958938594022844, 0.18132136224659612],
            },
            512: {
                "mean": [0.49569857915740306, 0.49564579707226114, 0.4927175974949744],
                "std": [0.280003501286239, 0.2782336760044972, 0.2515219633188661],
            },
             224: {
                "mean": [0.4798033789035061, 0.4753913263012517, 0.4721207172806979],
                "std": [0.29983359231305656, 0.28349450929746645, 0.2526360182164682],
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
        
        # Training augmentations
        self.train_transform = T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                # Combine Rotation, Scale, and Shear into one operation to preserve quality.
                # FILL=[128, 128, 128] is critical to match the new 50% gray background.
                T.RandomAffine(
                    degrees=(0, 360),
                    scale=(0.9, 1.1),
                    shear=(-11, 11), # Added slight shear mimicking Ben Graham's original paper
                    fill=[128, 128, 128]
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),

                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),

                T.Normalize(mean=self.dataset_mean, std=self.dataset_std),
                
                # RandomErasing value set to roughly 0.5 to drop gray patches instead of black
                T.RandomErasing(p=0.3, scale=(0.02, 0.1), value=0.5),
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
            T.Normalize(mean=self.dataset_mean, std=self.dataset_std),
            T.RandomErasing(p=0.2, scale=(0.05, 0.05), ratio=(0.5, 0.5)),
        ])
        # Val/test: deterministic resize + normalize only
        self.val_transform = T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=self.dataset_mean, std=self.dataset_std),
            ]
        )

    def setup(self, stage=None):
        """Set up datasets for training, validation, and optionally test."""
        self.sampler = None
        self.class_weights = None

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
        elif self.balancing_mode in {"sampler", "weighted_sampler"}:
            self.train_dataset = DRDataset(self.train_csv_path, transform=self.train_transform)
            labels = self.train_dataset.labels.numpy()
            counts = np.bincount(labels)
            class_sample_weights = 1.0 / counts
            sample_weights = class_sample_weights[labels]
            self.sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=len(labels), replacement=True
            )
        elif self.balancing_mode == "weighted_loss":
            # No oversampling — compute class weights for the loss function instead
            self.train_dataset = DRDataset(self.train_csv_path, transform=self.train_transform)
            labels = self.train_dataset.labels.numpy()
            weights = compute_class_weight(
                class_weight="balanced", classes=np.unique(labels), y=labels
            )
            self.class_weights = torch.tensor(weights, dtype=torch.float32)
        else:
            raise ValueError(
                "Invalid balancing_mode. Expected one of: naive_oversample, sampler, weighted_loss"
            )

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
