"""
Generate synthetic minority-class training images via feature-space SMOTE variants.

Pipeline
--------
1. Load a timm backbone in feature-extraction mode (from a .ckpt checkpoint or
   directly from pretrained timm weights).
2. Extract L2-normalised embeddings for the target minority classes.
3. Run an imbalanced-learn oversampler on the feature matrix to produce
   synthetic feature vectors:
     - smote          : standard SMOTE (baseline)
     - borderline     : BorderlineSMOTE — focuses on samples near the class
                        boundary (grade-3/4 images closest to grade-2)
     - adasyn         : ADASYN — generates more samples where the class
                        density is sparse / harder to learn
     - kmeans         : KMeansSMOTE — clusters minority class first so
                        synthetic points stay within realistic sub-groups
4. Recover pixel-blending triplets from each synthetic vector:
   Each synthetic feature lies on a line segment between two real samples.
   Find the 2-NN of f_syn in the original feature set → anchor + neighbour.
   Compute λ = projection of f_syn onto that segment.
5. Blend:  img_syn = img_anchor + λ * (img_neighbour - img_anchor)
6. Save as JPEG, write augmented CSV.

Usage
-----
# BorderlineSMOTE from a trained checkpoint (recommended):
python scripts/generate_smote_images.py \\
    --checkpoint artifacts/checkpoints/<run_id>/epoch=N-....ckpt \\
    --train-csv  data/.../train.csv \\
    --output-dir data/.../smote_synthetic \\
    --output-csv data/.../train_smote.csv \\
    --method borderline \\
    --target-classes 3 4 \\
    --target-count 2000

# ADASYN without a checkpoint (uses pretrained ImageNet features):
python scripts/generate_smote_images.py \\
    --model-name efficientnet_b2 \\
    --train-csv  data/.../train.csv \\
    --output-dir data/.../smote_synthetic \\
    --method adasyn

Available methods: smote | borderline | adasyn | kmeans
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from timm.data import resolve_data_config, create_transform
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Minimal image dataset for feature extraction (no augmentation)
# ---------------------------------------------------------------------------

class _ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[str], transform) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), idx


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_backbone(
    checkpoint: str | None,
    model_name: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Returns (feature_extractor, timm_data_config)."""
    if checkpoint:
        print(f"Loading backbone from checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        hp = ckpt.get("hyper_parameters", {})
        model_name_ckpt = hp.get("model_name", model_name)
        print(f"  → model_name in checkpoint: {model_name_ckpt}")
        try:
            from src.models.factory import TIMM_MODEL_REGISTRY
            timm_id, _ = TIMM_MODEL_REGISTRY[model_name_ckpt]
        except (ImportError, KeyError):
            timm_id = model_name_ckpt

        backbone = timm.create_model(timm_id, pretrained=False, num_classes=0)
        state = ckpt["state_dict"]
        for prefix in ("model.backbone.", "model.model."):
            extracted = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if extracted:
                backbone.load_state_dict(extracted, strict=False)
                break
        else:
            print("  [warn] Could not extract backbone — falling back to pretrained timm weights.")
            backbone = timm.create_model(timm_id, pretrained=True, num_classes=0)
    else:
        print(f"No checkpoint — using pretrained timm: {model_name}")
        try:
            from src.models.factory import TIMM_MODEL_REGISTRY
            timm_id, _ = TIMM_MODEL_REGISTRY[model_name]
        except (ImportError, KeyError):
            timm_id = model_name
        backbone = timm.create_model(timm_id, pretrained=True, num_classes=0)

    backbone.eval().to(device)
    return backbone, resolve_data_config({}, model=backbone)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    backbone: torch.nn.Module,
    image_paths: list[str],
    data_config: dict,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> np.ndarray:
    """Returns L2-normalised feature matrix [N, D]."""
    transform = create_transform(
        input_size=data_config["input_size"],
        mean=data_config["mean"],
        std=data_config["std"],
        is_training=False,
    )
    loader = DataLoader(
        _ImagePathDataset(image_paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    feats = []
    for imgs, _ in tqdm(loader, desc="Extracting features", leave=False):
        f = backbone(imgs.to(device))
        feats.append(F.normalize(f, dim=1).cpu().float().numpy())
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Imbalanced-learn oversampler factory
# ---------------------------------------------------------------------------

def build_sampler(method: str, k_neighbors: int, target_count: int, labels: np.ndarray):
    """
    Returns a fitted imbalanced-learn sampler.

    sampling_strategy: dict mapping each target class to the desired
    *total* number of samples (original + synthetic).
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    sampling_strategy = {
        int(cls): int(target_count)
        for cls, count in zip(unique_labels, counts)
        if int(count) < int(target_count)
    }

    if not sampling_strategy:
        return None

    common = dict(sampling_strategy=sampling_strategy, random_state=42)

    if method == "smote":
        from imblearn.over_sampling import SMOTE
        return SMOTE(k_neighbors=k_neighbors, **common)

    elif method == "borderline":
        from imblearn.over_sampling import BorderlineSMOTE
        # kind="borderline-1": only borderline samples used as seeds
        return BorderlineSMOTE(k_neighbors=k_neighbors, kind="borderline-1", **common)

    elif method == "adasyn":
        from imblearn.over_sampling import ADASYN
        # n_neighbors controls local density estimation
        return ADASYN(n_neighbors=k_neighbors, **common)

    elif method == "kmeans":
        from imblearn.over_sampling import KMeansSMOTE
        return KMeansSMOTE(k_neighbors=k_neighbors, **common)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose: smote, borderline, adasyn, kmeans")


# ---------------------------------------------------------------------------
# Recover blending triplets from synthetic feature vectors
# ---------------------------------------------------------------------------

def recover_triplets(
    f_synthetic: np.ndarray,           # [N_syn, D]  — synthetic feature vectors
    f_real: np.ndarray,                 # [N_real, D] — original class features
) -> list[tuple[int, int, float]]:
    """
    For each synthetic vector f_syn, find the two nearest real samples.
    Those are the anchor (n1) and neighbour (n2) that were interpolated.
    λ = projection of f_syn onto the (n1 → n2) segment, clamped to [0, 1].

    Returns list of (anchor_idx, neighbour_idx, lambda) into f_real.
    """
    # Find 2 nearest real neighbours for every synthetic point
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean", algorithm="auto")
    nn.fit(f_real)
    distances, indices = nn.kneighbors(f_synthetic)   # [N_syn, 2]

    triplets: list[tuple[int, int, float]] = []
    for i in range(len(f_synthetic)):
        a_idx, b_idx = int(indices[i, 0]), int(indices[i, 1])
        f_a = f_real[a_idx]
        f_b = f_real[b_idx]
        f_s = f_synthetic[i]

        # Project f_s onto segment f_a → f_b
        ab = f_b - f_a
        ab_sq = float(np.dot(ab, ab))
        if ab_sq < 1e-12:
            lam = 0.0
        else:
            lam = float(np.clip(np.dot(f_s - f_a, ab) / ab_sq, 0.0, 1.0))

        triplets.append((a_idx, b_idx, lam))

    return triplets


# ---------------------------------------------------------------------------
# Image blending
# ---------------------------------------------------------------------------

def blend_images(path_a: str, path_b: str, lam: float) -> Image.Image:
    """img_a + λ * (img_b - img_a), clamped to [0, 255]."""
    img_a = np.array(Image.open(path_a).convert("RGB"), dtype=np.float32)
    img_b = np.array(Image.open(path_b).convert("RGB"), dtype=np.float32)
    if img_a.shape != img_b.shape:
        img_b = np.array(
            Image.open(path_b).convert("RGB").resize(
                (img_a.shape[1], img_a.shape[0]), Image.BILINEAR
            ),
            dtype=np.float32,
        )
    blended = np.clip(img_a + lam * (img_b - img_a), 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Feature-space SMOTE image generation for minority DR classes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",    default=None,
                   help="Path to Lightning .ckpt (recommended — domain-adapted features).")
    p.add_argument("--model-name",    default="efficientnet_b2",
                   help="Timm short name used if --checkpoint is not provided.")
    p.add_argument("--train-csv",     required=True,
                   help="Input training CSV (image_path, label columns).")
    p.add_argument("--output-dir",    required=True,
                   help="Directory where synthetic images will be saved.")
    p.add_argument("--output-csv",    default=None,
                   help="Augmented CSV path. Defaults to <train_csv_dir>/train_smote.csv.")
    p.add_argument("--method",        default="borderline",
                   choices=["smote", "borderline", "adasyn", "kmeans"],
                   help="Oversampling algorithm (borderline recommended for DR).")
    p.add_argument("--target-classes", nargs="+", type=int, default=[3, 4],
                   help="Grades to oversample.")
    p.add_argument("--target-count",  type=int, default=None,
                   help="Desired total samples per target class. Defaults to majority count.")
    p.add_argument("--k-neighbors",   type=int, default=5,
                   help="k for nearest-neighbour search in feature space.")
    p.add_argument("--batch-size",    type=int, default=64)
    p.add_argument("--num-workers",   type=int, default=4)
    p.add_argument("--jpeg-quality",  type=int, default=95)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device    = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv or str(Path(args.train_csv).with_name("train_smote.csv"))

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.train_csv)
    if "image_path" not in df.columns or "label" not in df.columns:
        sys.exit("CSV must contain 'image_path' and 'label' columns.")

    class_counts = df["label"].value_counts().sort_index()
    print("\nClass distribution:")
    for g, c in class_counts.items():
        print(f"  Grade {g}: {c:>6d}")

    present_classes = set(class_counts.index.tolist())
    missing_targets = [c for c in args.target_classes if c not in present_classes]
    if missing_targets:
        sys.exit(
            "Requested target classes are not present in train-csv: "
            + ", ".join(str(c) for c in missing_targets)
        )

    target_count = args.target_count or int(class_counts.max())
    print(f"\nMethod: {args.method}  |  target count/class: {target_count}")
    print(f"Target classes: {args.target_classes}\n")

    # ── Feature extraction (target classes only) ──────────────────────────────
    backbone, data_config = load_backbone(args.checkpoint, args.model_name, device)

    target_mask = df["label"].isin(args.target_classes)
    target_df   = df[target_mask].reset_index(drop=True)

    print(f"Extracting features for {len(target_df)} target-class images …")
    all_features = extract_features(
        backbone, target_df["image_path"].tolist(),
        data_config, device, args.batch_size, args.num_workers,
    )
    all_labels = target_df["label"].values

    # ── Run imblearn sampler on features ─────────────────────────────────────
    print(f"Running {args.method} oversampling …")
    sampler = build_sampler(args.method, args.k_neighbors, target_count, all_labels)
    if sampler is None:
        print("All selected classes already meet target_count. Nothing to oversample.")
        X_synthetic = np.empty((0, all_features.shape[1]), dtype=all_features.dtype)
        y_synthetic = np.empty((0,), dtype=all_labels.dtype)
    else:
        X_res, y_res = sampler.fit_resample(all_features, all_labels)

        # Synthetic samples are appended after originals
        X_synthetic = X_res[len(all_features):]
        y_synthetic  = y_res[len(all_features):]
    print(f"  → {len(X_synthetic)} synthetic feature vectors generated.")

    # ── Recover blending triplets and generate images ─────────────────────────
    synthetic_rows: list[dict] = []

    for grade in args.target_classes:
        syn_mask  = y_synthetic == grade
        f_syn_cls = X_synthetic[syn_mask]
        if len(f_syn_cls) == 0:
            print(f"Grade {grade}: no synthetic samples needed.")
            continue

        real_mask   = all_labels == grade
        f_real_cls  = all_features[real_mask]
        paths_cls   = target_df["image_path"].values[real_mask]

        print(f"Grade {grade}: recovering triplets for {len(f_syn_cls)} synthetic samples …")
        triplets = recover_triplets(f_syn_cls, f_real_cls)

        grade_dir = output_dir / f"grade_{grade}"
        grade_dir.mkdir(exist_ok=True)

        for i, (a_idx, b_idx, lam) in enumerate(
            tqdm(triplets, desc=f"  blending grade {grade}", leave=False)
        ):
            img = blend_images(paths_cls[a_idx], paths_cls[b_idx], lam)
            out_path = grade_dir / f"{args.method}_{i:06d}.jpg"
            img.save(out_path, "JPEG", quality=args.jpeg_quality)
            synthetic_rows.append({"image_path": str(out_path), "label": grade})

    # ── Write augmented CSV ───────────────────────────────────────────────────
    if synthetic_rows:
        augmented_df = pd.concat(
            [df, pd.DataFrame(synthetic_rows)], ignore_index=True
        )
    else:
        augmented_df = df.copy()
        print("No synthetic samples generated.")

    augmented_df.to_csv(output_csv, index=False)

    print(f"\nAugmented CSV → {output_csv}")
    print("Final class distribution:")
    for g, c in augmented_df["label"].value_counts().sort_index().items():
        marker = " *" if g in args.target_classes else ""
        print(f"  Grade {g}: {c:>6d}{marker}")


if __name__ == "__main__":
    main()
