"""
Meta-learner training script.

Workflow
--------
1. Load each base checkpoint (DRModel or DualStreamDRModel), freeze it.
2. Collect per-model softmax probability vectors on the val set.
3. (Optional) cache probabilities to disk so reruns are fast.
4. Train MetaLearner on the cached probs.
5. Save the MetaLearner checkpoint.
6. Evaluate on the test set if --test-csv is provided.

Usage
-----
python train_meta.py \\
    --base-checkpoints  ckpt1.ckpt ckpt2.ckpt ckpt3.ckpt \\
    --val-csv           data/.../val.csv \\
    --test-csv          data/.../test.csv \\
    --image-sizes       224 224 260 \\
    --fusion-type       mlp \\
    --epochs            100 \\
    --output-dir        artifacts/meta
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from src.data_module import DRDataModule
from src.dataset import DRDataset
from src.ensemble_meta import MetaLearner, ProbDataset
from src.model import DRModel
from src.models.factory import ModelFactory
from src.utils import generate_run_id


# ---------------------------------------------------------------------------
# Probability collection
# ---------------------------------------------------------------------------

@torch.inference_mode()
def collect_probs(
    checkpoints: list[str],
    csv_path: str,
    image_sizes: list[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    cache_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run each frozen checkpoint on the dataset and collect softmax probs.

    Returns
    -------
    member_probs : [N_models, N_samples, num_classes]
    labels       : [N_samples]
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[cache] Loading cached probs from {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        return data["member_probs"], data["labels"]

    all_probs: list[torch.Tensor] = []
    labels: torch.Tensor | None = None

    for ckpt_path, img_size in zip(checkpoints, image_sizes):
        print(f"Collecting probs: {os.path.basename(ckpt_path)} (size={img_size})")

        # Load checkpoint — try DRModel first, then DualStreamDRModel
        try:
            model = DRModel.load_from_checkpoint(ckpt_path, map_location=device)
        except Exception:
            from src.dual_stream_model import DualStreamDRModel
            model = DualStreamDRModel.load_from_checkpoint(ckpt_path, map_location=device)

        model.freeze()
        model.to(device)
        model.eval()

        # Build a minimal normalization using the model's timm data_config
        timm_cfg = getattr(model.model, "data_config", None) if hasattr(model, "model") \
                   else getattr(model.rgb_backbone, "data_config", None)
        if timm_cfg:
            mean, std = list(timm_cfg["mean"]), list(timm_cfg["std"])
        else:
            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]

        transform = T.Compose([
            T.Resize((img_size, img_size), antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
        ])
        dataset = DRDataset(csv_path, transform=transform)
        loader  = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True,
        )

        model_probs: list[torch.Tensor] = []
        model_labels: list[torch.Tensor] = []

        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs  = torch.softmax(logits, dim=1).cpu()
            model_probs.append(probs)
            model_labels.append(y)

        all_probs.append(torch.cat(model_probs, dim=0))   # [N, C]
        if labels is None:
            labels = torch.cat(model_labels, dim=0)

    member_probs = torch.stack(all_probs, dim=0)  # [N_models, N, C]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"member_probs": member_probs, "labels": labels}, cache_path)
        print(f"[cache] Saved probs to {cache_path}")

    return member_probs, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train meta-learner ensemble fusion")
    p.add_argument("--base-checkpoints", nargs="+", required=True,
                   help="Paths to frozen base-model checkpoints")
    p.add_argument("--val-csv",  required=True,
                   help="Validation CSV (image_path, label)")
    p.add_argument("--test-csv", default=None,
                   help="Test CSV for final evaluation (optional)")
    p.add_argument("--image-sizes", nargs="+", type=int, default=None,
                   help="Input image size per checkpoint. Defaults to 224 for all.")
    p.add_argument("--fusion-type",
                   choices=["temperature", "learned_weights", "mlp", "cross_attention"],
                   default="mlp")
    p.add_argument("--hidden",      type=int,   default=64)
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--num-heads",   type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--epochs",      type=int,   default=150)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--num-workers", type=int,   default=7)
    p.add_argument("--output-dir",  default="artifacts/meta")
    p.add_argument("--cache-dir",   default="artifacts/meta/prob_cache",
                   help="Directory for caching collected probability tensors")
    p.add_argument("--no-cache",    action="store_true",
                   help="Disable probability caching")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_models = len(args.base_checkpoints)
    image_sizes = args.image_sizes or [224] * n_models
    if len(image_sizes) != n_models:
        raise ValueError(
            f"--image-sizes must have {n_models} values (one per checkpoint)."
        )

    run_id = generate_run_id()
    print(f"Run ID    : {run_id}")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect val probabilities ──────────────────────────────────────────
    cache_path = None if args.no_cache else str(
        Path(args.cache_dir) / f"val_probs_{run_id}.pt"
    )
    val_probs, val_labels = collect_probs(
        checkpoints=args.base_checkpoints,
        csv_path=args.val_csv,
        image_sizes=image_sizes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        cache_path=cache_path,
    )
    # val_probs : [N_models, N_val, num_classes]
    num_classes = val_probs.shape[2]
    print(f"Val probs shape  : {val_probs.shape}")
    print(f"Val labels shape : {val_labels.shape}")

    # ── Baseline: uniform soft voting ─────────────────────────────────────
    uniform_preds = val_probs.mean(dim=0).argmax(dim=1)
    baseline_kappa = cohen_kappa_score(
        val_labels.numpy(), uniform_preds.numpy(), weights="quadratic"
    )
    print(f"Baseline uniform ensemble QWK (val): {baseline_kappa:.4f}")

    # ── Build datasets ────────────────────────────────────────────────────
    # 80/20 split of val set for meta train/val (no data leakage on test)
    n_total = len(val_labels)
    n_meta_val = max(1, int(0.2 * n_total))
    n_meta_train = n_total - n_meta_val

    indices = torch.randperm(n_total)
    train_idx = indices[:n_meta_train]
    val_idx   = indices[n_meta_train:]

    meta_train_ds = ProbDataset(val_probs[:, train_idx], val_labels[train_idx])
    meta_val_ds   = ProbDataset(val_probs[:, val_idx],   val_labels[val_idx])

    meta_train_loader = DataLoader(
        meta_train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0,
    )
    meta_val_loader = DataLoader(
        meta_val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0,
    )

    # ── MetaLearner ───────────────────────────────────────────────────────
    meta = MetaLearner(
        num_models=n_models,
        num_classes=num_classes,
        fusion_type=args.fusion_type,
        hidden=args.hidden,
        dropout=args.dropout,
        num_heads=args.num_heads,
        lr=args.lr,
    )

    logger = TensorBoardLogger(
        save_dir=str(output_dir), name="", version="meta"
    )
    checkpoint_cb = ModelCheckpoint(
        monitor="meta_val_kappa",
        mode="max",
        save_top_k=1,
        dirpath=str(output_dir),
        filename="meta_learner-{epoch}-{meta_val_kappa:.4f}",
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[
            checkpoint_cb,
            EarlyStopping(monitor="meta_val_kappa", patience=20, mode="max"),
        ],
        log_every_n_steps=1,
    )
    trainer.fit(meta, meta_train_loader, meta_val_loader)
    print(f"Best meta checkpoint: {checkpoint_cb.best_model_path}")

    # ── Test evaluation (optional) ────────────────────────────────────────
    if args.test_csv:
        test_cache = None if args.no_cache else str(
            Path(args.cache_dir) / f"test_probs_{run_id}.pt"
        )
        test_probs, test_labels = collect_probs(
            checkpoints=args.base_checkpoints,
            csv_path=args.test_csv,
            image_sizes=image_sizes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            cache_path=test_cache,
        )
        best_meta = MetaLearner.load_from_checkpoint(checkpoint_cb.best_model_path)
        best_meta.to(device)
        preds = best_meta.predict_from_probs(test_probs)
        test_kappa = cohen_kappa_score(
            test_labels.numpy(), preds.cpu().numpy(), weights="quadratic"
        )
        print(f"\nTest QWK (meta-learner, {args.fusion_type}): {test_kappa:.4f}")

        # Baseline on test too
        baseline_test = test_probs.mean(dim=0).argmax(dim=1)
        baseline_test_kappa = cohen_kappa_score(
            test_labels.numpy(), baseline_test.numpy(), weights="quadratic"
        )
        print(f"Test QWK (uniform ensemble baseline):        {baseline_test_kappa:.4f}")


if __name__ == "__main__":
    main()
