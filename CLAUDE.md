# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Transfer learning with PyTorch Lightning for classifying retinal fundus images into 5 diabetic retinopathy severity grades (0–4). Training runs on a Slurm HPC cluster (H100 GPUs) inside an Apptainer container; preprocessing is done locally.

Dataset: Kaggle Diabetic Retinopathy Detection (`trainLabels.csv` → stratified train/val/test CSVs).

## Key commands

### Local development

```bash
# Install dependencies
pip install -r requirements.txt

# Train single-stream model (uses conf/config.yaml)
python train.py

# Train with Hydra overrides
python train.py model_name=convnext_base image_size=224 batch_size=32 num_gpus=1

# Train dual-stream model (uses conf/config_dual_stream.yaml)
python train_dual_stream.py
python train_dual_stream.py rgb_model_name=convnext_base wav_model_name=efficientnet_b0 fusion_type=cross_attention

# Train meta-learner (after base models are trained)
python train_meta.py \
  --base-checkpoints artifacts/checkpoints/run1/ckpt.ckpt artifacts/checkpoints/run2/ckpt.ckpt \
  --val-csv data/.../val.csv --test-csv data/.../test.csv \
  --fusion-type mlp

# OOF cross-validation (produces oof_predictions.csv + per-fold checkpoints)
# Uses conf/config.yaml; add oof_num_folds=5 or override via Hydra
python train_oof.py model_name=convnext_base oof_num_folds=5

# XGBoost meta-learner from OOF probability CSVs (one CSV per base model)
python train_meta_xgb.py \
  --oof-csvs artifacts/oof/run1/oof_predictions.csv artifacts/oof/run2/oof_predictions.csv \
  --test-prob-csvs artifacts/oof/run1/test_predictions.csv artifacts/oof/run2/test_predictions.csv \
  --output-dir artifacts/meta_xgb

# Evaluate a single checkpoint
python test.py --checkpoint artifacts/checkpoints/<run_id>/epoch=N-....ckpt

# Evaluate ensemble (soft voting + optional meta-learner)
python test_ensemble.py \
  --ensemble-checkpoints ckpt1.ckpt ckpt2.ckpt \
  --tune-ensemble-weights \
  --val-csv data/.../val.csv --test-csv data/.../test.csv

# Preprocessing: crop & resize raw images locally (run for both train and test)
python scripts/crop_and_resize.py \
  --src data/.../train --dest data/.../resized/train \
  --workers 4 --skip-existing --sigmaX 10
python scripts/crop_and_resize.py \
  --src data/.../test --dest data/.../resized/test \
  --workers 4 --skip-existing --sigmaX 10

# Generate train/val/test CSV splits using the official test labels
python scripts/split_dataset.py \
  --data-dir data/.../resized/train \
  --csv-path data/trainLabels.csv \
  --test-labels-csv data/testLabels.csv \
  --test-data-dir data/.../resized/test \
  --train-csv-path data/.../train.csv \
  --val-csv-path data/.../val.csv \
  --test-csv-path data/.../test.csv
# Add --test-usage Public or --test-usage Private to filter the test set by Usage column
```

### Cluster (Slurm + Apptainer)

```bash
# Build container (one-time, local) — CUDA 12.8 + PyTorch 2.9.1
./scripts/build_apptainer_local.sh

# Sync to cluster
./scripts/sync_to_fep.sh && ./scripts/sync_dataset_to_fep.sh && ./scripts/sync_apptainer_to_fep.sh

# Single-model training (up to 3 H100s)
NUM_GPUS=3 \
APPTAINER_IMAGE=~/apptainer-images/dr-detection-cu128.sif \
DATASET_DIR=~/diabetic-retinopathy-detection/data/... \
./scripts/submit_slurm_train_apptainer.sh model_name=convnext_base image_size=224 batch_size=128

# Dual-stream training
NUM_GPUS=2 APPTAINER_IMAGE=... DATASET_DIR=... \
./scripts/submit_slurm_train_dual_stream.sh \
  rgb_model_name=convnext_base wav_model_name=efficientnet_b0 fusion_type=mlp

# OOF cross-validation
APPTAINER_IMAGE=... DATASET_DIR=... \
./scripts/submit_slurm_train_oof.sh model_name=convnext_base oof_num_folds=5

# Meta-learner training (1 GPU, fast)
APPTAINER_IMAGE=... DATASET_DIR=... \
./scripts/submit_slurm_train_meta.sh \
  --base-checkpoints ckpt1.ckpt ckpt2.ckpt --val-csv data/.../val.csv --fusion-type mlp

# XGBoost meta-learner on OOF CSVs
APPTAINER_IMAGE=... DATASET_DIR=... \
./scripts/submit_slurm_train_meta_xgb.sh \
  --oof-csvs oof1.csv oof2.csv --test-prob-csvs test1.csv test2.csv

# Ensemble test
APPTAINER_IMAGE=... DATASET_DIR=... \
./scripts/submit_slurm_test_ensemble_apptainer.sh \
  --ensemble-checkpoints ckpt1.ckpt ckpt2.ckpt --tune-ensemble-weights \
  --test-csv data/.../test.csv --val-csv data/.../val.csv
```

Slurm stdout/stderr → `slurm/`. TensorBoard logs → `logs/`. Checkpoints → `artifacts/checkpoints/<run_id>/`.

## Architecture

### Single-stream training pipeline

```
train.py  (Hydra, conf/config.yaml)
  ├─ ModelFactory → TimmModel  (src/models/factory.py)  creates timm backbone
  ├─ DRDataModule              (src/data_module.py)      CSV + augmentation
  └─ DRModel                   (src/model.py)            LightningModule
```

`train.py` creates the model first, reads `model.data_config` (timm's recommended mean/std), then passes it to `DRDataModule` for normalization.

### Dual-stream training pipeline

```
train_dual_stream.py  (Hydra, conf/config_dual_stream.yaml)
  ├─ DRDataModule                 (shared, normalised with RGB backbone's stats)
  └─ DualStreamDRModel            (src/dual_stream_model.py)
       ├─ TimmModel (RGB, heavy)  e.g. convnext_base → f_rgb [B, D_rgb]
       ├─ WaveletChannelTransform → TimmModel (wav, light)  → f_wav [B, D_wav]
       └─ FusionHead              (src/fusion.py)
            "mlp"             → MLPFusionHead (concat + 2-layer MLP)
            "cross_attention" → CrossAttentionFusionHead (RGB queries wav)
```

### OOF cross-validation pipeline

```
train_oof.py  (Hydra, conf/config.yaml)
  ├─ StratifiedKFold over oof_source_csv_path
  ├─ Per fold: DRDataModule + DRModel (same as single-stream)
  └─ Outputs to artifacts/oof/<run_id>/
       ├─ fold_N/checkpoints/        best checkpoint per fold
       ├─ oof_predictions.csv        prob_0..4, pred_label, fold per sample
       └─ test_predictions.csv       mean probs over all fold models
```

OOF predictions feed directly into `train_meta_xgb.py`.

### Meta-learner ensemble pipeline

```
train_meta.py  (argparse)
  ├─ Frozen base checkpoints  → collect val softmax probs [N_models, N_val, 5]
  └─ MetaLearner              (src/ensemble_meta.py)
       fusion_type:
         "temperature"     → per-model temperature scaling
         "learned_weights" → softmax-normalised scalar weights
         "mlp"             → flatten → 2-layer MLP
         "cross_attention" → per-model token → self-attention → pool → head

train_meta_xgb.py  (argparse)
  ├─ OOF CSV per base model (from train_oof.py outputs)
  ├─ XGBoost with fold-aware OOF CV evaluation
  └─ Outputs to artifacts/meta_xgb/<run_id>/
       ├─ meta_xgb_model.json        final model trained on all OOF data
       ├─ meta_oof_predictions.csv   CV meta predictions
       └─ metrics.json               OOF/test QWK, fold scores
```

### Key source modules

| File | Role |
|---|---|
| `src/models/factory.py` | `TimmModel`, `TIMM_MODEL_REGISTRY`, normalization helpers |
| `src/model.py` | `DRModel` LightningModule; enumerated D4 TTA; MC Dropout |
| `src/dual_stream_model.py` | `DualStreamDRModel` LightningModule |
| `src/fusion.py` | `MLPFusionHead`, `CrossAttentionFusionHead` |
| `src/ensemble.py` | `EnsemblePredictor` (inference-only soft voting) |
| `src/ensemble_meta.py` | `MetaLearner` LightningModule (temperature / weights / MLP / cross-attention) |
| `src/data_module.py` | `DRDataModule`; dataset, augmentation, class balancing |
| `src/config_checks.py` | `validate_training_recipe` — raises on hard conflicts (e.g. MC Dropout + drop_rate=0), warns on soft ones (MixUp + Focal) |

### Key config options

**`conf/config.yaml`** (single-stream):

| Key | Values | Notes |
|---|---|---|
| `model_name` | see `TIMM_MODEL_REGISTRY` in `factory.py` | e.g. `convnext_base`, `efficientnetv2_m`, `vit_base` |
| `normalization_mode` | `timm` \| `imagenet` \| `dataset_by_size` \ | `timm` reads per-model stats automatically |
| `balancing_mode` | `naive_oversample` \| `sampler` \| `weighted_loss` \| `smote` | `smote` requires pre-generated SMOTE images via `scripts/generate_smote_images.py` |
| `loss_name` | `cross_entropy` \| `focal` | |
| `use_mixup` | `true` \| `false` | MixUp + CutMix via timm |
| `num_gpus` | `1`–`3` | DDP activated automatically when > 1 |
| `drop_path_rate` | `0.0`–`0.4` | stochastic depth; key for ViTs/ConvNeXt |
| `freq_transform` | `none` \| `wavelet` \| `dct` \| `fourier` \| `fourier_hpf` | appends frequency channels to RGB input in single-stream model |
| `tta_enabled` | `true` \| `false` | test-time augmentation at inference |
| `mc_dropout_enabled` | `true` \| `false` | MC Dropout uncertainty at test time; requires `drop_rate > 0` |
| `layer_lr_decay` | `0.5`–`1.0` | layer-wise LR decay for ViT families |

**`conf/config_dual_stream.yaml`** adds: `rgb_model_name`, `wav_model_name`, `fusion_type`, `wavelet_name` (`sym2` default), `wavelet_levels` (2 default), `fusion_hidden`, `fusion_num_heads`.

### Timm model registry

Models are registered in `src/models/factory.py:TIMM_MODEL_REGISTRY`. Add new models there. The factory is now entirely `timm`-based — torchvision is no longer used for backbones.

Supported models (call `from src.models.factory import list_supported_models`):
- `efficientnetv2_m`, `efficientnetv2_s`
- `convnext_base`, `convnext_large`
- `coatnet_2`, `maxvit_base`, `swin_base`, `vit_base`
- `efficientnet_b0`, `efficientnet_b2`, `mobilenetv3_large` (lightweight / wavelet stream)

### Data flow

1. Raw images → `scripts/crop_and_resize.py` (Ben Graham `sigmaX`, CLAHE, gray background fill → 512×512 on disk)
2. `scripts/split_dataset.py` → `train.csv`, `val.csv`, `test.csv`
3. `DRDataModule` reads CSVs, applies train augmentation (RandAugment + MixUp/CutMix) or val resize
4. If `balancing_mode=naive_oversample`, a sidecar `train_oversampled.csv` is generated on first run

### Metrics

Primary: `val_kappa` (quadratic weighted Cohen's kappa). Also logged: `acc`, `precision`, `recall`, `f1`. Full per-class report printed and logged to TensorBoard at end of test.

### Output directories

| Path | Contents |
|---|---|
| `artifacts/checkpoints/<run_id>/` | single-stream and dual-stream checkpoints |
| `artifacts/oof/<run_id>/` | OOF fold checkpoints + `oof_predictions.csv` |
| `artifacts/meta_xgb/<run_id>/` | XGBoost model JSON + metrics |
| `logs/` | TensorBoard event files |
| `slurm/` | Slurm stdout/stderr |

## Important notes

- `normalization_mode='timm'` requires the model to be instantiated before the DataModule so the data_config can be read. `train.py` and `train_dual_stream.py` handle this order.
- `dataset_by_size` normalization only supports `image_size` 224 or 260. For other sizes use `imagenet`.
- DDP (`num_gpus > 1`): `sync_dist=True` is set on all `.log()` calls. The `strategy` is `ddp_find_unused_parameters_false` for efficiency.
- The container targets CUDA 12.8 + PyTorch 2.9.1 (`container/apptainer.def`). Rebuild the `.sif` before running on H100s if you haven't already.
- `src/config_checks.py:validate_training_recipe` is called at the start of `train.py`, `train_dual_stream.py`, and `train_oof.py`. It raises `ValueError` for hard incompatibilities (e.g. `mc_dropout_enabled=true` with `drop_rate=0`) and emits warnings for soft conflicts (e.g. MixUp + FocalLoss). Set `strict_compatibility_checks=false` to downgrade errors to warnings.
- `freq_transform` in single-stream (`none`/`wavelet`/`dct`/`fourier`/`fourier_hpf`) appends extra channels to the 3-channel RGB input inside the model. The DataModule always delivers 3-channel images; channel expansion happens in `DRModel.forward`.
- OOF output CSVs (`oof_predictions.csv`, `test_predictions.csv`) are the direct inputs to `train_meta_xgb.py`. Column format: `image_path`, `label`, `fold`, `prob_0`…`prob_4`, `pred_label`.
