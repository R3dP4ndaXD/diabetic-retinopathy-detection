---
title: Diabetic Retinopathy Detection
license: mit
---

# Diabetic Retinopathy Detection

Transfer learning with PyTorch Lightning for classifying retinal images into five diabetic retinopathy severity grades (0–4). Training runs on a Slurm cluster inside an Apptainer container; preprocessing is done locally.

Dataset: [Kaggle Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)

## Project layout

```
train.py                # training entrypoint (Hydra config)
test.py                 # evaluate a checkpoint on the labeled test split
app.py                  # Gradio demo
conf/
  config.yaml           # local/default config
  config.cluster.yaml   # cluster-specific config
src/
  model.py              # LightningModule (DRModel)
  data_module.py        # LightningDataModule (train/val/test)
  dataset.py            # torch Dataset reading CSV with image_path,label
  models/factory.py     # backbone factory (densenet, resnet, …)
scripts/
  download-dr-dataset.sh
  merge_and_extract.sh
  crop_and_resize.py
  split_dataset.py      # stratified train/val/test split
  build_apptainer_local.sh
  sync_to_fep.sh
  sync_dataset_to_fep.sh
  sync_apptainer_to_fep.sh
  submit_slurm_apptainer.sh
  slurm_train_apptainer.sh
container/
  apptainer.def
  requirements.apptainer.txt
artifacts/               # checkpoints and predictions
logs/                    # TensorBoard runs
```

## Prerequisites

- Kaggle API key configured (`~/.kaggle/kaggle.json`)
- Python 3.10+ virtual environment with `pip install -r requirements.txt`
- Apptainer installed locally (for building the container image)
- SSH access to the cluster login node (`ssh fep`)

---

## Step 1 — Download the dataset (local)

```bash
./scripts/download-dr-dataset.sh
./scripts/merge_and_extract.sh
```

This places raw images under `data/diabetic-retinopathy-dataset/train/`.

## Step 2 — Crop and resize (local)

```bash
python scripts/crop_and_resize.py \
  --src  data/diabetic-retinopathy-dataset/train \
  --dest data/diabetic-retinopathy-dataset/resized/train \
  --workers 4 \
  --skip-existing
```

Resized images (~3–8 GB) go to `data/diabetic-retinopathy-dataset/resized/train/`.

## Step 3 — Build the Apptainer image (local, one-time)

```bash
./scripts/build_apptainer_local.sh
```

Produces `container/build/dr-detection-cu121.sif` (CUDA 12.1 + all Python deps).

Rebuild whenever `container/apptainer.def` or `container/requirements.apptainer.txt` changes.

## Step 4 — Sync to the cluster

```bash
# sync the project code
./scripts/sync_to_fep.sh

# sync the preprocessed images + trainLabels.csv
./scripts/sync_dataset_to_fep.sh

# sync the Apptainer image
./scripts/sync_apptainer_to_fep.sh
```

On FEP the dataset lands at `~/diabetic-retinopathy-detection/data/diabetic-retinopathy-dataset/`, mirroring the local layout. Only preprocessed images are transferred, not raw ones.

## Step 5 — Generate train/val/test CSV splits (cluster, one-time)

```bash
ssh fep
cd ~/diabetic-retinopathy-detection

apptainer exec --nv \
  --bind ~/diabetic-retinopathy-detection/data/diabetic-retinopathy-dataset:/data \
  --bind ~/diabetic-retinopathy-detection:/workspace \
  --pwd /workspace \
  ~/apptainer-images/dr-detection-cu121.sif \
  python scripts/split_dataset.py \
    --data_dir /data/resized/train \
    --csv_path /data/trainLabels.csv \
    --train_csv_path data/diabetic-retinopathy-dataset/train.csv \
    --val_csv_path   data/diabetic-retinopathy-dataset/val.csv \
    --test_csv_path  data/diabetic-retinopathy-dataset/test.csv
```

Default proportions: 80% train, 10% val, 10% test (stratified, `--random_state 42`). Adjust with `--val_size` and `--test_size`. Image paths in the CSVs use the `/data/…` prefix matching the container bind-mount.

Re-run this whenever you want fresh splits.

## Step 6 — Submit a training job

```bash
ssh fep
cd ~/diabetic-retinopathy-detection

export APPTAINER_IMAGE=~/apptainer-images/dr-detection-cu121.sif
export DATASET_DIR=~/diabetic-retinopathy-detection/data/diabetic-retinopathy-dataset_ben

./scripts/submit_slurm_apptainer.sh
```

The Slurm batch job binds the dataset directly as `/data` inside the container (no staging copy to `$TMPDIR`).

After `trainer.fit()` completes, `trainer.test()` runs automatically on the test split and prints `test_loss`, `test_acc`, and `test_kappa`.

### Hydra overrides

Pass any config overrides after the submit script:

```bash
PARTITION=dgxa100 \
GRES=gpu:1 \
CPUS_PER_TASK=16 \
MEMORY=64G \
TIME_LIMIT=12:00:00 \
./scripts/submit_slurm_apptainer.sh batch_size=64 num_workers=16 model_name=resnet50
```

### Slurm environment variables

| Variable | Default | Description |
|---|---|---|
| `APPTAINER_IMAGE` | *(required)* | Path to `.sif` image |
| `DATASET_DIR` | *(required)* | Dataset path on shared storage to bind as `/data` |
| `PARTITION` | `dgxa100` | Slurm partition |
| `GRES` | `gpu:1` | GPU request |
| `CPUS_PER_TASK` | `8` | CPU cores |
| `MEMORY` | `32G` | RAM |
| `TIME_LIMIT` | `11:00:00` | Walltime |
| `ACCOUNT` | *(optional)* | Slurm account |
| `QOS` | *(optional)* | Slurm QoS |
| `APPTAINER_EXTRA_BINDS` | *(optional)* | Extra bind mounts |

Slurm stdout/stderr go to `slurm/` in the project directory.

## Step 7 — Evaluate on the test set

Test metrics are computed automatically at the end of each training run. To evaluate a specific checkpoint separately:

```bash
python test.py --checkpoint artifacts/dr-model.ckpt
```

This prints `test_loss`, `test_acc`, and `test_kappa` (quadratic weighted Cohen's kappa).

Save per-image predictions:

```bash
python test.py \
  --checkpoint artifacts/dr-model.ckpt \
  --predictions-csv artifacts/test_predictions.csv
```

## Monitoring with TensorBoard

TensorBoard logs are written under `logs/`. To view them locally via SSH port forwarding:

```bash
ssh -L 6006:localhost:6006 fep
cd ~/diabetic-retinopathy-detection
apptainer exec \
  --bind ~/diabetic-retinopathy-detection:/workspace \
  --pwd /workspace \
  ~/apptainer-images/dr-detection-cu121.sif \
  tensorboard --logdir logs --host 127.0.0.1 --port 6006
```

Then open http://localhost:6006 in your browser.

## Configuration

Training config is managed by Hydra. Two configs are provided:

- `conf/config.yaml` — local defaults
- `conf/config.cluster.yaml` — cluster paths

Key settings in `conf/config.yaml`:

```yaml
train_csv_path: data/diabetic-retinopathy-dataset/train.csv
val_csv_path:   data/diabetic-retinopathy-dataset/val.csv
test_csv_path:  data/diabetic-retinopathy-dataset/test.csv

seed: 42
batch_size: 128
num_workers: 32
use_class_weighting: true
model_name: "densenet169"
max_epochs: 20
image_size: 224
learning_rate: 3e-4
use_scheduler: true
```

## Data splits

The labeled Kaggle training set (`trainLabels.csv`) is split into three stratified subsets:

| Split | Default % | Purpose |
|---|---|---|
| Train | 80% | Model training |
| Validation | 10% | Hyperparameter tuning, early stopping |
| Test | 10% | Final held-out evaluation |

The original Kaggle test set is unlabeled and is not used.


./scripts/submit_slurm_test_ensemble_apptainer.sh   --ensemble-checkpoints artifacts/checkpoints/run-2026-03-30-20-26-43-swin_b_naive_oversample/epoch\=10-step\=8877-val_loss\=1.09-val_acc\=0.75-val_kappa\=0.61.ckpt artifacts/checkpoints/run-2026-03-30-18-24-09-resnet50_naive_oversample/epoch\=11-step\=9684-val_loss\=1.32-val_acc\=0.74-val_kappa\=0.55.ckpt artifacts/checkpoints/run-2026-03-30-20-23-09-densenet121_naive_oversample/epoch\=14-step\=12105-val_loss\=1.22-val_acc\=0.75-val_kappa\=0.56.ckpt artifacts/checkpoints/run-2026-03-30-21-21-28-efficientnet_b0_naive_oversample/epoch\=9-step\=8070-val_loss\=1.05-val_acc\=0.67-val_kappa\=0.56.ckpt  artifacts/checkpoints/run-2026-03-30-15-56-31-efficientnet_b2_naive_oversample/epoch\=13-step\=11298-val_loss\=1.28-val_acc\=0.76-val_kappa\=0.62.ckpt   --tune-ensemble-weights   --test-csv data/diabetic-retinopathy-dataset_ben_2
24/test.csv  --val-csv data/diabetic-retinopathy-dataset_ben_224/val.csv


Test QWK (meta-learner, mlp):               0.7596
Test QWK (meta-learner, learned_weights):   0.7684
Test QWK (meta-learner, cross_attention):   0.7792
Test QWK (uniform ensemble baseline):       0.7682

./scripts/submit_slurm_train_meta.sh \
  --base-checkpoints artifacts/checkpoints/run-2026-04-05-04-24-52-efficientnetv2_m_naive_oversample/epoch\=23-step\=38688-val_loss\=0.83-val_acc\=0.82-val_kappa\=0.72.ckpt artifacts/checkpoints/run-2026-04-05-02-51-31-coatnet_2_naive_oversample/epoch\=5-step\=4836-val_loss\=0.87-val_acc\=0.78-val_kappa\=0.71.ckpt artifacts/checkpoints/run-2026-04-04-19-53-42-swin_base_naive_oversample_no_ema/epoch\=33-step\=54808-val_loss\=0.86-val_acc\=0.83-val_kappa\=0.73.ckpt artifacts/checkpoints/run-2026-04-04-17-26-17-convnext_base_efficientnet_b0_mlp_dual/epoch\=23-step\=77400-val_loss\=0.87-val_acc\=0.82-val_kappa\=0.71.ckpt artifacts/checkpoints/run-2026-04-04-16-19-21-efficientnet_b2_naive_oversample_no_ema/epoch\=30-step\=24986-val_loss\=0.88-val_acc\=0.78-val_kappa\=0.69.ckpt artifacts/checkpoints/run-2026-04-04-14-33-54-convnext_base_naive_oversample/epoch\=23-step\=38688-val_loss\=0.88-val_acc\=0.81-val_kappa\=0.72.ckpt --image-sizes 416 224 256 224 260 224 --test-csv data/diabetic-retinopathy-dataset/test.csv  --val-csv data/diabetic-retinopathy-dataset/val.csv --fusion-type mlp