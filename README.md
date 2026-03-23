---
title: Diabetic Retinopathy Detection App
emoji: 🐢
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.22.0
app_file: app.py
license: mit
---


# Diabetic Retinopathy Detection: Utilizing Multiprocessing for Processing Large Datasets and Transfer Learning to Fine-Tune Deep Learning Models

Efficiently process large datasets & develop advanced model pipelines for diabetic retinopathy detection. Streamlining diagnosis.

## TL;DR: 
In this project, large datasets are efficiently handled by downloading, extracting, and preparing them for analysis. Utilizing PyTorch Lightning, a robust system for diabetic retinopathy detection is developed, categorizing images into distinct disease stages. The model pipeline is enriched with various pretrained backbone models, with progress tracked using TensorBoard. Furthermore, a user-friendly web app is created to showcase the model's capabilities. The approach pursued aims to streamline both data processing and model development, facilitating accurate and accessible diabetic retinopathy diagnosis.

## Getting Started
**Introduction:**
Diabetic retinopathy (DR) remains a significant global health concern, with early detection playing a critical role in preventing vision loss. For those eager to contribute to this vital area of research, a comprehensive project studio is readily available. This studio has already tackled many essential tasks involved in DR detection, providing researchers and enthusiasts with a ready-to-use platform for experimentation.

**Get Started with the Project Studio:**
Researchers and enthusiasts alike can access the necessary tools and resources by duplicating this project studio. This streamlined solution offers an immediate starting point for experimentation on the [Diabetic Retinopathy Dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection).

**What the Studio Offers:**
- Efficient Handling of Large Datasets: The studio automates the management of large datasets, including downloading, extracting, and data preparation.
- Advanced Model Development: Utilizing PyTorch Lightning, the studio facilitates the development of a sophisticated system for DR detection, categorizing images into different disease stages.
- Integration of Pretrained Backbone Models: Various pretrained backbone models are integrated into the pipeline, allowing for experimentation with different architectures.
- Progress Tracking with TensorBoard: Researchers can monitor progress seamlessly with TensorBoard integration, tracking metrics and visualizing model performance.
- User-Friendly Web Application: A user-friendly web application is provided for showcasing model capabilities and sharing findings effortlessly.


Here's a more structured and standardized version of the steps in a blog format:

---

## Downloading and Preprocessing Diabetic Retinopathy Dataset:

> Note: You can skip this entire step, as this studio already has it done for you.

In this step, we'll walk through the process of downloading and preprocessing the Diabetic Retinopathy Detection dataset. This dataset is commonly used for developing algorithms to identify diabetic retinopathy in eye images.

### Prerequisites

Before we begin, ensure you have the following prerequisites:

- Kaggle API key (Get one [here](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2Faccount%2Flogin%3Fphase%3Dregister))
- `kaggle` library installed (`pip install kaggle`)

**Note:** Before proceeding with the steps below, make sure to change your current directory to `dr-detection` and install the required dependencies by running the following commands:
```bash
cd dr-detection
pip install -r requirements.txt
```

### Step 1: Download the Dataset

There are two ways to download the dataset:

#### First Way: Downloading as a Complete Zip File

```bash
kaggle competitions download -c diabetic-retinopathy-detection

# Extract
unzip diabetic-retinopathy-detection.zip -d data/diabetic-retinopathy-detection
rm diabetic-retinopathy-detection.zip
```

#### Second Way: Downloading as Parts

```bash
./scripts/download-dr-dataset.sh

# Merge and extract the parts
./scripts/merge_and_extract.sh
```

### Step 2: Preprocess Images

Once the dataset is downloaded, preprocess the images to crop and resize them.

<div align="center">
  <img src="https://github.com/bhimrazy/diabetic-retinopathy-detection/assets/46085301/9fa28dea-38cd-4fba-abb0-0ed8001a8075" alt="Preprocessing Image" height="400" width="auto">
  <p style="text-align:center;">Example of cropping and resizing</p>
</div>

```bash
python scripts/crop_and_resize.py --src data/diabetic-retinopathy-dataset/train data/diabetic-retinopathy-dataset/resized/train
python scripts/crop_and_resize.py --src data/diabetic-retinopathy-dataset/test data/diabetic-retinopathy-dataset/resized/test
```

### Step 3: Split Data and Save to CSV

Finally, split the data into train and validation sets and save them to CSV files.

```bash
python scripts/split_dataset.py
```

## Training Model and Monitoring Progress with TensorBoard

In the previous section, we covered how to set up your dataset and configure your training pipeline using a `Config` class. Now, let's dive into training your model and monitoring its progress using TensorBoard.

### Important: Working with Cluster Home Directory Limits

If your cluster caps the home directory (e.g. 50GB), preprocess the dataset **locally** and sync only the smaller preprocessed images to the cluster login node. Jobs then stage those images onto compute-node local disk (`$TMPDIR`) at runtime, which is outside the home quota.

**Step 1 — Preprocess locally** (on your development machine):

```bash
# Download the raw dataset
./scripts/download-dr-dataset.sh

# # Merge and extract the parts
./scripts/merge_and_extract.sh

# Crop and resize images to 512×512 (only train set is needed for training)
python scripts/crop_and_resize.py \
  --src  data/diabetic-retinopathy-dataset/train \
  --dest data/diabetic-retinopathy-dataset/resized/train
```

**Step 2 — Sync preprocessed images to FEP** (~3–8 GB, fits in home quota):

```bash
./scripts/sync_dataset_to_fep.sh
```

This syncs `data/diabetic-retinopathy-dataset/resized/` and `trainLabels.csv` to `~/diabetic-retinopathy-detection/data/diabetic-retinopathy-dataset/` on FEP, mirroring your local project layout. Raw images are not transferred.

**Step 3 — Generate train/val CSV splits on FEP** (run once, inside the container):

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
    --val_csv_path data/diabetic-retinopathy-dataset/val.csv
```

This writes `data/diabetic-retinopathy-dataset/train.csv` and `val.csv` into the project directory with image paths that match the `/data` bind-mount used at training time.

**Step 4 — Submit training jobs**:

```bash
export APPTAINER_IMAGE=~/apptainer-images/dr-detection-cu121.sif
export DATASET_SRC_DIR=~/diabetic-retinopathy-detection/data/diabetic-retinopathy-dataset
./scripts/submit_slurm_apptainer.sh
```

At runtime the Slurm batch job stages `~/dr-dataset` into `$TMPDIR/dr-dataset` on the compute node's local disk (not counted against home quota) and binds it as `/data` inside the container.

### Exploring Data Transformations and Augmentations

If you're looking for examples of data transformations and augmentations, you can explore the provided `notebook.ipynb` file. This notebook contains various examples of data preprocessing techniques, such as resizing, cropping, rotation, and more.

To open and explore the notebook:
1. Navigate to the directory containing the `notebook.ipynb` file.
3. Open the notebook and run the cells to see different transformation and augmentation examples.


### Training the Model

To train your model, you can use the provided `train.py` script. Make sure you have set up your environment correctly and installed all dependencies as mentioned earlier. Here's how you can run the training pipeline:

1. Open your terminal or command prompt.
2. Navigate to the directory containing the `train.py` script.
3. Run the following command:

```bash
python train.py
```

This command will execute the training script and start training your model based on the parameters specified in your `Config` class.

### Training on a Slurm Cluster with Apptainer

If your cluster requires jobs to run inside an Apptainer image, build the image locally, sync both the repository and the `.sif` file to the login node, then submit through Slurm from there.

1. Build the Apptainer image locally:

```bash
./scripts/build_apptainer_local.sh
```

This builds [container/apptainer.def](container/apptainer.def) into `container/build/dr-detection-cu121.sif` by default. The image includes CUDA 12.1 user-space libraries and installs the Python dependencies needed by this repository.

2. Sync the repository to the cluster login node:

```bash
./scripts/sync_to_fep.sh
```

3. Sync the image to the login node:

```bash
./scripts/sync_apptainer_to_fep.sh
```

4. SSH to the login node and move into the synced project:

```bash
ssh fep
cd ~/diabetic-retinopathy-detection
```

5. Submit a GPU training job using the synced image:

```bash
export APPTAINER_IMAGE=~/apptainer-images/dr-detection-cu121.sif
./scripts/submit_slurm_apptainer.sh
```

6. Pass Hydra overrides directly to the submit script when needed:

```bash
export APPTAINER_IMAGE=~/apptainer-images/dr-detection-cu121.sif
PARTITION=dgxa100 \
GRES=gpu:1 \
CPUS_PER_TASK=16 \
MEMORY=64G \
TIME_LIMIT=12:00:00 \
./scripts/submit_slurm_apptainer.sh batch_size=64 num_workers=16 model_name=resnet50
```

The submission wrapper calls `sbatch`, and the batch job runs:

```bash
apptainer exec --nv ... python train.py ...
```

Useful environment variables for cluster submission:

- `APPTAINER_IMAGE`: required path to the `.sif` image
- `PARTITION`: Slurm partition, defaults to `dgxa100`
- `GRES`: GPU request, defaults to `gpu:1`
- `CPUS_PER_TASK`: CPU cores, defaults to `8`
- `MEMORY`: RAM request, defaults to `32G`
- `TIME_LIMIT`: walltime, defaults to `11:00:00`
- `ACCOUNT`: optional Slurm account
- `QOS`: optional Slurm QoS
- `APPTAINER_EXTRA_BINDS`: optional extra bind mounts, comma-separated

Slurm stdout and stderr are written under `slurm/` in the project directory.

Useful environment variables for local image builds:

- `BUILD_MODE`: `fakeroot` by default, or `sudo`
- `IMAGE_NAME`: output image base name, defaults to `dr-detection-cu121`
- `OUTPUT_DIR`: output folder, defaults to `container/build`
- `APPTAINER_DEF`: alternate definition file path

### Monitoring Training Progress with TensorBoard

TensorBoard is a powerful tool for visualizing and monitoring the training process. You can use it to track metrics such as loss, accuracy, and learning rate over time, as well as visualize model graphs and embeddings.

To load TensorBoard logs and monitor your training progress:

1. Ensure you have TensorBoard installed. You can install it via pip:

```bash
pip install tensorboard
```

2. Once your model starts training, TensorBoard logs will be generated in the specified directory (e.g., `"logs/"`). You can launch TensorBoard using the following command:

```bash
tensorboard --logdir=logs/
```

This command will start a TensorBoard server locally, allowing you to view your training metrics and visualizations in your web browser.


## Gradio - Diabetic Retinopathy Detection App
<!-- 
<iframe src="https://bhimrazy-diabetic-retinopathy-detection.hf.space" frameborder="0" width="1920" height="1080"></iframe>
-->
### Overview
Welcome to Diabetic Retinopathy Detection App! This app utilizes deep learning models to detect diabetic retinopathy in retinal images. Diabetic retinopathy is a common complication of diabetes and early detection is crucial for effective treatment.

### Try It Out
Use the interactive interface below to upload retinal images and get predictions on diabetic retinopathy severity.

[Open Diabetic Retinopathy Detection App](https://bhimrazy-diabetic-retinopathy-detection.hf.space)

[![Gradio App](https://github.com/bhimrazy/diabetic-retinopathy-detection/assets/46085301/4e0788dd-84a1-427e-a38a-e22c2aa86c50)](https://bhimrazy-diabetic-retinopathy-detection.hf.space)

### How to Use
1. Click on the "Open Diabetic Retinopathy Detection App" button above.
2. Upload a retinal image by clicking on the "Upload Image" button.
3. Once the image is uploaded, the model will process it and provide predictions on the severity of diabetic retinopathy.
4. Interpret the results provided by the model.



