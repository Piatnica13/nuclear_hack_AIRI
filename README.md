# Weather AutoEncoder for ERA5 Compression

> Hackathon project for learning compact latent representations of global atmospheric fields from ERA5.

## Overview

This project implements a convolutional autoencoder for compressing and reconstructing global atmospheric states from the ERA5 reanalysis dataset.

The model is trained on 28 atmospheric variables and evaluates reconstruction quality using RMSE, NRMSE, Surface Score, Pressure Score and Overall Score.

The repository also contains tools for:

- downloading ERA5 datasets;
- training models;
- validating checkpoints;
- generating compressed binary outputs;
- visualizing reconstructed weather fields;
- analyzing the minimum dataset size required for successful training.

---

# Features

- 28-channel ERA5 input
- Automatic dataset download
- Docker-based environment
- One-command project startup
- Automatic validation
- Binary output generation
- Visualization of all atmospheric channels
- Mathematical analysis of data efficiency
- JSON and CSV reports

---

# Project structure

```text
.
├── app
│   ├── train.py                # Model training
│   ├── validate.py             # Validation and metrics
│   ├── download.py             # Dataset downloader
│   ├── visualize.py            # Reconstruction visualization
│   │
│   ├── visual/
│   │   └── analysis.py         # Dataset size analysis
│   │
│   ├── results/
│   │   ├── results.json
│   │   ├── per_sample_metrics.csv
│   │   └── bitstreams/
│   │
│   ├── checkpoints/
│   │   ├── model_128.pth
│   │   ├── model_256.pth
│   │   └── model_512.pth
│   │
│   └── ...
│
├── data
│
├── docker-compose.yml
│
├── presentation.pptx
│
└── README.md
```

---

# Architecture

The project follows a simple reconstruction pipeline.

```
ERA5 Dataset
      │
      ▼
 DataLoader
      │
      ▼
 AutoEncoder
      │
      ├────────► Reconstruction
      │
      ├────────► Metrics
      │
      └────────► Optional Binary Output
```

Training is performed using paired input-target atmospheric states.

Validation computes all quality metrics and optionally stores binary outputs for each reconstructed sample.

---

# Installation

The entire project can be started using Docker.

```bash
docker compose up --build
```

---

## Dataset

The repository intentionally does **not** include the complete ERA5 dataset or generated binary files.

This decision was made because:

- the dataset occupies several gigabytes;
- generated bitstreams are reproducible and can be regenerated at any time;
- excluding large binary artifacts keeps the repository lightweight and easy to clone.

The repository already contains the full data processing pipeline.

### Download ERA5

Specify the required time interval inside

```text
app/download.py
```

and run

```bash
python app/download.py
```

The downloaded data will automatically be stored inside

```text
data/
```

### Generate bitstreams

Bitstreams can be reproduced from any checkpoint by running

```bash
python app/validate.py \
    --model checkpoints/model_512.pth \
    --save-bitstream
```

Generated files will appear in

```text
app/results/bitstreams/
```

This guarantees that every result included in the paper and presentation can be reproduced from scratch using only the repository contents.

# Training

Train the model

```bash
python app/train.py
```

Checkpoints are automatically saved during training.

The repository contains pretrained checkpoints for

- 128 samples
- 256 samples
- 512 samples (final model)

The original task also considered experiments up to 1024 samples.

---

# Validation

Evaluate any checkpoint

```bash
python app/validate.py \
    --model checkpoints/model_512.pth
```

Generate binary outputs

```bash
python app/validate.py \
    --model checkpoints/model_512.pth \
    --save-bitstream
```

Outputs are written into

```
app/results/
```

including

```
results.json
per_sample_metrics.csv
bitstreams/
```

---

# Visualization

Visualize reconstruction quality

```bash
python app/visualize.py
```

The script displays all 28 atmospheric channels simultaneously.

For every variable it shows

- original field;
- reconstructed field;
- absolute reconstruction error.

---

# Data Analysis

One of the main goals of the hackathon is answering the question:

> How much atmospheric data is actually required to train an effective compression model?

The repository contains a mathematical analysis tool.

Run

```bash
python app/visual/analysis.py
```

The script fits the experimental results and estimates the minimum amount of training data required to satisfy the desired reconstruction quality.

It also generates publication-ready figures illustrating the relationship between dataset size and reconstruction performance.

---

# Results

Validation automatically reports

- RMSE
- MAE
- PSNR
- Surface Score
- Pressure Score
- Overall Score
- Overall NRMSE
- Average inference time

All results are exported into JSON and CSV files.

---

# Presentation

A presentation describing the project and experimental results is included in the project root.

---

# Reproducibility

The project is fully reproducible.

Docker ensures identical execution environments across different operating systems.

---

# Authors

Мантров Александр Юрьевич @mandoronov
Горьковенко Константин Андреевич @Piatnica_13
Петрович Игорь Сергеевич @us2ern7ame

Hackathon MEPhI Team
