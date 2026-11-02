# GB-STGNN

Granular-ball spatio-temporal graph neural network for river discharge forecasting
on LamaH-CE and LamaH-Ice. This repository contains the model, granular-ball
construction, preprocessing, training, evaluation metrics, and core tests.

## Installation

Use Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for your CUDA environment.
The core implementation does not require torch-geometric or baseline model code.

## Prepare data

Download the **daily-resolution** versions of both datasets:

- **LamaH-CE:** [Download page (Zenodo)](https://zenodo.org/records/5153305).
  Download `2_LamaH-CE_daily.tar.gz`, which contains daily-resolution data.
- **LamaH-Ice:** [Download page (HydroShare)](https://www.hydroshare.org/resource/86117a5f36cc4b7c90a5d54e18161c91/).
  Select `lamah_ice`, which contains daily-resolution data.

Place `2_LamaH-CE_daily.tar.gz` under `dataset/`, then run:

```bash
python preprocess_lamah_ce.py --archive dataset/2_LamaH-CE_daily.tar.gz --out-dir dataset/processed_lamah_ce
```

For LamaH-Ice, place the extracted dataset under `dataset/lamah_ice`:

```bash
python preprocess_lamah_ice.py --data-dir dataset/lamah_ice --out-dir dataset/processed_lamah_ice
```


## Train and evaluate

```bash
python train_gb_stgnn_variance_seed_topology_penalty.py --dataset lamah_ce --device cuda
python train_gb_stgnn_variance_seed_topology_penalty.py --dataset lamah_ice --device cuda
# Run both datasets sequentially:
python train_gb_stgnn_variance_seed_topology_penalty.py --dataset all --device cuda
```
