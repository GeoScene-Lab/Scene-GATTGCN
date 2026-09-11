# GATTGCN-PyTorch

This repository provides a PyTorch implementation of **GATTGCN** proposed in:

> **A Graph Attention-Based GNSS Time-Series Prediction Model for Large-Scale Ground Deformation Monitoring: A Case Study of the Sichuan–Yunnan Region**

GATTGCN is designed for large-scale surface displacement prediction by jointly modeling the **spatial relationships among monitoring stations** and the **temporal dependencies of displacement time series**.


---

## ✨ Features

- PyTorch implementation of the **GATTGCN** model.
- Combines **Graph Attention Network (GAT)**, graph convolution, and recurrent neural networks for spatiotemporal prediction.
- Supports the integration of **dynamic features**, **static features**, and **spatial adjacency information**.
- Uses a sliding-window strategy for time-series prediction.
- Supports configurable input sequence length, prediction length, hidden dimensions, and attention heads.
- Includes training, validation, evaluation, and TensorBoard logging.

The default experiment uses:

```text
16 days of historical observations → next 1 day prediction
```

---

## 📦 Requirements

Main dependencies:

- numpy
- matplotlib
- pandas
- torch
- pytorch-lightning >= 1.3.0
- torchmetrics >= 0.3.0
- python-dotenv

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/GeoScene-Lab/Scene-GATTGCN.git
cd Scene-GATTGCN
```

---

## 📊 Datasets

The example dataset is stored in the `data/` directory.

The model uses four main types of input data:

```text
data/
├── dynamic data.csv
├── static data.csv
├── adjacency_matrix_with_weights.csv
└── target.csv
```

- **Dynamic data**: time-varying environmental features, including **rainfall** and **humidity**.
- **Static data**: time-invariant spatial attributes, including **DEM (Digital Elevation Model)** and **distance to the nearest fault**.
- **Adjacency matrix**: weighted spatial relationships among monitoring stations.
- **Target data**: displacement observations used for model training and prediction.

For a custom dataset, make sure that the station order is consistent across all input files.

---

## 🧠 Model Architecture

GATTGCN jointly learns spatial and temporal dependencies in the monitoring network.

The overall architecture can be summarized as:

```text
Dynamic Features + Static Features
              │
              ▼
        Node Features
              │
              ▼
      Spatial Graph Network
        ┌───────────────┐
        │               │
        ▼               ▼
      GAT          Graph Convolution
        │               │
        └───────┬───────┘
                ▼
          Feature Fusion
                │
                ▼
        Recurrent Modeling
                │
                ▼
      Displacement Prediction
```

The graph attention mechanism learns adaptive spatial dependencies among monitoring stations, while the recurrent component captures temporal changes in displacement.

---

## ▶️ Usage

Train the GATTGCN model with:

```bash
python main.py   --model_name GATTGCN   --max_epochs 200   --learning_rate 0.0001   --weight_decay 0.0015   --batch_size 64   --hidden_dim 32   --loss mse_with_regularizer   --settings supervised   --gpus 1
```

You can also adjust parameters such as:

```text
--data
--seq_len
--pre_len
--hidden_dim
--num_heads
```

For example:

```bash
python main.py --model_name GATTGCN --seq_len 16 --pre_len 1 --hidden_dim 32
```

To monitor the training process using TensorBoard:

```bash
tensorboard --logdir lightning_logs
```

Then open:

```text
http://localhost:6006
```

---


