# GATTGCN-PyTorch

[![GitHub stars](https://img.shields.io/github/stars/martinwhl/T-GCN-PyTorch?label=stars&maxAge=2592000)](https://gitHub.com/martinwhl/T-GCN-PyTorch/stargazers/) [![issues](https://img.shields.io/github/issues/martinwhl/T-GCN-PyTorch)](https://github.com/martinwhl/T-GCN-PyTorch/issues) [![License](https://img.shields.io/github/license/martinwhl/T-GCN-PyTorch)](./LICENSE) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/martinwhl/T-GCN-PyTorch/graphs/commit-activity) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![Codefactor](https://www.codefactor.io/repository/github/martinwhl/T-GCN-PyTorch/badge)

This is a PyTorch implementation of GATTGCN in the following paper: [Scene-GATTGCN: A Graph Attention-Based Time-Series Prediction Model for Earth Surface Displacement in Large-Scale Monitoring Scenarios ].

## Requirements

* numpy
* matplotlib
* pandas
* torch
* pytorch-lightning>=1.3.0
* torchmetrics>=0.3.0
* python-dotenv

## Model Training

```bash
# GATTGCN
python main.py --model_name GATTGCN --max_epochs 200 --learning_rate 0.0001 --weight_decay 0.0015 --batch_size 64 --hidden_dim 32 --loss mse_with_regularizer --settings supervised --gpus 1
```

You can also adjust the `--data`, `--seq_len` and `--pre_len` parameters.

Run `tensorboard --logdir lightning_logs/version_0` to monitor the training progress and view the prediction results.
