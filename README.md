# GATTGCN-PyTorch

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
