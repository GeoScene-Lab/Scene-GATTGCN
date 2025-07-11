import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses
import pickle

class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        regressor="linear",
        loss="mse_with_regularizer",
        pre_len: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0,
        **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.regressor = (
            nn.Linear(
                self.model.hyperparameters.get("hidden_dim"),
                self.hparams.pre_len
            )
            if regressor == "linear"
            else regressor
        )
        self._loss = loss



    def forward(self, x):
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.model(x)
        # (batch_size, num_nodes, pre_len)
        predictions = self.regressor(hidden)
        return predictions

    def shared_step(self, batch, batch_idx):
        # (batch_size, seq_len, num_nodes, input_dim), (batch_size, pre_len, num_nodes, input_dim)
        x, y = batch
        num_nodes = x.size(2)
        # [batch_size,num_nodes,pre_len]
        predictions = self(x)
        predictions = predictions.reshape((-1, num_nodes))
        y = y.reshape((-1, num_nodes))
        return predictions, y

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        # (batch_size, pre_len, num_nodes)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)

        # 加载目标数据的归一化参数
        with open('target_min_max.pkl', 'rb') as f:
            target_min_max = pickle.load(f)
            target_min = torch.tensor(target_min_max['target_min']).to(predictions.device)
            target_max = torch.tensor(target_min_max['target_max']).to(predictions.device)
        predictions = predictions * (target_max - target_min) + target_min
        y = y * (target_max - target_min) + target_min
        # print(predictions.shape)
        # print(y.shape)
        loss = self.loss(predictions, y)
        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        # accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        explained_variance = utils.metrics.explained_variance(predictions, y)
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            # "accuracy": accuracy,
            "R2": r2,
            "ExplainedVar": explained_variance,
        }
        self.log_dict(metrics)
        print(metrics)
        return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)#1.5e-3
        parser.add_argument("--loss", type=str, default="mse_with_regularizer")
        return parser
