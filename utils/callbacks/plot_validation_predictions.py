import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from utils.callbacks.base import BestEpochCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy.linalg as la
import pickle
from sklearn import preprocessing
import torch
import torchmetrics

class PlotValidationPredictionsCallback(BestEpochCallback):
    def __init__(self, monitor="", mode="min"):
        super(PlotValidationPredictionsCallback, self).__init__(monitor=monitor, mode=mode)
        self.ground_truths = []
        self.predictions = []

    def on_fit_start(self, trainer, pl_module):
        self.ground_truths.clear()
        self.predictions.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if trainer.current_epoch != self.best_epoch:
            return
        self.ground_truths.clear()
        self.predictions.clear()
        predictions, y = outputs
        predictions = predictions.cpu().numpy()
        y = y.cpu().numpy()
        self.ground_truths.append(y[:, 0, :])
        self.predictions.append(predictions[:, 0, :])

    def on_fit_end(self, trainer, pl_module):
        def evaluation1(a,b):
            a = torch.tensor(a)
            b = torch.tensor(b)
            rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(a, b))
            mae = torchmetrics.functional.mean_absolute_error(a, b)
            return rmse, mae
        def evaluation2(a,b):
            # F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
            r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
            var = 1-(np.var(a-b))/np.var(a)
            return  r2, var

        def evaluate_per_site(y_true, y_pred):
            """
            输入形状: [num_timesteps, num_sites]
            返回: DataFrame 包含每个站点的RMSE, MAE, R2, Var
            """
            # 计算每个站点的指标
            rmse_per_site = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
            mae_per_site = np.mean(np.abs(y_pred - y_true), axis=0)

            # 计算R2
            ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
            ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
            r2_per_site = 1 - (ss_res / ss_tot)

            # 计算方差解释
            var_per_site = 1 - (np.var(y_true - y_pred, axis=0) / (np.var(y_true, axis=0)))

            # 组合结果
            df = pd.DataFrame({
                'Site_ID': range(y_true.shape[1]),
                'RMSE': rmse_per_site,
                'MAE': mae_per_site,
                'R2': r2_per_site,
                'Var_Explained': var_per_site
            })
            return df


        ground_truth = np.concatenate(self.ground_truths, 0)
        predictions = np.concatenate(self.predictions, 0)
        tensorboard = pl_module.logger.experiment


        result1 = predictions
        testY1 = ground_truth
        result1=result1.reshape(result1.shape[0],-1)
        testY1 = testY1.reshape(testY1.shape[0], -1)

        out = 'D:\预测西南的形变\实验模型\去掉问题站点后数据结果\本体加静态'
        path1 = 'GATTGCN本体加静态预测结果seq=16'
        path = os.path.join(out,path1)
        if not os.path.exists(path):
                os.makedirs(path)


        rmse1, mae1 = evaluation1(testY1, result1)
        r2,var = evaluation2(testY1, result1)
        result = pd.DataFrame(result1)
        testYY = pd.DataFrame(testY1)

        print('GCN_rmse:%r'%rmse1,
        'GCN_mae:%r'%mae1,
        # 'GCN_acc:%r'%acc1,
        'GCN_r2:%r'%r2,
        'GCN_var:%r'%var)

        # 计算所有站点的综合指标
        result.to_csv(path+'/test_prediction.csv',index = False,header = False)
        testYY.to_csv(path+'/test_true.csv',index = False,header = False)
        evalution = []
        evalution.append(rmse1)
        evalution.append(mae1)
        # evalution.append(acc1)
        evalution.append(r2)
        evalution.append(var)
        evalution = pd.DataFrame(evalution)
        evalution.to_csv(path+'/evalution.csv',index=False,header=None)

        # 计算每个站点的指标
        site_metrics_df = evaluate_per_site(testY1, result1)
        site_metrics_df.to_csv(path + '/pre_evalution.csv', index=False)

        for node_idx in range(ground_truth.shape[1]):
            plt.clf()
            #plt.rcParams["font.family"] = "Times New Roman"
            fig = plt.figure(figsize=(7, 2), dpi=300)
            plt.plot(
                ground_truth[:, node_idx],
                color="dimgray",
                linestyle="-",
                label="Ground truth",
            )
            plt.plot(
                predictions[:, node_idx],
                color="deepskyblue",
                linestyle="-",
                label="Predictions",
            )
            plt.legend(loc="best", fontsize=10)
            plt.xlabel("Time")
            plt.ylabel("Traffic Speed")
            tensorboard.add_figure(
                "Prediction result of node " + str(node_idx),
                fig,
                global_step=len(trainer.train_dataloader) * self.best_epoch,
                close=True,
            )
