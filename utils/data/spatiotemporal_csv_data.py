import argparse
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import utils.data.functions


class SpatioTemporalCSVDataModule(pl.LightningDataModule):
    def __init__(
            self,
            feat_path: str,
            static_feat_path: str,  # 新增静态特征路径
            adj_path: str,
            target_path: str,
            num_nodes: int,
            num_dynamic_features: int,  # 动态特征数
            batch_size: int = 64,
            seq_len: int = 5,
            pre_len: int = 1,
            split_ratio: float = 0.8,
            normalize: bool = True,
            **kwargs
    ):
        super().__init__()
        self._feat_path = feat_path
        self._static_feat_path = static_feat_path  #静态特征
        self._adj_path = adj_path
        self._target_path = target_path
        self.num_nodes = num_nodes
        self.num_features = num_dynamic_features + 2  # 总特征数=动态+静态
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self._adj = utils.data.functions.load_adjacency_matrix(self._adj_path)
        self._feat = utils.data.functions.load_features(
            self._feat_path,
            self._static_feat_path,
            self.num_nodes,
            num_dynamic_features
        )
        self._targets = utils.data.functions.load_targets(
            self._target_path,
        )
        self._feat_max_val = np.max(self._feat)

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--seq_len", type=int, default=16)
        parser.add_argument("--pre_len", type=int, default=1)
        parser.add_argument("--split_ratio", type=float, default=0.8)
        parser.add_argument("--normalize", type=bool, default=True)
        return parser

    def setup(self, stage: str = None):
        self.train_dataset, self.val_dataset = utils.data.functions.generate_torch_datasets(
            self._feat,
            self._targets,
            self.seq_len,
            self.pre_len,
            split_ratio=self.split_ratio,
            normalize=self.normalize,
        )


    def train_dataloader(self):
        # print('train_dataset',self.train_dataset[0][0])
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        # print('val_dataset',self.val_dataset[0][0])
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset))


    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def adj(self):
        return self._adj
