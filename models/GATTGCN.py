import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph_conv import calculate_laplacian_with_self_loop


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.6):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout

        # 定义图注意力的参数
        # (in_features,out_features)
        self.W = nn.Parameter(torch.Tensor(in_features, out_features * num_heads))
        self.attn = nn.Parameter(torch.Tensor(2 * out_features, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        # 初始化图注意力的参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, h, adj_mask=None):
        # h: (batch, num_nodes, in_features)
        batch_size, num_nodes, _ = h.shape
        # 线性变换 + 分头
        h_trans = torch.matmul(h, self.W)  # (batch, num_nodes, out_features*num_heads)
        h_trans = h_trans.view(batch_size, num_nodes, self.num_heads, -1)
        h_trans = h_trans.permute(2, 0, 1, 3)  # (num_heads, batch, num_nodes, out_features)
        attn_scores = []
        for head in range(self.num_heads):
            h_head = h_trans[head]
            # 拼接所有节点对 (i,j)
            # h_i = (batch, num_nodes, num_nodes, out_features)
            h_i = h_head.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            # h_j = (batch, num_nodes, num_nodes, out_features)
            h_j = h_head.unsqueeze(1).expand(-1, num_nodes, -1, -1)
            # pair = (batch, num_nodes, num_nodes, 2*out_features)
            pair = torch.cat([h_i, h_j], dim=-1)
            # e = (batch, num_nodes, num_nodes)
            e = torch.matmul(pair, self.attn).squeeze(-1)
            e = self.leaky_relu(e)
            if adj_mask is not None:
                e = e.masked_fill(adj_mask == 0, -1e9)
            # 按行对得分e进行归一化
            attention = F.softmax(e, dim=-1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            # 每个头的注意力权重矩阵
            attn_scores.append(attention)
        # 多头特征聚合
        outputs = []
        for head in range(self.num_heads):
            # (batch, num_nodes, out_features)
            h_head = h_trans[head]
            # (batch, num_nodes, num_nodes)
            attn = attn_scores[head]
            # (batch, num_nodes, out_features)
            h_new = torch.matmul(attn, h_head)
            outputs.append(h_new)
        # (batch, num_nodes, num_heads * out_features)
        output = torch.cat(outputs, dim=-1)
        return output


class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, input_dim: int, num_gru_units: int, output_dim: int, num_heads=8, fusion="concat"):
        """
        fusion: 'add' 表示将 GAT 与 Laplacian 分支输出相加，
               'concat' 表示将两者拼接。
        """
        super(TGCNGraphConvolution, self).__init__()
        self._input_dim = input_dim
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self.fusion = fusion

        # GAT 分支
        self.gat = GATLayer(
            in_features=input_dim + num_gru_units,
            out_features=output_dim // num_heads,  # 确保输出维度匹配
            num_heads=num_heads,
            dropout=0.6
        )
        self.register_buffer("adj_mask", torch.FloatTensor(adj))

        # Laplacian 分支
        self.register_buffer("laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self.weights = nn.Parameter(torch.randn(input_dim + num_gru_units, output_dim))
        self.biases = nn.Parameter(torch.randn(output_dim))

    def forward(self, inputs, hidden_state):
        # inputs: (batch_size, num_nodes, input_dim)
        # hidden_state: (batch_size, num_nodes, num_gru_units)
        concatenation = torch.cat((inputs, hidden_state), dim=-1)  # (batch, num_nodes, input_dim+num_gru_units)

        # GAT 分支输出
        # (batch_size, num_nodes, output_dim)
        gat_output = self.gat(concatenation, self.adj_mask.unsqueeze(0))

        # Laplacian 分支输出
        batch_size, num_nodes, _ = concatenation.shape
        laplacian_batch = self.laplacian.unsqueeze(0).expand(batch_size, -1, -1)
        # torch.bmm批量矩阵乘法（矩阵1形状为 (batch_size, m, n)，矩阵2形状为 (batch_size, n, p)，返回值为(batch_size, m, p)）
        # (batch_size, num_nodes, input_dim + num_gru_units)
        # 邻域聚合
        a_times_concat = torch.bmm(laplacian_batch, concatenation)
        # (batch_size, num_nodes, output_dim)
        # 特征变换
        laplacian_output = torch.matmul(a_times_concat, self.weights) + self.biases

        # 融合：相加或拼接
        if self.fusion == "add":
            output = gat_output + laplacian_output
        elif self.fusion == "concat":
            output = torch.cat((gat_output, laplacian_output), dim=-1)
        else:
            raise ValueError("输入正确的fusion的值: {}".format(self.fusion))
        return output

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim,
                "num_gru_units": self._num_gru_units,
                "output_dim": self._output_dim,
                "fusion": self.fusion}


class GATTGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int, num_heads=8, fusion="concat"):
        super(GATTGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        # 根据融合方式调整output_dim参数
        if fusion == "concat":
            # graph_conv1的输出应为 hidden_dim *2，所以设置output_dim为hidden_dim
            gconv1_output = hidden_dim
            # graph_conv2的输出应为 hidden_dim，所以设置output_dim为 hidden_dim//2
            gconv2_output = hidden_dim // 2
        else:
            gconv1_output = hidden_dim * 2
            gconv2_output = hidden_dim
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj, self._input_dim, self._hidden_dim, gconv1_output, num_heads=num_heads, fusion=fusion
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.adj, self._input_dim, self._hidden_dim, gconv2_output, num_heads=num_heads, fusion=fusion
        )

    def forward(self, inputs, hidden_state):
        # 第一个图卷积
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        r, u = torch.chunk(concatenation, chunks=2, dim=-1)
        # 第二个图卷积
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class GATTGCN(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int, num_heads=2, fusion="add", **kwargs):
        super(GATTGCN, self).__init__()
        self._input_dim = input_dim  # 每个节点的特征数量
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.gattgcn_cell = GATTGCNCell(self.adj, self._input_dim, self._hidden_dim, num_heads=num_heads, fusion=fusion)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, num_nodes, input_dim)
        batch_size, seq_len, num_nodes, _ = inputs.shape
        hidden_state = torch.zeros(batch_size, num_nodes, self._hidden_dim).type_as(inputs)
        for i in range(seq_len):
            hidden_state = self.gattgcn_cell(inputs[:, i, :, :], hidden_state)
        return hidden_state

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=32)
        parser.add_argument("--num_heads", type=int, default=2)
        parser.add_argument("--fusion", type=str, default="add", choices=["add", "concat"])
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
