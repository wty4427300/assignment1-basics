import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # 定义权重参数，形状为 (d_out, d_in)
        # 初始权重通常需要更精细的初始化（如 Xavier/Kaiming），这里先用基础定义
        self.weight = nn.Parameter(torch.empty(d_out, d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 einsum 执行 y = xW^T
        # '...i' 表示输入 x 的最后维度 (d_in)
        # 'oi' 表示权重参数 (d_out, d_in)
        # '...o' 表示输出的最后维度 (d_out)
        return torch.einsum('...i, oi -> ...o', x, self.weight)

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # 定义嵌入矩阵参数，形状为 (vocab_size, d_model)
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 是包含 Token ID 的整数张量，形状为 (...)
        # 直接使用 x 索引权重矩阵，返回形状为 (..., d_model) 的张量
        return self.weight[x]
