import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    实现层归一化(Layer Normalization)

    参数:
        emb_dim (int): 输入张量的特征维度大小
        eps (float, optional): 稳定计算的数值偏移量, 默认为1e-5

    属性:
        eps (float): 数值偏移量, 用于防止除零错误
        scale (nn.Parameter): 可学习的比例参数, 形状为(emb_dim,)
        shift (nn.Parameter): 可学习的偏移参数, 形状为(emb_dim,)
    """

    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps                                   # 防止分母为零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习的比例参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的偏移参数

    def forward(self, x):
        """
        前向传播函数, 对输入张量进行层归一化

        参数:
            x (torch.Tensor): 输入张量, 形状为(batch_size, ..., emb_dim)

        返回:
            torch.Tensor: 归一化后的张量, 形状与输入相同
        """
        # 计算最后一个维度上的均值和方差
        mean = x.mean(dim=-1, keepdim=True)                # 均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 方差

        # 对输入张量进行归一化
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # 应用可学习的缩放和平移参数
        return self.scale * norm_x + self.shift
