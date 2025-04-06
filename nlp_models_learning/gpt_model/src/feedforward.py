import torch
import torch.nn as nn


class GELU(nn.Module):
    """
    高斯误差线性单元(Gaussian Error Linear Unit)激活函数

    GELU 是一种平滑的非线性激活函数, 其定义为: 
    GELU(x) = x * P(X <= x) = 0.5 * x * (1 + tanh(sqrt(2 / π) * (x + 0.044715 * x^3)))

    属性:
        无额外属性, 继承自 nn.Module

    方法:
        forward(x): 前向传播方法, 计算输入张量 x 的 GELU 激活值
    """

    def __init__(self):
        """
        初始化 GELU 模块
        """
        super().__init__()

    def forward(self, x):
        """
        前向传播方法, 计算输入张量 x 的 GELU 激活值

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 应用 GELU 激活后的输出张量
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    前馈网络模块, 用于Transformer架构中的每个Transformer块

    参数:
        cfg (dict): 配置字典, 包含键: 
            - emb_dim (int): 输入和输出的嵌入维度大小

    属性:
        layers (nn.Sequential): 包含三个层的顺序容器: 
            1. 线性层: 将输入维度扩展为4倍 (emb_dim -> 4 * emb_dim)
            2. GELU激活函数: 提供非线性变换
            3. 线性层: 将维度缩小回原始大小 (4 * emb_dim -> emb_dim)
    """

    def __init__(self, cfg):
        super().__init__()
        # 定义前馈网络的层序列
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 扩展维度
            GELU(),                                         # 非线性激活函数(等价于nn.GELU())
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])   # 缩小回原始维度
        )

    def forward(self, x):
        """
        前向传播方法, 将输入张量通过前馈网络

        参数:
            x (torch.Tensor): 输入张量, 形状为 (batch_size, num_tokens, emb_dim)

        返回:
            torch.Tensor: 输出张量, 形状与输入相同 (batch_size, num_tokens, emb_dim)
        """
        return self.layers(x)
