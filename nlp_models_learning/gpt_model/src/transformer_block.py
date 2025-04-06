import torch.nn as nn
from src.multihead_attention import MultiHeadAttention
from src.layernorm import LayerNorm
from src.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Transformer块模块, 包含多头注意力机制和前馈网络

    参数:
        cfg (dict): 配置字典, 包含以下键: 
            - emb_dim (int): 嵌入维度大小
            - context_length (int): 上下文序列的最大长度
            - n_heads (int): 注意力头的数量
            - drop_rate (float): Dropout概率值
            - qkv_bias (bool, optional): 是否在线性变换中使用偏置, 默认为False

    属性:
        att (MultiHeadAttention): 多头注意力机制模块
        ff (FeedForward): 前馈网络模块
        norm1 (LayerNorm): 第一个层归一化模块
        norm2 (LayerNorm): 第二个层归一化模块
        drop_shortcut (nn.Dropout): Dropout层, 用于在残差连接中随机丢弃一些神经元, 以避免过拟合
    """

    def __init__(self, cfg):
        super().__init__()
        # 使用从零开始构造的多头自注意力机制(不使用MHAPyTorchScaledDotProduct)
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        # 初始化前馈网络
        self.ff = FeedForward(cfg)
        # 初始化第一个层归一化层
        self.norm1 = LayerNorm(cfg["emb_dim"])
        # 初始化第二个层归一化层
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # 初始化残差连接中的Dropout层
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        前向传播方法, 将输入张量通过Transformer块

        参数:
            x (torch.Tensor): 输入张量, 形状为 (batch_size, num_tokens, emb_dim)

        返回:
            torch.Tensor: 输出张量, 形状与输入相同 (batch_size, num_tokens, emb_dim)
        """
        # 残差连接1: 注意力机制
        shortcut = x
        x = self.norm1(x)          # 层归一化
        x = self.att(x)            # 多头注意力
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut           # 残差连接

        # 残差连接2: 前馈网络
        shortcut = x
        x = self.norm2(x)          # 层归一化
        x = self.ff(x)             # 前馈网络
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut           # 残差连接

        return x
