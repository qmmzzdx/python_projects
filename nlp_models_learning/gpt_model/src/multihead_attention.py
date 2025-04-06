import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块

    参数:
        d_in (int): 输入特征维度
        d_out (int): 输出特征维度, 必须是num_heads的倍数
        context_length (int): 上下文序列的最大长度
        dropout (float): Dropout概率值
        num_heads (int): 注意力头的数量
        qkv_bias (bool, optional): 是否在线性变换中使用偏置默认为False

    属性:
        d_out (int): 输出特征维度
        num_heads (int): 注意力头的数量
        head_dim (int): 每个注意力头的维度大小
        w_query (nn.Linear): 用于生成查询向量的线性层
        w_key (nn.Linear): 用于生成键向量的线性层
        w_value (nn.Linear): 用于生成值向量的线性层
        out_proj (nn.Linear): 输出投影层, 将多头注意力的结果映射回原始维度
        dropout (nn.Dropout): Dropout层, 防止过拟合
        mask (torch.Tensor): 因果掩码矩阵, 用于自注意力计算
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 确保d_out可以被num_heads整除
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # 计算每个头的维度大小
        self.head_dim = d_out // num_heads

        # 创建三个线性层, 分别用于生成查询、键和值向量
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 创建输出投影层, 将多头注意力的结果映射回原始维度
        self.out_proj = nn.Linear(d_out, d_out)

        # 创建Dropout层, 用于在训练过程中随机丢弃一些神经元, 以避免过拟合
        self.dropout = nn.Dropout(dropout)

        # 创建一个上三角矩阵作为因果掩码, 并将其注册为缓冲区
        # 该掩码用于在计算注意力权重时屏蔽未来信息
        self.register_buffer("mask", torch.triu(
            torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # 获取输入张量的批量大小、序列长度
        batch_size, num_tokens = x.size(0), x.size(1)

        # 对输入进行线性投影, 以获得查询、键和值的表示, 调整维度顺序以适应注意力计算
        # Shape: (batch_size, num_tokens, d_out) -> (batch_size, num_tokens, num_heads, head_dim)
        # Shape: (batch_size, num_tokens, num_heads, head_dim) -> (batch_size, num_heads, num_tokens, head_dim)
        queries = self.w_query(x).view(
            batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.w_key(x).view(batch_size, num_tokens,
                                  self.num_heads, self.head_dim).transpose(1, 2)
        values = self.w_value(x).view(
            batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用因果掩码计算缩放点积注意力(即自注意力机制)
        # 注意力分数计算: queries与keys的点积
        attn_scores = queries @ keys.transpose(2, 3)

        # 截取掩码以匹配当前序列长度, 并转换为布尔类型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数矩阵中的无效位置为-inf
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算注意力权重, 并应用dropout以防止过拟合
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 根据注意力权重加权求和得到上下文向量
        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并多个头的输出, 并通过可选的输出投影层
        # self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        # 返回最终的上下文向量
        return context_vec


class MHAPyTorchScaledDotProduct(nn.Module):
    """
    多头缩放点积注意力机制模块

    参数:
        d_in (int): 输入特征维度
        d_out (int): 输出特征维度, 必须是num_heads的倍数
        context_length (int): 上下文序列的最大长度
        dropout (float): Dropout概率值
        num_heads (int): 注意力头的数量
        qkv_bias (bool, optional): 是否在线性变换中使用偏置, 默认为False

    属性:
        d_out (int): 输出特征维度
        num_heads (int): 注意力头的数量
        head_dim (int): 每个注意力头的维度大小
        qkv (nn.Linear): 用于生成查询、键和值向量的线性层
        proj (nn.Linear): 输出投影层, 将多头注意力的结果映射回原始维度
        dropout (float): Dropout概率值
    """

    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        # 确保d_out可以被num_heads整除
        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        # 计算每个头的维度大小
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        # 创建一个线性层, 用于生成查询、键和值的组合
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        # 创建输出投影层, 将多头注意力的结果映射回原始维度
        self.proj = nn.Linear(d_out, d_out)
        # 设置Dropout概率值
        self.dropout = dropout

    def forward(self, x):
        # 获取输入张量的批量大小、序列长度
        batch_size, num_tokens = x.size(0), x.size(1)

        # 对输入进行线性投影, 以获得查询、键和值的表示
        # Shape: (batch_size, num_tokens, 3 * d_out)
        qkv = self.qkv(x)

        # 将查询、键和值的维度重新调整为多头形式
        # (batch_size, num_tokens, 3 * d_out) -> (batch_size, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3,
                       self.num_heads, self.head_dim)

        # 调整维度顺序以适应注意力计算
        # (batch_size, num_tokens, 3, num_heads, head_dim) -> (3, batch_size, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 提取查询、键和值
        # (3, batch_size, num_heads, num_tokens, head_dim) -> 3 * (batch_size, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        # 根据训练状态设置dropout概率
        use_dropout = 0.0 if not self.training else self.dropout

        # 应用缩放点积注意力计算
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # 合并多个头的输出, 并通过可选的输出投影层
        # self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.d_out)

        # 通过输出投影层映射回原始维度
        context_vec = self.proj(context_vec)

        # 返回最终的上下文向量
        return context_vec
