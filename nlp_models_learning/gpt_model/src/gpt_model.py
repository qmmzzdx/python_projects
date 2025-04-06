import torch
import numpy as np
import torch.nn as nn
from src.transformer_block import TransformerBlock
from src.layernorm import LayerNorm


class GPTModel(nn.Module):
    """
    GPT模型类, 包含嵌入层、位置编码、Transformer块、层归一化和输出头

    参数:
        cfg (dict): 配置字典, 包含以下键: 
            - vocab_size (int): 词汇表大小
            - context_length (int): 上下文序列的最大长度
            - emb_dim (int): 嵌入维度大小
            - n_heads (int): 注意力头的数量
            - n_layers (int): Transformer块的数量
            - drop_rate (float): Dropout概率值
            - qkv_bias (bool, optional): 是否在线性变换中使用偏置, 默认为False

    属性:
        tok_emb (nn.Embedding): 词嵌入层, 将词汇表中的每个词映射到嵌入向量
        pos_emb (nn.Embedding): 位置嵌入层, 为每个位置提供位置信息
        drop_emb (nn.Dropout): Dropout层, 用于在嵌入层之后随机丢弃一些神经元, 以避免过拟合
        trf_blocks (nn.Sequential): 包含多个Transformer块的顺序容器
        final_norm (LayerNorm): 最终的层归一化层
        out_head (nn.Linear): 输出头, 将嵌入向量映射到词汇表大小的概率分布
    """

    def __init__(self, cfg):
        super().__init__()
        # 初始化词嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 初始化位置嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 初始化嵌入层之后的Dropout层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 初始化多个Transformer块
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 初始化最终的层归一化层
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 初始化输出头
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        前向传播方法, 将输入索引张量通过GPT模型

        参数:
            in_idx (torch.Tensor): 输入索引张量, 形状为 (batch_size, num_tokens)

        返回:
            torch.Tensor: 输出对数概率分布张量, 形状为 (batch_size, num_tokens, vocab_size)
        """
        # 获取词嵌入
        # Shape: (batch_size, num_tokens, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        # 获取位置嵌入
        # Shape: (num_tokens, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(
            in_idx.size(1), device=in_idx.device))
        # 将词嵌入和位置嵌入相加
        x = tok_embeds + pos_embeds  # Shape: (batch_size, num_tokens, emb_dim)
        # 应用Dropout
        x = self.drop_emb(x)
        # 通过Transformer块
        x = self.trf_blocks(x)
        # 应用最终的层归一化
        x = self.final_norm(x)
        # 通过输出头, 得到对数概率分布
        # Shape: (batch_size, num_tokens, vocab_size)
        logits = self.out_head(x)

        return logits

    def assign_check(self, current_param, target_param):
        """参数加载验证(包含空值检查)"""
        if current_param is None or target_param is None:
            raise ValueError(
                f"current_param type: {type(current_param)}, target_param type: {type(target_param)}")
        if current_param.shape != target_param.shape:
            raise ValueError(
                f"Shape mismatch. current_param: {current_param.shape}, Rightarget_paramt: {target_param.shape}")
        return torch.nn.Parameter(target_param.clone().detach())

    def load_weights(self, cfg, gpt_hf):
        """
        从GPT模型中加载权重到当前模型中

        参数:
        - cfg: 包含模型配置的字典, 例如层数等
        - gpt_hf: 预训练的GPT模型实例

        此函数不会返回任何值, 但它会将gpt_hf模型中的权重加载到当前模型中
        """
        # 获取GPT模型的状态字典
        d = gpt_hf.state_dict()

        # 加载位置嵌入和词嵌入权重
        self.pos_emb.weight = self.assign_check(
            self.pos_emb.weight, d["wpe.weight"])
        self.tok_emb.weight = self.assign_check(
            self.tok_emb.weight, d["wte.weight"])

        # 遍历每一层transformer块, 加载权重
        for b in range(cfg["n_layers"]):
            # 处理注意力层的权重和偏置
            q_w, k_w, v_w = np.split(
                d[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
            self.trf_blocks[b].att.w_query.weight = self.assign_check(
                self.trf_blocks[b].att.w_query.weight, q_w.T)
            self.trf_blocks[b].att.w_key.weight = self.assign_check(
                self.trf_blocks[b].att.w_key.weight, k_w.T)
            self.trf_blocks[b].att.w_value.weight = self.assign_check(
                self.trf_blocks[b].att.w_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(d[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
            if self.trf_blocks[b].att.w_query.bias:
                self.trf_blocks[b].att.w_query.bias = self.assign_check(
                    self.trf_blocks[b].att.w_query.bias, q_b)
            else:
                self.trf_blocks[b].att.w_query.bias = torch.nn.Parameter(
                    torch.zeros_like(q_b))
            if self.trf_blocks[b].att.w_key.bias:
                self.trf_blocks[b].att.w_key.bias = self.assign_check(
                    self.trf_blocks[b].att.w_key.bias, k_b)
            else:
                self.trf_blocks[b].att.w_key.bias = torch.nn.Parameter(
                    torch.zeros_like(k_b))
            if self.trf_blocks[b].att.w_value.bias:
                self.trf_blocks[b].att.w_value.bias = self.assign_check(
                    self.trf_blocks[b].att.w_value.bias, v_b)
            else:
                self.trf_blocks[b].att.w_value.bias = torch.nn.Parameter(
                    torch.zeros_like(v_b))

            # 处理注意力层的输出投影权重和偏置
            self.trf_blocks[b].att.out_proj.weight = self.assign_check(
                self.trf_blocks[b].att.out_proj.weight, d[f"h.{b}.attn.c_proj.weight"].T)
            self.trf_blocks[b].att.out_proj.bias = self.assign_check(
                self.trf_blocks[b].att.out_proj.bias, d[f"h.{b}.attn.c_proj.bias"])

            # 处理前馈网络的权重和偏置
            self.trf_blocks[b].ff.layers[0].weight = self.assign_check(
                self.trf_blocks[b].ff.layers[0].weight, d[f"h.{b}.mlp.c_fc.weight"].T)
            self.trf_blocks[b].ff.layers[0].bias = self.assign_check(
                self.trf_blocks[b].ff.layers[0].bias, d[f"h.{b}.mlp.c_fc.bias"])
            self.trf_blocks[b].ff.layers[2].weight = self.assign_check(
                self.trf_blocks[b].ff.layers[2].weight, d[f"h.{b}.mlp.c_proj.weight"].T)
            self.trf_blocks[b].ff.layers[2].bias = self.assign_check(
                self.trf_blocks[b].ff.layers[2].bias, d[f"h.{b}.mlp.c_proj.bias"])

            # 处理层归一化的权重和偏置
            self.trf_blocks[b].norm1.scale = self.assign_check(
                self.trf_blocks[b].norm1.scale, d[f"h.{b}.ln_1.weight"])
            self.trf_blocks[b].norm1.shift = self.assign_check(
                self.trf_blocks[b].norm1.shift, d[f"h.{b}.ln_1.bias"])
            self.trf_blocks[b].norm2.scale = self.assign_check(
                self.trf_blocks[b].norm2.scale, d[f"h.{b}.ln_2.weight"])
            self.trf_blocks[b].norm2.shift = self.assign_check(
                self.trf_blocks[b].norm2.shift, d[f"h.{b}.ln_2.bias"])

        # 加载最终归一化层和输出头的权重
        self.final_norm.scale = self.assign_check(
            self.final_norm.scale, d[f"ln_f.weight"])
        self.final_norm.shift = self.assign_check(
            self.final_norm.shift, d[f"ln_f.bias"])
        self.out_head.weight = self.assign_check(
            self.out_head.weight, d["wte.weight"])
