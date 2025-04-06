import torch


def generate_text(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    根据给定的模型和初始上下文生成新的文本, 支持温度采样、top-k过滤和提前终止

    参数:
        model (GPTModel): 训练好的GPT模型
        idx (torch.Tensor): 初始上下文的索引张量, 形状为 (batch_size, T)
        max_new_tokens (int): 需要生成的最大新令牌数量
        context_size (int): 模型支持的最大上下文长度
        temperature (float, 可选): 控制采样随机性的温度参数, 默认为0.0(贪婪采样)
        top_k (int, 可选): 从概率最高的k个token中采样的数量, 默认为None(不启用top-k过滤)
        eos_id (int, 可选): 结束符的token ID, 生成该ID时提前终止, 默认为None

    返回:
        torch.Tensor: 包含生成文本的索引张量, 可能由于eos_id提前终止, 形状不超过(batch_size, T + max_new_tokens)
    """

    # idx 是 (B, T) 形状的张量, 包含当前上下文的索引
    for _ in range(max_new_tokens):
        # 如果当前上下文超过模型支持的最大长度, 截取最后context_size个token
        idx_cond = idx[:, -context_size:]

        # 禁用梯度计算以节省内存(推理阶段不需要反向传播)
        with torch.no_grad():
            # 获取模型输出的对数概率分布, 形状(batch_size, n_token, vocab_size)
            outputs_model = model(idx_cond)

        # 仅关注最后一个时间步的预测结果
        # 形状从(batch_size, n_token, vocab_size)变为(batch_size, vocab_size)
        if hasattr(outputs_model, "logits"):
            logits = outputs_model.logits
        elif hasattr(outputs_model, "last_hidden_state"):
            logits = outputs_model.last_hidden_state
        elif isinstance(outputs_model, torch.Tensor):
            logits = outputs_model
        logits = logits[:, -1, :]

        # top-k过滤: 仅保留概率最高的k个token
        if top_k is not None:
            # 获取前k大的对数概率值及其索引
            top_logits, _ = torch.topk(logits, top_k)
            # 确定第k大的值作为阈值
            min_val = top_logits[:, -1]
            # 将小于阈值的logits设为负无穷(对应概率为0)
            logits = torch.where(logits < min_val, torch.tensor(
                float("-inf")).to(logits.device), logits)

        # 温度调节: 温度越高分布越平滑, 温度趋近0时接近贪婪采样
        if temperature > 0.0:
            # 通过温度参数缩放对数概率
            logits = logits / temperature
            # 将缩放后的logits转换为概率分布
            probs = torch.softmax(logits, dim=-1)
            # 根据概率分布进行随机采样
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # 温度=0时直接取概率最大的token(确定性输出)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 如果生成结束符则提前终止生成过程
        if idx_next == eos_id:
            break

        # 将新生成的token索引追加到当前上下文中
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
