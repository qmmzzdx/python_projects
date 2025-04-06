import math
import torch
from src.generate_text import generate_text


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    计算单个批次的损失值

    参数:
        input_batch (torch.Tensor): 输入序列张量
        target_batch (torch.Tensor): 目标序列张量
        model (torch.nn.Module): GPT 模型实例
        device (torch.device): 运行设备(CPU 或 GPU)

    返回:
        torch.Tensor: 当前批次的损失值
    """
    input_batch, target_batch = input_batch.to(
        device), target_batch.to(device)            # 将数据移动到指定设备
    logits = model(input_batch)                     # 前向传播, 获取模型输出
    loss = torch.nn.functional.cross_entropy(       # 计算交叉熵损失
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    计算数据加载器中多个批次的平均损失

    参数:
        data_loader (DataLoader): 数据加载器
        model (torch.nn.Module): GPT 模型实例
        device (torch.device): 运行设备(CPU 或 GPU)
        num_batches (int, optional): 要计算的批次数量, 默认为 None(计算所有批次)

    返回:
        float: 平均损失值
    """
    total_loss = 0.0
    if len(data_loader) == 0:  # 如果数据加载器为空, 返回 NaN
        return float("nan")
    elif num_batches is None:  # 如果未指定批次数量, 使用所有批次
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:                                # 当前批次在范围内
            loss = calc_loss_batch(
                input_batch, target_batch, model, device)  # 计算单个批次损失
            total_loss += loss.item()                      # 累加损失值
        else:
            break
    return total_loss / num_batches  # 平均损失


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    在训练集和验证集上评估模型性能

    参数:
        model (torch.nn.Module): GPT 模型实例
        train_loader (DataLoader): 训练集数据加载器
        val_loader (DataLoader): 验证集数据加载器
        device (torch.device): 运行设备(CPU 或 GPU)
        eval_iter (int): 用于评估的迭代次数

    返回:
        tuple: 训练集和验证集的平均损失
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter)  # 计算训练集损失
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter)    # 计算验证集损失
    model.train()
    return train_loss, val_loss


def text_to_token_ids(text, tokenizer):
    """
    将输入文本转换为 token ID 张量, 并添加批次维度

    参数:
        text (str): 输入文本字符串
        tokenizer (Tokenizer): 分词器实例

    返回:
        torch.Tensor: 包含 token ID 的张量, 形状为 (1, T), 其中 T 是 token 数量
    """
    encoded = tokenizer.encode(text)                     # 将文本转换为 token ID 列表
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 将列表转换为张量并添加批次维度
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    将 token ID 张量转换回文本, 并移除批次维度

    参数:
        token_ids (torch.Tensor): 包含 token ID 的张量, 形状通常为 (1, T)
        tokenizer (Tokenizer): 分词器实例

    返回:
        str: 解码后的文本字符串
    """
    flat = token_ids.squeeze(0)             # 移除批次维度, 得到形状为 (T,) 的一维张量
    return tokenizer.decode(flat.tolist())  # 将张量转换为 Python 列表并解码为文本


def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    使用模型生成文本样本并打印结果

    参数:
        model (torch.nn.Module): GPT 模型实例
        tokenizer (Tokenizer): 分词器实例
        device (torch.device): 运行设备(CPU 或 GPU)
        start_context (str): 生成文本的起始上下文
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]  # 获取模型的最大上下文长度
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size, temperature=1.0, top_k=5
        )
        decoded_text = token_ids_to_text(
            token_ids, tokenizer)  # 将生成的 token ID 转换为文本
        print(f"{decoded_text.replace('\n', ' ')}\n")
    model.train()


def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6):
    """
    训练 GPT 模型

    参数:
        model (torch.nn.Module): GPT 模型实例
        train_loader (DataLoader): 训练集数据加载器
        val_loader (DataLoader): 验证集数据加载器
        optimizer (torch.optim.Optimizer): 优化器实例
        device (torch.device): 运行设备(CPU 或 GPU)
        n_epochs (int): 总训练轮数
        eval_freq (int): 评估频率(每多少步进行一次评估)
        eval_iter (int): 评估时使用的迭代次数
        start_context (str): 生成文本的起始上下文
        tokenizer (Tokenizer): 分词器实例
        warmup_steps (int): 学习率预热步数
        initial_lr (float): 初始学习率
        min_lr (float): 最小学习率

    返回:
        tuple: 训练损失、验证损失、已见token数和学习率的历史记录
    """
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    # 初始化已见token数和全局步数
    tokens_seen, global_step = 0, -1

    peak_lr = optimizer.param_groups[0]["lr"]             # 获取最大学习率
    total_training_steps = len(train_loader) * n_epochs   # 计算总训练步数
    lr_increment = (peak_lr - initial_lr) / warmup_steps  # 计算学习率增量

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清零梯度
            global_step += 1       # 更新全局步数

            # 预热阶段
            if global_step < warmup_steps:
                lr = initial_lr + global_step * lr_increment
            else:
                # 预热后余弦衰减
                progress = ((global_step - warmup_steps) /
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * \
                    (1 + math.cos(math.pi * progress))

             # 更新优化器的学习率
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # 记录当前学习率

            # 前向传播和反向传播
            loss = calc_loss_batch(
                input_batch, target_batch, model, device)
            loss.backward()
            if global_step >= warmup_steps:     # 预热阶段后应用梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
            optimizer.step()                    # 更新模型参数
            tokens_seen += input_batch.numel()  # 累加已见token数

            # 定期评估模型
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)        # 记录训练损失
                val_losses.append(val_loss)            # 记录验证损失
                track_tokens_seen.append(tokens_seen)  # 记录已见令牌数
                print(f"Ep {epoch+1} (Iter {global_step:06d}):", end=' ')
                print(f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 每个 epoch 结束时生成并打印样本
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen, track_lrs  # 返回训练记录
