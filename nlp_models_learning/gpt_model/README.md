# GPT2-124M LLM

一个基于 Transformer 架构的 GPT2-124M 模型实现，旨在提供一个本地复现可训练的文本生成模型。主要实现了文本生成的核心功能，支持多种配置的 GPT2 模型。

## 项目结构

```
gpt_model/
├── datasets/                   # 存放训练数据集
│   └── the-verdict.txt         # 用于训练的文本文件
├── params/                     # 存放模型参数配置
│   └── gpt_params.json         # 各种模型配置参数
├── src/                        # 源代码目录
│   ├── feedforward.py          # 前馈网络模块
│   ├── generate_text.py        # 文本生成模块
│   ├── gpt_model.py            # GPT 模型定义
│   ├── layernorm.py            # 层归一化模块
│   ├── multihead_attention.py  # 多头注意力模块
│   └── transformer_block.py    # Transformer 块模块
└── tests/                      # 测试代码目录
    ├── gpt_training.py         # 训练模型的测试
    └── test_gpt_model.py       # GPT 模型的单元测试
```

## 项目运行流程

- Terminal> python -m test.test_gpt_model

## 项目总览

该项目实现了一个简化的 GPT 模型，主要包括以下几个部分：

- **数据处理**：从文本文件中读取数据并进行预处理，生成训练所需的输入和目标序列。
- **模型定义**：实现了 GPT 模型的各个组件，包括嵌入层、位置编码、多头注意力机制、前馈网络和层归一化。
- **训练与评估**：提供训练模型的功能，并在训练过程中评估模型性能。
- **文本生成**：根据训练好的模型和给定的上下文生成新的文本。

## GPT 模型内部细节

### 1. 嵌入层 (Embedding Layer)

- **词嵌入 (Token Embedding)**：使用 `nn.Embedding` 类将输入的词汇 ID 转换为对应的嵌入向量。每个词汇都有一个对应的向量表示，模型通过这些向量来理解词汇的语义。
- **位置嵌入 (Positional Embedding)**：由于 Transformer 模型不具备序列信息，位置嵌入用于为每个词汇提供位置信息。位置嵌入通过正弦和余弦函数生成，确保模型能够理解词汇在句子中的顺序。
- `src/gpt_model.py`中的嵌入层相关代码：
    ```python
    # 初始化词嵌入层
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    # 初始化位置嵌入层
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
    ......
    ......
    ......
    # 获取词嵌入
    # Shape: (batch_size, num_tokens, emb_dim)
    tok_embeds = self.tok_emb(in_idx)
    # 获取位置嵌入
    # Shape: (num_tokens, emb_dim)
    pos_embeds = self.pos_emb(torch.arange(
        in_idx.size(1), device=in_idx.device))
    # 将词嵌入和位置嵌入相加
    x = tok_embeds + pos_embeds  # Shape: (batch_size, num_tokens, emb_dim)
    ```

### 2. Transformer 块 (Transformer Block)

每个 Transformer 块由以下几个部分组成，完整 Block 块可查看`src/transformer_block.py`：

**多头注意力机制 (Multi-Head Attention)**：
  - 通过多个注意力头并行计算，模型能够关注输入序列中的不同部分，从而捕捉更丰富的上下文信息。
  - 每个头独立计算注意力权重，使用以下公式：

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```
  
  - 其中Q是查询，K是键，V是值， $d_k$ 是键的维度。
  - 最后，将所有头的输出拼接在一起，经过线性变换得到最终的输出。
  - 详情可查看`src/multihead_attention.py`

**前馈网络 (FeedForward Network)**：
  - 该网络负责对每个位置的表示进行非线性变换，增强模型的表达能力。
  - 在每个 Transformer 块中，经过注意力机制的输出会传递到前馈网络。前馈网络通常由两个线性层和一个激活函数（如 GELU）组成，公式如下：

```math
\text{FFN}(x) = \text{GeLU}(xW_1 + b_1)W_2 + b_2
```

  - 详情可查看`src/FeedForward.py`

**层归一化 (Layer Normalization)**：
  - 在每个子层（注意力和前馈网络）之后，应用层归一化以稳定训练过程，减少内部协变量偏移。层归一化的公式为：

```math
    \text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta
```
  - 其中 $\mu$ 和 $\sigma$ 分别是均值和标准差， $\gamma$ 和 $\beta$ 是可学习的参数。
  - 详情可查看`src/layernorm.py`

**残差连接 (Residual Connection)**：
  - 在每个子层的输出与输入之间添加残差连接，帮助模型更好地学习和训练。公式为：

```math
    \text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))
```
  - `src/transformer_block.py`中的残差连接相关代码：
    ```python
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
    ```

### 3. 输出层 (Output Layer)

**线性层**：将 Transformer 块的输出映射到词汇表大小的概率分布，模型通过 softmax 函数生成每个词汇的预测概率：

```math
  P(x_t | x_{1:t-1}) = \text{softmax}(W_{out}h_t)
```

  - 其中 $h_t$ 是当前时间步的隐藏状态， $W_{out}$ 是输出权重矩阵。
  - `src/gpt_model.py`中输出层相关代码：
    ```python
    # 应用Dropout
    x = self.drop_emb(x)
    # 通过Transformer块
    x = self.trf_blocks(x)
    # 应用最终的层归一化
    x = self.final_norm(x)
    # 通过输出头, 得到对数概率分布
    # Shape: (batch_size, num_tokens, vocab_size)
    logits = self.out_head(x)
    ```

## 训练策略

### 1. 学习率调度

**学习率预热 (Learning Rate Warmup)**：在训练的初期，逐渐增加学习率，以避免模型在初始阶段的震荡。通常在前几个 epoch 中逐步增加学习率，达到设定的最大值后再进行衰减。具体实现中，学习率在前 N 个步骤内线性增加，公式为：
```math
  \text{lr}(t) = \frac{t}{N} \cdot \text{lr}_{max}
```
  - 其中 $t$ 是当前步骤， $N$ 是预热的步骤数， $\text{lr}_{max}$ 是最大学习率。
  - `tests/gpt_training.py`中学习率预热相关代码：
    ```python
    peak_lr = optimizer.param_groups[0]["lr"]             # 获取最大学习率
    total_training_steps = len(train_loader) * n_epochs   # 计算总训练步数
    lr_increment = (peak_lr - initial_lr) / warmup_steps  # 计算学习率增量
    ```

**余弦衰减 (Cosine Decay)**：一种平滑降低学习率的策略，让学习率从初始值逐渐下降到最小值，以帮助模型在接近收敛时更精细地调整参数，变化曲线像余弦函数的一半周期（从0到π）。
**核心思想**：
  - 训练初期：用较大的学习率快速收敛。  
  - 训练后期：用较小的学习率精细调整参数，避免震荡。
  - 比线性衰减更平滑，能更好地逼近最优解。  
  - 具体公式：
```math
\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos\left(\frac{t}{T} \pi\right)\right)
```
  - $T$ ：总训练步数（从最大学习率衰减到最小学习率的步数）。
  - $t$ ：当前训练步数（ 0 $\leq t$ $\leq T$ ）。
  - $\eta_{\text{min}}$ ：最小学习率（通常设为0或接近0）。
  - $\eta_{\text{max}}$ ：初始学习率（最大值）。
  - `tests/gpt_training.py`中余弦衰减相关代码：
    ```python
    progress = ((global_step - warmup_steps) / (total_training_steps - warmup_steps))
    lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    ```

**梯度裁剪 (Gradient Clipping)**：一种防止梯度爆炸（Gradient Explosion）的技术，在训练深度学习模型（如GPT）时，梯度可能会变得非常大，导致模型参数更新不稳定，甚至无法收敛。梯度裁剪的作用就是限制梯度的大小，使其不超过某个设定的阈值（threshold）

梯度裁剪的公式（L2范数裁剪）
  - 计算梯度的L2范数（长度）：

```math
   \text{grad\_norm} = \sqrt{\sum_{i} g_i^2}
```
  - $g_i$ 是梯度的每个分量（每个参数的梯度）。
  - `tests/gpt_training.py`中梯度裁剪相关代码：
    ```python
    if global_step >= warmup_steps:     # 预热阶段后应用梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    ```

梯度裁剪（Gradient Clipping）公式的简单解释：计算所有梯度的平方和，再开平方，得到梯度的总长度。

**判断是否裁剪**：
  - 如果 grad_norm > threshold，说明梯度太大，需要裁剪。
  - 如果 grad_norm ≤ threshold，梯度正常，不需要裁剪。

**裁剪方法**：

如果梯度太大，就按比例缩小梯度，使其长度等于 threshold：

```math
\text{clipped\_grad} = \frac{\text{threshold}}{\text{grad\_norm}} \cdot \text{grad}
```
这样，裁剪后的梯度长度就是 threshold，不会过大。

**梯度裁剪的作用**：
  - 防止梯度爆炸：特别是在RNN、LSTM、Transformer等模型中，梯度可能会变得非常大，导致训练失败。
  - 稳定训练：让梯度保持在一个合理的范围内，避免参数更新过大或过小。
  - 提高收敛性：避免模型因梯度爆炸而无法收敛。

### 2. 批量大小 (Batch Size)

- 选择合适的批量大小以平衡训练速度和内存使用。较大的批量大小可以加速训练，但可能会导致内存不足。通常在训练过程中会进行实验以找到最佳的批量大小。

### 3. 正则化 (Regularization)

**Dropout**：在前馈网络和注意力层中使用 Dropout 技术，以减少过拟合。通过随机丢弃一定比例的神经元，增强模型的泛化能力。通常设置为 0.1 到 0.3 之间。

**权重衰减 (Weight Decay)**：在损失函数中添加 L2 正则化项，以防止模型过拟合。损失函数的形式为：
```math
  L = L_{original} + \lambda \sum_{i} w_i^2
```
  - 其中 $L_{original}$ 是原始损失， $\lambda$ 是正则化强度， $w_i$ 是模型的权重。

### 4. 训练过程监控

- 在训练过程中，定期评估模型在验证集上的性能，监控训练损失和验证损失，以便及时调整训练策略。

## 输出方式的细节

### 1. 文本生成

- **生成策略**：在生成文本时，可以使用不同的策略，如贪婪搜索、随机采样、温度采样和 top-k 采样等。
  - **贪婪搜索**：每一步选择概率最高的词汇，生成的文本可能缺乏多样性。
  - **随机采样**：根据预测的概率分布随机选择下一个词汇，生成的文本更加多样化。
  - **温度采样**：通过调整温度参数来控制生成文本的多样性。较高的温度会导致更随机的选择，较低的温度则会使选择更集中。温度的公式如下，其中 $T$ 是温度参数：
```math
P(x) = \frac{e^{\frac{log(P(x))}{T}}}{\sum_{i} e^{\frac{log(P(x_i))}{T}}}
```
  - **Top-k 采样**：在每一步中，仅考虑概率最高的 k 个词汇进行采样，避免低概率词汇的影响。
  - 具体策略实现可查看`src/generate_text.py`中的代码

### 2. 输出格式

- 生成的文本将以字符串形式返回，可以根据需要进行后续处理或直接输出到文件中。

## 项目运行流程

### 1. 数据准备

- 训练数据在 `datasets/` 文件夹中，文件名为 `the-verdict.txt`。该文件包含用于训练的文本数据。

### 2. 配置参数

- 在 `params/gpt_params.json` 中，根据需要调整模型的参数配置。该文件包含不同配置的 GPT 模型参数，例如词汇表大小、上下文长度、嵌入维度等。

### 3. 训练模型

- `tests/gpt_training.py`中调用 `train_model` 函数进行模型训练。

- 训练过程中，模型将读取 `the-verdict.txt` 文件中的数据，进行训练并输出训练损失和验证损失。

### 4. 生成文本

- 训练完成后，使用 `src/generate_text.py` 中的 `generate_text` 函数，根据训练好的模型生成文本。可以在 `tests/gpt_training.py` 中设置生成文本的起始上下文。

### 5. 查看结果

- 训练过程中，模型会定期输出生成的文本样本，根据这些样本评估模型的生成能力。

### 6. 调整和重训

- 根据生成的文本质量，可以调整 `params/gpt_params.json` 中的参数，或更改训练数据，然后重新训练模型以提高性能。

## 项目主要步骤

### 1. 数据处理

- **数据集类**：`GPTDataset` 类用于将文本切分为输入和目标序列对，支持滑动窗口生成重叠序列。

### 2. 模型定义

- **GPTModel**：实现了 GPT 模型的核心结构，包括嵌入层、位置编码、多个 Transformer 块和输出层。

### 3. 训练与评估

- **训练函数**：`train_model` 函数负责模型的训练过程，包括前向传播、损失计算和参数更新。
- **评估函数**：在训练过程中定期评估模型性能，计算训练集和验证集的损失。

### 4. 文本生成

- **生成函数**：`generate_text` 函数根据给定的上下文生成新的文本，支持温度采样和 top-k 过滤。

## 测试与验证

- **性能测试**：对模型的训练和生成速度进行性能测试，评估模型的效率。
