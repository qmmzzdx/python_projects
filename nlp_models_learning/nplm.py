import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

# 创建简单的训练数据集
sentences = ["i like apple", "i love banana", "i hate rat"]

# 构建词汇表
word_list = list(set(" ".join(sentences).split()))
word_to_idx = {word: idx for idx, word in enumerate(word_list)}  # 英文单词到索引的映射
idx_to_word = {idx: word for idx, word in enumerate(word_list)}  # 索引到英文单词的映射
vocab_size = len(word_list)
print('Vocabulary:', word_to_idx)
print('Vocabulary size:', vocab_size)

# 超参数配置
BATCH_SIZE = 2        # 每个训练批次包含的样本数
CONTEXT_SIZE = 2      # 使用前2个单词预测下一个单词
EMBEDDING_DIM = 2     # 词向量维度
HIDDEN_DIM = 2        # 隐藏层神经元数量
LEARNING_RATE = 0.1   # 学习率
EPOCHS = 5000         # 训练总轮次


def make_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    """随机生成训练批次数据"""
    inputs, targets = [], []
    selected = random.sample(sentences, BATCH_SIZE)

    for sentence in selected:
        words = sentence.split()
        # 构造输入（英文单词索引）和目标（英文单词索引）
        inputs.append([word_to_idx[w] for w in words[:-1]])
        targets.append(word_to_idx[words[-1]])

    return torch.LongTensor(inputs), torch.LongTensor(targets)


# 展示数据批次
input_batch, target_batch = make_batch()
print("\nBatch example:")
print("Input indices:", input_batch)
print("Input tokens:", [[idx_to_word[idx.item()]
      for idx in seq] for seq in input_batch])
print("Target indices:", target_batch)
print("Target tokens:", [idx_to_word[idx.item()] for idx in target_batch])


class NeuralProbabilisticLM(nn.Module):
    """神经概率语言模型"""

    def __init__(self, vocab_size: int, embed_dim: int, context_size: int, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)       # 词嵌入层
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)  # 全连接层
        self.fc2 = nn.Linear(hidden_dim, vocab_size)                # 输出层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播过程
        batch_size = x.size(0)
        embeds = self.embeddings(x).view(batch_size, -1)  # 展平操作
        hidden = torch.tanh(self.fc1(embeds))             # 激活函数
        logits = self.fc2(hidden)                         # 输出逻辑值
        return logits


# 初始化模型组件
model = NeuralProbabilisticLM(
    vocab_size=vocab_size,
    embed_dim=EMBEDDING_DIM,
    context_size=CONTEXT_SIZE,
    hidden_dim=HIDDEN_DIM
)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nModel architecture:", model)

# 模型训练
print("\nTraining progress:")
for epoch in range(EPOCHS):
    optimizer.zero_grad()               # 梯度清零
    inputs, targets = make_batch()      # 生成数据
    outputs = model(inputs)             # 前向计算
    loss = criterion(outputs, targets)  # 计算损失

    # 定期打印训练进度（英文输出）
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch: {epoch+1:04d}, Loss: {loss.item():.6f}")

    loss.backward()    # 反向传播
    optimizer.step()   # 参数更新

test_inputs = [['i', 'hate'], ['i', 'like']]
input_indices = torch.LongTensor(
    [[word_to_idx[w] for w in seq] for seq in test_inputs])

with torch.no_grad():
    predictions = model(input_indices).argmax(dim=1)

print("\nPredictions:")
for seq, pred_idx in zip(test_inputs, predictions):
    print(f"Context: {seq} -> Prediction: {idx_to_word[pred_idx.item()]}")
