import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

# 数据集配置
sentences = ["i like apple", "i love banana", "i hate rat"]

# 构建词汇表
word_list = list(set(" ".join(sentences).split()))
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
idx_to_word = {idx: word for idx, word in enumerate(word_list)}
vocab_size = len(word_list)
print('Vocabulary:', word_to_idx)
print('Vocabulary size:', vocab_size)

# 超参数设置
BATCH_SIZE = 2
CONTEXT_SIZE = 2    # 使用前两个词预测下一个词
EMBEDDING_DIM = 2   # 词向量维度
HIDDEN_DIM = 2      # LSTM隐藏层维度
LR = 0.1
EPOCHS = 5000


def make_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    """生成训练批次数据"""
    inputs, targets = [], []
    selected = random.sample(sentences, BATCH_SIZE)

    for sentence in selected:
        words = sentence.split()
        inputs.append([word_to_idx[w] for w in words[:-1]])
        targets.append(word_to_idx[words[-1]])

    return torch.LongTensor(inputs), torch.LongTensor(targets)


# 数据样例展示
input_batch, target_batch = make_batch()
print("\nSample Batch:")
print("Input indices:", input_batch)
print("Input tokens:", [[idx_to_word[idx.item()]
      for idx in seq] for seq in input_batch])
print("Target indices:", target_batch)
print("Target tokens:", [idx_to_word[idx.item()] for idx in target_batch])


class LSTMLanguageModel(nn.Module):
    """LSTM语言模型实现"""

    def __init__(self):
        super().__init__()
        # 词嵌入层：将离散词索引映射为连续向量
        self.embed = nn.Embedding(vocab_size, EMBEDDING_DIM)
        # LSTM层：捕捉序列时序特征
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True)
        # 输出层：将隐藏状态映射到词汇表空间
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, x):
        # 输入形状: [batch_size, seq_len]
        x = self.embed(x)  # 转换为: [batch_size, seq_len, embed_dim]

        # LSTM处理序列
        # lstm_out形状: [batch_size, seq_len, hidden_dim]
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 生成预测结果
        return self.fc(last_output)


# 初始化模型组件
model = LSTMLanguageModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("\nModel Architecture:")
print(model)

# 模型训练流程
print("\nTraining Progress:")
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    inputs, targets = make_batch()
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch: {epoch+1:04d} | Loss: {loss.item():.6f}")

    loss.backward()
    optimizer.step()

# 预测演示
test_cases = [['i', 'hate'], ['i', 'like']]
input_tensor = torch.LongTensor(
    [[word_to_idx[w] for w in case] for case in test_cases])

with torch.no_grad():
    preds = model(input_tensor).argmax(dim=1)

print("\nPredictions:")
for case, pred_idx in zip(test_cases, preds):
    print(f"Input: {case} -> Prediction: {idx_to_word[pred_idx.item()]}")
