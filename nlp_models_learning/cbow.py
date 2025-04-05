import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

# 定义训练句子列表
sentences = [
    "I am a student",
    "He is a customer",
    "She is a customer",
    "Mzzdx is author",
    "Xjzs is mzzdx",
]

words = ' '.join(sentences).split()

# 构建词汇表（去重并排序以确保一致性）
word_list = list(sorted(set(words)))
vocab_size = len(word_list)

# 创建词到索引和索引到词的映射字典
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
idx_to_word = {idx: word for idx, word in enumerate(word_list)}


def create_cbow_pairs(sentences: List[str], window_size: int = 2) -> List[Tuple[str, List[str]]]:
    """生成CBOW格式的(目标词, 上下文词)对
    Args:
        sentences: 原始句子列表
        window_size: 单侧上下文窗口大小
    Returns:
        目标词和上下文词对的列表，格式如[("is", ["Mzzdx", "author"]), ...]
    """
    pairs = []
    for sentence in sentences:
        tokens = sentence.split()
        # 遍历每个位置作为目标词
        for target_pos in range(len(tokens)):
            target_word = tokens[target_pos]
            # 计算上下文窗口的起始和结束位置
            start = max(0, target_pos - window_size)
            end = min(len(tokens), target_pos + window_size + 1)
            # 排除目标词本身，获取上下文词
            context_words = tokens[start:target_pos] + tokens[target_pos+1:end]
            pairs.append((target_word, context_words))
    return pairs


def get_one_hot(word: str) -> torch.Tensor:
    """生成单词的one-hot编码向量
    Args:
        word: 需要编码的单词
    Returns:
        Tensor形状 [vocab_size]，对应位置为1
    """
    tensor = torch.zeros(vocab_size)
    tensor[word_to_idx[word]] = 1.0
    return tensor


class CBOW(nn.Module):
    """CBOW神经网络模型
    结构：输入层 -> 隐藏层(嵌入平均) -> 输出层
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Args:
            vocab_size: 词汇表大小
            embed_dim: 词向量维度
        """
        super().__init__()
        # 输入到隐藏层的权重矩阵 (嵌入矩阵)
        self.embeddings = nn.Linear(vocab_size, embed_dim, bias=False)
        # 隐藏层到输出的权重矩阵
        self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, context_words: torch.Tensor) -> torch.Tensor:
        """前向传播
        Args:
            context_words: 上下文词的one-hot张量 [num_context, vocab_size]
        Returns:
            预测的目标词概率分布 [1, vocab_size]
        """
        # 获取所有上下文词的嵌入 [num_context, embed_dim]
        embeddings = self.embeddings(context_words)
        # 平均嵌入得到隐藏层表示 [embed_dim]
        hidden = torch.mean(embeddings, dim=0)
        # 生成输出预测 [1, vocab_size]
        output = self.decoder(hidden.unsqueeze(0))
        return output


if __name__ == "__main__":
    # 创建训练数据并展示样例
    cbow_pairs = create_cbow_pairs(sentences)
    print("\nCBOW Data Sample:")
    for pair in cbow_pairs[:3]:
        print(f"- Target Word: {pair[0]:<8} Context: {', '.join(pair[1])}")

    # 模型参数设置
    EMBED_DIM = 2  # 选择2维便于可视化
    model = CBOW(vocab_size, EMBED_DIM)
    print("\nModel Structure:")
    print(model)

    LEARNING_RATE = 0.1  # 增大学习率以加快收敛
    EPOCHS = 1000
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print("\nTraining started...")
    loss_history = []

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for target, context in cbow_pairs:
            # 准备输入输出数据
            X = torch.stack([get_one_hot(w) for w in context]
                            ).float()  # [num_context, vocab_size]
            y_true = torch.tensor([word_to_idx[target]], dtype=torch.long)

            # 前向传播
            y_pred = model(X)
            loss = loss_fn(y_pred, y_true)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 记录平均损失
        avg_loss = total_loss / len(cbow_pairs)
        loss_history.append(avg_loss)

        # 每100轮打印进度
        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.4f}")

    print("\nWord Embeddings after Training:")
    for word, idx in word_to_idx.items():
        embed = model.embeddings.weight[:, idx].detach().numpy()
        print(f"{word:<10}: {embed.round(3)}")
