import torch
import torch.nn as nn
import torch.optim as optim

# 定义训练语料
sentences = [
    "I am a student",
    "He is a customer",
    "She is a customer",
    "mzzdx is author",
    "Xjzs is mzzdx",
]


def build_vocabulary(sentences):
    """构建词汇表和索引映射"""
    # 合并句子并拆分单词
    words = ' '.join(sentences).split()
    # 创建唯一词汇表
    unique_words = list(set(words))
    # 生成双向映射字典
    word_to_index = {word: idx for idx, word in enumerate(unique_words)}
    index_to_word = {idx: word for idx, word in enumerate(unique_words)}
    return unique_words, word_to_index, index_to_word


def generate_skipgram_data(sentences, word_to_idx, window_size=2):
    """生成Skip-Gram格式的训练数据"""
    training_pairs = []

    for sentence in sentences:
        tokens = sentence.split()
        for center_pos in range(len(tokens)):
            # 获取中心词
            center_word = tokens[center_pos]
            center_idx = word_to_idx[center_word]

            # 定义上下文窗口边界
            start = max(0, center_pos - window_size)
            end = min(len(tokens), center_pos + window_size + 1)

            # 收集上下文词
            for context_pos in range(start, end):
                if context_pos != center_pos:  # 排除中心词本身
                    context_word = tokens[context_pos]
                    context_idx = word_to_idx[context_word]
                    training_pairs.append((center_idx, context_idx))

    return training_pairs


class SkipGramModel(nn.Module):
    """Skip-Gram模型架构"""

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # 词嵌入层（中心词->嵌入）
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 输出层（嵌入->上下文词概率）
        self.projection = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, center_word):
        """前向传播过程"""
        embedded = self.embeddings(center_word)
        logits = self.projection(embedded)
        return logits


if __name__ == "__main__":
    # 构建词汇表
    vocab, word2idx, idx2word = build_vocabulary(sentences)
    VOCAB_SIZE = len(vocab)

    print("Vocabulary:", vocab)
    print("Vocabulary size:", VOCAB_SIZE)

    # 生成训练数据（直接使用索引格式）
    skipgram_data = generate_skipgram_data(sentences, word2idx)
    print("\nSkip-Gram training samples (first 3):",
          [(idx2word[c], idx2word[t]) for c, t in skipgram_data[:3]])

    # 初始化模型参数
    EMBEDDING_DIM = 2
    model = SkipGramModel(VOCAB_SIZE, EMBEDDING_DIM)
    print("\nModel structure:", model)

    LEARNING_RATE = 0.025  # 优化学习率
    EPOCHS = 1000          # 训练轮数

    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # 记录训练过程的损失值
    training_loss = []

    print("\nTraining started...")
    for epoch in range(EPOCHS):
        total_loss = 0

        for center_idx, context_idx in skipgram_data:
            # 准备输入输出数据
            input_tensor = torch.tensor([center_idx], dtype=torch.long)
            target_tensor = torch.tensor([context_idx], dtype=torch.long)

            # 前向传播
            predictions = model(input_tensor)
            loss = loss_function(predictions, target_tensor)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 记录平均损失
        avg_loss = total_loss / len(skipgram_data)
        training_loss.append(avg_loss)

        # 每100轮输出训练状态
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Average loss: {avg_loss:.4f}")

    print("\nWord embeddings after training:")
    for word in vocab:
        idx = word2idx[word]
        embedding = model.embeddings.weight[idx].detach().numpy()
        print(f"{word:8}: {embedding.round(4)}")
