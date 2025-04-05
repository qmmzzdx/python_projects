import numpy as np
from typing import List, Dict, Tuple

# 英文示例数据集
corpus = [
    "I really really love watching movies",
    "This movie is truly a great movie",
    "The weather today is really nice weather",
    "I went to see a movie today",
    "The movies in cinema are all very good"
]


def tokenize_english(text: str) -> List[str]:
    """英文分词处理（简单按空格分割）"""
    return text.lower().split()


def build_bow_vectors(corpus: List[str]) -> Tuple[Dict[str, int], List[List[int]]]:
    """
    构建词袋模型向量

    参数:
        corpus: 原始文本列表

    返回:
        vocabulary: 词汇表字典 {单词: 索引}
        bow_vectors: 词袋向量列表
    """
    # 分词处理
    tokenized_corpus = [tokenize_english(text) for text in corpus]

    # 构建词汇表
    vocabulary = {}
    for sentence in tokenized_corpus:
        for word in sentence:
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)

    # 生成词袋向量
    bow_vectors = []
    for sentence in tokenized_corpus:
        vector = [0] * len(vocabulary)
        for word in sentence:
            vector[vocabulary[word]] += 1
        bow_vectors.append(vector)

    return vocabulary, bow_vectors


def cosine_similarity(vec_a: List[int], vec_b: List[int]) -> float:
    """计算两个向量的余弦相似度"""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    vocab, bow_vectors = build_bow_vectors(corpus)

    print("\n[Vocabulary Index Mapping]")
    for word, idx in sorted(vocab.items(), key=lambda x: x[1]):
        print(f"{word:<12} -> {idx}")

    print("\n[Bag-of-Words Vectors]")
    for i, vec in enumerate(bow_vectors, 1):
        print(f"Text {i}: {vec}")

    similarity_matrix = np.zeros((len(corpus), len(corpus)))
    for i in range(len(corpus)):
        for j in range(len(corpus)):
            similarity_matrix[i][j] = cosine_similarity(
                bow_vectors[i], bow_vectors[j])

    print("\n[Cosine Similarity Matrix]")
    print(np.round(similarity_matrix, 2))
