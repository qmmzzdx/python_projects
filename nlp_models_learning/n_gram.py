from collections import defaultdict, Counter
from typing import List, Tuple, Dict

# 英文示例数据集
corpus = [
    "I like eating apples",
    "I like eating bananas",
    "She likes eating grapes",
    "He dislikes eating bananas",
    "He likes eating apples",
    "She loves eating strawberries"
]

# 分词函数（按空格分割为单词）


def tokenize(text: str) -> List[str]:
    """将文本分割为单词列表"""
    return text.split()

# n-gram计数函数


def count_ngrams(corpus: List[str], n: int) -> Dict[Tuple[str, ...], Counter]:
    """
    统计语料库中所有n-gram的出现频率
    Args:
        corpus: 文本列表
        n: n-gram的阶数
    Returns:
        字典格式的n-gram计数，键为前缀元组，值为Counter对象
    """
    ngram_counts = defaultdict(Counter)
    for text in corpus:
        tokens = tokenize(text)
        # 滑动窗口生成n-gram
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])            # 将n-gram转换为不可变元组
            prefix, target = ngram[:-1], ngram[-1]  # 分离前缀和目标词
            ngram_counts[prefix][target] += 1       # 更新计数
    return ngram_counts

# 概率计算函数


def calculate_probabilities(ngram_counts: Dict[Tuple[str, ...], Counter]) -> Dict[Tuple[str, ...], Dict[str, float]]:
    """
    将n-gram计数转换为概率分布
    Args:
        ngram_counts: count_ngrams()的输出
    Returns:
        字典格式的条件概率分布，P(target | prefix)
    """
    return {
        prefix: {word: count/total for word, count in counter.items()}
        for prefix, counter in ngram_counts.items()
        if (total := sum(counter.values())) > 0  # 使用海象运算符计算总数
    }

# 文本生成函数


def generate_text(
    prefix: str,
    probs: Dict[Tuple[str, ...], Dict[str, float]],
    n: int,
    max_length: int = 8
) -> str:
    """
    基于n-gram概率生成文本
    Args:
        prefix: 起始前缀字符串
        probs: 概率字典
        n: n-gram阶数
        max_length: 最大生成长度
    Returns:
        生成的文本字符串
    """
    tokens = tokenize(prefix)
    # 检查前缀长度是否足够
    if len(tokens) < n-1:
        return "Prefix too short for n-gram model"

    # 迭代生成直到达到最大长度
    for _ in range(max_length - len(tokens)):
        current_prefix = tuple(tokens[-(n-1):])  # 取最后n-1个词作为当前前缀
        # 选择概率最高的下一个词
        next_word = max(
            probs.get(current_prefix, {}),
            key=probs.get(current_prefix, {}).get,
            default=None
        )
        if not next_word:  # 没有可用候选词时停止
            break
        tokens.append(next_word)
    return " ".join(tokens)


if __name__ == "__main__":
    # 打印分词结果
    print("Word lists:")
    for text in corpus:
        print(tokenize(text))

    # 统计bigram词频
    bigram_counts = count_ngrams(corpus, 2)
    print("\nBigram frequencies:")
    for prefix, counts in bigram_counts.items():
        print(f"{' '.join(prefix)}: {dict(counts)}")

    # 计算bigram概率
    bigram_probs = calculate_probabilities(bigram_counts)
    print("\nBigram probabilities:")
    for prefix, probs in bigram_probs.items():
        print(f"{' '.join(prefix)}: {probs}")

    # 测试文本生成
    test_prefix = "I"
    generated = generate_text(test_prefix, bigram_probs, 2)
    print(f"\nGenerated text: {generated}")

    # 预期输出示例：
    # Generated text: I like eating apples
